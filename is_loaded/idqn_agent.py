# idqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 超参数配置 ---
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
MEMORY_CAPACITY = 2000
TARGET_UPDATE = 100

# 设定最大观测任务数 (神经网络输入固定维度)
MAX_OBSERVED_TASKS = 10
# 状态维度: 穿梭车坐标(2) + 电量(1) + 每个任务的特征(4: r, c, type, dist) * MAX_TASKS
STATE_DIM = 3 + 4 * MAX_OBSERVED_TASKS
# 动作维度: 选第几个任务 (0 ~ MAX_TASKS-1)
ACTION_DIM = MAX_OBSERVED_TASKS


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, ACTION_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class IDQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.epsilon = EPSILON_START
        self.learn_step_counter = 0

    def select_action(self, state, valid_actions):
        """
        根据状态选择动作 (即选择执行哪个任务)
        valid_actions: 当前有效的任务索引列表 (例如 [0, 1, 2])，防止选择不存在的任务
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                # Masking: 将无效动作的 Q 值设为负无穷，确保不被选中
                full_mask = torch.full((1, ACTION_DIM), -float('inf')).to(self.device)
                for act in valid_actions:
                    if act < ACTION_DIM:
                        full_mask[0][act] = 0  # 有效位置设为 0 (加法不影响)

                masked_q = q_values + full_mask
                return masked_q.argmax().item()
        else:
            # 随机探索
            return random.choice(valid_actions) if valid_actions else 0

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        state, action, reward, next_state = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)

        # 计算当前 Q 值
        q_eval = self.policy_net(state).gather(1, action)

        # 计算目标 Q 值
        q_next = self.target_net(next_state).detach()
        q_target = reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = nn.MSELoss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 Epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # 更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())