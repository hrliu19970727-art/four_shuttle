# main.py
import simpy
import threading
import time
import random
import numpy as np
import logging
from map import warehouse
from path_planning import improved_a_star_search, is_valid_position, manhattan_distance
from scheduling import Task
from visualization import run_visualization
from config import *
from shared_state import shared_state, state_lock

# --- 引入 DRL 模块 ---
try:
    from idqn_agent import IDQNAgent, MAX_OBSERVED_TASKS

    DRL_ENABLED = True
except ImportError:
    print("未找到 idqn_agent.py 或 torch，将回退到默认逻辑。")
    DRL_ENABLED = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
SIMULATION_DELAY = 0.05
simulation_ready = threading.Event()
visualization_started = threading.Event()

# 坐标定义
INPUT_POINTS = [(ROWS - 13, 8 - 1), (ROWS - 20, 6 - 1)]
OUTPUT_POINTS = [(ROWS - 14, 47 - 1), (ROWS - 20, 49 - 1)]

# 初始化全局 Agent (所有小车共享大脑，即参数共享的 IDQN)
if DRL_ENABLED:
    global_agent = IDQNAgent()


def get_nearest(points, current_pos):
    best_p = points[0]
    min_dist = float('inf')
    for p in points:
        d = manhattan_distance(current_pos, p)
        if d < min_dist:
            min_dist = d
            best_p = p
    return best_p


def is_occupied(pos, my_id):
    with state_lock:
        for s in shared_state["shuttles"]:
            if s["id"] != my_id and tuple(s["pos"]) == pos:
                return True
    return False


def initialize_shared_state():
    with state_lock:
        valid_pos = []
        for r in range(ROWS):
            for c in range(COLS):
                if warehouse.grid_type[r][c] == TYPE_STORAGE:
                    valid_pos.append((r, c))
        random.shuffle(valid_pos)
        chosen = valid_pos[:SHUTTLES]

        shared_state["shuttles"] = [
            {"id": i, "pos": list(p), "busy": False, "Load": False, "path": [], "current_task": None, "status": "空闲",
             "battery": 100}
            for i, p in enumerate(chosen)
        ]

        shared_state["slots"] = warehouse.slots.tolist()
        shared_state["release_tasks"] = []
        shared_state["pick_tasks"] = []
        shared_state["completed_tasks"] = 0
        shared_state["failed_tasks"] = 0
        shared_state["completed_release_tasks"] = 0
        shared_state["completed_pick_tasks"] = 0
        shared_state["total_planning_time"] = 0.0
        shared_state["planning_count"] = 0
        shared_state["total_path_length"] = 0
        shared_state["path_count"] = 0
        shared_state["time"] = 0
        shared_state["done"] = False
        print(f"初始化完成: {SHUTTLES}辆小车 (DRL Mode: {DRL_ENABLED})")


def generate_tasks(env):
    """
    【高负载版】任务生成器
    用于压力测试算法性能
    """
    print(">>> 高负载任务生成器启动 (High Load Mode)")

    # 初始爆发：一开始就生成一批任务，让小车动起来
    yield env.timeout(1)
    with state_lock:
        print("--- 生成初始波任务 ---")
        # 初始填充 10 个任务
        for _ in range(10):
            # (复用下方的生成逻辑，这里简化处理，依靠循环即可)
            pass

    while not shared_state["done"]:
        # 1. 极短的生成间隔 (0.5 ~ 2秒)
        yield env.timeout(random.uniform(0.5, 2.0))

        if shared_state["done"]: break

        with state_lock:
            # 统计空闲和占用货位
            empties = []
            filled = []
            for r in range(ROWS):
                for c in range(COLS):
                    if warehouse.grid_type[r][c] == TYPE_STORAGE:
                        slot_status = shared_state["slots"][r][c]
                        if slot_status == 1:
                            empties.append((r, c))
                        elif slot_status == 2:
                            filled.append((r, c))

            # 过滤掉已经是任务目标的点
            active_pos = set()
            for t in shared_state["release_tasks"] + shared_state["pick_tasks"]:
                active_pos.add(t.position)
            for s in shared_state["shuttles"]:
                if s["current_task"]: active_pos.add(s["current_task"].position)

            empties = [p for p in empties if p not in active_pos]
            filled = [p for p in filled if p not in active_pos]

            current_task_count = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])

            # 2. 提高队列上限 (例如允许积压 50 个任务)
            if current_task_count < 50:

                # 3. 批量生成 (每次随机生成 1-3 个任务)
                batch_size = random.randint(1, 3)

                for _ in range(batch_size):
                    # 重新检查剩余可用位置 (防止一波生成把位置耗尽报错)
                    if not empties and not filled: break

                    # 随机逻辑：70%概率生成入库(填满仓库)，30%出库
                    # 或者如果空位很少，强制出库；货很少，强制入库
                    is_release = False

                    if not filled:
                        is_release = True
                    elif not empties:
                        is_release = False
                    else:
                        is_release = (random.random() < 0.7)

                    if is_release and empties:
                        pos = random.choice(empties)
                        shared_state["release_tasks"].append(Task(0, pos, "release"))
                        empties.remove(pos)  # 避免同批次重复
                        print(f"[压力测试] 新增入库 -> {pos}")
                    elif not is_release and filled:
                        pos = random.choice(filled)
                        shared_state["pick_tasks"].append(Task(0, pos, "pick"))
                        filled.remove(pos)
                        print(f"[压力测试] 新增出库 -> {pos}")


def move_shuttle(env, s_id, path):
    if not path: return
    start_idx = 1 if path[0] == tuple(shared_state["shuttles"][s_id]["pos"]) else 0
    actual_path = path[start_idx:]

    for i, step in enumerate(actual_path):
        while True:
            if not is_occupied(step, s_id):
                move_success = False
                with state_lock:
                    if not is_occupied(step, s_id):
                        shared_state["shuttles"][s_id]["pos"] = list(step)
                        shared_state["shuttles"][s_id]["path"] = actual_path[i:]
                        shared_state["shuttles"][s_id]["status"] = "移动中"
                        shared_state["shuttles"][s_id]["battery"] -= 0.05
                        move_success = True
                if move_success: break
            yield env.timeout(0.5)

        yield env.timeout(MOVE_TIME)
        if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY)


def execute_op_sim(env, s_id, duration, status_text):
    with state_lock:
        shared_state["shuttles"][s_id]["status"] = status_text
    yield env.timeout(duration)
    if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY * 2)


def run_path_planning(start, end, others, obstacles):
    t_start = time.perf_counter()
    path = improved_a_star_search(warehouse.grid_type, start, end, others, obstacles)
    t_end = time.perf_counter()
    with state_lock:
        shared_state["total_planning_time"] += (t_end - t_start) * 1000
        shared_state["planning_count"] += 1
        if path:
            shared_state["total_path_length"] += len(path)
            shared_state["path_count"] += 1
    return path


def get_planning_params(s_id):
    with state_lock:
        c = tuple(shared_state["shuttles"][s_id]["pos"])
        oth = [s["path"] for s in shared_state["shuttles"] if s["id"] != s_id]
        obs = [tuple(s["pos"]) for s in shared_state["shuttles"] if s["id"] != s_id]
    return c, oth, obs


# --- DRL 辅助函数 ---
def construct_state(s_id, available_tasks):
    """
    构建状态向量: [Self_X, Self_Y, Battery, Task1_Features..., TaskN_Features...]
    Task Features: [Rel_X, Rel_Y, Type(0/1), Distance]
    """
    with state_lock:
        s_info = shared_state["shuttles"][s_id]
        sr, sc = s_info["pos"]
        bat = s_info["battery"]

    state = [sr, sc, bat]

    # 填充任务特征
    for i in range(MAX_OBSERVED_TASKS):
        if i < len(available_tasks):
            t = available_tasks[i]
            tr, tc = t.position
            t_type = 0 if t.task_type == "release" else 1
            dist = abs(sr - tr) + abs(sc - tc)
            state.extend([tr - sr, tc - sc, t_type, dist])
        else:
            # Padding (无任务补0)
            state.extend([0, 0, -1, 0])

    return np.array(state, dtype=np.float32)


def controller(env, s_id, input_locks, output_locks):
    yield env.timeout(random.random())

    while True:
        with state_lock:
            if shared_state["done"]: break

        # --- DRL 任务选择逻辑 ---
        task = None
        task_idx = -1
        t_type = None
        state_vec = None

        # 1. 获取所有候选任务
        with state_lock:
            # 合并两个队列供 AI 选择
            candidates = shared_state["pick_tasks"] + shared_state["release_tasks"]
            # 限制候选数量，防止输入过大
            candidates = candidates[:MAX_OBSERVED_TASKS]

        if not candidates:
            with state_lock: shared_state["shuttles"][s_id]["status"] = "待命"
            yield env.timeout(1)
            continue

        if DRL_ENABLED:
            # 2. 构建状态并询问 AI
            state_vec = construct_state(s_id, candidates)
            valid_actions = list(range(len(candidates)))
            action_idx = global_agent.select_action(state_vec, valid_actions)

            # 3. 根据 AI 选择取任务
            if action_idx < len(candidates):
                task = candidates[action_idx]
                # 从原队列中移除
                with state_lock:
                    if task in shared_state["pick_tasks"]:
                        shared_state["pick_tasks"].remove(task)
                        t_type = "pick"
                    elif task in shared_state["release_tasks"]:
                        shared_state["release_tasks"].remove(task)
                        t_type = "release"
            else:
                # AI 选了无效动作 (padding区域)，随机分配一个兜底
                task = candidates[0]
                # (同上移除逻辑...)
                with state_lock:
                    if task in shared_state["pick_tasks"]:
                        shared_state["pick_tasks"].pop(0)
                        t_type = "pick"
                    elif task in shared_state["release_tasks"]:
                        shared_state["release_tasks"].pop(0)
                        t_type = "release"
        else:
            # 回退到旧逻辑 (FIFO)
            with state_lock:
                if shared_state["pick_tasks"]:
                    task = shared_state["pick_tasks"].pop(0)
                    t_type = "pick"
                elif shared_state["release_tasks"]:
                    task = shared_state["release_tasks"].pop(0)
                    t_type = "release"

        # --- 执行任务 ---
        start_time = env.now
        try:
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = True
                shared_state["shuttles"][s_id]["current_task"] = task

            success = False

            # ... (执行逻辑与之前相同，调用 A*) ...
            # 为了简洁，此处直接复用之前的执行逻辑
            # === 入库流程 ===
            if t_type == "release":
                cur, _, _ = get_planning_params(s_id)
                target_port = get_nearest(INPUT_POINTS, cur)

                print(f"小车{s_id} (DRL) 选定入库任务 -> 请求锁...")
                with input_locks[target_port].request() as req:
                    yield req
                    cur, others, obs = get_planning_params(s_id)
                    path1 = run_path_planning(cur, target_port, others, obs)
                    if path1:
                        yield env.process(move_shuttle(env, s_id, path1))
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                        with state_lock:
                            shared_state["shuttles"][s_id]["Load"] = True

                        cur, others, obs = get_planning_params(s_id)
                        path2 = run_path_planning(cur, task.position, others, obs)
                        if path2:
                            if len(path2) > 1:
                                yield env.process(move_shuttle(env, s_id, path2[:2]))
                                remaining = path2[1:]
                            else:
                                remaining = []
                            success = True

                if success and remaining:
                    yield env.process(move_shuttle(env, s_id, remaining))
                    yield env.process(execute_op_sim(env, s_id, PARK_TIME, "放货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 2
                        warehouse.update_slot(r, c, 2)
                        shared_state["shuttles"][s_id]["Load"] = False

            # === 出库流程 ===
            elif t_type == "pick":
                cur, others, obs = get_planning_params(s_id)
                print(f"小车{s_id} (DRL) 选定出库任务")
                path1 = run_path_planning(cur, task.position, others, obs)
                if path1:
                    yield env.process(move_shuttle(env, s_id, path1))
                    yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "取货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 1
                        warehouse.update_slot(r, c, 1)
                        shared_state["shuttles"][s_id]["Load"] = True

                    cur, _, _ = get_planning_params(s_id)
                    target_port = get_nearest(OUTPUT_POINTS, cur)
                    with output_locks[target_port].request() as req:
                        yield req
                        cur, others, obs = get_planning_params(s_id)
                        path2 = run_path_planning(cur, target_port, others, obs)
                        if path2:
                            yield env.process(move_shuttle(env, s_id, path2))
                            yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "卸货"))
                            with state_lock: shared_state["shuttles"][s_id]["Load"] = False
                            success = True

                            yield env.timeout(0.5)
                            # 驶离逻辑 (简略)
                            with state_lock:
                                cur_pos = tuple(shared_state["shuttles"][s_id]["pos"])
                                # ... (同之前的驶离代码)

            # --- DRL 训练步骤 ---
            end_time = env.now
            # 奖励设计: 时间越短奖励越高 (负值)
            # 例如: 耗时 50s -> Reward = -50
            reward = -(end_time - start_time)

            if DRL_ENABLED and state_vec is not None:
                # 获取完成后的新状态 (Next State)
                # 注意: 这里的 next_state 其实是新的任务列表状态，为了简化，我们重新采样一次
                with state_lock:
                    next_candidates = shared_state["pick_tasks"] + shared_state["release_tasks"]
                    next_candidates = next_candidates[:MAX_OBSERVED_TASKS]
                next_state_vec = construct_state(s_id, next_candidates)

                # 存入经验并学习
                global_agent.store_transition(state_vec, action_idx, reward, next_state_vec)
                global_agent.learn()

            # 结算
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = False
                shared_state["shuttles"][s_id]["current_task"] = None
                if success:
                    shared_state["completed_tasks"] += 1
                    if t_type == "release":
                        shared_state["completed_release_tasks"] += 1
                    else:
                        shared_state["completed_pick_tasks"] += 1
                    print(f"小车{s_id} 完成 (Reward: {reward:.1f})")
                else:
                    shared_state["failed_tasks"] += 1
                    # 失败惩罚
                    if DRL_ENABLED: global_agent.store_transition(state_vec, action_idx, -200, state_vec)
                    if t_type == "pick":
                        shared_state["pick_tasks"].append(task)
                    else:
                        shared_state["release_tasks"].append(task)

        except Exception as e:
            print(f"Err: {e}")
            import traceback
            traceback.print_exc()


def run_sim():
    with state_lock:
        shared_state["simulation_started"] = True
    env = simpy.Environment()

    in_locks = {pt: simpy.Resource(env, capacity=1) for pt in INPUT_POINTS}
    out_locks = {pt: simpy.Resource(env, capacity=1) for pt in OUTPUT_POINTS}

    for i in range(SHUTTLES):
        env.process(controller(env, i, in_locks, out_locks))

    env.process(generate_tasks(env))

    def monitor():
        while not shared_state["done"]:
            with state_lock: shared_state["time"] = env.now
            yield env.timeout(0.5)

    env.process(monitor())

    simulation_ready.set()
    while not visualization_started.is_set(): time.sleep(0.1)
    try:
        env.run(until=3600)
    finally:
        with state_lock:
            shared_state["done"] = True


def run_viz():
    visualization_started.set()
    run_visualization()


if __name__ == "__main__":
    initialize_shared_state()
    t = threading.Thread(target=run_sim, daemon=True)
    t.start()
    if simulation_ready.wait(5): run_viz()
    t.join(1)