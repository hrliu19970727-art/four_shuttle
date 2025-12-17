# simulation_core.py
import simpy
import threading
import time
import random
import numpy as np
import logging
from map import warehouse
# 引入不同算法
from path_planning import improved_a_star_search  # A*

# 确保你创建了 path_bfs.py
try:
    from path_bfs import bfs_search
except ImportError:
    bfs_search = None

from scheduling import Task
from visualization import run_visualization
from config import *
from shared_state import shared_state, state_lock

# DRL 可选
try:
    from idqn_agent import IDQNAgent, MAX_OBSERVED_TASKS

    DRL_ENABLED = True
except ImportError:
    DRL_ENABLED = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
SIMULATION_DELAY = 0.001  # 极速模式，便于快速跑完数据
simulation_ready = threading.Event()
visualization_started = threading.Event()

INPUT_POINTS = [(ROWS - 13, 8 - 1), (ROWS - 20, 6 - 1)]
OUTPUT_POINTS = [(ROWS - 14, 47 - 1), (ROWS - 20, 49 - 1)]

# 全局 Agent (如果是 IDQN 模式)
global_agent = IDQNAgent() if DRL_ENABLED else None


def get_nearest(points, current_pos):
    # 简单曼哈顿距离
    best_p = points[0]
    min_dist = float('inf')
    for p in points:
        d = abs(current_pos[0] - p[0]) + abs(current_pos[1] - p[1])
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


def move_shuttle(env, s_id, path):
    if not path: return True
    start_idx = 1 if path[0] == tuple(shared_state["shuttles"][s_id]["pos"]) else 0
    actual_path = path[start_idx:]

    for i, step in enumerate(actual_path):
        wait_time = 0
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
            wait_time += 0.5
            if wait_time > 3.0: return False  # 超时判死锁

        yield env.timeout(MOVE_TIME)
        if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY)

    return True


def execute_op_sim(env, s_id, duration, status_text):
    with state_lock:
        shared_state["shuttles"][s_id]["status"] = status_text
    yield env.timeout(duration)
    if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY * 2)


# === 核心算法分发器 ===
def run_path_planning(s_id, start, end, others, obstacles, algorithm):
    """根据指定的算法执行路径规划"""
    t_start = time.perf_counter()

    path = []
    if algorithm == "BFS":
        if bfs_search:
            # BFS 只需要静态障碍 (obstacles)
            # 注意：obstacles 是 (r,c,t) 集合，BFS 需要过滤掉 t 维度，或者只看当前障碍
            # 为了能在 BFS 中复用 obstacles，我们需要将其转换为 2D 坐标集合
            # 这里简单做：只要某个坐标在未来 10 步内被占，就视为不可通行（悲观策略）
            static_obs = set()
            for (r, c, t) in obstacles:
                if t < 20:  # 只关心近期的障碍
                    static_obs.add((r, c))

            path = bfs_search(warehouse.grid_type, start, end, obstacles=static_obs)
        else:
            print("Error: path_bfs.py not found!")
            path = []

    elif algorithm == "A_STAR" or algorithm == "IDQN":
        # A* 考虑时空预约和载重
        with state_lock:
            current_slots = np.array(shared_state["slots"])
            is_loaded = shared_state["shuttles"][s_id]["Load"]

        path = improved_a_star_search(
            warehouse.grid_type, start, end,
            other_paths=others, reservations=obstacles,
            slots_map=current_slots, is_loaded=is_loaded
        )

    t_end = time.perf_counter()

    # 统计规划耗时
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
        obs = set()
        for s in shared_state["shuttles"]:
            if s["id"] == s_id: continue
            other_pos = tuple(s["pos"])
            # 简单碰撞体积：锁定当前位置及未来一小段时间
            for t in range(10): obs.add((other_pos[0], other_pos[1], t))
            if s["path"]:
                for t, p in enumerate(s["path"]):
                    obs.add((p[0], p[1], t))
                    obs.add((p[0], p[1], t + 1))
    return c, oth, obs


def robust_move(env, s_id, target_pos, algorithm):
    """封装移动逻辑，支持重规划"""
    max_retries = 5
    for attempt in range(max_retries):
        cur, others, obs = get_planning_params(s_id)
        # 传入算法参数
        path = run_path_planning(s_id, cur, target_pos, others, obs, algorithm)

        if not path:
            yield env.timeout(1.0)
            continue

        success = yield env.process(move_shuttle(env, s_id, path))
        if success:
            return True
        else:
            continue

    return False


def construct_state(s_id, available_tasks):
    with state_lock:
        s_info = shared_state["shuttles"][s_id]
        sr, sc = s_info["pos"]
        bat = s_info["battery"]

    state = [sr, sc, bat]
    for i in range(MAX_OBSERVED_TASKS):
        if i < len(available_tasks):
            t = available_tasks[i]
            tr, tc = t.position
            t_type = 0 if t.task_type == "release" else 1
            dist = abs(sr - tr) + abs(sc - tc)
            state.extend([tr - sr, tc - sc, t_type, dist])
        else:
            state.extend([0, 0, -1, 0])
    return np.array(state, dtype=np.float32)


# === 任务分发器 (替代原来的 generate_tasks) ===
def task_dispatcher(env, task_list):
    """将预生成的任务列表按需下发"""
    print(f">>> [Core] 任务分发器启动，共 {len(task_list)} 个任务待执行")
    yield env.timeout(1)

    queue_limit = 20  # 允许的任务积压上限

    while task_list or len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"]) > 0:
        with state_lock:
            # 检查是否全部做完
            total_processed = shared_state["completed_tasks"] + shared_state["failed_tasks"]
            if not task_list and total_processed >= TOTAL_TASKS_COUNT_GLOBAL:
                shared_state["done"] = True
                break

            current_backlog = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])

            if current_backlog < queue_limit and task_list:
                # 补充任务
                batch = min(5, len(task_list))
                for _ in range(batch):
                    t = task_list.pop(0)
                    if t.task_type == "release":
                        shared_state["release_tasks"].append(t)
                    else:
                        shared_state["pick_tasks"].append(t)

        yield env.timeout(1.0)


def controller(env, s_id, input_locks, output_locks, algorithm):
    """控制器，接收 algorithm 参数"""
    yield env.timeout(random.random())

    while True:
        with state_lock:
            if shared_state["done"]: break

        # --- 任务选择逻辑 ---
        task = None
        state_vec = None
        action_idx = 0

        with state_lock:
            candidates = shared_state["pick_tasks"] + shared_state["release_tasks"]
            candidates = candidates[:MAX_OBSERVED_TASKS]

        if not candidates:
            with state_lock: shared_state["shuttles"][s_id]["status"] = "待命"
            yield env.timeout(1)
            continue

        # IDQN 特有的任务选择
        if algorithm == "IDQN" and DRL_ENABLED:
            state_vec = construct_state(s_id, candidates)
            valid_actions = list(range(len(candidates)))
            action_idx = global_agent.select_action(state_vec, valid_actions)
            if action_idx < len(candidates):
                task = candidates[action_idx]
            else:
                task = candidates[0]
        else:
            # BFS / A* 默认 FIFO
            task = candidates[0]

        # 锁定并移除任务
        t_type = None
        with state_lock:
            if task in shared_state["pick_tasks"]:
                shared_state["pick_tasks"].remove(task)
                t_type = "pick"
            elif task in shared_state["release_tasks"]:
                shared_state["release_tasks"].remove(task)
                t_type = "release"
            else:
                continue  # 被人抢了

        start_time = env.now
        try:
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = True
                shared_state["shuttles"][s_id]["current_task"] = task

            success = False

            # === 入库流程 ===
            if t_type == "release":
                cur, _, _ = get_planning_params(s_id)
                target_port = get_nearest(INPUT_POINTS, cur)

                # print(f"小车{s_id} 入库 -> 申请锁")
                with input_locks[target_port].request() as req:
                    yield req
                    # 1. 去端口 (【修复】传入 algorithm 参数)
                    reached_port = yield env.process(robust_move(env, s_id, target_port, algorithm))

                    if reached_port:
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                        with state_lock:
                            shared_state["shuttles"][s_id]["Load"] = True

                        # 2. 去货位 (【修复】传入 algorithm 参数)
                        reached_slot = yield env.process(robust_move(env, s_id, task.position, algorithm))
                        if reached_slot:
                            success = True

                if success:
                    yield env.process(execute_op_sim(env, s_id, PARK_TIME, "放货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 2
                        warehouse.update_slot(r, c, 2)
                        shared_state["shuttles"][s_id]["Load"] = False

            # === 出库流程 ===
            elif t_type == "pick":
                # 1. 去货位 (【修复】传入 algorithm 参数)
                reached_slot = yield env.process(robust_move(env, s_id, task.position, algorithm))
                if reached_slot:
                    yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "取货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 1
                        warehouse.update_slot(r, c, 1)
                        shared_state["shuttles"][s_id]["Load"] = True

                    # 2. 去端口
                    cur, _, _ = get_planning_params(s_id)
                    target_port = get_nearest(OUTPUT_POINTS, cur)
                    with output_locks[target_port].request() as req:
                        yield req
                        # 3. 去出库点 (【修复】传入 algorithm 参数)
                        reached_port = yield env.process(robust_move(env, s_id, target_port, algorithm))

                        if reached_port:
                            yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "卸货"))
                            with state_lock:
                                shared_state["shuttles"][s_id]["Load"] = False
                            success = True

                            # 必须驶离
                            yield env.timeout(0.5)
                            rest_pos = None
                            with state_lock:
                                cur = tuple(shared_state["shuttles"][s_id]["pos"])
                                for dr in range(-5, 6):
                                    for dc in range(-5, 6):
                                        nr, nc = cur[0] + dr, cur[1] + dc
                                        if 0 <= nr < ROWS and 0 <= nc < COLS and warehouse.grid_type[nr][
                                            nc] == TYPE_STORAGE:
                                            rest_pos = (nr, nc)
                                            break
                                    if rest_pos: break
                            if rest_pos:
                                # 4. 驶离 (【修复】传入 algorithm 参数)
                                env.process(robust_move(env, s_id, rest_pos, algorithm))

            # --- DRL 训练 (仅 IDQN 模式) ---
            if algorithm == "IDQN" and DRL_ENABLED:
                end_time = env.now
                reward = -(end_time - start_time)
                if state_vec is not None:
                    with state_lock:
                        next_cands = shared_state["pick_tasks"] + shared_state["release_tasks"]
                        next_cands = next_cands[:MAX_OBSERVED_TASKS]
                    next_state_vec = construct_state(s_id, next_cands)
                    global_agent.store_transition(state_vec, action_idx, reward, next_state_vec)
                    global_agent.learn()

            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = False
                shared_state["shuttles"][s_id]["current_task"] = None
                if success:
                    shared_state["completed_tasks"] += 1
                else:
                    shared_state["failed_tasks"] += 1
                    # 回退
                    if t_type == "pick":
                        shared_state["pick_tasks"].append(task)
                    elif t_type == "release":
                        shared_state["release_tasks"].append(task)

        except Exception as e:
            print(f"Err: {e}")
            import traceback
            traceback.print_exc()


# === 核心入口函数 ===
TOTAL_TASKS_COUNT_GLOBAL = 0


def run_comparison_sim(algorithm_name, task_list_copy):
    """
    运行一次完整的仿真
    :param algorithm_name: "BFS", "A_STAR", "IDQN"
    :param task_list_copy: 任务列表
    :return: 统计结果 dict
    """
    global TOTAL_TASKS_COUNT_GLOBAL
    TOTAL_TASKS_COUNT_GLOBAL = len(task_list_copy)

    # 1. 重置全局状态
    with state_lock:
        valid_pos = []
        for r in range(ROWS):
            for c in range(COLS):
                if warehouse.grid_type[r][c] == TYPE_STORAGE:
                    valid_pos.append((r, c))
        random.shuffle(valid_pos)

        shared_state["shuttles"] = [
            {"id": i, "pos": list(valid_pos[i]), "busy": False, "Load": False, "path": [], "current_task": None,
             "status": "空闲", "battery": 100}
            for i in range(SHUTTLES)
        ]
        # 重置统计
        shared_state["release_tasks"] = []
        shared_state["pick_tasks"] = []
        shared_state["completed_tasks"] = 0
        shared_state["failed_tasks"] = 0
        shared_state["total_planning_time"] = 0.0
        shared_state["planning_count"] = 0
        shared_state["total_path_length"] = 0
        shared_state["path_count"] = 0
        shared_state["time"] = 0
        shared_state["done"] = False

        # 重置货位状态
        shared_state["slots"] = [[1 if warehouse.grid_type[r][c] == TYPE_STORAGE else 0 for c in range(COLS)] for r in
                                 range(ROWS)]
        for t in task_list_copy:
            if t.task_type == "pick":
                r, c = t.position
                shared_state["slots"][r][c] = 2

    print(f"\n====== [SimCore] 启动仿真: {algorithm_name} ======")
    env = simpy.Environment()

    in_locks = {pt: simpy.Resource(env, capacity=1) for pt in INPUT_POINTS}
    out_locks = {pt: simpy.Resource(env, capacity=1) for pt in OUTPUT_POINTS}

    # 启动控制器
    for i in range(SHUTTLES):
        env.process(controller(env, i, in_locks, out_locks, algorithm_name))

    # 启动任务分发
    env.process(task_dispatcher(env, task_list_copy))

    # 运行
    try:
        env.run(until=5000)
    except Exception as e:
        print(f"Sim Error: {e}")

    # 收集结果
    result = {
        "Algorithm": algorithm_name,
        "Completed": shared_state["completed_tasks"],
        "Failed": shared_state["failed_tasks"],
        "Time_Sim": env.now,
        "Avg_Plan_Time_ms": (shared_state["total_planning_time"] / shared_state["planning_count"]) if shared_state[
            "planning_count"] else 0,
        "Avg_Path_Len": (shared_state["total_path_length"] / shared_state["path_count"]) if shared_state[
            "path_count"] else 0
    }
    return result