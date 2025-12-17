# simulation_core.py
import simpy
import threading
import time
import random
import numpy as np
import logging
from map import warehouse

# --- 引入不同算法 ---
from path_planning import improved_a_star_search  # 改进版 A*

try:
    from path_standard_a_star import standard_a_star_search  # 标准版 A*
except ImportError:
    standard_a_star_search = None
try:
    from path_bfs import bfs_search  # BFS
except ImportError:
    bfs_search = None

from scheduling import Task
from visualization import run_visualization
from config import *
from shared_state import shared_state, state_lock

# DRL 可选配置
try:
    from idqn_agent import IDQNAgent, MAX_OBSERVED_TASKS

    DRL_ENABLED = True
except ImportError:
    DRL_ENABLED = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
SIMULATION_DELAY = 0.001
simulation_ready = threading.Event()
visualization_started = threading.Event()

INPUT_POINTS = [(ROWS - 13, 8 - 1), (ROWS - 20, 6 - 1)]
OUTPUT_POINTS = [(ROWS - 14, 47 - 1), (ROWS - 20, 49 - 1)]

global_agent = IDQNAgent() if DRL_ENABLED else None


# === 基础辅助函数 ===
def get_nearest(points, current_pos):
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
    """
    执行移动动作
    返回: True(成功), False(阻塞超时)
    """
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
            # 超时判断: 如果卡住太久，返回 False 触发重规划
            if wait_time > 3.0: return False

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
    """
    根据 algorithm 参数调度不同的路径规划函数
    """
    t_start = time.perf_counter()
    path = []

    # 统一获取物理状态 (所有算法都受物理约束限制)
    with state_lock:
        current_slots = np.array(shared_state["slots"])
        is_loaded = shared_state["shuttles"][s_id]["Load"]

    # 1. 普通 A* (Baseline)
    if algorithm == "STANDARD_A*" or algorithm == "BFS":
        # 这里为了简单，把 BFS 归类到标准寻路，或者你可以单独处理
        if algorithm == "STANDARD_A*":
            if standard_a_star_search:
                # 传入 slots_map 和 is_loaded 以遵守物理规则
                path = standard_a_star_search(
                    warehouse.grid_type, start, end,
                    slots_map=current_slots, is_loaded=is_loaded
                )
            else:
                print("Error: standard_a_star_search not imported")
        elif algorithm == "BFS":
            if bfs_search:
                # BFS 也可以加上物理约束，这里假设 bfs_search 已实现或暂时只用障碍物过滤
                # 简单处理：将有货的格子加入静态 obstacles
                bfs_obs = set()
                # 将动态障碍转为静态点
                for (r, c, t) in obstacles:
                    if t < 5: bfs_obs.add((r, c))
                # 将满货位加入障碍 (如果载货)
                if is_loaded:
                    for r in range(ROWS):
                        for c in range(COLS):
                            if current_slots[r][c] == 2 and (r, c) != end:
                                bfs_obs.add((r, c))
                path = bfs_search(warehouse.grid_type, start, end, obstacles=bfs_obs)

    # 2. 改进 A* (Ours) / IDQN
    elif algorithm == "IMPROVED_A*" or algorithm == "IDQN":
        path = improved_a_star_search(
            warehouse.grid_type, start, end,
            other_paths=others, reservations=obstacles,
            slots_map=current_slots, is_loaded=is_loaded
        )

    t_end = time.perf_counter()

    # 统计
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
            for t in range(10): obs.add((other_pos[0], other_pos[1], t))
            if s["path"]:
                for t, p in enumerate(s["path"]):
                    obs.add((p[0], p[1], t))
                    obs.add((p[0], p[1], t + 1))
    return c, oth, obs


def robust_move(env, s_id, target_pos, algorithm):
    """
    封装移动逻辑，支持重规划
    【重要】必须接收 algorithm 参数并传递给 run_path_planning
    """
    max_retries = 5
    for attempt in range(max_retries):
        cur, others, obs = get_planning_params(s_id)

        # 传入 algorithm
        path = run_path_planning(s_id, cur, target_pos, others, obs, algorithm)

        if not path:
            # 无解时等待
            yield env.timeout(1.0)
            continue

        # 执行移动
        success = yield env.process(move_shuttle(env, s_id, path))
        if success:
            return True
        else:
            # 移动失败(拥堵)，重试
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


def task_dispatcher(env, task_list):
    print(f">>> [Core] 任务分发器启动，共 {len(task_list)} 个任务待执行")
    yield env.timeout(1)
    queue_limit = 20

    while task_list or len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"]) > 0:
        with state_lock:
            total_processed = shared_state["completed_tasks"] + shared_state["failed_tasks"]
            if not task_list and total_processed >= TOTAL_TASKS_COUNT_GLOBAL:
                shared_state["done"] = True
                break

            current_backlog = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
            if current_backlog < queue_limit and task_list:
                batch = min(5, len(task_list))
                for _ in range(batch):
                    t = task_list.pop(0)
                    if t.task_type == "release":
                        shared_state["release_tasks"].append(t)
                    else:
                        shared_state["pick_tasks"].append(t)
        yield env.timeout(1.0)


def controller(env, s_id, input_locks, output_locks, algorithm):
    """
    小车控制器
    【重要】接收 algorithm 参数并透传给 robust_move
    """
    yield env.timeout(random.random())
    while True:
        with state_lock:
            if shared_state["done"]: break

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

        if algorithm == "IDQN" and DRL_ENABLED:
            state_vec = construct_state(s_id, candidates)
            valid_actions = list(range(len(candidates)))
            action_idx = global_agent.select_action(state_vec, valid_actions)
            task = candidates[action_idx] if action_idx < len(candidates) else candidates[0]
        else:
            task = candidates[0]

        t_type = None
        with state_lock:
            if task in shared_state["pick_tasks"]:
                shared_state["pick_tasks"].remove(task)
                t_type = "pick"
            elif task in shared_state["release_tasks"]:
                shared_state["release_tasks"].remove(task)
                t_type = "release"
            else:
                continue

        start_time = env.now
        try:
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = True
                shared_state["shuttles"][s_id]["current_task"] = task

            success = False

            if t_type == "release":
                cur, _, _ = get_planning_params(s_id)
                target_port = get_nearest(INPUT_POINTS, cur)

                with input_locks[target_port].request() as req:
                    yield req
                    # 传入 algorithm
                    if yield env.process(robust_move(env, s_id, target_port, algorithm)):
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                        with state_lock:
                            shared_state["shuttles"][s_id]["Load"] = True
                        # 传入 algorithm
                        if yield env.process(robust_move(env, s_id, task.position, algorithm)):
                            success = True
                if success:
                    yield env.process(execute_op_sim(env, s_id, PARK_TIME, "放货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 2
                        warehouse.update_slot(r, c, 2)
                        shared_state["shuttles"][s_id]["Load"] = False

            elif t_type == "pick":
                # 传入 algorithm
                if yield env.process(robust_move(env, s_id, task.position, algorithm)):
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
                        # 传入 algorithm
                        if yield env.process(robust_move(env, s_id, target_port, algorithm)):
                            yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "卸货"))
                            with state_lock:
                                shared_state["shuttles"][s_id]["Load"] = False
                            success = True
                            yield env.timeout(0.5)

                            # 驶离逻辑 (传入 algorithm)
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
                                env.process(robust_move(env, s_id, rest_pos, algorithm))

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
                    if t_type == "pick":
                        shared_state["pick_tasks"].append(task)
                    elif t_type == "release":
                        shared_state["release_tasks"].append(task)

        except Exception as e:
            print(f"Err: {e}")
            import traceback
            traceback.print_exc()


TOTAL_TASKS_COUNT_GLOBAL = 0


def run_comparison_sim(algorithm_name, task_list_copy):
    global TOTAL_TASKS_COUNT_GLOBAL
    TOTAL_TASKS_COUNT_GLOBAL = len(task_list_copy)

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

    for i in range(SHUTTLES):
        env.process(controller(env, i, in_locks, out_locks, algorithm_name))

    env.process(task_dispatcher(env, task_list_copy))

    try:
        env.run(until=5000)
    except Exception as e:
        print(f"Sim Error: {e}")

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