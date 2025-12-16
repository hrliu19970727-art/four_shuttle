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
SIMULATION_DELAY = 0.001
simulation_ready = threading.Event()
visualization_started = threading.Event()

INPUT_POINTS = [(ROWS - 13, 8 - 1), (ROWS - 20, 6 - 1)]
OUTPUT_POINTS = [(ROWS - 14, 47 - 1), (ROWS - 20, 49 - 1)]

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
        print(f"初始化完成: {SHUTTLES}辆小车")


def generate_tasks(env):
    print(">>> 任务生成器启动")
    while not shared_state["done"]:
        yield env.timeout(random.uniform(1.0, 3.0))
        if shared_state["done"]: break

        with state_lock:
            empties = []
            filled = []
            for r in range(ROWS):
                for c in range(COLS):
                    if warehouse.grid_type[r][c] == TYPE_STORAGE:
                        st = shared_state["slots"][r][c]
                        if st == 1:
                            empties.append((r, c))
                        elif st == 2:
                            filled.append((r, c))

            active_pos = set()
            for t in shared_state["release_tasks"] + shared_state["pick_tasks"]:
                active_pos.add(t.position)
            for s in shared_state["shuttles"]:
                if s["current_task"]: active_pos.add(s["current_task"].position)

            empties = [p for p in empties if p not in active_pos]
            filled = [p for p in filled if p not in active_pos]

            cnt = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
            if cnt < 20:
                if empties and random.random() < 0.6:
                    pos = random.choice(empties)
                    shared_state["release_tasks"].append(Task(0, pos, "release"))
                    print(f"新入库 -> {pos}")
                elif filled:
                    pos = random.choice(filled)
                    shared_state["pick_tasks"].append(Task(0, pos, "pick"))
                    print(f"新出库 -> {pos}")


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
            if wait_time > 2.0: return False

        yield env.timeout(MOVE_TIME)
        if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY)

    return True


def execute_op_sim(env, s_id, duration, status_text):
    with state_lock:
        shared_state["shuttles"][s_id]["status"] = status_text
    yield env.timeout(duration)
    if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY * 2)


def run_path_planning(s_id, start, end, others, obstacles):
    """
    统一规划入口，获取当前状态并调用 A*
    """
    t_start = time.perf_counter()

    # 获取实时货位状态和载重状态
    with state_lock:
        current_slots = np.array(shared_state["slots"])  # 转为numpy方便传递
        is_loaded = shared_state["shuttles"][s_id]["Load"]

    path = improved_a_star_search(
        warehouse.grid_type,
        start,
        end,
        other_paths=others,
        reservations=obstacles,
        slots_map=current_slots,  # 传入货位地图
        is_loaded=is_loaded  # 传入载重状态
    )

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
        # 构建动态障碍物
        reservations = set()
        for s in shared_state["shuttles"]:
            if s["id"] == s_id: continue
            # 静态位置锁定
            other_pos = tuple(s["pos"])
            for t in range(20): reservations.add((other_pos[0], other_pos[1], t))
            # 动态路径锁定
            if s["path"]:
                for t, p in enumerate(s["path"]):
                    reservations.add((p[0], p[1], t))
                    reservations.add((p[0], p[1], t + 1))

    return c, oth, reservations


def robust_move(env, s_id, target_pos):
    max_retries = 5
    for attempt in range(max_retries):
        cur, others, obs = get_planning_params(s_id)
        # 传入 s_id
        path = run_path_planning(s_id, cur, target_pos, others, obs)

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


def controller(env, s_id, input_locks, output_locks):
    yield env.timeout(random.random())

    while True:
        with state_lock:
            if shared_state["done"]: break

        task = None
        state_vec = None

        with state_lock:
            candidates = shared_state["pick_tasks"] + shared_state["release_tasks"]
            candidates = candidates[:MAX_OBSERVED_TASKS]

        if not candidates:
            with state_lock: shared_state["shuttles"][s_id]["status"] = "待命"
            yield env.timeout(1)
            continue

        action_idx = 0
        if DRL_ENABLED:
            state_vec = construct_state(s_id, candidates)
            valid_actions = list(range(len(candidates)))
            action_idx = global_agent.select_action(state_vec, valid_actions)

            if action_idx < len(candidates):
                task = candidates[action_idx]
            else:
                task = candidates[0]
        else:
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
                # 任务可能被别人抢走了
                continue

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

                print(f"小车{s_id} 选定入库 -> 请求锁...")
                with input_locks[target_port].request() as req:
                    yield req
                    # 1. 去入库点
                    reached_port = yield env.process(robust_move(env, s_id, target_port))

                    if reached_port:
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                        with state_lock:
                            shared_state["shuttles"][s_id]["Load"] = True

                        # 2. 去货位
                        reached_slot = yield env.process(robust_move(env, s_id, task.position))
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
                # 1. 去货位
                reached_slot = yield env.process(robust_move(env, s_id, task.position))
                if reached_slot:
                    yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "取货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 1
                        warehouse.update_slot(r, c, 1)
                        shared_state["shuttles"][s_id]["Load"] = True

                    # 2. 去出库点
                    cur, _, _ = get_planning_params(s_id)
                    target_port = get_nearest(OUTPUT_POINTS, cur)
                    with output_locks[target_port].request() as req:
                        yield req
                        reached_port = yield env.process(robust_move(env, s_id, target_port))

                        if reached_port:
                            yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "卸货"))
                            with state_lock:
                                shared_state["shuttles"][s_id]["Load"] = False
                            success = True

                            yield env.timeout(0.5)
                            # 驶离
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
                                env.process(robust_move(env, s_id, rest_pos))

            # --- DRL 训练 ---
            end_time = env.now
            reward = -(end_time - start_time)

            if DRL_ENABLED and state_vec is not None:
                with state_lock:
                    next_candidates = shared_state["pick_tasks"] + shared_state["release_tasks"]
                    next_candidates = next_candidates[:MAX_OBSERVED_TASKS]
                next_state_vec = construct_state(s_id, next_candidates)
                global_agent.store_transition(state_vec, action_idx, reward, next_state_vec)
                global_agent.learn()

            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = False
                shared_state["shuttles"][s_id]["current_task"] = None
                if success:
                    shared_state["completed_tasks"] += 1
                    if t_type == "release":
                        shared_state["completed_release_tasks"] += 1
                    else:
                        shared_state["completed_pick_tasks"] += 1
                    print(f"小车{s_id} 完成")
                else:
                    shared_state["failed_tasks"] += 1
                    if DRL_ENABLED: global_agent.store_transition(state_vec, action_idx, -200, state_vec)
                    # 归还任务
                    if t_type == "pick":
                        shared_state["pick_tasks"].append(task)
                    elif t_type == "release":
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