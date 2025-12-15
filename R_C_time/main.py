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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
SIMULATION_DELAY = 0.05
simulation_ready = threading.Event()
visualization_started = threading.Event()

INPUT_POINTS = [(ROWS - 13, 8 - 1), (ROWS - 20, 6 - 1)]
OUTPUT_POINTS = [(ROWS - 14, 47 - 1), (ROWS - 20, 49 - 1)]


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
    print(">>> 任务生成器启动 (平衡模式)")
    while not shared_state["done"]:
        yield env.timeout(random.uniform(2.0, 5.0))  # 降低频率，给车辆留出通行时间
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
            # 限制队列长度为 8，防止过度拥堵
            if cnt < 8:
                if empties and random.random() < 0.6:
                    pos = random.choice(empties)
                    shared_state["release_tasks"].append(Task(0, pos, "release"))
                    print(f"新入库 -> {pos}")
                elif filled:
                    pos = random.choice(filled)
                    shared_state["pick_tasks"].append(Task(0, pos, "pick"))
                    print(f"新出库 -> {pos}")


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


def controller(env, s_id, input_locks, output_locks):
    yield env.timeout(random.random())

    while True:
        with state_lock:
            if shared_state["done"]: break

        task = None
        t_type = None
        with state_lock:
            if shared_state["pick_tasks"]:
                task = shared_state["pick_tasks"].pop(0)
                t_type = "pick"
            elif shared_state["release_tasks"]:
                task = shared_state["release_tasks"].pop(0)
                t_type = "release"

        if not task:
            with state_lock: shared_state["shuttles"][s_id]["status"] = "待命"
            yield env.timeout(1)
            continue

        try:
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = True
                shared_state["shuttles"][s_id]["current_task"] = task

            success = False

            # === 入库流程 ===
            if t_type == "release":
                cur, _, _ = get_planning_params(s_id)
                target_port = get_nearest(INPUT_POINTS, cur)

                print(f"小车{s_id} 等待入库锁 {target_port}...")
                with state_lock:
                    shared_state["shuttles"][s_id]["status"] = "排队中"

                with input_locks[target_port].request() as req:
                    yield req

                    # 1. 前往入库点
                    cur, others, obs = get_planning_params(s_id)
                    path1 = run_path_planning(cur, target_port, others, obs)
                    if path1:
                        yield env.process(move_shuttle(env, s_id, path1))
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                        with state_lock:
                            shared_state["shuttles"][s_id]["Load"] = True

                        # 2. 规划前往货位
                        cur, others, obs = get_planning_params(s_id)
                        path2 = run_path_planning(cur, task.position, others, obs)

                        if path2:
                            # 【优化】先移出入库点一步，然后立即释放锁
                            # 这样下一辆车就可以开始往入库点走了
                            if len(path2) > 1:
                                first_step_path = [path2[0], path2[1]]
                                yield env.process(move_shuttle(env, s_id, first_step_path))
                                remaining_path = path2[1:]
                            else:
                                remaining_path = []  # 已经在目标点(极少见)

                            # 此处退出with块，锁自动释放
                            success = True
                        else:
                            # 规划失败，也要退出释放锁
                            success = False
                            remaining_path = []

                # 3. 在锁外完成剩余路程
                if success and remaining_path:
                    yield env.process(move_shuttle(env, s_id, remaining_path))
                    yield env.process(execute_op_sim(env, s_id, PARK_TIME, "放货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 2
                        warehouse.update_slot(r, c, 2)
                        shared_state["shuttles"][s_id]["Load"] = False

            # === 出库流程 ===
            elif t_type == "pick":
                # 1. 去货位取货
                cur, others, obs = get_planning_params(s_id)
                path1 = run_path_planning(cur, task.position, others, obs)
                if path1:
                    yield env.process(move_shuttle(env, s_id, path1))
                    yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "取货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 1
                        warehouse.update_slot(r, c, 1)
                        shared_state["shuttles"][s_id]["Load"] = True

                    # 2. 申请出库锁
                    cur, _, _ = get_planning_params(s_id)
                    target_port = get_nearest(OUTPUT_POINTS, cur)

                    print(f"小车{s_id} 等待出库锁 {target_port}...")
                    with state_lock:
                        shared_state["shuttles"][s_id]["status"] = "排队中"

                    with output_locks[target_port].request() as req:
                        yield req

                        cur, others, obs = get_planning_params(s_id)
                        path2 = run_path_planning(cur, target_port, others, obs)
                        if path2:
                            yield env.process(move_shuttle(env, s_id, path2))
                            yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "卸货"))
                            with state_lock:
                                shared_state["shuttles"][s_id]["Load"] = False
                            success = True

                            # 【优化】任务完成后，必须驶离出库点，否则会堵死下一辆车
                            # 简单策略：随机找一个附近的空闲存储位作为“休息区”
                            yield env.timeout(0.5)  # 稍微停顿
                            leave_success = False
                            with state_lock:
                                # 找一个最近的空闲存储位
                                cur_pos = tuple(shared_state["shuttles"][s_id]["pos"])
                                candidates = []
                                for r in range(max(0, cur_pos[0] - 5), min(ROWS, cur_pos[0] + 5)):
                                    for c in range(max(0, cur_pos[1] - 5), min(COLS, cur_pos[1] + 5)):
                                        if warehouse.grid_type[r][c] == TYPE_STORAGE:
                                            candidates.append((r, c))
                                if candidates:
                                    rest_pos = random.choice(candidates)
                                    leave_success = True

                            if leave_success:
                                print(f"小车{s_id} 驶离出库口 -> {rest_pos}")
                                cur, others, obs = get_planning_params(s_id)
                                path_leave = run_path_planning(cur, rest_pos, others, obs)
                                if path_leave:
                                    yield env.process(move_shuttle(env, s_id, path_leave))

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
                    print(f"小车{s_id} 任务完成")
                else:
                    shared_state["failed_tasks"] += 1
                    print(f"小车{s_id} 任务失败")
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
        env.run(until=6000)
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