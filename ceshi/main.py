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
SIMULATION_DELAY = 0.0001
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
        shared_state["completed_release_tasks"] = 0  # 新增初始化
        shared_state["completed_pick_tasks"] = 0  # 新增初始化
        shared_state["time"] = 0
        shared_state["done"] = False
        print(f"初始化完成: {SHUTTLES}辆小车")


def generate_tasks(env):
    print("任务生成器启动")
    while not shared_state["done"]:
        yield env.timeout(random.randint(5, 15))
        if shared_state["done"]: break
        with state_lock:
            empties = []
            filled = []
            for r in range(ROWS):
                for c in range(COLS):
                    if warehouse.grid_type[r][c] == TYPE_STORAGE:
                        if shared_state["slots"][r][c] == 1:
                            empties.append((r, c))
                        elif shared_state["slots"][r][c] == 2:
                            filled.append((r, c))

            active_pos = set()
            for t in shared_state["release_tasks"] + shared_state["pick_tasks"]: active_pos.add(t.position)
            for s in shared_state["shuttles"]:
                if s["current_task"]: active_pos.add(s["current_task"].position)

            empties = [p for p in empties if p not in active_pos]
            filled = [p for p in filled if p not in active_pos]

            cnt = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
            if cnt < 6:
                if empties and random.random() < 0.6:
                    pos = random.choice(empties)
                    shared_state["release_tasks"].append(Task(0, pos, "release"))
                    print(f"新入库任务: 目标{pos}")
                elif filled:
                    pos = random.choice(filled)
                    shared_state["pick_tasks"].append(Task(0, pos, "pick"))
                    print(f"新出库任务: 来源{pos}")


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


def controller(env, s_id):
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

            def get_planning_params(target_pos):
                with state_lock:
                    c_pos = tuple(shared_state["shuttles"][s_id]["pos"])
                    others_p = [s["path"] for s in shared_state["shuttles"] if s["id"] != s_id]
                    obstacles = [tuple(s["pos"]) for s in shared_state["shuttles"] if s["id"] != s_id]
                return c_pos, others_p, obstacles

            if t_type == "release":
                cur, others, obs = get_planning_params(INPUT_POINTS[0])
                input_pt = get_nearest(INPUT_POINTS, cur)
                print(f"小车{s_id} (入库) -> 入库点 {input_pt}")
                path1 = improved_a_star_search(warehouse.grid_type, cur, input_pt, others, obs)

                if path1:
                    yield env.process(move_shuttle(env, s_id, path1))
                    yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                    with state_lock:
                        shared_state["shuttles"][s_id]["Load"] = True

                    cur, others, obs = get_planning_params(task.position)
                    print(f"小车{s_id} (入库) -> 货位 {task.position}")
                    path2 = improved_a_star_search(warehouse.grid_type, cur, task.position, others, obs)

                    if path2:
                        yield env.process(move_shuttle(env, s_id, path2))
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "放货"))
                        with state_lock:
                            r, c = task.position
                            shared_state["slots"][r][c] = 2
                            warehouse.update_slot(r, c, 2)
                            shared_state["shuttles"][s_id]["Load"] = False
                        success = True

            elif t_type == "pick":
                cur, others, obs = get_planning_params(task.position)
                print(f"小车{s_id} (出库) -> 货位 {task.position}")
                path1 = improved_a_star_search(warehouse.grid_type, cur, task.position, others, obs)

                if path1:
                    yield env.process(move_shuttle(env, s_id, path1))
                    yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "取货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 1
                        warehouse.update_slot(r, c, 1)
                        shared_state["shuttles"][s_id]["Load"] = True

                    cur, others, obs = get_planning_params(OUTPUT_POINTS[0])
                    output_pt = get_nearest(OUTPUT_POINTS, cur)
                    print(f"小车{s_id} (出库) -> 出库点 {output_pt}")
                    path2 = improved_a_star_search(warehouse.grid_type, cur, output_pt, others, obs)

                    if path2:
                        yield env.process(move_shuttle(env, s_id, path2))
                        yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "卸货"))
                        with state_lock: shared_state["shuttles"][s_id]["Load"] = False
                        success = True

            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = False
                shared_state["shuttles"][s_id]["current_task"] = None
                if success:
                    shared_state["completed_tasks"] += 1
                    # --- 更新：分别统计 ---
                    if t_type == "release":
                        shared_state["completed_release_tasks"] += 1
                    else:
                        shared_state["completed_pick_tasks"] += 1
                    # --------------------
                    print(f"小车{s_id} 完成")
                else:
                    shared_state["failed_tasks"] += 1
                    print(f"小车{s_id} 失败")
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
    for i in range(SHUTTLES): env.process(controller(env, i))
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