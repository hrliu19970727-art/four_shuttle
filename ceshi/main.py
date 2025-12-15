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
SIMULATION_DELAY = 0.3
simulation_ready = threading.Event()
visualization_started = threading.Event()


def initialize_shared_state():
    with state_lock:
        # 初始位置：选择存储巷道 (TYPE_STORAGE)，避开主干道
        valid_pos = []
        for r in range(1, ROWS - 1):
            for c in range(1, COLS - 1):
                if warehouse.grid_type[r][c] == TYPE_STORAGE:
                    valid_pos.append((r, c))

        # 随机打乱并选择
        random.shuffle(valid_pos)
        chosen = valid_pos[:SHUTTLES]

        shared_state["shuttles"] = [
            {"id": i, "pos": list(p), "busy": False, "Load": False, "path": [], "current_task": None, "status": "空闲",
             "battery": 100}
            for i, p in enumerate(chosen)
        ]

        # 同步货位状态
        shared_state["slots"] = warehouse.slots.tolist()

        shared_state["release_tasks"] = []
        shared_state["pick_tasks"] = []
        shared_state["completed_tasks"] = 0
        shared_state["failed_tasks"] = 0
        shared_state["time"] = 0
        shared_state["done"] = False

        print(f"初始化完成: {SHUTTLES}辆小车")


def get_available_slots(status_req):
    """获取可用货位: status_req=1(找空位放货), status_req=2(找有货取货)"""
    avail = []
    with state_lock:
        active = set()
        active.update(t.position for t in shared_state["release_tasks"])
        active.update(t.position for t in shared_state["pick_tasks"])
        for s in shared_state["shuttles"]:
            if s.get("current_task"): active.add(s["current_task"].position)

        for r in range(ROWS):
            for c in range(COLS):
                # 必须是存储巷道且状态匹配
                if (warehouse.grid_type[r][c] == TYPE_STORAGE and
                        shared_state["slots"][r][c] == status_req and
                        (r, c) not in active):
                    avail.append((r, c))
    return avail


def generate_tasks(env):
    print("任务生成器启动")
    while not shared_state["done"]:
        yield env.timeout(random.randint(5, 10))
        if shared_state["done"]: break

        # 动态平衡存取
        empties = get_available_slots(1)
        filled = get_available_slots(2)

        cnt = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
        if cnt > 6: continue

        with state_lock:
            # 如果仓库空位多(<20个有货)，优先生成放货
            occupancy = sum(1 for row in shared_state["slots"] for c in row if c == 2)

            if occupancy < 20 or (empties and random.random() < 0.6):
                if empties:
                    pos = random.choice(empties)
                    shared_state["release_tasks"].append(Task(0, pos, "release"))
                    print(f"新放货: {pos}")
            elif filled:
                pos = random.choice(filled)
                shared_state["pick_tasks"].append(Task(0, pos, "pick"))
                print(f"新取货: {pos}")


def move_shuttle(env, s_id, path):
    if not path: return
    start = 1 if path[0] == tuple(shared_state["shuttles"][s_id]["pos"]) else 0
    actual = path[start:]

    for i, step in enumerate(actual):
        with state_lock:
            shared_state["shuttles"][s_id]["pos"] = list(step)
            shared_state["shuttles"][s_id]["path"] = actual[i:]
            shared_state["shuttles"][s_id]["status"] = "移动中"
            shared_state["shuttles"][s_id]["battery"] -= 0.05

        yield env.timeout(MOVE_TIME)
        if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY)


def execute_op(env, s_id, task, t_type):
    r, c = task.position
    # 验证
    with state_lock:
        cur = tuple(shared_state["shuttles"][s_id]["pos"])
        if manhattan_distance(cur, task.position) > 1: return False

        target_st = shared_state["slots"][r][c]
        if t_type == "release" and target_st != 1: return False
        if t_type == "pick" and target_st != 2: return False

        shared_state["shuttles"][s_id]["status"] = f"执行{t_type}"

    t_cost = PARK_TIME if t_type == "release" else RETRIEVE_TIME
    yield env.timeout(t_cost)
    if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY * 2)

    with state_lock:
        new_st = 2 if t_type == "release" else 1
        shared_state["slots"][r][c] = new_st
        warehouse.update_slot(r, c, new_st)  # 同步到地图对象

        if t_type == "release":
            shared_state["completed_tasks"] += 1
            print(f"小车{s_id} 放货成功 {task.position}")
        else:
            shared_state["completed_tasks"] += 1
            print(f"小车{s_id} 取货成功 {task.position}")
    return True


def controller(env, s_id):
    yield env.timeout(random.random())
    while True:
        with state_lock:
            if shared_state["done"]: break
            bat = shared_state["shuttles"][s_id]["battery"]
            if bat < 10:
                shared_state["shuttles"][s_id]["status"] = "充电中"
                yield env.timeout(5)
                shared_state["shuttles"][s_id]["battery"] = 100
                continue

        # 取任务
        task, t_type = None, None
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

        # 规划
        with state_lock:
            shared_state["shuttles"][s_id]["busy"] = True
            shared_state["shuttles"][s_id]["current_task"] = task
            shared_state["shuttles"][s_id]["Load"] = (t_type == "release")
            cur = tuple(shared_state["shuttles"][s_id]["pos"])

            others = [s["path"] for s in shared_state["shuttles"] if s["id"] != s_id and s["path"]]

        # 传入 grid_type 进行交通规则感知规划
        path = improved_a_star_search(warehouse.grid_type, cur, task.position, others)

        success = False
        if path:
            yield env.process(move_shuttle(env, s_id, path))
            success = yield env.process(execute_op(env, s_id, task, t_type))
        else:
            print(f"小车{s_id} 规划失败 -> {task.position}")
            with state_lock:
                if t_type == "pick":
                    shared_state["pick_tasks"].append(task)
                else:
                    shared_state["release_tasks"].append(task)

        with state_lock:
            shared_state["shuttles"][s_id]["busy"] = False
            shared_state["shuttles"][s_id]["current_task"] = None
            if not success: shared_state["failed_tasks"] += 1


def run_sim():
    with state_lock:
        shared_state["simulation_started"] = True
    env = simpy.Environment()

    # 初始任务
    initial_pos = [tuple(s["pos"]) for s in shared_state["shuttles"]]
    # ... (简化的初始生成)

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
        env.run(until=600)
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