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

# --- 坐标定义 (系统坐标系) ---
# 需要将用户坐标 (行, 列) 转换为系统坐标 (ROW-r, c-1)
# 入库点: (13, 8), (20, 6)
INPUT_POINTS = [
    (ROWS - 13, 8 - 1),
    (ROWS - 20, 6 - 1)
]
# 出库点: (14, 47), (20, 49)
OUTPUT_POINTS = [
    (ROWS - 14, 47 - 1),
    (ROWS - 20, 49 - 1)
]


def get_nearest(points, current_pos):
    """找到最近的入/出库点"""
    best_p = points[0]
    min_dist = float('inf')
    for p in points:
        d = manhattan_distance(current_pos, p)
        if d < min_dist:
            min_dist = d
            best_p = p
    return best_p


def initialize_shared_state():
    with state_lock:
        # 初始位置：随机选择存储巷道
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
        shared_state["time"] = 0
        shared_state["done"] = False

        print(f"初始化完成: {SHUTTLES}辆小车")


def generate_tasks(env):
    """任务生成器"""
    print("任务生成器启动")
    while not shared_state["done"]:
        yield env.timeout(random.randint(5, 15))
        if shared_state["done"]: break

        # 简单的任务生成逻辑
        with state_lock:
            # 统计空闲和占用货位
            empties = []
            filled = []
            for r in range(ROWS):
                for c in range(COLS):
                    if warehouse.grid_type[r][c] == TYPE_STORAGE:
                        if shared_state["slots"][r][c] == 1:
                            empties.append((r, c))
                        elif shared_state["slots"][r][c] == 2:
                            filled.append((r, c))

            # 避免重复任务
            active_pos = set()
            for t in shared_state["release_tasks"] + shared_state["pick_tasks"]:
                active_pos.add(t.position)
            for s in shared_state["shuttles"]:
                if s["current_task"]: active_pos.add(s["current_task"].position)

            empties = [p for p in empties if p not in active_pos]
            filled = [p for p in filled if p not in active_pos]

            cnt = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
            if cnt < 10:
                # 随机生成入库或出库
                if empties and random.random() < 0.6:
                    pos = random.choice(empties)
                    shared_state["release_tasks"].append(Task(0, pos, "release"))
                    print(f"新入库任务: 目标{pos}")
                elif filled:
                    pos = random.choice(filled)
                    shared_state["pick_tasks"].append(Task(0, pos, "pick"))
                    print(f"新出库任务: 来源{pos}")


def move_shuttle(env, s_id, path):
    """执行移动"""
    if not path: return
    # 如果已经在起点，跳过第一个点
    start_idx = 1 if path[0] == tuple(shared_state["shuttles"][s_id]["pos"]) else 0
    actual_path = path[start_idx:]

    for i, step in enumerate(actual_path):
        with state_lock:
            shared_state["shuttles"][s_id]["pos"] = list(step)
            shared_state["shuttles"][s_id]["path"] = actual_path[i:]
            shared_state["shuttles"][s_id]["status"] = "移动中"
            shared_state["shuttles"][s_id]["battery"] -= 0.05

        yield env.timeout(MOVE_TIME)
        if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY)


def execute_op_sim(env, s_id, duration, status_text):
    """模拟操作耗时（无状态改变）"""
    with state_lock:
        shared_state["shuttles"][s_id]["status"] = status_text
    yield env.timeout(duration)
    if SIMULATION_DELAY: time.sleep(SIMULATION_DELAY * 2)


def controller(env, s_id):
    """
    小车控制器 - 实现两段式逻辑
    入库: 1.去入库点(取) -> 2.去货位(放)
    出库: 1.去货位(取) -> 2.去出库点(放)
    """
    yield env.timeout(random.random())

    while True:
        with state_lock:
            if shared_state["done"]: break

        # 1. 获取任务
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

        # 2. 执行任务逻辑
        try:
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = True
                shared_state["shuttles"][s_id]["current_task"] = task

            success = False

            # --- 入库流程 (Release) ---
            if t_type == "release":
                # 第一步: 去入库点取货
                with state_lock:
                    cur = tuple(shared_state["shuttles"][s_id]["pos"])
                    input_pt = get_nearest(INPUT_POINTS, cur)
                    others = [s["path"] for s in shared_state["shuttles"] if s["id"] != s_id]

                print(f"小车{s_id} (入库) -> 前往入库点 {input_pt}")
                path1 = improved_a_star_search(warehouse.grid_type, cur, input_pt, others)

                if path1:
                    yield env.process(move_shuttle(env, s_id, path1))
                    yield env.process(execute_op_sim(env, s_id, PARK_TIME, "入库点取货"))
                    with state_lock:
                        shared_state["shuttles"][s_id]["Load"] = True

                    # 第二步: 去存储位放货
                    with state_lock:
                        cur = tuple(shared_state["shuttles"][s_id]["pos"])
                        others = [s["path"] for s in shared_state["shuttles"] if s["id"] != s_id]

                    print(f"小车{s_id} (入库) -> 前往货位 {task.position}")
                    path2 = improved_a_star_search(warehouse.grid_type, cur, task.position, others)

                    if path2:
                        yield env.process(move_shuttle(env, s_id, path2))
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "货位放货"))
                        # 更新货位状态
                        with state_lock:
                            r, c = task.position
                            shared_state["slots"][r][c] = 2  # 占用
                            warehouse.update_slot(r, c, 2)
                            shared_state["shuttles"][s_id]["Load"] = False
                        success = True

            # --- 出库流程 (Pick) ---
            elif t_type == "pick":
                # 第一步: 去存储位取货
                with state_lock:
                    cur = tuple(shared_state["shuttles"][s_id]["pos"])
                    others = [s["path"] for s in shared_state["shuttles"] if s["id"] != s_id]

                print(f"小车{s_id} (出库) -> 前往货位 {task.position}")
                path1 = improved_a_star_search(warehouse.grid_type, cur, task.position, others)

                if path1:
                    yield env.process(move_shuttle(env, s_id, path1))
                    yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "货位取货"))

                    # 更新货位状态 (取走变空)
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 1  # 空闲
                        warehouse.update_slot(r, c, 1)
                        shared_state["shuttles"][s_id]["Load"] = True

                    # 第二步: 去出库点放货
                    with state_lock:
                        cur = tuple(shared_state["shuttles"][s_id]["pos"])
                        output_pt = get_nearest(OUTPUT_POINTS, cur)
                        others = [s["path"] for s in shared_state["shuttles"] if s["id"] != s_id]

                    print(f"小车{s_id} (出库) -> 前往出库点 {output_pt}")
                    path2 = improved_a_star_search(warehouse.grid_type, cur, output_pt, others)

                    if path2:
                        yield env.process(move_shuttle(env, s_id, path2))
                        yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "出库点卸货"))
                        with state_lock: shared_state["shuttles"][s_id]["Load"] = False
                        success = True

            # 3. 结算
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = False
                shared_state["shuttles"][s_id]["current_task"] = None
                if success:
                    shared_state["completed_tasks"] += 1
                    print(f"小车{s_id} 任务完成")
                else:
                    shared_state["failed_tasks"] += 1
                    print(f"小车{s_id} 任务失败 (路径规划或状态错误)")
                    # 失败任务回退
                    if t_type == "pick":
                        shared_state["pick_tasks"].append(task)
                    else:
                        shared_state["release_tasks"].append(task)

        except Exception as e:
            print(f"Controller Error: {e}")
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