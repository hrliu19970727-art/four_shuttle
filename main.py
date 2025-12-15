# main.py (逻辑修复版 - 解决放货冲突)
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
from config import ROWS, COLS, SHUTTLES, MOVE_TIME, PARK_TIME, RETRIEVE_TIME, ALLEY_COLS, ALLEY_ROWS
from shared_state import shared_state, state_lock

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
task_log = []

# 仿真延时 (秒) - 控制动画速度
SIMULATION_DELAY = 0.02

# 同步事件
simulation_ready = threading.Event()
visualization_started = threading.Event()


def debug_alley_configuration():
    """调试巷道配置"""
    print("\n=== 巷道配置调试 ===")
    print(f"巷道行: {ALLEY_ROWS}")
    print(f"巷道列: {ALLEY_COLS}")
    print(f"地图尺寸: {ROWS} x {COLS}")

    # 检查小车初始位置
    with state_lock:
        for i, shuttle in enumerate(shared_state["shuttles"]):
            pos = tuple(shuttle["pos"])
            valid = is_valid_position(pos)
            alley_info = ""
            if pos[0] in ALLEY_ROWS:
                alley_info += " (在巷道行)"
            if pos[1] in ALLEY_COLS:
                alley_info += " (在巷道列)"
            print(f"小车{i} 初始位置: {pos}, 有效: {valid}{alley_info}")

    # 检查一些关键位置
    test_positions = [
        (1, 3), (1, 10),  # 小车初始位置
        (8, 6), (8, 13),  # 之前卡住的位置
        (0, 5), (10, 5),  # 电梯位置
        (5, 5), (15, 15),  # 一般位置
        (9, 5), (5, 7)  # 巷道位置
    ]

    print("\n位置有效性检查:")
    for pos in test_positions:
        valid = is_valid_position(pos)
        alley_info = ""
        if pos[0] in ALLEY_ROWS:
            alley_info += " (巷道行)"
        if pos[1] in ALLEY_COLS:
            alley_info += " (巷道列)"
        status = "有效" if valid else "无效"
        print(f"  位置 {pos}: {status}{alley_info}")

    print("=== 巷道调试结束 ===\n")


def initialize_shared_state():
    """初始化共享状态"""
    with state_lock:
        initial_positions = [(1, 3), (1, 10), (1, 20), (1, 30)]
        valid_positions = [pos for pos in initial_positions if is_valid_position(pos)]
        if not valid_positions:
            valid_positions = []
            for row in range(1, min(5, ROWS)):
                for col in range(1, min(10, COLS), 3):
                    pos = (row, col)
                    if is_valid_position(pos) and pos not in valid_positions:
                        valid_positions.append(pos)
                    if len(valid_positions) >= SHUTTLES:
                        break
                if len(valid_positions) >= SHUTTLES:
                    break

        if not valid_positions:
            valid_positions = [(1, 2), (1, 4), (2, 2), (2, 4)]

        shared_state["shuttles"] = [
            {
                "id": i,
                "pos": list(valid_positions[i % len(valid_positions)]),
                "busy": False,
                "Load": False,
                "path": [],
                "current_task": None,
                "status": "空闲",
                "battery": 100
            }
            for i in range(SHUTTLES)
        ]

        shared_state["slots"] = [[1 for _ in range(COLS)] for _ in range(ROWS)]

        for row in ALLEY_ROWS:
            for col in range(COLS):
                if 0 <= row < ROWS and 0 <= col < COLS:
                    shared_state["slots"][row][col] = 0
        for col in ALLEY_COLS:
            for row in range(ROWS):
                if 0 <= row < ROWS and 0 <= col < COLS:
                    shared_state["slots"][row][col] = 0

        shared_state["release_tasks"] = []
        shared_state["pick_tasks"] = []
        shared_state["completed_tasks"] = 0
        shared_state["failed_tasks"] = 0
        shared_state["time"] = 0
        shared_state["done"] = False
        shared_state["simulation_started"] = False

        print(f"初始化完成: {SHUTTLES}辆穿梭车")

    debug_alley_configuration()


def calculate_available_positions():
    """
    计算可用的放货位置
    【关键修复】必须检查货位状态是否为空闲 (1)
    """
    available = []
    with state_lock:  # 加锁读取
        for row in range(ROWS):
            for col in range(COLS):
                # 检查: 1.坐标有效 2.不是电梯 3.必须是空闲状态(1)
                if (is_valid_position((row, col)) and
                        (row, col) not in [(0, 5), (10, 5)] and
                        shared_state["slots"][row][col] == 1):
                    available.append((row, col))
    return available


def get_all_active_task_positions():
    """获取所有活动任务的位置"""
    positions = set()
    with state_lock:
        positions.update(task.position for task in shared_state["release_tasks"])
        positions.update(task.position for task in shared_state["pick_tasks"])
        for shuttle in shared_state["shuttles"]:
            if shuttle.get("current_task"):
                positions.add(shuttle["current_task"].position)
    return positions


def generate_nearby_release_tasks(n, shuttle_positions):
    """生成附近的放货任务"""
    available = calculate_available_positions()  # 现在只返回真正空闲的位置

    existing_positions = get_all_active_task_positions()
    available = [pos for pos in available if pos not in existing_positions]

    if not available:
        return []

    tasks = []
    for i in range(min(n, len(available))):
        nearby_positions = []
        for pos in available:
            min_distance = min(manhattan_distance(pos, shuttle_pos) for shuttle_pos in shuttle_positions)
            if min_distance <= 10:
                nearby_positions.append((pos, min_distance))

        pos = None
        if nearby_positions:
            nearby_positions.sort(key=lambda x: x[1])
            pos = nearby_positions[0][0]
        else:
            pos = random.choice(available)

        task = Task(i, pos, "release")
        tasks.append(task)
        available.remove(pos)
        existing_positions.add(pos)
        print(f"  生成放货任务 {i}: {pos}")

    return tasks


def generate_release_tasks(n):
    """生成放货任务"""
    available = calculate_available_positions()

    existing_positions = get_all_active_task_positions()
    available = [pos for pos in available if pos not in existing_positions]

    if not available:
        return []

    n = min(n, len(available))
    tasks = []
    for i in range(n):
        pos = random.choice(available)
        task = Task(i, pos, "release")
        tasks.append(task)
        available.remove(pos)
        existing_positions.add(pos)
        print(f"  放货任务 {i}: {pos}")
    return tasks


def generate_pick_tasks(n):
    """生成取货任务"""
    pick_tasks = []
    existing_positions = get_all_active_task_positions()

    with state_lock:
        occupied_positions = []
        for row in range(ROWS):
            for col in range(COLS):
                # 检查有货(2)且未被预订
                if shared_state["slots"][row][col] == 2 and (row, col) not in existing_positions:
                    occupied_positions.append((row, col))

        if not occupied_positions:
            return pick_tasks

        n = min(n, len(occupied_positions))
        for i in range(n):
            pos = random.choice(occupied_positions)
            task = Task(i, pos, "pick")
            pick_tasks.append(task)
            occupied_positions.remove(pos)
            existing_positions.add(pos)
            print(f"  取货任务 {i}: {pos}")

    return pick_tasks


def log_task_completion(task_type, position, start_time, end_time, success=True):
    """记录任务完成日志"""
    task_log.append({
        "type": task_type,
        "position": position,
        "start_time": start_time,
        "end_time": end_time,
        "success": success
    })

    status = "成功" if success else "失败"
    logging.info(f"[{end_time:.1f}] {task_type}任务 {status} at {position}")

    with state_lock:
        if success:
            shared_state["completed_tasks"] += 1
        else:
            shared_state["failed_tasks"] += 1


def move_shuttle(env, shuttle_id, path):
    """移动小车"""
    if not path:
        return

    actual_path = path[1:] if path and path[0] == tuple(shared_state["shuttles"][shuttle_id]["pos"]) else path

    if not actual_path:
        return

    for i, step in enumerate(actual_path):
        if not is_valid_position(step) and not (0 <= step[0] < ROWS and 0 <= step[1] < COLS):
            continue

        with state_lock:
            old_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])
            shared_state["shuttles"][shuttle_id]["pos"] = list(step)
            shared_state["shuttles"][shuttle_id]["path"] = actual_path[i:]
            shared_state["shuttles"][shuttle_id]["status"] = f"移动中({i + 1}/{len(actual_path)})"
            shared_state["shuttles"][shuttle_id]["battery"] = max(0,
                                                                  shared_state["shuttles"][shuttle_id]["battery"] - 1)

            if hasattr(warehouse, 'clear_position'):
                warehouse.clear_position(old_pos[0], old_pos[1])
            if hasattr(warehouse, 'add_shuttle'):
                warehouse.add_shuttle(step[0], step[1], shuttle_id)

        yield env.timeout(MOVE_TIME)

        if SIMULATION_DELAY > 0:
            time.sleep(SIMULATION_DELAY)

    print(f"小车{shuttle_id} 移动完成")


def execute_task_operation(env, shuttle_id, task, task_type):
    """执行任务操作"""
    r, c = task.position

    # 1. 检查和验证
    with state_lock:
        current_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])
        distance = manhattan_distance(current_pos, task.position)

        if distance > 1:
            adj = [(task.position[0] + dx, task.position[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            if current_pos not in adj:
                print(f"✗ 小车{shuttle_id} 距离过远: {distance}")
                log_task_completion(task_type, task.position, env.now, env.now, False)
                return False

        shared_state["shuttles"][shuttle_id]["status"] = f"执行{task_type}"

        # 双重检查货位状态
        if task_type == "release":
            if shared_state["slots"][r][c] != 1:
                print(f"✗ 放货失败: 位置{task.position}非空闲 (状态: {shared_state['slots'][r][c]})")
                log_task_completion("放货", task.position, env.now, env.now, False)
                return False
        else:  # pick
            if shared_state["slots"][r][c] != 2:
                print(f"✗ 取货失败: 位置{task.position}无货 (状态: {shared_state['slots'][r][c]})")
                log_task_completion("取货", task.position, env.now, env.now, False)
                return False

        op_time = PARK_TIME if task_type == "release" else RETRIEVE_TIME

    # 2. 执行耗时
    print(f"小车{shuttle_id} {task_type}中...")
    yield env.timeout(op_time)
    if SIMULATION_DELAY > 0:
        time.sleep(SIMULATION_DELAY * 2)

    # 3. 更新状态
    with state_lock:
        if task_type == "release":
            shared_state["slots"][r][c] = 2
            if hasattr(warehouse, 'update_slot'):
                warehouse.update_slot(r, c, 2)
            log_task_completion("放货", task.position, env.now - PARK_TIME, env.now, True)
            print(f"✓ 放货成功 at {task.position}")
        else:  # pick
            shared_state["slots"][r][c] = 1
            if hasattr(warehouse, 'update_slot'):
                warehouse.update_slot(r, c, 1)
            log_task_completion("取货", task.position, env.now - RETRIEVE_TIME, env.now, True)
            print(f"✓ 取货成功 at {task.position}")

        return True


def shuttle_controller(env, shuttle_id):
    """小车控制器"""
    print(f"小车{shuttle_id} 控制器启动")
    consecutive_failures = 0
    yield env.timeout(random.uniform(0, 3))

    while True:
        if consecutive_failures > 3:
            yield env.timeout(10)
            consecutive_failures = 0
            continue

        with state_lock:
            if shared_state["done"]: return
            if shared_state["shuttles"][shuttle_id]["battery"] < 10:
                shared_state["shuttles"][shuttle_id]["status"] = "充电中"
                yield env.timeout(5)
                shared_state["shuttles"][shuttle_id]["battery"] = 100
                continue

        task = None
        task_type = None

        with state_lock:
            if shared_state["pick_tasks"]:
                task = shared_state["pick_tasks"].pop(0)
                task_type = "pick"
            elif shared_state["release_tasks"]:
                task = shared_state["release_tasks"].pop(0)
                task_type = "release"

        if task is None:
            with state_lock:
                shared_state["shuttles"][shuttle_id]["status"] = "等待任务"
            yield env.timeout(2)
            continue

        with state_lock:
            shared_state["shuttles"][shuttle_id]["busy"] = True
            shared_state["shuttles"][shuttle_id]["Load"] = (task_type == "release")
            shared_state["shuttles"][shuttle_id]["current_task"] = task
            shared_state["shuttles"][shuttle_id]["status"] = "规划路径"
            current_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])

            # ---【新增逻辑】收集其他小车的路径 ---
            other_paths = []
            for s in shared_state["shuttles"]:
                # 如果是其他小车，且它有规划好的路径
                if s["id"] != shuttle_id and s.get("path"):
                    other_paths.append(s["path"])
            # ----------------------------------

        # 路径规划 (传入 other_paths)
        grid = None
        # 调用新的接口
        path = improved_a_star_search(grid, current_pos, task.position, other_paths)

        if not path:
            print(f"✗ 路径规划失败 -> {task.position}")
            consecutive_failures += 1
            with state_lock:
                shared_state["shuttles"][shuttle_id]["busy"] = False
                shared_state["shuttles"][shuttle_id]["current_task"] = None
                if task_type == "pick":
                    shared_state["pick_tasks"].append(task)
                else:
                    shared_state["release_tasks"].append(task)
            yield env.timeout(2)
            continue

        task_success = False
        try:
            with state_lock:
                shared_state["shuttles"][shuttle_id]["status"] = "移动中"

            yield env.process(move_shuttle(env, shuttle_id, path))

            final_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])
            dist = manhattan_distance(final_pos, task.position)

            if dist <= 1:
                task_success = yield env.process(execute_task_operation(env, shuttle_id, task, task_type))
            else:
                print(f"小车{shuttle_id} 未到达目标: {final_pos} != {task.position}")
                log_task_completion(task_type, task.position, env.now, env.now, False)
                consecutive_failures += 1

        except Exception as e:
            print(f"执行异常: {e}")
            consecutive_failures += 1

        with state_lock:
            shared_state["shuttles"][shuttle_id]["busy"] = False
            shared_state["shuttles"][shuttle_id]["Load"] = False
            shared_state["shuttles"][shuttle_id]["current_task"] = None
            shared_state["shuttles"][shuttle_id]["status"] = "任务完成" if task_success else "任务失败"
            if task_success: consecutive_failures = 0


def dynamic_task_generator(env):
    """动态任务生成器"""
    print("动态任务生成器启动")
    while not shared_state["done"]:
        yield env.timeout(random.randint(5, 10))

        if shared_state["done"]: break

        with state_lock:
            current_task_count = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
            shuttle_positions = [tuple(shuttle["pos"]) for shuttle in shared_state["shuttles"]]
            occupied_count = sum(1 for row in shared_state["slots"] for cell in row if cell == 2)

        if current_task_count > 6:
            continue

        # 逻辑: 优先填满仓库，如果货物多则生成取货
        if occupied_count < 20 or random.random() < 0.6:
            new_tasks = generate_nearby_release_tasks(1, shuttle_positions)
            with state_lock:
                shared_state["release_tasks"].extend(new_tasks)
        else:
            new_tasks = generate_pick_tasks(1)
            with state_lock:
                shared_state["pick_tasks"].extend(new_tasks)


def run_simulation():
    """运行仿真"""
    print("=== 仓库仿真系统启动 ===")
    with state_lock:
        shared_state["simulation_started"] = True

    env = simpy.Environment()

    print("生成初始任务...")
    with state_lock:
        shuttle_positions = [tuple(shuttle["pos"]) for shuttle in shared_state["shuttles"]]

    initial_release_tasks = generate_nearby_release_tasks(4, shuttle_positions)

    with state_lock:
        shared_state["release_tasks"] = initial_release_tasks

    for i in range(SHUTTLES):
        env.process(shuttle_controller(env, i))

    env.process(dynamic_task_generator(env))

    def system_monitor():
        while not shared_state["done"]:
            with state_lock:
                shared_state["time"] = env.now
            yield env.timeout(0.5)

    env.process(system_monitor())

    simulation_ready.set()

    start_wait = time.time()
    while not visualization_started.is_set():
        if time.time() - start_wait > 10: break
        time.sleep(0.1)

    try:
        print("=== 仿真开始运行 ===")
        env.run(until=3000)
        print("=== 仿真时间结束 ===")
    except Exception as e:
        print(f"仿真异常: {e}")
    finally:
        with state_lock:
            shared_state["done"] = True

        completed = shared_state.get("completed_tasks", 0)
        failed = shared_state.get("failed_tasks", 0)
        print(f"\n=== 仿真统计 ===")
        print(f"完成: {completed}, 失败: {failed}")
        if completed + failed > 0:
            print(f"成功率: {completed / (completed + failed) * 100:.1f}%")


def run_visualization_wrapper():
    try:
        visualization_started.set()
        run_visualization()
    except Exception as e:
        print(f"可视化错误: {e}")


if __name__ == "__main__":
    print("初始化...")
    warehouse.add_elevator(0, 5)
    warehouse.add_elevator(10, 5)
    initialize_shared_state()

    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()

    if simulation_ready.wait(timeout=10):
        print("仿真已就绪，启动界面...")
        run_visualization_wrapper()
    else:
        print("仿真启动超时")

    sim_thread.join(timeout=2)