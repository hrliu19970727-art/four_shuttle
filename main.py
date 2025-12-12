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
from config import ROWS, COLS, SHUTTLES, MOVE_TIME, PARK_TIME, RETRIEVE_TIME, ALLEY_COLS, ALLEY_ROWS
from shared_state import shared_state, state_lock

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
task_log = []

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
        # 初始化小车状态 - 确保在有效位置
        initial_positions = [(1, 3), (1, 10), (1, 20), (1, 30)]
        # 过滤无效的初始位置
        valid_positions = [pos for pos in initial_positions if is_valid_position(pos)]
        if not valid_positions:
            # 如果预设位置都无效，使用备用位置
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

        # 如果还是找不到有效位置，使用默认位置
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

        # 初始化货架状态 (0=不可用, 1=空闲, 2=占用)
        shared_state["slots"] = [[1 for _ in range(COLS)] for _ in range(ROWS)]

        # 设置巷道为不可用
        for row in ALLEY_ROWS:
            for col in range(COLS):
                if 0 <= row < ROWS and 0 <= col < COLS:
                    shared_state["slots"][row][col] = 0
        for col in ALLEY_COLS:
            for row in range(ROWS):
                if 0 <= row < ROWS and 0 <= col < COLS:
                    shared_state["slots"][row][col] = 0

        # 初始化任务队列
        shared_state["release_tasks"] = []
        shared_state["pick_tasks"] = []

        # 初始化统计信息
        shared_state["completed_tasks"] = 0
        shared_state["failed_tasks"] = 0

        # 初始化其他状态
        shared_state["time"] = 0
        shared_state["done"] = False
        shared_state["simulation_started"] = False

        print(f"初始化完成: {SHUTTLES}辆穿梭车")

    # 调试巷道配置
    debug_alley_configuration()


def calculate_available_positions():
    """计算可用的位置"""
    available = []
    for row in range(ROWS):
        for col in range(COLS):
            if is_valid_position((row, col)) and (row, col) not in [(0, 5), (10, 5)]:  # 排除电梯位置
                available.append((row, col))
    return available


def generate_nearby_release_tasks(n, shuttle_positions):
    """生成附近的放货任务"""
    available = calculate_available_positions()
    if not available:
        print("警告: 没有可用的放货位置")
        return []

    tasks = []

    # 获取当前已有的任务位置，避免任务重复生成
    existing_task_positions = set()
    with state_lock:
        existing_task_positions.update(task.position for task in shared_state["release_tasks"])
        existing_task_positions.update(task.position for task in shared_state["pick_tasks"])

    available = [pos for pos in available if pos not in existing_task_positions]

    if not available:
        return []

    for i in range(min(n, len(available))):
        # 优先选择靠近小车的位置
        nearby_positions = []
        for pos in available:
            # 计算到最近小车的距离
            min_distance = min(manhattan_distance(pos, shuttle_pos) for shuttle_pos in shuttle_positions)
            if min_distance <= 10:  # 只考虑距离10以内的位置
                nearby_positions.append((pos, min_distance))

        pos = None
        min_distance = -1

        if nearby_positions:
            # 选择最近的位置
            nearby_positions.sort(key=lambda x: x[1])
            pos = nearby_positions[0][0]
            min_distance = nearby_positions[0][1]
        else:
            # 如果没有附近位置，随机选择
            pos = random.choice(available)

        task = Task(i, pos, "release")
        tasks.append(task)
        available.remove(pos)
        print(f"  附近放货任务 {i}: {pos} (距离: {min_distance if min_distance != -1 else '随机选择'})")

    return tasks


def generate_release_tasks(n):
    """生成放货任务"""
    available = calculate_available_positions()
    if not available:
        print("警告: 没有可用的放货位置")
        return []

    n = min(n, len(available))
    tasks = []

    # 获取当前已有的任务位置
    existing_task_positions = set()
    with state_lock:
        existing_task_positions.update(task.position for task in shared_state["release_tasks"])
        existing_task_positions.update(task.position for task in shared_state["pick_tasks"])

    available = [pos for pos in available if pos not in existing_task_positions]

    if not available:
        return []

    for i in range(n):
        pos = random.choice(available)
        task = Task(i, pos, "release")
        tasks.append(task)
        available.remove(pos)
        print(f"  放货任务 {i}: {pos}")
    return tasks


def generate_pick_tasks(n):
    """生成取货任务"""
    pick_tasks = []
    with state_lock:
        # 找出所有被占用的位置
        occupied_positions = []
        for row in range(ROWS):
            for col in range(COLS):
                if shared_state["slots"][row][col] == 2:  # 被占用的位置
                    occupied_positions.append((row, col))

        if not occupied_positions:
            print("提示: 没有可用的取货位置（尚无货物）")
            return pick_tasks

        n = min(n, len(occupied_positions))
        for i in range(n):
            pos = random.choice(occupied_positions)
            task = Task(i, pos, "pick")
            pick_tasks.append(task)
            occupied_positions.remove(pos)
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
    """移动小车 - 修复版"""
    if not path:
        print(f"小车{shuttle_id} 移动路径为空")
        return

    print(f"小车{shuttle_id} 开始移动，路径长度: {len(path)}")

    # 使用完整路径，不再限制步数
    actual_path = path[1:] if path and path[0] == tuple(shared_state["shuttles"][shuttle_id]["pos"]) else path

    if not actual_path:
        print(f"小车{shuttle_id} 无需移动")
        return

    print(f"  实际移动: {len(actual_path)} 步")

    for i, step in enumerate(actual_path):
        # 检查步骤有效性
        if not is_valid_position(step):
            print(f"  警告: 小车{shuttle_id} 移动步骤无效: {step}")
            continue

        with state_lock:
            old_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])
            shared_state["shuttles"][shuttle_id]["pos"] = list(step)
            shared_state["shuttles"][shuttle_id]["path"] = actual_path[i:]
            shared_state["shuttles"][shuttle_id]["status"] = f"移动中({i + 1}/{len(actual_path)})"

            # 消耗电量
            shared_state["shuttles"][shuttle_id]["battery"] = max(0,
                                                                  shared_state["shuttles"][shuttle_id]["battery"] - 1)

            # 更新仓库地图
            if hasattr(warehouse, 'clear_position'):
                warehouse.clear_position(old_pos[0], old_pos[1])
            if hasattr(warehouse, 'add_shuttle'):
                warehouse.add_shuttle(step[0], step[1], shuttle_id)

            # print(f"  小车{shuttle_id} 移动到: {step}")

        # 移动时间
        yield env.timeout(MOVE_TIME)

    print(f"小车{shuttle_id} 移动完成")


def execute_task_operation(env, shuttle_id, task, task_type):
    """
    执行任务操作 - 解决死锁问题
    【关键修复】将需要长时间 SimPy yield 的部分移到锁外。
    """
    r, c = task.position

    # 1. 检查和更新前置状态 (需要锁)
    with state_lock:
        current_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])
        distance = manhattan_distance(current_pos, task.position)

        # 检查任务执行位置
        if distance > 1:
            adjacent_positions = [
                (task.position[0] - 1, task.position[1]),
                (task.position[0] + 1, task.position[1]),
                (task.position[0], task.position[1] - 1),
                (task.position[0], task.position[1] + 1)
            ]
            if current_pos not in adjacent_positions:
                print(f"✗ 小车{shuttle_id} 距离目标太远: {distance} 步，任务失败")
                log_task_completion(task_type, task.position, env.now, env.now, False)
                return False
            else:
                print(f"  在相邻位置，允许执行任务")

        shared_state["shuttles"][shuttle_id]["status"] = f"执行{task_type}"

        if task_type == "release":
            if shared_state["slots"][r][c] != 1:  # 位置非空闲
                print(f"✗ 小车{shuttle_id} 放货失败: 位置被占用")
                log_task_completion("放货", task.position, env.now, env.now, False)
                return False
        else:  # pick
            if shared_state["slots"][r][c] != 2:  # 位置无货
                print(f"✗ 小车{shuttle_id} 取货失败: 位置无货")
                log_task_completion("取货", task.position, env.now, env.now, False)
                return False

        # 确定需要消耗的时间
        op_time = PARK_TIME if task_type == "release" else RETRIEVE_TIME

    # 2. 消耗时间 (不需要锁，避免死锁)
    print(f"小车{shuttle_id} {task_type}中... (消耗时间: {op_time})")
    yield env.timeout(op_time)

    # 3. 任务完成后的状态更新 (需要锁)
    with state_lock:
        if task_type == "release":
            shared_state["slots"][r][c] = 2
            if hasattr(warehouse, 'update_slot'):
                warehouse.update_slot(r, c, 2)
            log_task_completion("放货", task.position, env.now - PARK_TIME, env.now, True)
            print(f"✓ 小车{shuttle_id} 放货完成 at {task.position}")
        else:  # pick
            shared_state["slots"][r][c] = 1
            if hasattr(warehouse, 'update_slot'):
                warehouse.update_slot(r, c, 1)
            log_task_completion("取货", task.position, env.now - RETRIEVE_TIME, env.now, True)
            print(f"✓ 小车{shuttle_id} 取货完成 at {task.position}")

        return True  # 任务成功

    # 由于步骤 2 是 SimPy yield，我们不需要 try...except 来处理 SimPy 异常，
    # 只需要处理数据访问的异常 (已在 lock 内部处理)。


def shuttle_controller(env, shuttle_id):
    """小车控制器"""
    print(f"小车{shuttle_id} 控制器启动")

    # 记录连续失败次数
    consecutive_failures = 0

    # 初始等待，避免所有小车同时启动
    yield env.timeout(random.uniform(0, 3))

    while True:
        # 如果连续失败太多，暂停一段时间
        if consecutive_failures > 3:
            print(f"小车{shuttle_id} 连续失败过多，暂停工作")
            yield env.timeout(10)
            consecutive_failures = 0
            continue

        with state_lock:
            if shared_state["done"]:
                print(f"小车{shuttle_id} 控制器退出")
                return

            # 检查电量
            if shared_state["shuttles"][shuttle_id]["battery"] < 10:
                shared_state["shuttles"][shuttle_id]["status"] = "需要充电"
                yield env.timeout(5)  # 充电时间
                shared_state["shuttles"][shuttle_id]["battery"] = 100
                shared_state["shuttles"][shuttle_id]["status"] = "充电完成"
                print(f"小车{shuttle_id} 充电完成")

        # 获取任务
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
            # 没有任务，等待
            with state_lock:
                shared_state["shuttles"][shuttle_id]["status"] = "等待任务"
            yield env.timeout(2)
            continue

        print(f"小车{shuttle_id} 获取{task_type}任务 -> {task.position}")

        with state_lock:
            shared_state["shuttles"][shuttle_id]["busy"] = True
            shared_state["shuttles"][shuttle_id]["Load"] = (task_type == "release")
            shared_state["shuttles"][shuttle_id]["current_task"] = task
            shared_state["shuttles"][shuttle_id]["status"] = "规划路径"
            current_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])

        # 路径规划
        with state_lock:
            grid = np.array(shared_state["slots"])

        path = improved_a_star_search(grid, current_pos, task.position)

        if not path:
            print(f"✗ 小车{shuttle_id} 所有路径规划都失败")
            consecutive_failures += 1
            with state_lock:
                shared_state["shuttles"][shuttle_id]["busy"] = False
                shared_state["shuttles"][shuttle_id]["status"] = "路径规划失败"
                shared_state["shuttles"][shuttle_id]["current_task"] = None
                # 任务重新入队
                if task_type == "pick":
                    shared_state["pick_tasks"].append(task)
                else:
                    shared_state["release_tasks"].append(task)
            yield env.timeout(2)
            continue

        task_success = False
        try:
            # 移动阶段
            with state_lock:
                shared_state["shuttles"][shuttle_id]["status"] = "移动中"
            yield env.process(move_shuttle(env, shuttle_id, path))

            # 检查是否到达目标
            final_pos = tuple(shared_state["shuttles"][shuttle_id]["pos"])

            # 任务执行
            distance = manhattan_distance(final_pos, task.position)
            if distance <= 1:
                print(f"小车{shuttle_id} 到达目标区域，距离: {distance}")
                task_success = yield env.process(
                    execute_task_operation(env, shuttle_id, task, task_type)
                )
            else:
                print(f"小车{shuttle_id} 未精确到达目标: 当前位置 {final_pos}, 目标 {task.position}, 距离: {distance}")
                log_task_completion(task_type, task.position, env.now, env.now, False)
                consecutive_failures += 1


        except Exception as e:
            print(f"✗ 小车{shuttle_id} 任务执行异常: {e}")
            import traceback
            traceback.print_exc()
            consecutive_failures += 1

        # 重置状态
        with state_lock:
            shared_state["shuttles"][shuttle_id]["busy"] = False
            shared_state["shuttles"][shuttle_id]["Load"] = False
            shared_state["shuttles"][shuttle_id]["current_task"] = None
            if task_success:
                shared_state["shuttles"][shuttle_id]["status"] = "任务完成"
                consecutive_failures = 0  # 重置失败计数
            else:
                shared_state["shuttles"][shuttle_id]["status"] = "任务失败"


def dynamic_task_generator(env):
    """动态任务生成器 - 修复版"""
    print("动态任务生成器启动")
    task_id_counter = 100

    while not shared_state["done"]:
        # 等待随机时间
        yield env.timeout(random.randint(10, 20))

        if shared_state["done"]:
            break

        # 检查当前任务数量，避免堆积
        with state_lock:
            current_task_count = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
            # 获取小车当前位置
            shuttle_positions = [tuple(shuttle["pos"]) for shuttle in shared_state["shuttles"]]

        # 如果任务太多，暂停生成
        if current_task_count > 4:
            print("任务队列已满，暂停生成新任务")
            yield env.timeout(15)
            continue

        # 决定任务类型
        with state_lock:
            occupied_count = sum(1 for row in shared_state["slots"] for cell in row if cell == 2)

        if occupied_count < 5 or random.random() < 0.7:  # 货物较少或70%概率生成放货
            # 生成近距离任务
            new_tasks = generate_nearby_release_tasks(1, shuttle_positions)
            with state_lock:
                shared_state["release_tasks"].extend(new_tasks)
                if new_tasks:
                    print(f"添加 {len(new_tasks)} 个放货任务")
        else:  # 取货任务
            new_tasks = generate_pick_tasks(1)
            with state_lock:
                shared_state["pick_tasks"].extend(new_tasks)
                if new_tasks:
                    print(f"添加 {len(new_tasks)} 个取货任务")

        task_id_counter += len(new_tasks)

    print("动态任务生成器退出")


def run_simulation():
    """运行仿真"""
    print("=== 仓库仿真系统启动 ===")

    # 标记仿真开始
    with state_lock:
        shared_state["simulation_started"] = True

    env = simpy.Environment()

    # 生成初始任务 - 使用附近任务
    print("生成初始任务...")
    with state_lock:
        shuttle_positions = [tuple(shuttle["pos"]) for shuttle in shared_state["shuttles"]]

    initial_release_tasks = generate_nearby_release_tasks(2, shuttle_positions)
    initial_pick_tasks = generate_pick_tasks(0)  # 初始没有取货任务

    with state_lock:
        shared_state["release_tasks"] = initial_release_tasks
        shared_state["pick_tasks"] = initial_pick_tasks

    print(f"初始任务: {len(initial_release_tasks)} 放货, {len(initial_pick_tasks)} 取货")

    # 启动所有小车
    for i in range(SHUTTLES):
        env.process(shuttle_controller(env, i))
        print(f"启动小车 {i}")

    # 启动动态任务生成器
    env.process(dynamic_task_generator(env))

    # 系统状态监控
    def system_monitor():
        monitor_count = 0
        while not shared_state["done"]:
            with state_lock:
                shared_state["time"] = env.now
            yield env.timeout(0.5)

            monitor_count += 1
            if monitor_count % 20 == 0:  # 每10秒报告一次
                with state_lock:
                    busy_shuttles = sum(1 for s in shared_state["shuttles"] if s.get("busy", False))
                    total_tasks = (len(shared_state["release_tasks"]) +
                                   len(shared_state["pick_tasks"]))
                    completed = shared_state.get("completed_tasks", 0)
                    failed = shared_state.get("failed_tasks", 0)
                    total_attempts = completed + failed
                    success_rate = (completed / total_attempts * 100) if total_attempts > 0 else 0

                    print(f"[状态] 时间: {env.now:.1f}s, 忙碌小车: {busy_shuttles}/{SHUTTLES}, "
                          f"待处理任务: {total_tasks}, 成功率: {success_rate:.1f}% ({completed}成功/{failed}失败)")

    env.process(system_monitor())

    # 通知主线程仿真已准备
    simulation_ready.set()
    print("仿真准备就绪，等待可视化启动...")

    # 等待可视化启动
    start_time = time.time()
    while not visualization_started.is_set():
        if time.time() - start_time > 15:
            print("可视化启动超时，继续运行仿真")
            break
        time.sleep(0.1)

    # 运行仿真
    try:
        print("=== 仿真开始运行 ===")
        env.run(until=300)  # 运行5分钟
        print("=== 仿真正常结束 ===")

    except Exception as e:
        print(f"!!! 仿真异常: {e}")
        import traceback
        traceback.print_exc()

    finally:
        with state_lock:
            shared_state["done"] = True

        # 最终统计
        with state_lock:
            completed = shared_state.get("completed_tasks", 0)
            failed = shared_state.get("failed_tasks", 0)
            total = completed + failed
            success_rate = (completed / total * 100) if total > 0 else 0

        print(f"\n=== 仿真统计 ===")
        print(f"总运行时间: {env.now:.1f}秒")
        print(f"完成任务: {completed}")
        print(f"失败任务: {failed}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"总任务数: {total}")


def run_visualization_wrapper():
    """可视化包装器"""
    try:
        print("启动可视化界面...")
        visualization_started.set()
        run_visualization()
    except Exception as e:
        print(f"可视化错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 50)
    print("      自动化仓库调度仿真系统")
    print("=" * 50)

    # 初始化仓库
    print("初始化仓库设施...")
    warehouse.add_elevator(0, 5)
    warehouse.add_elevator(10, 5)

    # 初始化共享状态
    print("初始化共享状态...")
    initialize_shared_state()

    # 启动仿真线程
    print("启动仿真线程...")
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()

    # 等待仿真准备
    print("等待仿真准备...")
    if simulation_ready.wait(timeout=20):
        print("仿真准备完成")
    else:
        print("仿真准备超时，尝试继续...")

    # 启动可视化（在主线程运行）
    run_visualization_wrapper()

    # 等待仿真线程结束
    sim_thread.join(timeout=5)

    print("\n" + "=" * 50)
    print("      程序执行完毕")
    print("=" * 50)
