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
SIMULATION_DELAY = 0.005  # 加快仿真速度以便快速看完一波任务
simulation_ready = threading.Event()
visualization_started = threading.Event()

INPUT_POINTS = [(ROWS - 13, 8 - 1), (ROWS - 20, 6 - 1)]
OUTPUT_POINTS = [(ROWS - 14, 47 - 1), (ROWS - 20, 49 - 1)]

# --- 工业场景配置 ---
WAVE_SIZE = 40  # 本次作业波次的总任务数 (例如 40 个任务)
TASK_QUEUE_LIMIT = 10  # 下发队列限制 (避免一次性把所有任务都塞给小车)

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


# --- 静态任务生成器 ---
class BatchScheduler:
    def __init__(self, wave_size):
        self.total_tasks = wave_size
        self.pending_tasks = []  # 待下发的静态任务池
        self.generated_count = 0
        self._prepare_static_wave()

    def _prepare_static_wave(self):
        """
        在仿真开始前，生成一张固定的任务清单 (模拟 WMS 订单池)
        """
        print(f">>> 正在生成静态作业波次 (共 {self.total_tasks} 单)...")
        with state_lock:
            # 获取当前所有空位和货位
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

            # 随机打乱以模拟真实分布
            random.shuffle(empties)
            random.shuffle(filled)

            for _ in range(self.total_tasks):
                # 简单逻辑：有货就出库，没货就入库，保持动态平衡
                # 实际工业中可能是 50% 入 50% 出
                is_release = False
                if filled and (not empties or random.random() < 0.5):
                    # 生成出库任务
                    pos = filled.pop(0)
                    t = Task(0, pos, "pick")
                    self.pending_tasks.append(t)
                    # 预占位：为了防止后续任务重复生成同一位置，
                    # 在静态生成阶段我们就假设它被处理了，但这只是逻辑上的，
                    # 实际 shared_state["slots"] 还是原样，等小车去处理。
                    # 这里为了简单，我们仅从临时列表中移除。
                elif empties:
                    # 生成入库任务
                    pos = empties.pop(0)
                    t = Task(0, pos, "release")
                    self.pending_tasks.append(t)

        print(f">>> 静态波次生成完毕: {len(self.pending_tasks)} 个任务待执行")

    def has_pending_tasks(self):
        return len(self.pending_tasks) > 0

    def pop_task(self):
        if self.pending_tasks:
            return self.pending_tasks.pop(0)
        return None


# 全局调度器实例
batch_scheduler = None


def generate_tasks(env):
    """
    负责将静态池中的任务“喂”给系统
    """
    global batch_scheduler
    batch_scheduler = BatchScheduler(WAVE_SIZE)

    print(">>> 任务分发器启动 (静态波次模式)")
    yield env.timeout(1)  # 启动缓冲

    while not shared_state["done"]:
        # 检查是否还有任务没发完
        if not batch_scheduler.has_pending_tasks():
            # 任务全部发完了，但这不代表做完了，只需停止生成即可
            # 等待所有任务完成的逻辑在 monitor 中
            yield env.timeout(1)
            continue

        with state_lock:
            # 检查当前积压情况
            current_backlog = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])

            # 如果系统有空闲处理能力，就下发新任务
            if current_backlog < TASK_QUEUE_LIMIT:
                # 每次下发 1-2 个，模拟订单到达节奏
                for _ in range(min(2, len(batch_scheduler.pending_tasks))):
                    task = batch_scheduler.pop_task()
                    if task:
                        if task.task_type == "release":
                            shared_state["release_tasks"].append(task)
                            print(f"[WCS下发] 入库任务 -> {task.position} (剩余: {len(batch_scheduler.pending_tasks)})")
                        else:
                            shared_state["pick_tasks"].append(task)
                            print(f"[WCS下发] 出库任务 -> {task.position} (剩余: {len(batch_scheduler.pending_tasks)})")

                    if current_backlog >= TASK_QUEUE_LIMIT: break

        # 发单间隔
        yield env.timeout(random.uniform(0.5, 1.5))


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


def run_path_planning(s_id, start, end, others, obstacles):
    t_start = time.perf_counter()
    with state_lock:
        current_slots = np.array(shared_state["slots"])
        is_loaded = shared_state["shuttles"][s_id]["Load"]

    path = improved_a_star_search(
        warehouse.grid_type,
        start, end,
        other_paths=others,
        reservations=obstacles,
        slots_map=current_slots,
        is_loaded=is_loaded
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
        obs = set()
        for s in shared_state["shuttles"]:
            if s["id"] == s_id: continue
            other_pos = tuple(s["pos"])
            for t in range(20): obs.add((other_pos[0], other_pos[1], t))
            if s["path"]:
                for t, p in enumerate(s["path"]):
                    obs.add((p[0], p[1], t))
                    obs.add((p[0], p[1], t + 1))
    return c, oth, obs


def robust_move(env, s_id, target_pos):
    max_retries = 5
    for attempt in range(max_retries):
        cur, others, obs = get_planning_params(s_id)
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

        # --- 任务选择 ---
        task = None
        task_idx = -1
        t_type = None
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
                with state_lock:
                    if task in shared_state["pick_tasks"]:
                        shared_state["pick_tasks"].remove(task)
                        t_type = "pick"
                    elif task in shared_state["release_tasks"]:
                        shared_state["release_tasks"].remove(task)
                        t_type = "release"
            else:
                task = candidates[0]
                with state_lock:
                    if task in shared_state["pick_tasks"]:
                        shared_state["pick_tasks"].pop(0)
                        t_type = "pick"
                    elif task in shared_state["release_tasks"]:
                        shared_state["release_tasks"].pop(0)
                        t_type = "release"
        else:
            with state_lock:
                if shared_state["pick_tasks"]:
                    task = shared_state["pick_tasks"].pop(0)
                    t_type = "pick"
                elif shared_state["release_tasks"]:
                    task = shared_state["release_tasks"].pop(0)
                    t_type = "release"

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

                print(f"小车{s_id} (DRL) 选定入库 -> 请求锁...")
                with input_locks[target_port].request() as req:
                    yield req
                    # 1. 拿到锁，使用 robust_move 前往入库点
                    # 【关键修复】拆分为两行，避免 SyntaxError
                    reached_port = yield env.process(robust_move(env, s_id, target_port))

                    if reached_port:
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                        with state_lock:
                            shared_state["shuttles"][s_id]["Load"] = True

                        # 2. 前往货位
                        cur, _, _ = get_planning_params(s_id)
                        # 【关键修复】拆分为两行
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
                # 【关键修复】拆分为两行
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
                        # 【关键修复】拆分为两行
                        reached_port = yield env.process(robust_move(env, s_id, target_port))

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
                                env.process(robust_move(env, s_id, rest_pos))

            # --- DRL 训练与结算 ---
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
                    print(f"小车{s_id} 完成 (Reward: {reward:.1f})")
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
            with state_lock:
                shared_state["time"] = env.now
                # 【关键】检查是否所有任务都完成了
                completed = shared_state["completed_tasks"]
                # 注意：失败的任务会回退，所以总量守恒
                # 如果 待下发=0 且 队列=0 且 所有车不忙，说明做完了？
                # 简单点：直接看 completed 是否达到 WAVE_SIZE
                # (考虑到可能有极少数永久失败死锁，可以加个超时)
                if completed >= WAVE_SIZE:
                    print(f"\n====== 波次完成! 耗时: {env.now:.2f}s ======")
                    shared_state["done"] = True

            yield env.timeout(0.5)

    env.process(monitor())

    simulation_ready.set()
    while not visualization_started.is_set(): time.sleep(0.1)

    # 运行直到 done 标志被置位，或者设置一个超长保底时间
    # 这里的 until 是 SimPy 的时间单位，不是秒
    while not shared_state["done"]:
        try:
            env.run(until=env.now + 10)  # 步进运行
        except Exception:
            break


def run_viz():
    visualization_started.set()
    run_visualization()


if __name__ == "__main__":
    initialize_shared_state()
    t = threading.Thread(target=run_sim, daemon=True)
    t.start()
    if simulation_ready.wait(5): run_viz()
    t.join(1)