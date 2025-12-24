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
    MAX_OBSERVED_TASKS = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
SIMULATION_DELAY = 0.005
simulation_ready = threading.Event()
visualization_started = threading.Event()

INPUT_POINTS = [(ROWS - 13, 8 - 1), (ROWS - 20, 6 - 1)]
OUTPUT_POINTS = [(ROWS - 14, 47 - 1), (ROWS - 20, 49 - 1)]

# --- 工业场景配置 ---
WAVE_SIZE = 100
TASK_QUEUE_LIMIT = 10

if DRL_ENABLED:
    global_agent = IDQNAgent()


def get_nearest(points, current_pos):
    """获取最近的点，如果距离相同则随机选择"""
    best_points = []
    min_dist = float('inf')
    for p in points:
        d = manhattan_distance(current_pos, p)
        if d < min_dist:
            min_dist = d
            best_points = [p]
        elif d == min_dist:
            best_points.append(p)
    return random.choice(best_points)


def is_occupied(pos, my_id):
    with state_lock:
        for s in shared_state["shuttles"]:
            if s["id"] != my_id and tuple(s["pos"]) == pos:
                return True
    return False


def find_nearby_rest_pos(my_id, current_pos, search_radius=6):
    """寻找附近的休息点（避让点）"""
    r, c = current_pos
    for dist in range(1, search_radius + 1):
        candidates = []
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                if abs(dr) != dist and abs(dc) != dist: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    if warehouse.grid_type[nr][nc] == TYPE_STORAGE:
                        if not is_occupied((nr, nc), my_id):
                            candidates.append((nr, nc))
        if candidates:
            return random.choice(candidates)
    return None


def initialize_shared_state():
    with state_lock:
        valid_pos = []
        # 收集所有存储位
        for r in range(ROWS):
            for c in range(COLS):
                if warehouse.grid_type[r][c] == TYPE_STORAGE:
                    valid_pos.append((r, c))

        # --- [新增] 初始化随机库存 (50% 有货) ---
        # 必须先于小车位置初始化，以免小车出生在有货的位置（虽然当前逻辑允许，但最好避开）
        random.shuffle(valid_pos)
        # 取一半位置作为初始有货
        n_initial_cargo = len(valid_pos) // 10
        for i in range(n_initial_cargo):
            r, c = valid_pos[i]
            warehouse.slots[r][c] = 2  # 物理地图更新
            # 注意：shared_state 在下面初始化，这里先改 warehouse 对象

        print(f"初始化库存: {n_initial_cargo} / {len(valid_pos)} 个货位有货")

        # 重新打乱剩余空位给小车出生
        empty_pos = valid_pos[n_initial_cargo:]
        random.shuffle(empty_pos)
        chosen_shuttle_pos = empty_pos[:SHUTTLES]

        shared_state["shuttles"] = [
            {"id": i, "pos": list(p), "busy": False, "Load": False, "path": [], "current_task": None, "status": "空闲",
             "battery": 100}
            for i, p in enumerate(chosen_shuttle_pos)
        ]

        shared_state["slots"] = warehouse.slots.tolist()  # 同步到共享状态
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


# --- [修改点1] 静态任务生成器 (强制穿插模式) ---
class BatchScheduler:
    def __init__(self, wave_size):
        self.total_tasks = wave_size
        self.pending_tasks = []
        self._prepare_static_wave()

    def _prepare_static_wave(self):
        print(f">>> 正在生成静态作业波次 (共 {self.total_tasks} 单) - [强交替模式]...")
        with state_lock:
            empties = []
            filled = []
            # 获取初始状态快照
            for r in range(ROWS):
                for c in range(COLS):
                    if warehouse.grid_type[r][c] == TYPE_STORAGE:
                        st = shared_state["slots"][r][c]
                        if st == 1 or st == 0:  # 空位
                            empties.append((r, c))
                        elif st == 2:  # 有货
                            filled.append((r, c))

            random.shuffle(empties)
            random.shuffle(filled)

            # 模拟任务生成过程中的状态流转
            for i in range(self.total_tasks):
                # 偶数 -> 入库 (Release)
                if i % 2 == 0:
                    if empties:
                        pos = empties.pop(0)
                        t = Task(i, pos, "release")
                        self.pending_tasks.append(t)
                        # [关键] 这个位置将来会有货，加入 filled 供后续出库用
                        # 我们把它加到末尾，模拟先进先出，或者随机插入
                        filled.append(pos)
                    elif filled:
                        # 实在没空位，被迫转出库
                        pos = filled.pop(0)
                        t = Task(i, pos, "pick")
                        self.pending_tasks.append(t)
                        empties.append(pos)

                # 奇数 -> 出库 (Pick)
                else:
                    if filled:
                        pos = filled.pop(0)
                        t = Task(i, pos, "pick")
                        self.pending_tasks.append(t)
                        # [关键] 这个位置将来会变空，加入 empties 供后续入库用
                        empties.append(pos)
                    elif empties:
                        # 实在没货，被迫转入库
                        pos = empties.pop(0)
                        t = Task(i, pos, "release")
                        self.pending_tasks.append(t)
                        filled.append(pos)

        print(f">>> 静态波次生成完毕: {len(self.pending_tasks)} 个任务 (已预设库存)")

    def has_pending_tasks(self):
        return len(self.pending_tasks) > 0

    def pop_task(self):
        if self.pending_tasks:
            return self.pending_tasks.pop(0)
        return None


batch_scheduler = None


def generate_tasks(env):
    global batch_scheduler
    batch_scheduler = BatchScheduler(WAVE_SIZE)

    print(">>> 任务分发器启动")
    yield env.timeout(1)

    while not shared_state["done"]:
        if not batch_scheduler.has_pending_tasks():
            yield env.timeout(1)
            continue

        with state_lock:
            current_backlog = len(shared_state["release_tasks"]) + len(shared_state["pick_tasks"])
            if current_backlog < TASK_QUEUE_LIMIT:
                for _ in range(min(2, len(batch_scheduler.pending_tasks))):
                    task = batch_scheduler.pop_task()
                    if task:
                        if task.task_type == "release":
                            shared_state["release_tasks"].append(task)
                            print(f"[WCS下发] 入库任务 -> {task.position} (ID:{task.id})")
                        else:
                            shared_state["pick_tasks"].append(task)
                            print(f"[WCS下发] 出库任务 -> {task.position} (ID:{task.id})")
                    if current_backlog >= TASK_QUEUE_LIMIT: break

        yield env.timeout(random.uniform(0.5, 1.5))


def move_shuttle(env, s_id, path):
    if not path: return True
    start_idx = 1 if len(path) > 0 and path[0] == tuple(shared_state["shuttles"][s_id]["pos"]) else 0
    actual_path = path[start_idx:]

    if not actual_path:
        return True

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
            if wait_time > 5.0: return False

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
    max_retries = 3
    for attempt in range(max_retries):
        cur, others, obs = get_planning_params(s_id)

        if cur == target_pos:
            return True

        path = run_path_planning(s_id, cur, target_pos, others, obs)

        if not path:
            yield env.timeout(0.5)
            continue

        success = yield env.process(move_shuttle(env, s_id, path))
        if success:
            return True
        else:
            yield env.timeout(0.5)
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


# --- [修改点2] 控制逻辑：按ID顺序选取任务，实现交替 ---
def controller(env, s_id, input_locks, output_locks):
    yield env.timeout(random.uniform(0, 2))

    while True:
        with state_lock:
            if shared_state["done"]: break

        # --- 任务选择 ---
        task = None
        task_idx = -1
        t_type = None
        state_vec = None
        action_idx = 0

        with state_lock:
            # [关键修改] 将出库和入库任务合并后，按 Task ID 排序
            # 这样小车就会优先选择 ID 较小的任务，从而严格遵循生成时的交替顺序
            all_tasks = shared_state["pick_tasks"] + shared_state["release_tasks"]
            candidates = sorted(all_tasks, key=lambda t: t.id)
            candidates = candidates[:MAX_OBSERVED_TASKS]

        if not candidates:
            with state_lock: shared_state["shuttles"][s_id]["status"] = "待命"
            yield env.timeout(1)
            continue

        if DRL_ENABLED:
            state_vec = construct_state(s_id, candidates)
            valid_actions = list(range(len(candidates)))
            action_idx = global_agent.select_action(state_vec, valid_actions)
            task = candidates[action_idx] if action_idx < len(candidates) else candidates[0]
        else:
            task = candidates[0]

        # 尝试从队列中移除任务
        task_acquired = False
        with state_lock:
            if task in shared_state["pick_tasks"]:
                shared_state["pick_tasks"].remove(task)
                t_type = "pick"
                task_acquired = True
            elif task in shared_state["release_tasks"]:
                shared_state["release_tasks"].remove(task)
                t_type = "release"
                task_acquired = True

        if not task_acquired:
            yield env.timeout(0.1)
            continue

        start_time = env.now
        with state_lock:
            shared_state["shuttles"][s_id]["busy"] = True
            shared_state["shuttles"][s_id]["current_task"] = task

        success = False

        try:
            cur_pos = tuple(shared_state["shuttles"][s_id]["pos"])

            # ================= 入库流程 =================
            if t_type == "release":
                # 负载均衡 + 拥堵感知
                best_port = None
                min_score = float('inf')
                candidate_ports = list(INPUT_POINTS)
                random.shuffle(candidate_ports)

                cur, _, _ = get_planning_params(s_id)
                for p in candidate_ports:
                    dist = manhattan_distance(cur, p)
                    congestion = input_locks[p].count + len(input_locks[p].queue)
                    score = dist + (congestion * 20)
                    if score < min_score:
                        min_score = score
                        best_port = p
                target_port = best_port

                # 繁忙避让
                if input_locks[target_port].count > 0 or len(input_locks[target_port].queue) > 0:
                    rest_pos = find_nearby_rest_pos(s_id, cur)
                    if rest_pos:
                        yield env.process(robust_move(env, s_id, rest_pos))

                reached_port = False

                with input_locks[target_port].request() as req:
                    yield req
                    reached_port = yield env.process(robust_move(env, s_id, target_port))

                    if reached_port:
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "取货"))
                        with state_lock:
                            shared_state["shuttles"][s_id]["Load"] = True

                if reached_port:
                    reached_slot = yield env.process(robust_move(env, s_id, task.position))
                    if reached_slot:
                        success = True
                        yield env.process(execute_op_sim(env, s_id, PARK_TIME, "放货"))
                        with state_lock:
                            r, c = task.position
                            shared_state["slots"][r][c] = 2
                            warehouse.update_slot(r, c, 2)
                            shared_state["shuttles"][s_id]["Load"] = False

            # ================= 出库流程 =================
            elif t_type == "pick":
                reached_slot = yield env.process(robust_move(env, s_id, task.position))

                if reached_slot:
                    yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "取货"))
                    with state_lock:
                        r, c = task.position
                        shared_state["slots"][r][c] = 1
                        warehouse.update_slot(r, c, 1)
                        shared_state["shuttles"][s_id]["Load"] = True

                    cur, _, _ = get_planning_params(s_id)
                    best_port = None
                    min_score = float('inf')
                    candidate_ports = list(OUTPUT_POINTS)
                    random.shuffle(candidate_ports)
                    for p in candidate_ports:
                        dist = manhattan_distance(cur, p)
                        congestion = output_locks[p].count + len(output_locks[p].queue)
                        score = dist + (congestion * 20)
                        if score < min_score:
                            min_score = score
                            best_port = p
                    target_port = best_port

                    if output_locks[target_port].count > 0 or len(output_locks[target_port].queue) > 0:
                        rest_pos = find_nearby_rest_pos(s_id, cur)
                        if rest_pos:
                            yield env.process(robust_move(env, s_id, rest_pos))

                    with output_locks[target_port].request() as req:
                        yield req
                        reached_port = yield env.process(robust_move(env, s_id, target_port))

                        if reached_port:
                            yield env.process(execute_op_sim(env, s_id, RETRIEVE_TIME, "卸货"))
                            with state_lock:
                                shared_state["shuttles"][s_id]["Load"] = False
                            success = True

                            yield env.timeout(0.5)
                            rest_pos = find_nearby_rest_pos(s_id, target_port)
                            if rest_pos:
                                yield env.process(robust_move(env, s_id, rest_pos))

            # --- 结果与处理 ---
            end_time = env.now
            reward = -(end_time - start_time)

            if DRL_ENABLED and state_vec is not None:
                with state_lock:
                    next_all = shared_state["pick_tasks"] + shared_state["release_tasks"]
                    next_candidates = sorted(next_all, key=lambda t: t.id)[:MAX_OBSERVED_TASKS]
                next_state_vec = construct_state(s_id, next_candidates)
                global_agent.store_transition(state_vec, action_idx, reward, next_state_vec)
                global_agent.learn()

            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = False
                shared_state["shuttles"][s_id]["current_task"] = None

            if success:
                with state_lock:
                    shared_state["completed_tasks"] += 1
                    if t_type == "release":
                        shared_state["completed_release_tasks"] += 1
                    else:
                        shared_state["completed_pick_tasks"] += 1
                print(f"小车{s_id} 完成 {t_type} (Reward: {reward:.1f})")
            else:
                with state_lock:
                    shared_state["failed_tasks"] += 1
                    if t_type == "pick":
                        shared_state["pick_tasks"].append(task)
                    elif t_type == "release":
                        shared_state["release_tasks"].append(task)

                if DRL_ENABLED:
                    global_agent.store_transition(state_vec, action_idx, -200, state_vec)

                print(f"小车{s_id} 任务失败，执行主动避让...")
                cur_pos = tuple(shared_state["shuttles"][s_id]["pos"])
                escape_pos = find_nearby_rest_pos(s_id, cur_pos)
                if escape_pos:
                    yield env.process(robust_move(env, s_id, escape_pos))
                else:
                    yield env.timeout(random.uniform(2.0, 5.0))

        except Exception as e:
            print(f"Err in controller {s_id}: {e}")
            import traceback
            traceback.print_exc()
            with state_lock:
                shared_state["shuttles"][s_id]["busy"] = False


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
                completed = shared_state["completed_tasks"]
                if completed >= WAVE_SIZE:
                    print(f"\n====== 波次完成! 耗时: {env.now:.2f}s ======")
                    shared_state["done"] = True
            yield env.timeout(0.5)

    env.process(monitor())

    simulation_ready.set()
    while not shared_state["done"]:
        try:
            env.run(until=env.now + 10)
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