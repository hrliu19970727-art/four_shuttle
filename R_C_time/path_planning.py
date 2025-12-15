# path_planning.py (Space-Time A* / CA*)
import heapq
import random
from config import *

# 拥堵惩罚系数 (依然保留，用于倾向性选择)
ALPHA_M = 0.1


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_traffic_cost(pos, other_paths_sets):
    # 这里仅做静态热度参考，动态避障由 reservation 处理
    occupancy = 0
    for path_set in other_paths_sets:
        if pos in path_set: occupancy += 1
    return 1.0 + (ALPHA_M * occupancy)


def get_valid_neighbors(grid_type, current, rows, cols):
    """
    获取邻居，允许原地等待 (Wait Action)
    current: (row, col) - 注意这里不包含 time，仅做空间判断
    """
    r, c = current
    g_type = grid_type[r][c]

    # 基础移动方向
    candidates = []
    if g_type == TYPE_MAIN_H_EAST:
        candidates = [(0, 1)]
    elif g_type == TYPE_MAIN_H_WEST:
        candidates = [(0, -1)]
    elif g_type == TYPE_MAIN_V_SOUTH:
        candidates = [(1, 0)]
    elif g_type == TYPE_MAIN_V_NORTH:
        candidates = [(-1, 0)]
    elif g_type == TYPE_STORAGE:
        candidates = [(1, 0), (-1, 0)]
    elif g_type in [TYPE_INTERSECTION, TYPE_ELEVATOR, TYPE_MAIN_BIDIRECTIONAL]:
        candidates = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # 【关键】总是允许原地等待 (0, 0)，除非在某些特殊区域(如高速路)禁止停车
    # 这里为了防死锁，允许在任何地方等待
    candidates.append((0, 0))

    neighbors = []
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            n_type = grid_type[nr][nc]
            if n_type == TYPE_NONE: continue

            # 逆行检查
            if n_type == TYPE_MAIN_H_EAST and dc == -1: continue
            if n_type == TYPE_MAIN_H_WEST and dc == 1: continue
            if n_type == TYPE_MAIN_V_SOUTH and dr == -1: continue
            if n_type == TYPE_MAIN_V_NORTH and dr == 1: continue

            neighbors.append((nr, nc))

    # 特殊规则：主路进支路
    if g_type in [TYPE_MAIN_H_EAST, TYPE_MAIN_H_WEST, TYPE_MAIN_V_SOUTH, TYPE_MAIN_V_NORTH, TYPE_MAIN_BIDIRECTIONAL]:
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                t_type = grid_type[nr][nc]
                if t_type == TYPE_STORAGE:
                    if dc != 0: continue
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))
                elif t_type == TYPE_ELEVATOR:
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))

    return neighbors


def improved_a_star_search(grid_type_map, start, goal, other_paths=None, reservations=None):
    """
    Space-Time A* (时空 A*)
    :param start: (row, col)
    :param goal: (row, col)
    :param reservations: set((row, col, time)) - 记录哪些时空点被占用了
    """
    if reservations is None: reservations = set()

    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    # OpenList: (f_score, g_score, row, col, time, path)
    # 状态空间增加 Time 维度
    open_set = []
    heapq.heappush(open_set, (0, 0, start[0], start[1], 0, [start]))

    # Visited: (row, col, time) -> min_g
    # 必须包含 time，因为 (10,10) 在 t=5 和 t=6 是两个不同状态
    visited = {}

    # 限制最大搜索深度 (时间步)
    max_time_steps = 200
    # 限制最大计算次数
    max_calculations = 30000
    calcs = 0

    while open_set and calcs < max_calculations:
        calcs += 1
        f, g, r, c, t, path = heapq.heappop(open_set)

        # 到达目标 (且不需要等待)
        if (r, c) == goal:
            return path

        if t >= max_time_steps: continue

        # 获取空间邻居
        neighbors = get_valid_neighbors(grid_type_map, (r, c), rows, cols)
        random.shuffle(neighbors)

        for nr, nc in neighbors:
            nt = t + 1  # 下一步的时间

            # --- 时空冲突检测 (Reservation Table) ---
            # 1. 顶点冲突: 我要去的地方，别人在 t+1 时刻是否在？
            if (nr, nc, nt) in reservations:
                continue

            # 2. 交换冲突 (Edge Collision): 防止两车在 t -> t+1 互换位置穿模
            # 即: 我去 B, 别人从 B 来 A
            # (如果需要极高精度仿真开启此项，SimPy离散仿真通常只需顶点冲突)

            # 计算代价
            step_cost = 1.0
            # 原地等待代价稍高，鼓励移动
            if (nr, nc) == (r, c): step_cost = 1.1

            new_g = g + step_cost

            # 状态记录 key: (row, col, time)
            state_key = (nr, nc, nt)

            if state_key not in visited or new_g < visited[state_key]:
                visited[state_key] = new_g
                h = manhattan_distance((nr, nc), goal)
                f_new = new_g + h

                new_path = path + [(nr, nc)]
                heapq.heappush(open_set, (f_new, new_g, nr, nc, nt, new_path))

    # print(f"规划失败: {start}->{goal} (时空路网无解)")
    return []


def is_valid_position(pos):
    return 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS