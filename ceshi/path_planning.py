# path_planning.py (修复：允许驶入提升机)
import heapq
import random
from config import *

ALPHA_M = 0.1


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_traffic_cost(pos, other_paths_sets):
    occupancy = 0
    for path_set in other_paths_sets:
        if pos in path_set: occupancy += 1
    return 1.0 + (ALPHA_M * occupancy)


def get_valid_neighbors(grid_type, current, rows, cols):
    """
    根据交通规则返回可行邻居
    【核心修复】允许从主轨道进入提升机/出入口
    """
    r, c = current
    g_type = grid_type[r][c]
    candidates = []

    # --- 1. 基础行驶方向 (单行道限制) ---
    if g_type == TYPE_MAIN_H_EAST:
        candidates = [(0, 1)]  # 东
    elif g_type == TYPE_MAIN_H_WEST:
        candidates = [(0, -1)]  # 西
    elif g_type == TYPE_MAIN_V_SOUTH:
        candidates = [(1, 0)]  # 南
    elif g_type == TYPE_MAIN_V_NORTH:
        candidates = [(-1, 0)]  # 北
    elif g_type == TYPE_STORAGE:
        candidates = [(1, 0), (-1, 0)]  # 存储巷道：仅限垂直移动
    elif g_type in [TYPE_INTERSECTION, TYPE_ELEVATOR, TYPE_MAIN_BIDIRECTIONAL]:
        # 全向区域
        candidates = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    neighbors = []

    # --- 2. 检查常规邻居 ---
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            n_type = grid_type[nr][nc]

            # 物理连接检查
            if n_type == TYPE_NONE: continue

            # 逆行检查 (不能逆向进入单行道)
            if n_type == TYPE_MAIN_H_EAST and dc == -1: continue
            if n_type == TYPE_MAIN_H_WEST and dc == 1: continue
            if n_type == TYPE_MAIN_V_SOUTH and dr == -1: continue
            if n_type == TYPE_MAIN_V_NORTH and dr == 1: continue

            neighbors.append((nr, nc))

    # --- 3. 特殊转向规则 (主轨道 -> 支路) ---
    # 如果当前在主轨道/双向道，允许转向进入特定区域
    if g_type in [TYPE_MAIN_H_EAST, TYPE_MAIN_H_WEST, TYPE_MAIN_V_SOUTH, TYPE_MAIN_V_NORTH, TYPE_MAIN_BIDIRECTIONAL]:
        all_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in all_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                target_type = grid_type[nr][nc]

                # 情况A: 进入存储巷道 (必须垂直进入)
                if target_type == TYPE_STORAGE:
                    if dc != 0: continue  # 禁止横向穿墙进入
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))

                # 情况B: 【关键修复】进入提升机/出入口 (允许任意方向)
                # 您的入库点 (16,7) 就在主巷道 (15,7) 的下方 (dr=1, dc=0)
                # 之前的代码漏掉了这里，导致小车路过门口进不去
                elif target_type == TYPE_ELEVATOR:
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))

    return neighbors


def improved_a_star_search(grid_type_map, start, goal, other_paths=None):
    """A* 路径规划"""
    if other_paths is None: other_paths = []
    other_paths_sets = [set(p) for p in other_paths]

    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    open_set = []
    heapq.heappush(open_set, (0, 0, start, [start]))
    g_scores = {start: 0}

    max_steps = 40000
    steps = 0

    while open_set and steps < max_steps:
        steps += 1
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        neighbors = get_valid_neighbors(grid_type_map, current, rows, cols)
        random.shuffle(neighbors)

        for neighbor in neighbors:
            move_cost = get_traffic_cost(neighbor, other_paths_sets)

            # 进入非主干道增加代价，鼓励在主路上跑
            n_type = grid_type_map[neighbor[0]][neighbor[1]]
            if n_type in [TYPE_STORAGE, TYPE_ELEVATOR]:
                move_cost += 0.5

            new_g = g + move_cost

            if neighbor not in g_scores or new_g < g_scores[neighbor]:
                g_scores[neighbor] = new_g
                h = manhattan_distance(neighbor, goal)
                f_new = new_g + h

                heapq.heappush(open_set, (f_new, new_g, neighbor, path + [neighbor]))

    return []


def is_valid_position(pos):
    return 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS