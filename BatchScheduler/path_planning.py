# path_planning.py
import heapq
import random
from config import *

# 拥堵惩罚系数
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
    获取几何上可行的邻居 (不考虑货位占用，仅考虑轨道连通性)
    """
    r, c = current
    g_type = grid_type[r][c]

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

    # 允许原地等待
    candidates.append((0, 0))

    neighbors = []
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            n_type = grid_type[nr][nc]
            if n_type == TYPE_NONE: continue

            # 基础逆行检查
            if n_type == TYPE_MAIN_H_EAST and dc == -1: continue
            if n_type == TYPE_MAIN_H_WEST and dc == 1: continue
            if n_type == TYPE_MAIN_V_SOUTH and dr == -1: continue
            if n_type == TYPE_MAIN_V_NORTH and dr == 1: continue

            neighbors.append((nr, nc))

    # 特殊规则：主路进支路 (侧向进入)
    if g_type in [TYPE_MAIN_H_EAST, TYPE_MAIN_H_WEST, TYPE_MAIN_V_SOUTH, TYPE_MAIN_V_NORTH, TYPE_MAIN_BIDIRECTIONAL]:
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                t_type = grid_type[nr][nc]
                if t_type == TYPE_STORAGE:
                    if dc != 0: continue  # 禁止横向穿墙入巷道
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))
                elif t_type == TYPE_ELEVATOR:
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))

    return neighbors


def improved_a_star_search(grid_type_map, start, goal, other_paths=None, reservations=None, slots_map=None,
                           is_loaded=False):
    """
    Space-Time A* (带载重约束)
    :param slots_map: 全局货位状态矩阵 (0:无, 1:空, 2:有货)
    :param is_loaded: 当前小车是否载货
    """
    if reservations is None: reservations = set()

    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    open_set = []
    # (f, g, r, c, t, path)
    heapq.heappush(open_set, (0, 0, start[0], start[1], 0, [start]))

    visited = {}
    max_time_steps = 300
    max_calculations = 40000
    calcs = 0

    while open_set and calcs < max_calculations:
        calcs += 1
        f, g, r, c, t, path = heapq.heappop(open_set)

        if (r, c) == goal:
            return path

        if t >= max_time_steps: continue

        neighbors = get_valid_neighbors(grid_type_map, (r, c), rows, cols)
        random.shuffle(neighbors)

        for nr, nc in neighbors:
            nt = t + 1

            # --- 1. 时空冲突 (动态障碍: 其他车) ---
            if (nr, nc, nt) in reservations:
                continue

            # --- 2. 载重物理约束 (静态障碍: 货位) ---
            # 规则: 载货小车不能穿过有货的存储位 (除非那是目标位)
            if is_loaded and slots_map is not None:
                # 如果是存储巷道
                if grid_type_map[nr][nc] == TYPE_STORAGE:
                    # 且该位置有货 (状态=2)
                    if slots_map[nr][nc] == 2:
                        # 且该位置不是终点 (如果是终点，说明是去放货，允许进入)
                        if (nr, nc) != goal:
                            continue

                            # 计算代价
            step_cost = 1.0
            if (nr, nc) == (r, c): step_cost = 1.1  # 等待代价

            # 载货时走存储巷道代价更高 (鼓励走主路)
            if is_loaded and grid_type_map[nr][nc] == TYPE_STORAGE:
                step_cost += 0.5

            new_g = g + step_cost
            state_key = (nr, nc, nt)

            if state_key not in visited or new_g < visited[state_key]:
                visited[state_key] = new_g
                h = manhattan_distance((nr, nc), goal)
                f_new = new_g + h

                new_path = path + [(nr, nc)]
                heapq.heappush(open_set, (f_new, new_g, nr, nc, nt, new_path))

    return []


def is_valid_position(pos):
    return 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS