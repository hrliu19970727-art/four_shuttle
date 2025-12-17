# path_standard_a_star.py
import heapq
from config import *


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_valid_neighbors_standard(grid_type, current, rows, cols):
    """
    获取静态地图上可行的邻居节点
    仅考虑轨道连通性，不考虑动态障碍
    """
    r, c = current
    g_type = grid_type[r][c]
    candidates = []

    # 基础移动方向规则
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
        # 边界检查
        if 0 <= nr < rows and 0 <= nc < cols:
            n_type = grid_type[nr][nc]
            if n_type != TYPE_NONE:
                # 逆行检查 (物理约束)
                if n_type == TYPE_MAIN_H_EAST and dc == -1: continue
                if n_type == TYPE_MAIN_H_WEST and dc == 1: continue
                if n_type == TYPE_MAIN_V_SOUTH and dr == -1: continue
                if n_type == TYPE_MAIN_V_NORTH and dr == 1: continue
                neighbors.append((nr, nc))

    # 进出货架的物理约束 (禁止横向穿墙进入巷道)
    final_neighbors = []
    for nr, nc in neighbors:
        dr, dc = nr - r, nc - c
        # 如果当前在主路上
        if grid_type[r][c] in [TYPE_MAIN_H_EAST, TYPE_MAIN_H_WEST, TYPE_MAIN_V_SOUTH, TYPE_MAIN_V_NORTH]:
            t_type = grid_type[nr][nc]
            # 且目标是存储位，必须垂直进入 (dc=0)
            if t_type == TYPE_STORAGE and dc != 0: continue
        final_neighbors.append((nr, nc))

    return final_neighbors


def standard_a_star_search(grid_type_map, start, goal, slots_map=None, is_loaded=False):
    """
    标准 A* 算法 (修正版)

    参数:
    - slots_map: 全局货位状态矩阵 (用于检查是否有货)
    - is_loaded: 当前小车是否载货

    特点:
    1. 【遵守】物理规则：如果载货，严禁穿过有货的存储位。
    2. 【忽略】动态规则：不考虑其他小车未来的位置 (reservations)。
    3. 【忽略】拥堵成本：所有路段代价相同 (Cost=1)。
    """
    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    # OpenList: (f_score, g_score, row, col, path)
    open_set = []
    heapq.heappush(open_set, (0, 0, start[0], start[1], [start]))

    # Visited 集合记录 (r, c) 及其最小代价
    g_scores = {start: 0}

    max_steps = 30000
    steps = 0

    while open_set and steps < max_steps:
        steps += 1
        f, g, r, c, path = heapq.heappop(open_set)

        if (r, c) == goal:
            return path

        neighbors = get_valid_neighbors_standard(grid_type_map, (r, c), rows, cols)

        for nr, nc in neighbors:
            # === 核心修改：载重物理约束检查 ===
            # 如果小车载货(is_loaded=True) 且 目标格有货(slots=2) 且 目标格不是终点
            # 则视为物理障碍，不可通行
            if is_loaded and slots_map is not None:
                if grid_type_map[nr][nc] == TYPE_STORAGE:
                    if slots_map[nr][nc] == 2:
                        if (nr, nc) != goal:
                            continue
                            # =================================

            # 标准 A* 代价恒定为 1 (不考虑拥堵)
            step_cost = 1.0
            new_g = g + step_cost

            if (nr, nc) not in g_scores or new_g < g_scores[(nr, nc)]:
                g_scores[(nr, nc)] = new_g
                h = manhattan_distance((nr, nc), goal)
                f_new = new_g + h

                new_path = path + [(nr, nc)]
                heapq.heappush(open_set, (f_new, new_g, nr, nc, new_path))

    return []