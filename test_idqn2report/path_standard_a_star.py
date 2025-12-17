# test_standastar2report/path_standard_a_star.py
import heapq
from config import *


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_valid_neighbors_standard(grid_type, current, rows, cols):
    """
    获取静态地图上可行的邻居节点
    仅考虑轨道连通性
    """
    r, c = current
    g_type = grid_type[r][c]
    candidates = []

    # 1. 基础移动方向规则 (单行道限制)
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

    # 2. 允许原地等待 (Wait Action) -> 关键：分时复用的基础
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

    # 3. 进出货架的物理约束 (禁止横向穿墙进入巷道)
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


def standard_a_star_search(grid_type_map, start, goal, slots_map=None, is_loaded=False, reservations=None):
    """
    标准 A* 算法 (Space-Time 升级版)

    参数:
    - reservations: 动态障碍物集合 set((r, c, t))，用于分时复用

    特点:
    1. 【遵守】物理规则：载重限行。
    2. 【遵守】时空规则：不与已知轨迹碰撞 (reservations)。
    3. 【忽略】拥堵优化：路径代价恒定为 1，不考虑哪里车多，只求能走通的最短路。
    """
    if reservations is None: reservations = set()

    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    # OpenList: (f_score, g_score, row, col, time, path)
    # 增加 time 维度
    open_set = []
    heapq.heappush(open_set, (0, 0, start[0], start[1], 0, [start]))

    # Visited 集合记录 (r, c, t) 及其最小代价
    # 如果只记录 (r,c)，会导致无法“等待” (因为等待后 t 变了，但 (r,c) 没变)
    visited = {}

    # 限制搜索深度，防止在无解时无限等待
    max_time_steps = 300
    max_calculations = 40000
    calcs = 0

    while open_set and calcs < max_calculations:
        calcs += 1
        f, g, r, c, t, path = heapq.heappop(open_set)

        if (r, c) == goal:
            return path

        if t >= max_time_steps: continue

        neighbors = get_valid_neighbors_standard(grid_type_map, (r, c), rows, cols)

        for nr, nc in neighbors:
            nt = t + 1

            # === 1. 动态障碍检查 (时空分时复用) ===
            # 如果下一时刻该位置被占用，则跳过
            if (nr, nc, nt) in reservations:
                continue
            # ==================================

            # === 2. 静态物理约束 (载重限行) ===
            if is_loaded and slots_map is not None:
                if grid_type_map[nr][nc] == TYPE_STORAGE:
                    if slots_map[nr][nc] == 2:
                        if (nr, nc) != goal:
                            continue
                            # =================================

            # 标准 A* 代价恒定为 1.0 (Wait 代价也为 1.0)
            step_cost = 1.0
            new_g = g + step_cost

            # 状态 Key 包含时间 t
            state_key = (nr, nc, nt)

            if state_key not in visited or new_g < visited[state_key]:
                visited[state_key] = new_g
                h = manhattan_distance((nr, nc), goal)
                f_new = new_g + h

                new_path = path + [(nr, nc)]
                heapq.heappush(open_set, (f_new, new_g, nr, nc, nt, new_path))

    return []