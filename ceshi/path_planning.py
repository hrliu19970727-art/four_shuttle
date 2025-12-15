# path_planning.py
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
    【关键修复】允许从主轨道驶入相邻的存储巷道
    """
    r, c = current
    g_type = grid_type[r][c]

    # 基础候选方向 (根据单向规则)
    candidates = []
    if g_type == TYPE_MAIN_H_EAST:
        candidates = [(0, 1)]
    elif g_type == TYPE_MAIN_H_WEST:
        candidates = [(0, -1)]
    elif g_type == TYPE_MAIN_V_SOUTH:
        candidates = [(1, 0)]
    elif g_type == TYPE_MAIN_V_NORTH:
        candidates = [(-1, 0)]
    else:
        candidates = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 交叉口/巷道/电梯全向

    neighbors = []

    # 1. 首先添加符合单向规则的邻居
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            n_type = grid_type[nr][nc]
            # 基础物理检查 + 逆行检查
            if n_type != TYPE_NONE:
                # 逆行保护
                if n_type == TYPE_MAIN_H_EAST and dc == -1:
                    pass
                elif n_type == TYPE_MAIN_H_WEST and dc == 1:
                    pass
                else:
                    neighbors.append((nr, nc))

    # 2. 【新增】特殊规则：允许从主轨道“侧向”进入存储巷道
    # 如果当前在主轨道，检查上下左右是否有存储巷道
    if g_type in [TYPE_MAIN_H_EAST, TYPE_MAIN_H_WEST, TYPE_MAIN_V_SOUTH, TYPE_MAIN_V_NORTH]:
        all_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in all_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # 如果邻居是存储位，允许转向进入（这就解决了无法下高速的问题）
                if grid_type[nr][nc] == TYPE_STORAGE:
                    if (nr, nc) not in neighbors:  # 避免重复添加
                        neighbors.append((nr, nc))

    return neighbors


def improved_a_star_search(grid_type_map, start, goal, other_paths=None):
    """带交通规则 A*"""
    if other_paths is None: other_paths = []
    other_paths_sets = [set(p) for p in other_paths]

    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    open_set = []
    heapq.heappush(open_set, (0, 0, start, [start]))
    g_scores = {start: 0}

    max_steps = 20000
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

            # 进入存储巷道增加一点代价，鼓励优先走主路
            if grid_type_map[neighbor[0]][neighbor[1]] == TYPE_STORAGE:
                move_cost += 0.5

            new_g = g + move_cost

            if neighbor not in g_scores or new_g < g_scores[neighbor]:
                g_scores[neighbor] = new_g
                h = manhattan_distance(neighbor, goal)
                # 目标在主路上，增加h权重，引导尽快汇入
                f_new = new_g + h

                heapq.heappush(open_set, (f_new, new_g, neighbor, path + [neighbor]))

    print(f"  规划失败: {start}->{goal}")
    return []


def is_valid_position(pos):
    return 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS