# path_planning.py
import heapq
import random
import collections
from config import *

# 论文参数: 路径拥堵系数
ALPHA_M = 0.1


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_traffic_cost(pos, other_paths_sets, obstacles):
    """A* 专用：计算拥堵代价"""
    cost = 1.0
    occupancy = 0
    for path_set in other_paths_sets:
        if pos in path_set: occupancy += 1
    cost += (ALPHA_M * occupancy)

    if pos in obstacles:
        cost += 100.0

    return cost


def get_valid_neighbors(grid_type, current, rows, cols):
    """通用邻居获取逻辑 (A* 和 BFS 共用)"""
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

    neighbors = []
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            n_type = grid_type[nr][nc]
            if n_type == TYPE_NONE: continue
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


def improved_a_star_search(grid_type_map, start, goal, other_paths=None, obstacles=None):
    """A* 算法 (考虑交通拥堵)"""
    if other_paths is None: other_paths = []
    other_paths_sets = [set(p) for p in other_paths]
    if obstacles is None:
        obstacles = set()
    else:
        obstacles = set(obstacles)

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
            move_cost = get_traffic_cost(neighbor, other_paths_sets, obstacles)

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


def bfs_search(grid_type_map, start, goal, other_paths=None, obstacles=None):
    """
    BFS 算法 (广度优先搜索)
    特点：保证最短路径(步数最少)，但不考虑交通权重，且搜索范围是发散的
    """
    if obstacles is None:
        obstacles = set()
    else:
        obstacles = set(obstacles)

    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    # 队列存储: (current_pos, path)
    queue = collections.deque([(start, [start])])
    visited = {start}

    max_steps = 50000
    steps = 0

    while queue and steps < max_steps:
        steps += 1
        current, path = queue.popleft()

        if current == goal:
            return path

        neighbors = get_valid_neighbors(grid_type_map, current, rows, cols)
        # BFS 不需要随机shuffle，顺序遍历即可，或者shuffle增加随机性
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited:
                # BFS 处理障碍：如果有车，视为不可通行 (硬约束)
                if neighbor in obstacles:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []


def is_valid_position(pos):
    return 0 <= pos[0] < ROWS and 0 <= pos[1] < COLS