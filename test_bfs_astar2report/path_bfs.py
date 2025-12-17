# path_bfs.py
import collections
import random
from config import *


def get_valid_neighbors_bfs(grid_type, current, rows, cols):
    r, c = current
    g_type = grid_type[r][c]
    candidates = []

    # BFS 的移动规则与 A* 保持一致
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

    candidates.append((0, 0))  # 等待

    neighbors = []
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            n_type = grid_type[nr][nc]
            if n_type == TYPE_NONE: continue
            # 简单逆行检查
            if n_type == TYPE_MAIN_H_EAST and dc == -1: continue
            if n_type == TYPE_MAIN_H_WEST and dc == 1: continue
            if n_type == TYPE_MAIN_V_SOUTH and dr == -1: continue
            if n_type == TYPE_MAIN_V_NORTH and dr == 1: continue
            neighbors.append((nr, nc))

    # 主路进支路检查
    if g_type in [TYPE_MAIN_H_EAST, TYPE_MAIN_H_WEST, TYPE_MAIN_V_SOUTH, TYPE_MAIN_V_NORTH, TYPE_MAIN_BIDIRECTIONAL]:
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                t_type = grid_type[nr][nc]
                if t_type == TYPE_STORAGE and dc == 0:  # 仅允许垂直进入
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))
                elif t_type == TYPE_ELEVATOR:
                    if (nr, nc) not in neighbors: neighbors.append((nr, nc))

    return neighbors


def bfs_search(grid_type_map, start, goal, obstacles=None):
    """
    广度优先搜索 (BFS)
    特点：不考虑边权(拥堵)，只找步数最少；遇到障碍无法绕行(或绕行代价极高)
    """
    if obstacles is None: obstacles = set()
    rows = len(grid_type_map)
    cols = len(grid_type_map[0])

    if start == goal: return [start]

    queue = collections.deque([(start, [start])])
    visited = {start}

    max_steps = 30000
    steps = 0

    while queue and steps < max_steps:
        steps += 1
        current, path = queue.popleft()

        if current == goal:
            return path

        neighbors = get_valid_neighbors_bfs(grid_type_map, current, rows, cols)
        random.shuffle(neighbors)  # 增加随机性

        for neighbor in neighbors:
            # 简单的障碍检查：如果邻居在障碍列表中，则视为不通
            # 注意：BFS 通常处理静态图，对于时空障碍，这里简化为"当前有车就不过"
            # 为了能在 simulation_core 中跑通，这里只检查 coordinate 是否在 obstacles (set of (r,c,t))
            # 但 BFS 没时间维，所以我们取 t=0,1..20 的投影？
            # 简化策略：如果 neighbor 在 obstacles 的位置集合中，跳过
            # obstacles 格式是 (r, c, t)，我们需要提取 (r, c)

            # 这里为了公平对比，我们只做静态位置检查：如果此时此刻有车挡着，就不走
            # 但 obstacles 传入的是 set((r,c,t))。
            # 修正：bfs 很难做时空规划。我们把它当作"无视拥堵，只避开绝对死路"的算法。
            # 或者：只避开 obstacles 中 t=path_len 的点？

            path_len = len(path)
            # 检查时空冲突：在 path_len 时刻，neighbor 是否被占
            if (neighbor[0], neighbor[1], path_len) in obstacles:
                continue

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []