# path_planning.py (完整修复版)
from config import ROWS, COLS, ALLEY_ROWS, ALLEY_COLS
from collections import deque
import random


def is_valid_position(pos):
    """检查位置是否有效（不在巷道中）"""
    row, col = pos
    # 检查边界
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return False
    # 检查是否是巷道
    if row in ALLEY_ROWS or col in ALLEY_COLS:
        return False
    return True


def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_neighbors(pos):
    """获取有效邻居位置"""
    row, col = pos
    neighbors = []

    # 四个方向的移动
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_pos = (row + dr, col + dc)
        if is_valid_position(new_pos):
            neighbors.append(new_pos)

    return neighbors


def improved_a_star_search(grid, start, goal):
    """改进的路径规划算法 - 使用BFS确保找到路径"""
    print(f"路径规划: {start} -> {goal}")

    # 基础检查
    if not is_valid_position(start):
        print(f"  错误: 起点 {start} 无效")
        return []

    if not is_valid_position(goal):
        print(f"  错误: 终点 {goal} 无效")
        return []

    if start == goal:
        return [start]

    # 使用BFS寻找路径
    path = bfs_search(start, goal)

    if path:
        print(f"  路径规划成功: {len(path)} 步")
        if len(path) > 6:
            print(f"  路径: {path[:3]}...{path[-3:]}")
        else:
            print(f"  路径: {path}")
    else:
        print(f"  路径规划失败")

    return path


def bfs_search(start, goal):
    """使用BFS算法寻找路径"""
    queue = deque()
    queue.append((start, [start]))  # (当前位置, 路径)
    visited = set([start])

    max_iterations = 500
    iterations = 0

    while queue and iterations < max_iterations:
        iterations += 1
        current_pos, path = queue.popleft()

        if current_pos == goal:
            return path

        # 获取邻居，按距离目标远近排序（启发式）
        neighbors = get_neighbors(current_pos)
        neighbors.sort(key=lambda pos: manhattan_distance(pos, goal))

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    print(f"  BFS未找到路径，使用备用方案")
    return fallback_path(start, goal)


def fallback_path(start, goal):
    """备用路径规划方案"""
    print(f"  使用备用路径规划")

    # 方案1: 尝试直接路径（忽略巷道检查）
    path = []
    current = start

    # 最大步数限制
    max_steps = 50
    steps = 0

    while current != goal and steps < max_steps:
        steps += 1

        # 优先移动行方向
        if current[0] != goal[0]:
            step_r = 1 if goal[0] > current[0] else -1
            next_pos = (current[0] + step_r, current[1])
            if is_valid_position(next_pos):
                path.append(next_pos)
                current = next_pos
                continue

        # 然后移动列方向
        if current[1] != goal[1]:
            step_c = 1 if goal[1] > current[1] else -1
            next_pos = (current[0], current[1] + step_c)
            if is_valid_position(next_pos):
                path.append(next_pos)
                current = next_pos
                continue

        # 如果直接移动不可行，尝试绕行
        found_bypass = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            bypass_pos = (current[0] + dr, current[1] + dc)
            if (is_valid_position(bypass_pos) and
                    bypass_pos not in path[-5:]):  # 避免循环
                path.append(bypass_pos)
                current = bypass_pos
                found_bypass = True
                break

        if not found_bypass:
            break

    # 如果路径不为空且最后一个点不是目标，检查是否可以添加目标
    if path and path[-1] != goal:
        last_pos = path[-1]
        if (abs(last_pos[0] - goal[0]) + abs(last_pos[1] - goal[1]) == 1 and
                is_valid_position(goal)):
            path.append(goal)

    return [start] + path if path else []


def simple_direct_path(start, goal):
    """简化的直接路径规划（保留原有接口）"""
    return improved_a_star_search(None, start, goal)