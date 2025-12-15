# path_planning.py (最终修复版)
from config import ROWS, COLS, ALLEY_ROWS, ALLEY_COLS
from collections import deque
import random


def is_valid_position(pos):
    """【货位检查】检查位置是否有效（不在巷道中，用于起点/终点验证）"""
    row, col = pos
    # 检查边界
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return False
    # 检查是否是巷道
    # 穿梭车不能在巷道位置执行存取货任务，因此此处标记为无效货位
    if row in ALLEY_ROWS or col in ALLEY_COLS:
        return False
    return True


def is_valid_movement_cell(pos):
    """
    【移动检查】检查位置是否是有效的移动单元格（允许在巷道内移动）
    穿梭车必须能够在整个地图范围内移动（包括巷道），只需检查边界。
    """
    row, col = pos
    # 只需要检查边界
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return False
    return True


def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_neighbors(pos):
    """
    获取有效邻居位置
    【关键修复】使用 is_valid_movement_cell，允许小车在巷道中移动。
    """
    row, col = pos
    neighbors = []

    # 四个方向的移动
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_pos = (row + dr, col + dc)
        if is_valid_movement_cell(new_pos):
            neighbors.append(new_pos)

    return neighbors


def improved_a_star_search(grid, start, goal):
    """改进的路径规划算法 - 使用BFS确保找到路径"""
    print(f"路径规划: {start} -> {goal}")

    # 基础检查 - 确保起点和终点是合法的货位
    if not is_valid_position(start):
        print(f"  错误: 起点 {start} 无效 (必须是货位)")
        return []

    if not is_valid_position(goal):
        print(f"  错误: 终点 {goal} 无效 (必须是货位)")
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

        # 获取邻居，get_neighbors 已修正为允许巷道移动
        neighbors = get_neighbors(current_pos)
        # 启发式排序，加速查找
        neighbors.sort(key=lambda pos: manhattan_distance(pos, goal))

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    print(f"  BFS未找到路径，使用备用方案")
    return fallback_path(start, goal)


def fallback_path(start, goal):
    """备用路径规划方案 - 简化为直接路径（仅边界检查）"""
    print(f"  使用简化的备用路径规划")

    path = []
    current = start

    # 最大步数限制，避免无限循环
    max_steps = 2 * (ROWS + COLS)
    steps = 0

    while current != goal and steps < max_steps:
        steps += 1

        # 优先移动行方向
        if current[0] != goal[0]:
            step_r = 1 if goal[0] > current[0] else -1
            next_pos = (current[0] + step_r, current[1])
            # 使用 is_valid_movement_cell 允许通过巷道
            if is_valid_movement_cell(next_pos):
                path.append(next_pos)
                current = next_pos
                continue

        # 然后移动列方向
        if current[1] != goal[1]:
            step_c = 1 if goal[1] > current[1] else -1
            next_pos = (current[0], current[1] + step_c)
            # 使用 is_valid_movement_cell 允许通过巷道
            if is_valid_movement_cell(next_pos):
                path.append(next_pos)
                current = next_pos
                continue

        # 如果无法移动，退出
        break

    # 检查备用路径是否到达终点
    return [start] + path if path and path[-1] == goal else []


def simple_direct_path(start, goal):
    """简化的直接路径规划（保留原有接口）"""
    return improved_a_star_search(None, start, goal)