# path_planning.py (论文复现版 - 带交通拥堵感知的 A*)
import heapq
from config import ROWS, COLS, ALLEY_ROWS, ALLEY_COLS

# 论文中的参数: 主轨道路径利用系数 alpha_m = 0.1
ALPHA_M = 0.1


def is_valid_position(pos):
    """【货位检查】用于起点/终点验证，必须是货位，不能是巷道"""
    row, col = pos
    if row < 0 or row >= ROWS or col < 0 or col >= COLS: return False
    if row in ALLEY_ROWS or col in ALLEY_COLS: return False
    return True


def is_valid_movement_cell(pos):
    """【移动检查】允许在巷道中移动"""
    row, col = pos
    if row < 0 or row >= ROWS or col < 0 or col >= COLS: return False
    return True


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_traffic_cost(pos, other_paths):
    """
    计算该位置的交通拥堵代价 (论文公式 8-3 的核心)
    Cost = (1 + alpha * occupancy)
    """
    occupancy = 0
    # 检查该位置出现在了多少辆其他小车的路径中
    for path in other_paths:
        if pos in path:
            occupancy += 1

    # 基础代价是 1，每多一辆车占用，代价增加 0.1
    # 这样小车会倾向于绕开拥堵路段
    return 1.0 + (ALPHA_M * occupancy)


def improved_a_star_search(grid, start, goal, other_paths=None):
    """
    基于论文的 A* 路径规划算法
    :param other_paths: 其他小车已经规划好的路径列表 [[(r,c),...], [(r,c),...]]
    """
    if other_paths is None: other_paths = []

    # 转换 other_paths 为集合列表以加速查找 (优化性能)
    # 我们只关心未来的路径，所以这里假设 other_paths 包含的是它们未来的轨迹
    other_paths_sets = [set(p) for p in other_paths]

    print(f"A* 规划: {start} -> {goal} (参考其他 {len(other_paths)} 条路径)")

    if not is_valid_position(start) and not is_valid_movement_cell(start):
        return []  # 起点完全越界
    if not is_valid_position(goal):
        return []  # 终点必须是货位

    # 优先队列: (f_score, g_score, current_pos, path)
    # f = g + h
    open_set = []
    heapq.heappush(open_set, (0, 0, start, [start]))

    # 记录到达每个点的最小代价 g_score，防止走回头路
    g_scores = {start: 0}

    # 为了避免死循环，设置最大搜索步数
    max_steps = 5000
    steps = 0

    while open_set and steps < max_steps:
        steps += 1
        # 取出 f 最小的节点
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            print(f"  规划成功: 长度 {len(path)}, 代价 {g:.2f}")
            return path

        row, col = current
        # 四个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            neighbor = (row + dr, col + dc)

            if is_valid_movement_cell(neighbor):
                # 1. 计算移动代价 (关键步骤)
                # 基础代价(1) + 拥堵惩罚
                move_cost = get_traffic_cost(neighbor, other_paths_sets)
                new_g = g + move_cost

                # 2. 如果找到了更优路径 (或者该节点未被访问过)
                if neighbor not in g_scores or new_g < g_scores[neighbor]:
                    g_scores[neighbor] = new_g
                    h = manhattan_distance(neighbor, goal)
                    f_new = new_g + h
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_new, new_g, neighbor, new_path))

    print("  A* 规划失败: 未找到路径")
    return []