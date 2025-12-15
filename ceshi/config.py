# config.py
# --- 地图尺寸 ---
ROWS = 20
COLS = 46
CELL_SIZE = 25

# --- 实体数量 ---
SHUTTLES = 4  # 4台小车

# --- 时间参数 (秒) ---
MOVE_TIME = 0.5  # 移动一格耗时
PARK_TIME = 2.0  # 放货耗时
RETRIEVE_TIME = 2.0  # 取货耗时

# --- 地图元素类型定义 ---
TYPE_NONE = 0
TYPE_STORAGE = 1  # 存储巷道 (双向，容量1)
TYPE_MAIN_H_EAST = 2  # 主轨道-东向 (>)
TYPE_MAIN_H_WEST = 3  # 主轨道-西向 (<)
TYPE_MAIN_V_SOUTH = 4  # 主轨道-南向 (v)
TYPE_MAIN_V_NORTH = 5  # 主轨道-北向 (^)
TYPE_ELEVATOR = 6  # 提升机/端口
TYPE_INTERSECTION = 7  # 交叉口

# --- 颜色定义 (Dark Mode) ---
COLORS = {
    "bg": (30, 32, 36),
    # 轨道颜色
    TYPE_NONE: (50, 50, 55),
    TYPE_STORAGE: (200, 200, 200),  # 浅灰
    TYPE_MAIN_H_EAST: (60, 160, 240),  # 亮蓝
    TYPE_MAIN_H_WEST: (60, 160, 240),
    TYPE_MAIN_V_SOUTH: (60, 160, 240),
    TYPE_MAIN_V_NORTH: (60, 160, 240),
    TYPE_INTERSECTION: (100, 200, 255),
    TYPE_ELEVATOR: (255, 200, 50),  # 金色

    # 状态
    "occupied": (255, 100, 100),  # 红色(被占用)
    "panel_bg": (50, 52, 57),
    "text": (240, 240, 240),
    "highlight": (255, 200, 50),
}

# 巷道配置 (用于旧逻辑兼容，新逻辑主要依靠 grid_type)
ALLEY_ROWS = [2, 9, 16]
ALLEY_COLS = [1, 15, 30, 44]