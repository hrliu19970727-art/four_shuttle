# config.py
# --- 地图尺寸 (User: 29 Rows, 50 Cols) ---
ROWS = 29
COLS = 60
CELL_SIZE = 20  # 稍微调小一点以便在屏幕上显示全貌

# --- 实体数量 ---
SHUTTLES = 6

# --- 时间参数 ---
MOVE_TIME = 0.2
PARK_TIME = 1.0
RETRIEVE_TIME = 1.0

# --- 地图元素类型 ---
TYPE_NONE = 0
TYPE_STORAGE = 1        # 存储巷道 (双向)
TYPE_MAIN_H_EAST = 2    # 主轨道 >
TYPE_MAIN_H_WEST = 3    # 主轨道 <
TYPE_MAIN_V_SOUTH = 4   # 主轨道 v
TYPE_MAIN_V_NORTH = 5   # 主轨道 ^
TYPE_ELEVATOR = 6       # 提升机
TYPE_INTERSECTION = 7   # 交叉口
TYPE_MAIN_BIDIRECTIONAL = 8 # 双向主轨道 (新增，用于断头路或连接路)

# --- 颜色定义 ---
COLORS = {
    "bg": (30, 32, 36),
    TYPE_NONE: (40, 42, 46),
    TYPE_STORAGE: (200, 200, 200),
    TYPE_MAIN_H_EAST: (60, 160, 240),
    TYPE_MAIN_H_WEST: (60, 160, 240),
    TYPE_MAIN_V_SOUTH: (60, 160, 240),
    TYPE_MAIN_V_NORTH: (60, 160, 240),
    TYPE_MAIN_BIDIRECTIONAL: (100, 180, 255), # 稍浅的蓝色
    TYPE_INTERSECTION: (150, 220, 255),
    TYPE_ELEVATOR: (255, 200, 50),
    "occupied": (255, 100, 100),
    "panel_bg": (50, 52, 57),
    "text": (240, 240, 240),
    "highlight": (255, 200, 50),
}

# 废弃旧配置，由 map.py 动态生成
ALLEY_ROWS = []
ALLEY_COLS = []