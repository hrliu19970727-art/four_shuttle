# config.py
# --- 地图尺寸 ---
ROWS = 29
COLS = 50
CELL_SIZE = 20

# --- 算法选择 ---
# 可选值: "A_STAR" 或 "BFS"
PLANNING_ALGORITHM = "A_STAR"
# PLANNING_ALGORITHM = "BFS"

# --- 实体数量 ---
SHUTTLES = 4

# --- 时间参数 ---
MOVE_TIME = 0.2
PARK_TIME = 1.0
RETRIEVE_TIME = 1.0

# --- 地图元素类型 ---
TYPE_NONE = 0
TYPE_STORAGE = 1
TYPE_MAIN_H_EAST = 2
TYPE_MAIN_H_WEST = 3
TYPE_MAIN_V_SOUTH = 4
TYPE_MAIN_V_NORTH = 5
TYPE_ELEVATOR = 6
TYPE_INTERSECTION = 7
TYPE_MAIN_BIDIRECTIONAL = 8

# --- 颜色定义 ---
COLORS = {
    "bg": (30, 32, 36),
    TYPE_NONE: (40, 42, 46),
    TYPE_STORAGE: (200, 200, 200),
    TYPE_MAIN_H_EAST: (60, 160, 240),
    TYPE_MAIN_H_WEST: (60, 160, 240),
    TYPE_MAIN_V_SOUTH: (60, 160, 240),
    TYPE_MAIN_V_NORTH: (60, 160, 240),
    TYPE_MAIN_BIDIRECTIONAL: (100, 180, 255),
    TYPE_INTERSECTION: (150, 220, 255),
    TYPE_ELEVATOR: (255, 200, 50),
    "occupied": (255, 100, 100),
    "panel_bg": (50, 52, 57),
    "text": (240, 240, 240),
    "highlight": (255, 200, 50),
}

ALLEY_ROWS = []
ALLEY_COLS = []