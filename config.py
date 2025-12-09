# config.py

# 地图尺寸
ROWS = 20  # 行
COLS = 46  # 列

# 穿梭车数量
SHUTTLES = 2  # 穿梭车数量

# 货物运输时间
MOVE_TIME = 1  # 单次移动时间
PARK_TIME = 2  # 单次启停时间
TRANS_TIME = 2  # 单次换向时间
RETRIEVE_TIME = 2  # 单次取（放）货时间

# 屏幕设置
CELL_SIZE = 25  # 单元格大小
SCREEN_WIDTH = COLS * CELL_SIZE + 300  # 增加信息面板宽度
SCREEN_HEIGHT = ROWS * CELL_SIZE  # 屏幕高度

# 巷道行列（从0开始）
ALLEY_COLS = [1, 7, 14, 21, 28, 35, 41]  # 巷道列
ALLEY_ROWS = [9, 15]  # 巷道行

# 颜色定义 - 修复颜色映射
COLORS = {
    # 基础颜色映射（对应 map.py 中的数字）
    0: (173, 216, 230),  # 巷道 - 浅蓝
    1: (240, 240, 240),  # 空闲货位 - 浅灰色
    2: (255, 215, 0),  # 占用货位 - 金色
    3: (255, 0, 0),  # 小车0 - 红色
    4: (0, 255, 0),  # 小车1 - 绿色
    5: (0, 0, 255),  # 小车2 - 蓝色
    6: (128, 0, 128),  # 小车3 - 紫色
    7: (255, 255, 0),  # 提升机 - 黄色
    8: (0, 0, 0),  # 目标点 - 黑色

    # 别名（保持向后兼容）
    "alley": (173, 216, 230),
    "empty": (240, 240, 240),
    "occupied": (255, 215, 0),
    "elevator": (255, 255, 0),
    "target": (0, 0, 0),

    # 穿梭车颜色数组
    "shuttle_colors": [
        (255, 0, 0),  # 红色穿梭车1
        (0, 255, 0),  # 绿色穿梭车2
        (0, 0, 255),  # 蓝色穿梭车3
        (128, 0, 128),  # 紫色穿梭车4
    ],

    # 新增：信息面板颜色
    "panel_bg": (60, 63, 65),
    "text": (220, 220, 220),
    "highlight": (255, 203, 0),
}

# 日志文件
LOG_FILE = 'simulation.log'
CSV_LOG_FILE = 'task_log.csv'


def validate_alley_config():
    """验证巷道配置"""
    print("验证巷道配置...")
    valid = True

    # 检查巷道行是否在范围内
    for row in ALLEY_ROWS:
        if row < 0 or row >= ROWS:
            print(f"错误: 巷道行 {row} 超出范围 [0, {ROWS - 1}]")
            valid = False

    # 检查巷道列是否在范围内
    for col in ALLEY_COLS:
        if col < 0 or col >= COLS:
            print(f"错误: 巷道列 {col} 超出范围 [0, {COLS - 1}]")
            valid = False

    if valid:
        print("巷道配置验证通过")
    else:
        print("巷道配置验证失败！")

    return valid


# 在导入时验证配置
if __name__ == "__main__":
    validate_alley_config()
else:
    validate_alley_config()