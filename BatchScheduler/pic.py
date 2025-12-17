import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_warehouse_layout():
    # --- 1. 画布与坐标系设置 ---
    # 创建一个宽敞的画布，比例适应仓库布局
    fig, ax = plt.subplots(figsize=(20, 12))
    # 设置背景色，模仿监控界面的风格
    ax.set_facecolor('#E6E6FA')

    # 设置坐标轴范围，留出边缘以容纳外部设备和标注
    # 根据原图，X轴主体从8到50，Y轴从4到29，加上外部结构，设定如下范围
    ax.set_xlim(0, 56)
    ax.set_ylim(0, 32)
    # 保证 X 和 Y 轴的单位长度在视觉上相等，使货格呈正方形
    ax.set_aspect('equal')

    # --- 2. 颜色定义 ---
    COLOR_SHELF = '#4682B4'  # 货架颜色 (钢蓝)
    COLOR_LANE = '#D3D3D3'  # 巷道/轨道颜色 (浅灰)
    COLOR_ELEVATOR = '#778899'  # 提升机颜色 (岩灰)
    COLOR_CONVEYOR = '#A9A9A9'  # 输送线区域颜色 (暗灰)
    COLOR_BORDER = 'black'  # 边框颜色

    # ==========================================
    # 核心绘制逻辑：从基础货架到复杂设备
    # ==========================================

    # --- 3. 绘制货架存储区 (Storage Racks) ---
    # 根据图中标尺，货架被水平巷道分隔成四个主要区块。
    # X轴方向主要集中在 8 到 46 之间。
    shelf_y_ranges = [
        range(5, 13),  # 底部区块 (Y=4巷道上方)
        range(14, 21),  # 中部区块1 (Y=13巷道上方)
        range(22, 27),  # 中部区块2 (Y=21巷道上方)
        range(28, 30)  # 顶部区块 (Y=27巷道上方)
    ]

    for y_range in shelf_y_ranges:
        for y in y_range:
            for x in range(8, 47):  # X范围: 8 至 46
                # 绘制单个货格，假设每个是一个 1x1 的正方形
                rect = patches.Rectangle((x, y), 1, 1,
                                         linewidth=0.5, edgecolor=COLOR_BORDER, facecolor=COLOR_SHELF)
                ax.add_patch(rect)

    # --- 4. 绘制主巷道系统 (Main Aisles) ---
    # A. 水平巷道：位于货架区块之间，用于穿梭车横向移动。
    # 它们横跨整个主要存储区，连接左右两条垂直巷道 (X=6 到 X=48)。
    horizontal_lanes_y = [4, 13, 21, 27]
    for y_lane in horizontal_lanes_y:
        # 宽度计算: 48 - 6 + 1 = 43
        rect = patches.Rectangle((6, y_lane), 43, 1,
                                 linewidth=0.5, edgecolor=COLOR_BORDER, facecolor=COLOR_LANE)
        ax.add_patch(rect)

    # B. 垂直巷道：位于货架区两侧，用于连接不同层和提升机。
    # 贯穿整个仓库高度。
    vertical_lanes_x = [6, 48]
    for x_lane in vertical_lanes_x:
        # 高度计算: 从底部 Y=4 到顶部 Y=29，高度约为 26
        rect = patches.Rectangle((x_lane, 4), 1, 26,
                                 linewidth=0.5, edgecolor=COLOR_BORDER, facecolor=COLOR_LANE)
        ax.add_patch(rect)

    # --- 5. 绘制关键设备 ---

    # A. 提升机 (Elevators)：位于垂直巷道的外侧，共4台。
    # 定义位置和标签：(x坐标, y坐标, 标签文本, 标签x轴偏移量)
    elevators = [
        (4, 19, "提升机1", -1.5),  # 左上
        (4, 11, "提升机2", -1.5),  # 左下
        (49, 19, "提升机3", 1.5),  # 右上
        (49, 11, "提升机4", 1.5)  # 右下
    ]
    for (ex, ey, label, off_x) in elevators:
        # 绘制提升机井道示意图 (假设为 2x1 的空间)
        rect = patches.Rectangle((ex, ey), 2, 1, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_ELEVATOR)
        ax.add_patch(rect)
        # 添加文本标签
        ax.text(ex + 1 + off_x, ey + 0.5, label, ha='center', va='center', fontsize=9, fontweight='bold')

    # B. 左侧入库区域 (Inbound Area)：包含输送线和入口。
    # 通过拼接多个矩形来模拟原图中的复杂结构。
    # 主入口连接段，连接到 Y=14 的水平巷道
    ax.add_patch(patches.Rectangle((2, 14), 4, 1, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_CONVEYOR))
    # 左下角的辅助输送线结构
    for y in [15, 16, 17, 18]:
        ax.add_patch(patches.Rectangle((2, y), 2, 1, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_CONVEYOR))
    for y in [12, 13]:
        ax.add_patch(patches.Rectangle((4, y), 2, 1, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_CONVEYOR))
    # 最左侧的入库口位置，用绿色高亮
    ax.add_patch(patches.Rectangle((0, 14), 2, 2, linewidth=1, edgecolor=COLOR_BORDER, facecolor='#90EE90'))
    # 底部最左侧的辅助区域
    ax.add_patch(patches.Rectangle((0, 4), 6, 2, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_CONVEYOR))

    # C. 右侧出库区域 (Outbound Area)：包含输送线、出口和叠盘机。
    # 主出口连接段，从 Y=14 的水平巷道延伸出来
    ax.add_patch(patches.Rectangle((49, 14), 4, 1, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_CONVEYOR))
    # 右侧的辅助结构
    ax.add_patch(patches.Rectangle((51, 15), 2, 4, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_CONVEYOR))
    # 右下角的“叠盘机”区域
    ax.add_patch(patches.Rectangle((49, 4), 6, 4, linewidth=1, edgecolor=COLOR_BORDER, facecolor=COLOR_CONVEYOR))
    ax.text(52, 6, "叠盘机", ha='center', va='center', fontsize=9, fontweight='bold')
    # 最右侧的出库口位置，用金色高亮
    ax.add_patch(patches.Rectangle((53, 14), 2, 1, linewidth=1, edgecolor=COLOR_BORDER, facecolor='#FFD700'))

    # --- 6. 标注与装饰 (Annotations) ---

    # A. 模拟原图的坐标轴标尺
    # X轴标尺：在顶部显示，刻度从 8 到 50，步长为 2
    x_ticks = list(range(8, 51, 2))
    for x in x_ticks:
        ax.text(x + 0.5, 30.5, str(x), ha='center', fontsize=8)

    # Y轴标尺：在左侧显示，刻度不均匀
    y_ticks = [4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 23, 25, 27, 29]
    for y in y_ticks:
        ax.text(5.5, y + 0.5, str(y), ha='right', va='center', fontsize=8)

    # B. 入库和出库指示箭头与文本
    # 入库指示 (左侧)
    ax.arrow(-2, 15, 2, 0, head_width=0.8, head_length=0.8, fc='green', ec='green')
    ax.text(-3, 15, "入库", ha='right', va='center', fontsize=12, fontweight='bold', color='green')
    # 出库指示 (右侧)
    ax.arrow(55, 14.5, 2, 0, head_width=0.8, head_length=0.8, fc='orange', ec='orange')
    ax.text(58, 14.5, "出库", ha='left', va='center', fontsize=12, fontweight='bold', color='orange')

    # 隐藏默认的坐标轴线和刻度，使用自定义的标尺
    ax.axis('off')

    # 添加图表标题
    plt.title("四向穿梭车仓库布局示意图 (基于原图绘制)", fontsize=14, pad=20)

    # 显示最终绘制的图像
    plt.show()


# 执行绘图函数
if __name__ == "__main__":
    draw_warehouse_layout()