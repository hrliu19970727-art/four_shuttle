# map.py
import numpy as np
import pygame
from config import ROWS, COLS, CELL_SIZE, COLORS, ALLEY_ROWS, ALLEY_COLS


class WarehouseMap:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        # 0:巷道, 1:空闲货位, 2:占用货位, 3-6:穿梭车, 7:提升机, 8:目标点
        self.map = np.zeros((self.rows, self.cols), dtype=int)
        # 0:不可用, 1:空闲, 2:占用
        self.slots = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        # 存储小车和电梯位置
        self.shuttles = {}  # {shuttle_id: (row, col)}
        self.elevators = []  # [(row, col)]

        # 初始化地图 - 根据巷道设置
        for row in range(self.rows):
            for col in range(self.cols):
                # 如果是巷道行或列，设置为巷道
                if row in ALLEY_ROWS or col in ALLEY_COLS:
                    self.map[row][col] = 0  # 巷道
                    self.slots[row][col] = 0  # 不可用
                else:
                    self.map[row][col] = 1  # 空闲货位
                    self.slots[row][col] = 1  # 空闲

        print(f"地图初始化完成: {self.rows} x {self.cols}")
        print(f"巷道行: {ALLEY_ROWS}, 巷道列: {ALLEY_COLS}")

    def update_slot(self, row, col, status):
        """更新货位状态"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # 只有不是巷道的位置才能更新状态
            if self.map[row][col] != 0:
                self.slots[row][col] = status
                if status == 1:  # 空闲
                    self.map[row][col] = 1
                elif status == 2:  # 占用
                    self.map[row][col] = 2
                return True
        return False

    def add_obstacle(self, row, col):
        """添加障碍物"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.map[row][col] = 2
            self.slots[row][col] = 2
            return True
        return False

    def add_shuttle(self, row, col, shuttle_id):
        """添加/更新小车位置"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # 记录小车位置
            self.shuttles[shuttle_id] = (row, col)
            # 更新地图显示 (3-6 对应小车0-3)
            if shuttle_id < 4:  # 最多支持4个小车
                self.map[row][col] = 3 + shuttle_id
            else:
                self.map[row][col] = 3  # 默认红色
            return True
        return False

    def add_elevator(self, row, col):
        """添加电梯"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.map[row][col] = 7
            self.elevators.append((row, col))
            # 电梯位置不可用作货位
            self.slots[row][col] = 0
            print(f"添加提升机 at ({row}, {col})")
            return True
        return False

    def clear_position(self, row, col):
        """清除位置上的小车"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # 检查这个位置是否有小车
            for shuttle_id, pos in list(self.shuttles.items()):
                if pos == (row, col):
                    # 移除小车
                    del self.shuttles[shuttle_id]
                    # 恢复为原来的货位状态
                    if self.slots[row][col] == 1:
                        self.map[row][col] = 1
                    elif self.slots[row][col] == 2:
                        self.map[row][col] = 2
                    else:
                        self.map[row][col] = 0  # 巷道
                    return True
        return False

    def draw(self, surface):
        """
        使用 pygame 绘制地图到指定的 surface 上。
        """
        try:
            for row in range(self.rows):
                for col in range(self.cols):
                    cell_value = self.map[row][col]

                    # 获取颜色
                    color = COLORS.get(cell_value, COLORS[0])  # 默认为巷道颜色

                    # 绘制单元格
                    rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(surface, color, rect)

                    # 绘制网格线
                    pygame.draw.rect(surface, (100, 100, 100), rect, 1)

                    # 在小车上显示ID
                    if 3 <= cell_value <= 6:
                        shuttle_id = cell_value - 3
                        try:
                            font = pygame.font.Font(None, 16)
                            text = font.render(str(shuttle_id), True, (255, 255, 255))
                            text_rect = text.get_rect(center=rect.center)
                            surface.blit(text, text_rect)
                        except:
                            pass  # 字体渲染失败时忽略

            # 绘制图例
            self.draw_legend(surface)

        except Exception as e:
            print(f"地图绘制错误: {e}")
            import traceback
            traceback.print_exc()

    def draw_legend(self, surface):
        """绘制图例"""
        try:
            legend_x = 10
            legend_y = 10
            legend_width = 150
            cell_size = 15

            # 绘制图例背景
            legend_bg = pygame.Rect(legend_x, legend_y, legend_width, 120)
            pygame.draw.rect(surface, (0, 0, 0, 128), legend_bg)
            pygame.draw.rect(surface, (255, 255, 255), legend_bg, 1)

            # 图例标题
            font = pygame.font.Font(None, 16)
            title = font.render("图例", True, (255, 255, 255))
            surface.blit(title, (legend_x + 5, legend_y + 5))

            # 图例项
            legend_items = [
                (COLORS[0], "巷道"),
                (COLORS[1], "空闲货位"),
                (COLORS[2], "占用货位"),
                (COLORS[3], "穿梭车"),
                (COLORS[7], "提升机")
            ]

            for i, (color, text) in enumerate(legend_items):
                y_pos = legend_y + 30 + i * 20
                # 颜色方块
                color_rect = pygame.Rect(legend_x + 5, y_pos, cell_size, cell_size)
                pygame.draw.rect(surface, color, color_rect)
                pygame.draw.rect(surface, (255, 255, 255), color_rect, 1)
                # 文字
                text_surface = font.render(text, True, (255, 255, 255))
                surface.blit(text_surface, (legend_x + 25, y_pos))

        except Exception as e:
            print(f"图例绘制错误: {e}")

    def get_available_positions(self):
        """获取所有可用位置（空闲货位）"""
        available = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.slots[row][col] == 1:  # 空闲货位
                    available.append((row, col))
        return available

    def get_occupied_positions(self):
        """获取所有占用位置"""
        occupied = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.slots[row][col] == 2:  # 占用货位
                    occupied.append((row, col))
        return occupied

    def print_map_info(self):
        """打印地图信息（调试用）"""
        available_count = len(self.get_available_positions())
        occupied_count = len(self.get_occupied_positions())
        shuttle_count = len(self.shuttles)
        elevator_count = len(self.elevators)

        print(f"地图信息: 可用位置={available_count}, 占用位置={occupied_count}, "
              f"小车数={shuttle_count}, 电梯数={elevator_count}")


# 创建全局仓库实例
warehouse = WarehouseMap()