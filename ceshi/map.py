# map.py
import numpy as np
import pygame
from config import *


class WarehouseMap:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.grid_type = np.zeros((self.rows, self.cols), dtype=int)
        self.slots = np.zeros((self.rows, self.cols), dtype=int)

        self.init_layout()

    def init_layout(self):
        """
        生成闭环交通网：
        - 水平轨道：Row 2(东), Row 9(西), Row 16(东)
        - 垂直轨道：Col 1(北), Col 15(南), Col 30(北), Col 44(南) -> 形成完美回路
        """
        # 1. 默认全为存储巷道
        self.grid_type.fill(TYPE_STORAGE)
        self.slots.fill(1)

        # 2. 定义主轨道
        main_rows = [
            (2, TYPE_MAIN_H_EAST),
            (9, TYPE_MAIN_H_WEST),
            (16, TYPE_MAIN_H_EAST)
        ]

        # 垂直轨道配置：(列号, 方向)
        # 形成循环：Row9(西) -> Col1(北) -> Row2(东) -> Col15(南) -> Row9(西)...
        main_cols = [
            (1, TYPE_MAIN_V_NORTH),
            (15, TYPE_MAIN_V_SOUTH),
            (30, TYPE_MAIN_V_NORTH),
            (44, TYPE_MAIN_V_SOUTH)
        ]

        # 绘制水平轨道
        for r, direction in main_rows:
            for c in range(self.cols):
                self.grid_type[r][c] = direction
                self.slots[r][c] = 0

        # 绘制垂直轨道
        for c, direction in main_cols:
            for r in range(self.rows):
                # 交叉口处理
                if any(r == mr[0] for mr in main_rows):
                    self.grid_type[r][c] = TYPE_INTERSECTION
                else:
                    self.grid_type[r][c] = direction

                self.slots[r][c] = 0

        # 3. 设置提升机 (左侧)
        elevator_pos = [(2, 0), (9, 0), (16, 0)]
        for r, c in elevator_pos:
            self.grid_type[r][c] = TYPE_ELEVATOR
            self.slots[r][c] = 0

    def update_slot(self, r, c, status):
        if self.grid_type[r][c] == TYPE_STORAGE:
            self.slots[r][c] = status

    def clear_position(self, r, c):
        pass

    def add_shuttle(self, r, c, shuttle_id):
        pass

    def draw(self, screen):
        """绘制地图"""
        for r in range(self.rows):
            for c in range(self.cols):
                g_type = self.grid_type[r][c]
                slot_val = self.slots[r][c]

                color = COLORS.get(g_type, COLORS[TYPE_NONE])
                if g_type == TYPE_STORAGE and slot_val == 2:
                    color = COLORS["occupied"]

                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

                # 绘制方向箭头
                center = (rect[0] + CELL_SIZE // 2, rect[1] + CELL_SIZE // 2)
                arrow_color = (255, 255, 255)

                if g_type == TYPE_MAIN_H_EAST:  # >
                    pygame.draw.polygon(screen, arrow_color,
                                        [(center[0] - 3, center[1] - 3), (center[0] - 3, center[1] + 3),
                                         (center[0] + 3, center[1])])
                elif g_type == TYPE_MAIN_H_WEST:  # <
                    pygame.draw.polygon(screen, arrow_color,
                                        [(center[0] + 3, center[1] - 3), (center[0] + 3, center[1] + 3),
                                         (center[0] - 3, center[1])])
                elif g_type == TYPE_MAIN_V_SOUTH:  # v
                    pygame.draw.polygon(screen, arrow_color,
                                        [(center[0] - 3, center[1] - 3), (center[0] + 3, center[1] - 3),
                                         (center[0], center[1] + 3)])
                elif g_type == TYPE_MAIN_V_NORTH:  # ^
                    pygame.draw.polygon(screen, arrow_color,
                                        [(center[0] - 3, center[1] + 3), (center[0] + 3, center[1] + 3),
                                         (center[0], center[1] - 3)])

                pygame.draw.rect(screen, (40, 44, 52), rect, 1)


warehouse = WarehouseMap()