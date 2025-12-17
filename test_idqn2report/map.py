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

    def u2c(self, u_row, u_col):
        """坐标转换"""
        c_row = self.rows - u_row
        c_col = u_col - 1
        return c_row, c_col

    def set_area(self, u_rows, u_cols, type_val):
        """批量设置区域"""
        if isinstance(u_rows, int): u_rows = [u_rows]
        if isinstance(u_cols, int): u_cols = [u_cols]

        target_rows = [r for r in u_rows]
        target_cols = [c for c in u_cols]

        for ur in target_rows:
            for uc in target_cols:
                if 1 <= ur <= self.rows and 1 <= uc <= self.cols:
                    cr, cc = self.u2c(ur, uc)
                    self.grid_type[cr][cc] = type_val
                    self.slots[cr][cc] = 1 if type_val == TYPE_STORAGE else 0

    def init_layout(self):
        self.grid_type.fill(TYPE_NONE)
        self.slots.fill(0)

        def R(start, end):
            return list(range(start, end + 1))

        # === 1. 存储巷道 ===
        self.set_area(R(1, 2), R(6, 45), TYPE_STORAGE)
        self.set_area(3, R(6, 45), TYPE_STORAGE)
        self.set_area(3, R(1, 4), TYPE_STORAGE)

        cols_5_13 = [12, 13] + R(15, 18) + R(20, 23) + R(25, 28) + R(30, 37) + R(39, 42) + [44]
        self.set_area(R(5, 13), cols_5_13, TYPE_STORAGE)

        cols_15_20 = R(8, 10) + R(12, 44) + [46, 47]
        self.set_area(R(15, 20), cols_15_20, TYPE_STORAGE)

        cols_22_24 = R(6, 10) + R(12, 44) + R(46, 50)
        self.set_area(R(22, 24), cols_22_24, TYPE_STORAGE)

        cols_25_26 = R(6, 8) + [12, 13] + R(15, 18) + R(20, 23) + R(25, 28) + R(30, 37) + R(39, 42) + [44, 46, 47, 49,
                                                                                                       50]
        self.set_area(R(25, 26), cols_25_26, TYPE_STORAGE)

        self.set_area(28, R(7, 50), TYPE_STORAGE)
        self.set_area(29, R(8, 50), TYPE_STORAGE)

        # === 2. 主巷道 (基础定向) ===
        self.set_area(4, R(1, 45), TYPE_MAIN_H_EAST)
        self.set_area(14, R(8, 47), TYPE_MAIN_H_WEST)
        self.set_area(21, R(6, 50), TYPE_MAIN_H_EAST)
        self.set_area(27, R(6, 50), TYPE_MAIN_H_WEST)

        # 垂直主巷道 (形成环路: C11下, C45上)
        v_rows = R(5, 13) + R(15, 20) + R(22, 24) + R(25, 26)
        self.set_area(v_rows, 11, TYPE_MAIN_V_SOUTH)
        self.set_area(v_rows, 45, TYPE_MAIN_V_NORTH)
        # self.set_area(R(5, 13), 6, TYPE_MAIN_BIDIRECTIONAL)

        # === 3. 【关键修复】末梢路段双向化 (解决死胡同) ===
        # 将连接核心环路(11, 45)与边缘端口的路段改为双向，允许掉头

        # Row 4: 左侧末端 1-11
        self.set_area(4, R(1, 11), TYPE_MAIN_BIDIRECTIONAL)

        # Row 14: 左侧末端 8-11 (连接入库1), 右侧末端 45-47 (连接出库1)
        self.set_area(14, R(8, 11), TYPE_MAIN_BIDIRECTIONAL)
        self.set_area(14, R(45, 47), TYPE_MAIN_BIDIRECTIONAL)

        # Row 21: 左侧末端 6-11 (连接入库2), 右侧末端 45-50 (连接出库2)
        self.set_area(21, R(6, 11), TYPE_MAIN_BIDIRECTIONAL)
        self.set_area(21, R(45, 50), TYPE_MAIN_BIDIRECTIONAL)

        # Row 27: 两端
        self.set_area(27, R(6, 11), TYPE_MAIN_BIDIRECTIONAL)
        self.set_area(27, R(45, 50), TYPE_MAIN_BIDIRECTIONAL)

        # === 4. 交叉口覆盖 ===
        # 在垂直和水平主路交汇处设为交叉口 (允许全向)
        cross_rows = [4, 14, 21, 27]
        cross_cols = [6, 11, 45]
        for r in cross_rows:
            for c in cross_cols:
                cr, cc = self.u2c(r, c)
                # 只有当该位置有轨道时才覆盖
                if self.grid_type[cr][cc] != TYPE_NONE and self.grid_type[cr][cc] != TYPE_STORAGE:
                    self.grid_type[cr][cc] = TYPE_INTERSECTION

        # === 5. 入出库点 ===
        # 入库
        self.set_area(13, 8, TYPE_ELEVATOR)
        self.set_area(20, 6, TYPE_ELEVATOR)
        # 出库
        self.set_area(14, 47, TYPE_ELEVATOR)
        self.set_area(20, 49, TYPE_ELEVATOR)

    def update_slot(self, r, c, status):
        if self.grid_type[r][c] == TYPE_STORAGE:
            self.slots[r][c] = status

    def draw(self, screen):
        for r in range(self.rows):
            for c in range(self.cols):
                g_type = self.grid_type[r][c]
                if g_type == TYPE_NONE: continue

                slot_val = self.slots[r][c]
                color = COLORS.get(g_type, COLORS[TYPE_NONE])

                if g_type == TYPE_STORAGE and slot_val == 2:
                    color = COLORS["occupied"]

                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

                # 绘制双向轨道标识 (双箭头)
                if g_type == TYPE_MAIN_BIDIRECTIONAL:
                    pygame.draw.circle(screen, (150, 200, 255), (rect[0] + 10, rect[1] + 10), 3)

                # 标记入出库点
                if g_type == TYPE_ELEVATOR:
                    pygame.draw.rect(screen, (255, 255, 255), rect, 2)

                pygame.draw.rect(screen, (40, 44, 52), rect, 1)


warehouse = WarehouseMap()