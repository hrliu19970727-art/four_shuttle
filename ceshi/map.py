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
        """坐标转换: 用户(1-based, Bottom-Left) -> 代码(0-based, Top-Left)"""
        # User Row 1 -> Code Row 28 (ROWS-1)
        # User Row 29 -> Code Row 0
        c_row = self.rows - u_row
        c_col = u_col - 1
        return c_row, c_col

    def set_area(self, u_rows, u_cols, type_val):
        """批量设置区域类型"""
        # 统一转为列表处理
        if isinstance(u_rows, int): u_rows = [u_rows]
        if isinstance(u_cols, int): u_cols = [u_cols]

        # 解析 user rows (可能包含范围)
        target_rows = []
        for r in u_rows: target_rows.append(r)

        target_cols = []
        for c in u_cols: target_cols.append(c)

        for ur in target_rows:
            for uc in target_cols:
                if 1 <= ur <= self.rows and 1 <= uc <= self.cols:
                    cr, cc = self.u2c(ur, uc)
                    self.grid_type[cr][cc] = type_val
                    # 如果是存储巷道，默认设为空闲(1)，否则为轨道(0)
                    self.slots[cr][cc] = 1 if type_val == TYPE_STORAGE else 0

    def init_layout(self):
        # 1. 初始化全为无效区域 (TYPE_NONE)
        self.grid_type.fill(TYPE_NONE)
        self.slots.fill(0)

        # --- 辅助函数：生成范围列表 ---
        def R(start, end):
            return list(range(start, end + 1))

        # === 1. 存储巷道配置 (根据您的描述) ===
        # R1,2: 6-45
        self.set_area(R(1, 2), R(6, 45), TYPE_STORAGE)
        # R3: 6-45 Storage, 1-4 Charging (treat as Storage for now)
        self.set_area(3, R(6, 45), TYPE_STORAGE)
        self.set_area(3, R(1, 4), TYPE_STORAGE)  # 充电位

        # R5-13
        storage_cols_5_13 = [12, 13] + R(15, 18) + R(20, 23) + R(25, 28) + R(30, 37) + R(39, 42) + [44]
        self.set_area(R(5, 13), storage_cols_5_13, TYPE_STORAGE)

        # R15-20
        storage_cols_15_20 = R(8, 10) + R(12, 44) + [46, 47]
        self.set_area(R(15, 20), storage_cols_15_20, TYPE_STORAGE)

        # R22-24
        storage_cols_22_24 = R(6, 10) + R(12, 44) + R(46, 50)
        self.set_area(R(22, 24), storage_cols_22_24, TYPE_STORAGE)

        # R25-26
        storage_cols_25_26 = R(6, 8) + [12, 13] + R(15, 18) + R(20, 23) + R(25, 28) + R(30, 37) + R(39, 42) + [44, 46,
                                                                                                               47, 49,
                                                                                                               50]
        self.set_area(R(25, 26), storage_cols_25_26, TYPE_STORAGE)

        # R28: 7-50
        self.set_area(28, R(7, 50), TYPE_STORAGE)
        # R29: 8-50
        self.set_area(29, R(8, 50), TYPE_STORAGE)

        # === 2. 主巷道配置 (Main Aisles) ===

        # 水平主巷道 (设定流向以形成环路)
        # R4: 1-45 (-> East)
        self.set_area(4, R(1, 45), TYPE_MAIN_H_EAST)

        # R14: 8-47 (<- West)
        self.set_area(14, R(8, 47), TYPE_MAIN_H_WEST)

        # R21: 6-50 (-> East)
        self.set_area(21, R(6, 50), TYPE_MAIN_H_EAST)

        # R27: 6-50 (<- West)
        self.set_area(27, R(6, 50), TYPE_MAIN_H_WEST)

        # 垂直主巷道
        # Cols 11, 45 是贯穿多层的骨干，设定单向以形成高速环路
        # C11: South (Down)
        # C45: North (Up)
        vertical_rows = R(5, 13) + R(15, 20) + R(22, 24) + R(25, 26)

        self.set_area(vertical_rows, 11, TYPE_MAIN_V_SOUTH)  # C11 下行
        self.set_area(vertical_rows, 45, TYPE_MAIN_V_NORTH)  # C45 上行


        # === 3. 交叉口处理 (Intersections) ===
        # 在主轨道交汇处，覆盖为交叉口类型，允许全向行驶
        # C11 与 R4, R14, R21, R27 的交点
        # C45 与 R4, R14, R21, R27 的交点
        # C6 与 R4, R21, R27 的交点

        cross_rows = [4, 14, 21, 27]
        cross_cols = [6, 11, 45]

        for r in cross_rows:
            for c in cross_cols:
                # 只有当该位置之前已经被定义为某种轨道时才覆盖为交叉口
                # (避免在 R14/C6 这种不相交的地方画交叉口)
                cr, cc = self.u2c(r, c)
                if self.grid_type[cr][cc] != TYPE_NONE and self.grid_type[cr][cc] != TYPE_STORAGE:
                    self.grid_type[cr][cc] = TYPE_INTERSECTION

        # === 4. 提升机/端口 (假设在左侧) ===
        # 假设在 R4, R14, R21, R27 的左端入口处
        self.set_area(13, 8, TYPE_ELEVATOR)
        self.set_area(20, 6, TYPE_ELEVATOR)
        self.set_area(14, 47, TYPE_ELEVATOR)
        self.set_area(20, 49, TYPE_ELEVATOR)

    def update_slot(self, r, c, status):
        if self.grid_type[r][c] == TYPE_STORAGE:
            self.slots[r][c] = status

    def draw(self, screen):
        for r in range(self.rows):
            for c in range(self.cols):
                g_type = self.grid_type[r][c]
                if g_type == TYPE_NONE: continue  # 不绘制无效区域

                slot_val = self.slots[r][c]
                color = COLORS.get(g_type, COLORS[TYPE_NONE])

                if g_type == TYPE_STORAGE and slot_val == 2:
                    color = COLORS["occupied"]

                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

                # 简单方向指示
                center = (rect[0] + CELL_SIZE // 2, rect[1] + CELL_SIZE // 2)
                if g_type == TYPE_MAIN_H_EAST:
                    pygame.draw.circle(screen, (255, 255, 255), center, 2)
                    pygame.draw.line(screen, (255, 255, 255), rect[:2], (rect[0] + CELL_SIZE, rect[1] + CELL_SIZE // 2),
                                     1)

                pygame.draw.rect(screen, (40, 44, 52), rect, 1)


warehouse = WarehouseMap()