# visualization.py
import pygame
import sys
import math
from config import ROWS, COLS, CELL_SIZE, COLORS, PLANNING_ALGORITHM
from map import warehouse
from shared_state import shared_state, state_lock

# --- 界面布局常量 ---
MAP_WIDTH = COLS * CELL_SIZE
MAP_HEIGHT = ROWS * CELL_SIZE
INFO_PANEL_HEIGHT = 260
TOTAL_WIDTH = MAP_WIDTH
TOTAL_HEIGHT = MAP_HEIGHT + INFO_PANEL_HEIGHT

# --- 视觉参数 ---
SHUTTLE_SIZE = int(CELL_SIZE * 0.8)
ANIMATION_SPEED = 0.2

# --- 字体配置 ---
FONT_SIZE_TITLE = 28
FONT_SIZE_LARGE = 24
FONT_SIZE_MEDIUM = 18
FONT_SIZE_SMALL = 14
FONT_SIZE_TINY = 12

# --- 配色方案 ---
COLOR_BG = (30, 32, 36)
COLOR_PANEL_BG = (40, 42, 46)
COLOR_CARD_BG = (50, 52, 56)
COLOR_TEXT_MAIN = (240, 240, 240)
COLOR_TEXT_DIM = (160, 160, 160)
COLOR_ACCENT = (70, 160, 240)
COLOR_HIGHLIGHT = (255, 200, 50)
COLOR_GOOD = (80, 200, 120)
COLOR_WARN = (230, 230, 60)
COLOR_BAD = (240, 80, 80)
COLOR_TASK_REL = (100, 180, 255)
COLOR_TASK_PICK = (255, 140, 80)

_cached_fonts = {}
_shuttle_visual_pos = {}


def get_font(size, bold=False):
    key = (size, bold)
    if key in _cached_fonts: return _cached_fonts[key]
    font_names = ["simhei", "microsoftyahei", "pingfangsc", "notosanscjk", "arial"]
    font = None
    for name in font_names:
        try:
            p = pygame.font.match_font(name)
            if p:
                font = pygame.font.Font(p, size)
                break
        except:
            continue
    if not font: font = pygame.font.Font(None, size)
    if bold: font.set_bold(True)
    _cached_fonts[key] = font
    return font


def draw_battery_bar(surface, x, y, w, h, level):
    pygame.draw.rect(surface, (30, 30, 30), (x, y, w, h), border_radius=3)
    pygame.draw.rect(surface, (80, 80, 80), (x, y, w, h), 1, border_radius=3)
    col = COLOR_GOOD
    if level < 40: col = COLOR_WARN
    if level < 20: col = COLOR_BAD
    fill_w = int((w - 2) * (level / 100))
    if fill_w > 0:
        pygame.draw.rect(surface, col, (x + 1, y + 1, fill_w, h - 2), border_radius=2)


def draw_visual_aids(screen):
    for task in list(shared_state.get("release_tasks", [])):
        r, c = task.position
        cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, COLOR_TASK_REL, (cx, cy), 5)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6, 1)
    for task in list(shared_state.get("pick_tasks", [])):
        r, c = task.position
        cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, COLOR_TASK_PICK, (cx, cy), 5)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6, 1)

    for s in shared_state.get("shuttles", []):
        if s.get("busy") and s.get("current_task"):
            sid = s["id"]
            if sid in _shuttle_visual_pos:
                curr_r, curr_c = _shuttle_visual_pos[sid]
            else:
                curr_r, curr_c = s["pos"]
            tr, tc = s["current_task"].position
            sx, sy = curr_c * CELL_SIZE + CELL_SIZE // 2, curr_r * CELL_SIZE + CELL_SIZE // 2
            ex, ey = tc * CELL_SIZE + CELL_SIZE // 2, tr * CELL_SIZE + CELL_SIZE // 2
            col = COLOR_HIGHLIGHT if sid % 2 == 0 else COLOR_ACCENT
            pygame.draw.line(screen, col, (sx, sy), (ex, ey), 1)
            pygame.draw.circle(screen, col, (ex, ey), 3)


def draw_shuttles(screen):
    shuttles = shared_state.get("shuttles", [])
    font_id = get_font(10, True)
    for s in shuttles:
        sid = s["id"]
        target_r, target_c = s["pos"]
        if sid not in _shuttle_visual_pos:
            _shuttle_visual_pos[sid] = [target_r, target_c]
        vis_r, vis_c = _shuttle_visual_pos[sid]
        vis_r += (target_r - vis_r) * ANIMATION_SPEED
        vis_c += (target_c - vis_c) * ANIMATION_SPEED
        _shuttle_visual_pos[sid] = [vis_r, vis_c]
        x = vis_c * CELL_SIZE + (CELL_SIZE - SHUTTLE_SIZE) / 2
        y = vis_r * CELL_SIZE + (CELL_SIZE - SHUTTLE_SIZE) / 2
        base_color = COLOR_HIGHLIGHT if sid % 2 == 0 else COLOR_ACCENT
        if not s.get("busy"): base_color = (100, 100, 100)
        rect = (x, y, SHUTTLE_SIZE, SHUTTLE_SIZE)
        pygame.draw.rect(screen, base_color, rect, border_radius=3)
        if s.get("Load"):
            pygame.draw.rect(screen, (255, 255, 255), (x + 4, y + 4, SHUTTLE_SIZE - 8, SHUTTLE_SIZE - 8),
                             border_radius=1)
        txt = font_id.render(str(sid), True, (20, 20, 20))
        screen.blit(txt, (x + SHUTTLE_SIZE // 2 - txt.get_width() // 2, y + SHUTTLE_SIZE // 2 - txt.get_height() // 2))


def draw_info_panel(screen):
    panel_y = MAP_HEIGHT
    pygame.draw.rect(screen, COLOR_PANEL_BG, (0, panel_y, TOTAL_WIDTH, INFO_PANEL_HEIGHT))
    pygame.draw.line(screen, (60, 60, 60), (0, panel_y), (TOTAL_WIDTH, panel_y), 2)

    ft = get_font(FONT_SIZE_TITLE, True)
    fl = get_font(FONT_SIZE_LARGE, True)
    fm = get_font(FONT_SIZE_MEDIUM)
    fs = get_font(FONT_SIZE_SMALL)

    current_time = shared_state.get("time", 0)

    # === 1. 系统概览 ===
    x_sys = 20
    y = panel_y + 20
    screen.blit(ft.render("WCS 调度监控", True, COLOR_ACCENT), (x_sys, y))

    pygame.draw.rect(screen, COLOR_CARD_BG, (x_sys, y + 45, 160, 70), border_radius=8)
    screen.blit(fs.render("SIMULATION TIME", True, COLOR_TEXT_DIM), (x_sys + 10, y + 55))
    screen.blit(fl.render(f"{current_time:.1f} s", True, COLOR_TEXT_MAIN), (x_sys + 10, y + 80))

    # --- 算法性能卡片 ---
    # 计算平均值
    total_time = shared_state.get("total_planning_time", 0)
    count = shared_state.get("planning_count", 0)
    total_len = shared_state.get("total_path_length", 0)
    path_cnt = shared_state.get("path_count", 0)

    avg_time = (total_time / count) if count > 0 else 0
    avg_len = (total_len / path_cnt) if path_cnt > 0 else 0

    y_perf = y + 125
    screen.blit(fs.render(f"Algorithm: {PLANNING_ALGORITHM}", True, COLOR_HIGHLIGHT), (x_sys, y_perf))
    screen.blit(fs.render(f"Avg Time: {avg_time:.2f} ms", True, COLOR_TEXT_MAIN), (x_sys, y_perf + 20))
    screen.blit(fs.render(f"Avg Steps: {avg_len:.1f}", True, COLOR_TEXT_MAIN), (x_sys, y_perf + 40))

    # === 2. 车辆状态 ===
    x_shuttle = 220
    screen.blit(fm.render("Fleet Status", True, COLOR_TEXT_DIM), (x_shuttle, y))

    shuttles = shared_state.get("shuttles", [])
    for i, s in enumerate(shuttles[:6]):
        col_idx = i % 3
        row_idx = i // 3
        cx = x_shuttle + col_idx * 150
        cy = y + 30 + row_idx * 90

        pygame.draw.rect(screen, COLOR_CARD_BG, (cx, cy, 140, 80), border_radius=6)
        id_col = COLOR_HIGHLIGHT if s.get("busy") else COLOR_TEXT_DIM
        screen.blit(get_font(20, True).render(f"#{s['id']}", True, id_col), (cx + 10, cy + 8))

        status = s.get("status", "Idle")
        if len(status) > 8: status = status[:7] + "."
        screen.blit(fs.render(status, True, COLOR_TEXT_MAIN), (cx + 50, cy + 12))
        screen.blit(get_font(12).render(f"Pos: {s['pos']}", True, COLOR_TEXT_DIM), (cx + 10, cy + 35))
        draw_battery_bar(screen, cx + 10, cy + 55, 120, 6, s.get("battery", 0))

        if s.get("current_task"):
            is_load = s.get("Load")
            task_col = COLOR_TASK_REL if is_load else COLOR_TASK_PICK
            pygame.draw.circle(screen, task_col, (cx + 130, cy + 15), 4)

    # === 3. 统计仪表盘 ===
    x_stats = 700
    w_stats = TOTAL_WIDTH - x_stats - 20
    pygame.draw.rect(screen, (35, 37, 40), (x_stats, y, w_stats, 200), border_radius=10)
    pygame.draw.rect(screen, (60, 60, 60), (x_stats, y, w_stats, 200), 1, border_radius=10)
    screen.blit(fm.render("Mission Statistics", True, COLOR_HIGHLIGHT), (x_stats + 15, y + 15))

    comp_rel = shared_state.get("completed_release_tasks", 0)
    comp_pick = shared_state.get("completed_pick_tasks", 0)
    fail = shared_state.get("failed_tasks", 0)

    # 入库完成
    cx_rel = x_stats + 20
    cy_rel = y + 50
    pygame.draw.rect(screen, (45, 50, 60), (cx_rel, cy_rel, 120, 70), border_radius=8)
    pygame.draw.line(screen, COLOR_TASK_REL, (cx_rel, cy_rel + 10), (cx_rel, cy_rel + 60), 3)
    screen.blit(get_font(36, True).render(str(comp_rel), True, COLOR_TEXT_MAIN), (cx_rel + 15, cy_rel + 5))
    screen.blit(fs.render("入库完成", True, COLOR_TEXT_DIM), (cx_rel + 15, cy_rel + 45))

    # 出库完成
    cx_pick = x_stats + 20
    cy_pick = y + 130
    pygame.draw.rect(screen, (45, 50, 60), (cx_pick, cy_pick, 120, 70), border_radius=8)
    pygame.draw.line(screen, COLOR_TASK_PICK, (cx_pick, cy_pick + 10), (cx_pick, cy_pick + 60), 3)
    screen.blit(get_font(36, True).render(str(comp_pick), True, COLOR_TEXT_MAIN), (cx_pick + 15, cy_pick + 5))
    screen.blit(fs.render("出库完成", True, COLOR_TEXT_DIM), (cx_pick + 15, cy_pick + 45))

    # 指标列表
    lx = x_stats + 160
    ly = y + 50
    gap = 35

    screen.blit(fs.render("Total Failures", True, COLOR_TEXT_DIM), (lx, ly))
    screen.blit(fl.render(str(fail), True, COLOR_BAD if fail > 0 else COLOR_TEXT_DIM), (lx + 120, ly - 5))

    throughput = 0.0
    if current_time > 0: throughput = ((comp_rel + comp_pick) / current_time) * 60
    screen.blit(fs.render("Throughput", True, COLOR_TEXT_DIM), (lx, ly + gap))
    t_surf = fl.render(f"{throughput:.1f}", True, COLOR_ACCENT)
    screen.blit(t_surf, (lx + 120, ly + gap - 5))
    screen.blit(get_font(12).render("tasks/min", True, COLOR_TEXT_DIM),
                (lx + 120 + t_surf.get_width() + 5, ly + gap + 5))

    # 队列微图
    q_rel = len(shared_state.get("release_tasks", []))
    q_pick = len(shared_state.get("pick_tasks", []))
    max_q = 10

    bx = x_stats + 320
    by = y + 50
    screen.blit(fs.render("In-Queue", True, COLOR_TEXT_DIM), (bx, by))
    bar_w = 80
    fill_rel = min(1.0, q_rel / max_q) * bar_w
    pygame.draw.rect(screen, (50, 50, 50), (bx, by + 20, bar_w, 8), border_radius=2)
    if fill_rel > 0: pygame.draw.rect(screen, COLOR_TASK_REL, (bx, by + 20, fill_rel, 8), border_radius=2)
    screen.blit(fs.render(str(q_rel), True, COLOR_TEXT_MAIN), (bx + bar_w + 10, by + 15))

    screen.blit(fs.render("Out-Queue", True, COLOR_TEXT_DIM), (bx, by + 60))
    fill_pick = min(1.0, q_pick / max_q) * bar_w
    pygame.draw.rect(screen, (50, 50, 50), (bx, by + 80, bar_w, 8), border_radius=2)
    if fill_pick > 0: pygame.draw.rect(screen, COLOR_TASK_PICK, (bx, by + 80, fill_pick, 8), border_radius=2)
    screen.blit(fs.render(str(q_pick), True, COLOR_TEXT_MAIN), (bx + bar_w + 10, by + 75))


def run_visualization():
    try:
        pygame.init()
        screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("智能仓库调度系统 V8.0 (算法对比版)")
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    with state_lock: shared_state["done"] = True
                    pygame.quit()
                    return

            with state_lock:
                screen.fill(COLOR_BG)
                if hasattr(warehouse, 'draw'): warehouse.draw(screen)
                draw_visual_aids(screen)
                draw_shuttles(screen)
                draw_info_panel(screen)

            pygame.display.flip()
            clock.tick(30)

    except Exception as e:
        print(f"Visualization Error: {e}")
        pygame.quit()


if __name__ == "__main__":
    run_visualization()