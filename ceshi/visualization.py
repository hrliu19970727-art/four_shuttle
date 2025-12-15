# visualization.py
import pygame
import sys
from config import ROWS, COLS, CELL_SIZE, COLORS
from map import warehouse
from shared_state import shared_state, state_lock

# --- 布局常量 ---
MAP_WIDTH = COLS * CELL_SIZE
MAP_HEIGHT = ROWS * CELL_SIZE
INFO_PANEL_HEIGHT = 240
TOTAL_WIDTH = MAP_WIDTH
TOTAL_HEIGHT = MAP_HEIGHT + INFO_PANEL_HEIGHT

# --- 视觉参数 ---
SHUTTLE_SIZE = int(CELL_SIZE * 0.8)  # 小车比格子稍小
ANIMATION_SPEED = 0.2  # 动画平滑系数 (0.1-0.3之间，越大越快)

# --- 字体设置 ---
FONT_SIZE_TITLE = 32
FONT_SIZE_LARGE = 24
FONT_SIZE_MEDIUM = 20
FONT_SIZE_SMALL = 16

COLOR_BG = (30, 32, 36)
COLOR_PANEL_BG = (50, 52, 57)
COLOR_TEXT_MAIN = (250, 250, 250)
COLOR_TEXT_DIM = (180, 180, 180)
COLOR_ACCENT = (70, 180, 255)
COLOR_HIGHLIGHT = (255, 200, 50)
COLOR_GOOD = (100, 220, 100)
COLOR_WARN = (230, 230, 60)
COLOR_BAD = (230, 80, 80)
COLOR_TASK_REL = (100, 180, 255)
COLOR_TASK_PICK = (255, 140, 80)

_cached_fonts = {}
# 用于存储显示坐标的缓存 {id: [float_row, float_col]}
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
    pygame.draw.rect(surface, (120, 120, 120), (x, y, w, h), 1, border_radius=3)
    col = COLOR_GOOD if level > 60 else (COLOR_WARN if level > 25 else COLOR_BAD)
    fw = int(w * (level / 100))
    if fw > 0: pygame.draw.rect(surface, col, (x + 1, y + 1, min(fw, w - 2), h - 2), border_radius=2)


def draw_visual_aids(screen):
    # 1. 任务点
    for task in list(shared_state.get("release_tasks", [])):
        r, c = task.position
        cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, COLOR_TASK_REL, (cx, cy), 6)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 7, 1)
    for task in list(shared_state.get("pick_tasks", [])):
        r, c = task.position
        cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, COLOR_TASK_PICK, (cx, cy), 6)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 7, 1)

    # 2. 意图连线
    for s in shared_state.get("shuttles", []):
        if s.get("busy") and s.get("current_task"):
            # 使用平滑后的显示坐标作为起点
            sid = s["id"]
            if sid in _shuttle_visual_pos:
                curr_r, curr_c = _shuttle_visual_pos[sid]
            else:
                curr_r, curr_c = s["pos"]

            tr, tc = s["current_task"].position
            sx, sy = curr_c * CELL_SIZE + CELL_SIZE // 2, curr_r * CELL_SIZE + CELL_SIZE // 2
            ex, ey = tc * CELL_SIZE + CELL_SIZE // 2, tr * CELL_SIZE + CELL_SIZE // 2

            col = COLOR_HIGHLIGHT if sid % 2 == 0 else COLOR_ACCENT
            # 画半透明虚线效果(这里用细实线代替)
            pygame.draw.line(screen, col, (sx, sy), (ex, ey), 1)
            pygame.draw.circle(screen, col, (ex, ey), 4, 1)


def draw_shuttles(screen):
    """
    【核心功能】绘制带有平滑动画的小车
    """
    shuttles = shared_state.get("shuttles", [])
    font_id = get_font(12, True)

    for s in shuttles:
        sid = s["id"]
        target_r, target_c = s["pos"]

        # --- 插值动画逻辑 ---
        if sid not in _shuttle_visual_pos:
            _shuttle_visual_pos[sid] = [target_r, target_c]

        # 获取当前显示位置
        vis_r, vis_c = _shuttle_visual_pos[sid]

        # 计算差值并移动 (Lerp)
        diff_r = target_r - vis_r
        diff_c = target_c - vis_c

        # 如果距离很小直接吸附，否则按比例移动
        if abs(diff_r) < 0.01:
            vis_r = target_r
        else:
            vis_r += diff_r * ANIMATION_SPEED

        if abs(diff_c) < 0.01:
            vis_c = target_c
        else:
            vis_c += diff_c * ANIMATION_SPEED

        # 更新缓存
        _shuttle_visual_pos[sid] = [vis_r, vis_c]

        # --- 绘制小车实体 ---
        # 转换坐标 (注意 xy 与 row/col 的关系: x=col, y=row)
        x = vis_c * CELL_SIZE + (CELL_SIZE - SHUTTLE_SIZE) / 2
        y = vis_r * CELL_SIZE + (CELL_SIZE - SHUTTLE_SIZE) / 2

        # 颜色: 忙碌/空闲区分
        base_color = COLOR_HIGHLIGHT if sid % 2 == 0 else COLOR_ACCENT
        if not s.get("busy"):
            # 空闲时稍微暗一点
            base_color = (max(0, base_color[0] - 50), max(0, base_color[1] - 50), max(0, base_color[2] - 50))

        # 车身
        rect = (x, y, SHUTTLE_SIZE, SHUTTLE_SIZE)
        pygame.draw.rect(screen, base_color, rect, border_radius=4)
        pygame.draw.rect(screen, (255, 255, 255), rect, 1, border_radius=4)  # 白边

        # 车载状态 (如果是放货任务且载货，画一个小箱子)
        if s.get("Load"):
            box_size = SHUTTLE_SIZE // 2
            bx = x + (SHUTTLE_SIZE - box_size) // 2
            by = y + (SHUTTLE_SIZE - box_size) // 2
            pygame.draw.rect(screen, (200, 100, 50), (bx, by, box_size, box_size), border_radius=2)

        # 编号
        txt = font_id.render(str(sid), True, (30, 30, 30))
        screen.blit(txt, (x + 2, y + 2))


def draw_info_panel(screen):
    py = MAP_HEIGHT
    pygame.draw.rect(screen, COLOR_PANEL_BG, (0, py, TOTAL_WIDTH, INFO_PANEL_HEIGHT))
    pygame.draw.line(screen, (100, 100, 100), (0, py), (TOTAL_WIDTH, py), 3)

    ft, fl, fm, fs = get_font(FONT_SIZE_TITLE, True), get_font(FONT_SIZE_LARGE, True), get_font(
        FONT_SIZE_MEDIUM), get_font(FONT_SIZE_SMALL)

    # 1. 系统状态
    x = 20
    y = py + 20
    screen.blit(ft.render("仓库调度监控", True, COLOR_HIGHLIGHT), (x, y))

    ct, done = shared_state.get("time", 0), shared_state.get("done", False)
    screen.blit(fl.render(f"{ct:.1f} s", True, COLOR_TEXT_MAIN), (x, y + 45))

    st_col, st_txt = (COLOR_GOOD, "运行中") if not done else (COLOR_BAD, "已结束")
    pygame.draw.rect(screen, st_col, (x + 120, y + 45, 100, 30), border_radius=5)
    st_surf = fs.render(st_txt, True, (30, 30, 30))
    screen.blit(st_surf, (x + 120 + 50 - st_surf.get_width() // 2, y + 52))

    # 2. 穿梭车卡片
    x_sh = 280
    screen.blit(fm.render("穿梭车状态", True, COLOR_ACCENT), (x_sh, y))
    for i, s in enumerate(shared_state.get("shuttles", [])[:4]):
        cx = x_sh + i * 165
        cy = y + 30
        pygame.draw.rect(screen, (60, 62, 68), (cx, cy, 150, 180), border_radius=8)

        id_col = COLOR_HIGHLIGHT if i % 2 == 0 else COLOR_ACCENT
        screen.blit(fl.render(f"NO.{i}", True, id_col), (cx + 10, cy + 10))

        status = s.get("status", "--")
        screen.blit(fm.render(status[:8], True, COLOR_TEXT_MAIN), (cx + 10, cy + 65))
        draw_battery_bar(screen, cx + 10, cy + 95, 130, 12, s.get("battery", 0))

        tsk = s.get("current_task")
        if tsk:
            info = f"{'放' if s.get('Load') else '取'}->{tsk.position}"
            screen.blit(fs.render(info, True, COLOR_TEXT_MAIN), (cx + 10, cy + 140))
        else:
            screen.blit(fs.render("待命...", True, COLOR_TEXT_DIM), (cx + 10, cy + 140))

    # 3. 统计
    xs = 960
    rel = len(shared_state.get("release_tasks", []))
    pick = len(shared_state.get("pick_tasks", []))

    pygame.draw.rect(screen, (60, 65, 70), (xs, y, 140, 50), border_radius=6)
    pygame.draw.circle(screen, COLOR_TASK_REL, (xs + 25, y + 25), 8)
    screen.blit(fs.render("待放货", True, COLOR_TEXT_DIM), (xs + 45, y + 5))
    screen.blit(fl.render(str(rel), True, COLOR_TEXT_MAIN), (xs + 45, y + 25))

    pygame.draw.rect(screen, (60, 65, 70), (xs, y + 60, 140, 50), border_radius=6)
    pygame.draw.circle(screen, COLOR_TASK_PICK, (xs + 25, y + 85), 8)
    screen.blit(fs.render("待取货", True, COLOR_TEXT_DIM), (xs + 45, y + 65))
    screen.blit(fl.render(str(pick), True, COLOR_TEXT_MAIN), (xs + 45, y + 85))

    comp, fail = shared_state.get("completed_tasks", 0), shared_state.get("failed_tasks", 0)
    tot = comp + fail
    rate = (comp / tot * 100) if tot > 0 else 0.0
    screen.blit(fm.render("成功率", True, COLOR_TEXT_DIM), (xs, y + 130))
    rc = COLOR_GOOD if rate > 80 else (COLOR_WARN if rate > 50 else COLOR_BAD)
    screen.blit(ft.render(f"{rate:.1f}%", True, rc), (xs, y + 155))
    screen.blit(fs.render(f"完:{comp} 败:{fail}", True, COLOR_TEXT_DIM), (xs + 100, y + 165))


def run_visualization():
    try:
        pygame.init()
        screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("智能仓库调度 V5.0 (平滑动画版)")
        clock = pygame.time.Clock()
        running = True

        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                    with state_lock: shared_state["done"] = True

            with state_lock:
                screen.fill(COLOR_BG)
                # 1. 画地图
                if hasattr(warehouse, 'draw'):
                    warehouse.draw(screen)
                else:
                    # 备用绘制
                    for r in range(ROWS):
                        for c in range(COLS):
                            pygame.draw.rect(screen, (40, 40, 40), (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                                             1)

                # 2. 画辅助信息 (任务点、连线)
                draw_visual_aids(screen)

                # 3. 画小车 (带动画)
                draw_shuttles(screen)

                # 4. 画信息面板
                draw_info_panel(screen)

            pygame.display.flip()
            clock.tick(30)  # 30 FPS
        pygame.quit()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == "__main__": run_visualization()