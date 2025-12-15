# visualization.py
import pygame
import sys
import os
from config import ROWS, COLS, CELL_SIZE, COLORS
from map import warehouse
from shared_state import shared_state, state_lock

# --- 界面布局常量 ---
INFO_PANEL_WIDTH = 340  # 加宽侧边栏以容纳更多信息
MAP_WIDTH = COLS * CELL_SIZE
TOTAL_WIDTH = MAP_WIDTH + INFO_PANEL_WIDTH
TOTAL_HEIGHT = max(ROWS * CELL_SIZE + 40, 650)  # 保证最小高度

# --- 配色方案 (专业暗色模式) ---
COLOR_BG = (30, 32, 36)  # 全局背景
COLOR_MAP_BG = (40, 44, 52)  # 地图背景
COLOR_PANEL_BG = (50, 52, 57)  # 面板背景
COLOR_TEXT_MAIN = (240, 240, 240)  # 主要文字
COLOR_TEXT_DIM = (160, 160, 160)  # 次要文字
COLOR_ACCENT = (70, 160, 240)  # 强调色(蓝)
COLOR_HIGHLIGHT = (255, 190, 40)  # 高亮色(金)
COLOR_GOOD = (100, 200, 100)  # 成功/健康(绿)
COLOR_WARN = (230, 230, 60)  # 警告(黄)
COLOR_BAD = (230, 80, 80)  # 失败/错误(红)
COLOR_TASK_REL = (100, 180, 255)  # 放货任务颜色
COLOR_TASK_PICK = (255, 140, 80)  # 取货任务颜色

# 全局字体缓存
_cached_fonts = {}


def get_font(size, bold=False):
    """
    智能获取中文字体
    自动尝试 Windows/Mac/Linux 的常见中文字体，解决乱码问题
    """
    key = (size, bold)
    if key in _cached_fonts:
        return _cached_fonts[key]

    # 常见中文字体文件名列表
    font_names = [
        "simhei", "microsoftyahei", "msyahei", "simsun",  # Windows
        "pingfangsc", "heiti", "stheitilight",  # macOS
        "wenquanyizenhei", "wqy-microhei", "notosanscjk"  # Linux
    ]

    font = None
    # 1. 尝试通过 match_font 找系统字体
    for name in font_names:
        try:
            font_path = pygame.font.match_font(name)
            if font_path:
                font = pygame.font.Font(font_path, size)
                break
        except:
            continue

    # 2. 如果没找到，尝试 SysFont
    if font is None:
        try:
            # 组合名称尝试
            font = pygame.font.SysFont(",".join(font_names), size, bold=bold)
        except:
            # 3. 最后的保底
            font = pygame.font.Font(None, size)

    if bold and font:
        font.set_bold(True)

    _cached_fonts[key] = font
    return font


def draw_battery_bar(surface, x, y, w, h, level):
    """绘制电量进度条"""
    # 背景
    pygame.draw.rect(surface, (30, 30, 30), (x, y, w, h), border_radius=3)
    # 边框
    pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 1, border_radius=3)

    # 颜色判断
    if level > 60:
        col = COLOR_GOOD
    elif level > 25:
        col = COLOR_WARN
    else:
        col = COLOR_BAD

    # 进度
    fill_w = int(w * (level / 100))
    if fill_w > 0:
        # 限制最大宽度防止溢出
        fill_w = min(fill_w, w - 2)
        pygame.draw.rect(surface, col, (x + 1, y + 1, fill_w, h - 2), border_radius=2)


def draw_visual_aids(screen):
    """绘制地图辅助信息(任务点、连线、坐标)"""

    # 1. 绘制任务点 (直接画在地图格子中心)
    # 复制列表防止迭代时修改
    release_tasks = list(shared_state.get("release_tasks", []))
    pick_tasks = list(shared_state.get("pick_tasks", []))

    for task in release_tasks:
        r, c = task.position
        cx = c * CELL_SIZE + CELL_SIZE // 2
        cy = r * CELL_SIZE + CELL_SIZE // 2
        # 蓝色实心圆 + 白色描边
        pygame.draw.circle(screen, COLOR_TASK_REL, (cx, cy), 5)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6, 1)

    for task in pick_tasks:
        r, c = task.position
        cx = c * CELL_SIZE + CELL_SIZE // 2
        cy = r * CELL_SIZE + CELL_SIZE // 2
        # 橙色实心圆 + 白色描边
        pygame.draw.circle(screen, COLOR_TASK_PICK, (cx, cy), 5)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6, 1)

    # 2. 绘制小车意图连线
    shuttles = shared_state.get("shuttles", [])
    for s in shuttles:
        if s.get("busy") and s.get("current_task"):
            # 起点
            curr_r, curr_c = s["pos"]
            sx = curr_c * CELL_SIZE + CELL_SIZE // 2
            sy = curr_r * CELL_SIZE + CELL_SIZE // 2

            # 终点
            tr, tc = s["current_task"].position
            ex = tc * CELL_SIZE + CELL_SIZE // 2
            ey = tr * CELL_SIZE + CELL_SIZE // 2

            # 颜色区分偶数/奇数号小车
            line_color = COLOR_HIGHLIGHT if s.get("id", 0) % 2 == 0 else COLOR_ACCENT

            # 画线
            pygame.draw.line(screen, line_color, (sx, sy), (ex, ey), 2)
            # 画终点小靶心
            pygame.draw.circle(screen, line_color, (ex, ey), 4, 1)

    # 3. 绘制坐标标尺 (仅在左侧和顶部绘制)
    font_axis = get_font(12)
    # 顶栏列号
    for c in range(0, COLS, 5):
        txt = font_axis.render(str(c), True, COLOR_TEXT_DIM)
        screen.blit(txt, (c * CELL_SIZE + 2, 2))
    # 左栏行号
    for r in range(0, ROWS, 2):
        txt = font_axis.render(str(r), True, COLOR_TEXT_DIM)
        screen.blit(txt, (2, r * CELL_SIZE + 2))


def draw_info_panel(screen):
    """绘制右侧信息面板"""
    # 面板背景
    panel_rect = pygame.Rect(MAP_WIDTH, 0, INFO_PANEL_WIDTH, TOTAL_HEIGHT)
    pygame.draw.rect(screen, COLOR_PANEL_BG, panel_rect)
    # 分割线
    pygame.draw.line(screen, (80, 80, 80), (MAP_WIDTH, 0), (MAP_WIDTH, TOTAL_HEIGHT), 2)

    x_base = MAP_WIDTH + 20
    y = 20

    # --- 1. 标题区 ---
    title = get_font(26, True).render("仓库调度监控系统", True, COLOR_HIGHLIGHT)
    screen.blit(title, (x_base, y))
    y += 45

    # --- 2. 全局状态卡片 ---
    current_time = shared_state.get("time", 0)
    done = shared_state.get("done", False)

    # 时间
    pygame.draw.rect(screen, (60, 60, 65), (x_base, y, 140, 36), border_radius=6)
    time_lbl = get_font(14).render("仿真时间", True, COLOR_TEXT_DIM)
    time_val = get_font(20, True).render(f"{current_time:.1f} s", True, COLOR_TEXT_MAIN)
    screen.blit(time_lbl, (x_base + 10, y + 2))
    screen.blit(time_val, (x_base + 10, y + 16))

    # 状态
    status_bg = COLOR_GOOD if not done else COLOR_BAD
    pygame.draw.rect(screen, status_bg, (x_base + 150, y, 100, 36), border_radius=6)
    status_txt = "运行中" if not done else "已结束"
    stat_surf = get_font(18, True).render(status_txt, True, (30, 30, 30))
    # 居中显示
    stat_rect = stat_surf.get_rect(center=(x_base + 150 + 50, y + 18))
    screen.blit(stat_surf, stat_rect)

    y += 55

    # --- 3. 穿梭车列表 ---
    screen.blit(get_font(18, True).render("穿梭车状态", True, COLOR_ACCENT), (x_base, y))
    y += 25

    shuttles = shared_state.get("shuttles", [])
    for i, s in enumerate(shuttles):
        if i >= 5: break  # 最多显示5个

        # 卡片背景
        pygame.draw.rect(screen, (70, 72, 78), (x_base, y, INFO_PANEL_WIDTH - 40, 68), border_radius=6)

        # 左侧: ID 和 坐标
        id_color = COLOR_HIGHLIGHT if s.get("busy") else COLOR_TEXT_DIM
        screen.blit(get_font(20, True).render(f"#{i}", True, id_color), (x_base + 10, y + 8))

        pos_str = str(s.get('pos', [0, 0]))
        screen.blit(get_font(14).render(pos_str, True, COLOR_TEXT_DIM), (x_base + 10, y + 35))

        # 中间: 状态文字
        status = s.get("status", "未知")
        status_surf = get_font(15).render(status, True, COLOR_TEXT_MAIN)
        screen.blit(status_surf, (x_base + 60, y + 10))

        # 任务描述
        if s.get("current_task"):
            is_load = s.get("Load", False)
            act = "放" if is_load else "取"
            tgt = s['current_task'].position
            task_info = f"{act} -> {tgt}"
            screen.blit(get_font(14).render(task_info, True, COLOR_ACCENT), (x_base + 60, y + 35))
        else:
            screen.blit(get_font(14).render("--", True, COLOR_TEXT_DIM), (x_base + 60, y + 35))

        # 右侧: 电量
        bat = s.get("battery", 100)
        draw_battery_bar(screen, x_base + 200, y + 12, 80, 10, bat)
        bat_txt = get_font(12).render(f"{bat}%", True, COLOR_TEXT_DIM)
        screen.blit(bat_txt, (x_base + 200 + 40 - bat_txt.get_width() // 2, y + 26))

        y += 75

    y += 10

    # --- 4. 任务队列统计 ---
    screen.blit(get_font(18, True).render("任务队列概览", True, COLOR_ACCENT), (x_base, y))
    y += 25

    rel_cnt = len(shared_state.get("release_tasks", []))
    pick_cnt = len(shared_state.get("pick_tasks", []))

    # 放货计数器
    pygame.draw.rect(screen, (60, 65, 70), (x_base, y, 130, 50), border_radius=6)
    pygame.draw.circle(screen, COLOR_TASK_REL, (x_base + 20, y + 25), 6)
    screen.blit(get_font(14).render("待放货", True, COLOR_TEXT_DIM), (x_base + 35, y + 8))
    screen.blit(get_font(22, True).render(str(rel_cnt), True, COLOR_TEXT_MAIN), (x_base + 35, y + 24))

    # 取货计数器
    pygame.draw.rect(screen, (60, 65, 70), (x_base + 140, y, 130, 50), border_radius=6)
    pygame.draw.circle(screen, COLOR_TASK_PICK, (x_base + 160, y + 25), 6)
    screen.blit(get_font(14).render("待取货", True, COLOR_TEXT_DIM), (x_base + 175, y + 8))
    screen.blit(get_font(22, True).render(str(pick_cnt), True, COLOR_TEXT_MAIN), (x_base + 175, y + 24))

    y += 70

    # --- 5. 历史统计数据 ---
    comp = shared_state.get("completed_tasks", 0)
    fail = shared_state.get("failed_tasks", 0)
    total = comp + fail
    rate = (comp / total * 100) if total > 0 else 0.0

    screen.blit(get_font(18, True).render("累计执行统计", True, COLOR_ACCENT), (x_base, y))
    y += 30

    # 成功率大字
    rate_txt = get_font(36, True).render(f"{rate:.1f}%", True, COLOR_GOOD if rate > 80 else COLOR_WARN)
    screen.blit(rate_txt, (x_base, y))
    screen.blit(get_font(14).render("成功率", True, COLOR_TEXT_DIM), (x_base + 110, y + 18))

    y += 45
    # 详细数据
    row1 = f"完成: {comp}"
    row2 = f"失败: {fail}"
    row3 = f"总计: {total}"
    screen.blit(get_font(16).render(row1, True, COLOR_GOOD), (x_base, y))
    screen.blit(get_font(16).render(row2, True, COLOR_BAD), (x_base + 100, y))
    screen.blit(get_font(16).render(row3, True, COLOR_TEXT_MAIN), (x_base + 200, y))


def draw_end_screen(screen):
    """仿真结束后的结算画面"""
    # 半透明蒙层
    overlay = pygame.Surface((TOTAL_WIDTH, TOTAL_HEIGHT))
    overlay.fill((0, 0, 0))
    overlay.set_alpha(200)
    screen.blit(overlay, (0, 0))

    cx, cy = TOTAL_WIDTH // 2, TOTAL_HEIGHT // 2

    # 结算信息
    comp = shared_state.get("completed_tasks", 0)
    fail = shared_state.get("failed_tasks", 0)
    time_total = shared_state.get("time", 0)

    lines = [
        ("仿真已结束", 50, COLOR_HIGHLIGHT),
        (f"总耗时: {time_total:.1f} 秒", 30, COLOR_TEXT_MAIN),
        (f"✅ 成功任务: {comp}", 30, COLOR_GOOD),
        (f"❌ 失败任务: {fail}", 30, COLOR_BAD if fail > 0 else COLOR_TEXT_DIM),
        ("按 ESC 键退出", 24, COLOR_ACCENT)
    ]

    offset = -80
    for txt, size, color in lines:
        surf = get_font(size, True).render(txt, True, color)
        rect = surf.get_rect(center=(cx, cy + offset))
        screen.blit(surf, rect)
        offset += size + 20


def run_visualization():
    """主程序入口"""
    try:
        pygame.init()
        screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("智能仓库调度仿真系统 v3.0")
        clock = pygame.time.Clock()

        print(f"可视化界面启动: {TOTAL_WIDTH}x{TOTAL_HEIGHT}")

        running = True
        while running:
            # 1. 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    with state_lock:
                        shared_state["done"] = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        with state_lock: shared_state["done"] = True

            # 2. 绘图循环
            with state_lock:
                done = shared_state.get("done", False)

                # 背景与地图
                screen.fill(COLOR_BG)
                if hasattr(warehouse, 'draw'):
                    warehouse.draw(screen)
                else:
                    # 备用网格绘制
                    for r in range(ROWS):
                        for c in range(COLS):
                            rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                            pygame.draw.rect(screen, COLOR_MAP_BG, rect)
                            pygame.draw.rect(screen, (60, 60, 60), rect, 1)

                # 辅助层与UI
                draw_visual_aids(screen)
                draw_info_panel(screen)

                # 结算画面
                if done:
                    draw_end_screen(screen)

            pygame.display.flip()
            clock.tick(30)  # 帧率限制

        pygame.quit()
        print("可视化界面已关闭")

    except Exception as e:
        print(f"可视化运行时错误: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == "__main__":
    # 独立测试模式
    shared_state["shuttles"] = [
        {"id": 0, "pos": [2, 4], "busy": True, "battery": 85, "status": "移动中", "current_task": None},
        {"id": 1, "pos": [5, 10], "busy": False, "battery": 30, "status": "充电中", "current_task": None}
    ]
    run_visualization()