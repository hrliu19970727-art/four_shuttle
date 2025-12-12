# visualization.py
import pygame
import sys
import time
from config import ROWS, COLS, CELL_SIZE, COLORS
from map import warehouse
from shared_state import shared_state, state_lock

# 优化常量
INFO_PANEL_WIDTH = 300
MAP_WIDTH = COLS * CELL_SIZE
TOTAL_WIDTH = MAP_WIDTH + INFO_PANEL_WIDTH
TOTAL_HEIGHT = ROWS * CELL_SIZE
FONT_SIZE_LARGE = 24
FONT_SIZE_MEDIUM = 18
FONT_SIZE_SMALL = 14

# 颜色定义
BACKGROUND_COLOR = (40, 44, 52)
PANEL_BG_COLOR = COLORS.get("panel_bg", (60, 63, 65))
TEXT_COLOR = COLORS.get("text", (220, 220, 220))
HIGHLIGHT_COLOR = COLORS.get("highlight", (255, 203, 0))


def safe_render_font(text, font_size, color):
    """安全创建字体"""
    try:
        # 使用常见的中文字体
        font = pygame.font.SysFont("simhei", font_size)
        return font.render(str(text), True, color)
    except:
        try:
            font = pygame.font.SysFont("arial", font_size)
            return font.render(str(text), True, color)
        except:
            try:
                font = pygame.font.Font(None, font_size)
                return font.render(str(text), True, color)
            except:
                # 最后的备用方案
                surface = pygame.Surface((len(str(text)) * 10, 20))
                surface.fill((255, 0, 0))  # 红色背景表示错误
                return surface


def draw_info_panel(screen):
    """
    绘制信息面板 -
    【关键修复】假设 state_lock 已经被调用者持有，直接读取 shared_state
    """
    try:
        # 绘制面板背景
        pygame.draw.rect(screen, PANEL_BG_COLOR, (MAP_WIDTH, 0, INFO_PANEL_WIDTH, TOTAL_HEIGHT))

        y_offset = 20

        # 标题
        title_text = safe_render_font("仓库调度系统", FONT_SIZE_LARGE, HIGHLIGHT_COLOR)
        screen.blit(title_text, (MAP_WIDTH + 20, y_offset))
        y_offset += 40

        # 当前时间
        current_time = shared_state.get("time", 0)
        time_text = safe_render_font(f"仿真时间: {current_time:.1f}s", FONT_SIZE_MEDIUM, TEXT_COLOR)
        screen.blit(time_text, (MAP_WIDTH + 20, y_offset))
        y_offset += 30

        # 小车状态标题
        shuttle_title = safe_render_font("穿梭车状态:", FONT_SIZE_MEDIUM, HIGHLIGHT_COLOR)
        screen.blit(shuttle_title, (MAP_WIDTH + 20, y_offset))
        y_offset += 25

        # 小车状态
        shuttles = shared_state.get("shuttles", [])

        busy_count = 0
        for i, shuttle in enumerate(shuttles):
            if i >= 4:  # 只显示前4个小车
                break

            pos = shuttle.get("pos", [0, 0])
            busy = shuttle.get("busy", False)
            load = shuttle.get("Load", False)
            current_task = shuttle.get("current_task")

            if busy:
                busy_count += 1

            status = "忙碌" if busy else "空闲"
            load_status = "载货" if load else "空载"
            color = HIGHLIGHT_COLOR if busy else TEXT_COLOR

            task_info = ""
            if current_task and hasattr(current_task, 'position'):
                task_info = f" -> {current_task.position}"

            shuttle_text = f"  小车{i}: {pos} {status} {load_status}{task_info}"
            text_surface = safe_render_font(shuttle_text, FONT_SIZE_SMALL, color)
            screen.blit(text_surface, (MAP_WIDTH + 20, y_offset))
            y_offset += 20

        y_offset += 15

        # 任务状态
        tasks_title = safe_render_font("任务队列:", FONT_SIZE_MEDIUM, HIGHLIGHT_COLOR)
        screen.blit(tasks_title, (MAP_WIDTH + 20, y_offset))
        y_offset += 25

        release_count = len(shared_state.get("release_tasks", []))
        pick_count = len(shared_state.get("pick_tasks", []))

        release_text = safe_render_font(f"  放货任务: {release_count}", FONT_SIZE_SMALL, (100, 200, 255))
        screen.blit(release_text, (MAP_WIDTH + 20, y_offset))
        y_offset += 20

        pick_text = safe_render_font(f"  取货任务: {pick_count}", FONT_SIZE_SMALL, (255, 200, 100))
        screen.blit(pick_text, (MAP_WIDTH + 20, y_offset))
        y_offset += 20

        # 系统状态
        y_offset += 10
        status_title = safe_render_font("系统状态:", FONT_SIZE_MEDIUM, HIGHLIGHT_COLOR)
        screen.blit(status_title, (MAP_WIDTH + 20, y_offset))
        y_offset += 25

        done = shared_state.get("done", False)
        total_tasks = release_count + pick_count

        status_text = "运行中" if not done else "已结束"
        status_color = (100, 255, 100) if not done else (255, 100, 100)
        status_surface = safe_render_font(f"  状态: {status_text}", FONT_SIZE_SMALL, status_color)
        screen.blit(status_surface, (MAP_WIDTH + 20, y_offset))
        y_offset += 20

        # 统计信息
        stats_text = safe_render_font(f"  忙碌小车: {busy_count}/{len(shuttles)}", FONT_SIZE_SMALL, TEXT_COLOR)
        screen.blit(stats_text, (MAP_WIDTH + 20, y_offset))
        y_offset += 20

        stats_text2 = safe_render_font(f"  总任务数: {total_tasks}", FONT_SIZE_SMALL, TEXT_COLOR)
        screen.blit(stats_text2, (MAP_WIDTH + 20, y_offset))
        y_offset += 20

        # 操作提示
        y_offset += 10
        hint_title = safe_render_font("操作提示:", FONT_SIZE_MEDIUM, HIGHLIGHT_COLOR)
        screen.blit(hint_title, (MAP_WIDTH + 20, y_offset))
        y_offset += 25

        hint1 = safe_render_font("  ESC: 退出程序", FONT_SIZE_SMALL, TEXT_COLOR)
        screen.blit(hint1, (MAP_WIDTH + 20, y_offset))
        y_offset += 20

        hint2 = safe_render_font("  R: 重新开始", FONT_SIZE_SMALL, TEXT_COLOR)
        screen.blit(hint2, (MAP_WIDTH + 20, y_offset))

    except Exception as e:
        print(f"信息面板错误: {e}")


def run_visualization():
    """启动可视化界面"""
    try:
        pygame.init()
        screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("自动化仓库调度模拟")
        clock = pygame.time.Clock()

        print("可视化界面启动成功")
        print(f"窗口尺寸: {TOTAL_WIDTH} x {TOTAL_HEIGHT}")
        print(f"地图区域: {MAP_WIDTH} x {TOTAL_HEIGHT}")
        print(f"信息面板: {INFO_PANEL_WIDTH} x {TOTAL_HEIGHT}")

        frame_count = 0
        running = True

        while running:
            frame_count += 1

            # 处理事件 (必须在锁外，以响应用户操作)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    with state_lock:
                        shared_state["done"] = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        with state_lock:
                            shared_state["done"] = True
                    elif event.key == pygame.K_r:
                        print("重新开始功能待实现")
                        # 可以在这里添加重新开始逻辑

            # 【关键修复】将所有对共享数据的读取和绘制放在一个锁内
            with state_lock:
                # 检查退出条件
                if shared_state.get("done", False):
                    running = False

                # 清屏
                screen.fill(BACKGROUND_COLOR)

                try:
                    # 绘制仓库地图 (访问 warehouse 全局状态，现在安全)
                    if hasattr(warehouse, 'draw'):
                        warehouse.draw(screen)
                    else:
                        # 备用绘制
                        for row in range(ROWS):
                            for col in range(COLS):
                                color = (100, 100, 150)
                                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                                pygame.draw.rect(screen, color, rect)
                                pygame.draw.rect(screen, (50, 50, 50), rect, 1)

                    # 绘制信息面板 (访问 shared_state，现在安全)
                    draw_info_panel(screen)

                except Exception as e:
                    print(f"绘制错误: {e}")
                    # 错误时显示简单界面
                    error_font = pygame.font.Font(None, 36)
                    error_text = error_font.render("绘制错误", True, (255, 0, 0))
                    screen.blit(error_text, (TOTAL_WIDTH // 2 - 50, TOTAL_HEIGHT // 2))

            # 更新显示 (必须在锁外，避免长时间占用锁)
            pygame.display.flip()

            # 控制帧率
            clock.tick(30)

            # 每100帧打印一次状态（调试用）
            if frame_count % 300 == 0:  # 每10秒打印一次
                with state_lock:
                    shuttles = shared_state.get("shuttles", [])
                    busy_shuttles = sum(1 for s in shuttles if s.get("busy", False))
                    print(f"帧 {frame_count}, 时间: {shared_state.get('time', 0):.1f}s, "
                          f"忙碌小车: {busy_shuttles}/{len(shuttles)}")

        pygame.quit()
        print("可视化界面正常退出")

    except Exception as e:
        print(f"可视化严重错误: {e}")
        import traceback
        traceback.print_exc()
