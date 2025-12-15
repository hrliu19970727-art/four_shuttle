import pygame
import sys

# 定义颜色字典
COLORS = {
    "alley": (173, 216, 230),  # 主巷道 - 浅蓝
    "1": (0, 128, 128),  # 货位巷道 - 青色
    "aisle": (165, 42, 42),       # 货位巷道 - 棕色
    "occupied": (255, 215, 0),  # 占用货位 - 金色
    "shuttle_colors": [
        (255, 0, 0),  # 红色穿梭车1
        (0, 255, 0),  # 绿色穿梭车2
        (0, 0, 255),  # 蓝色穿梭车3
        (128, 0, 128),  # 紫色穿梭车4
        (128, 0, 128),  # 紫色穿梭车5
        (0, 128, 128),  # 青色穿梭车6
    ],
    "elevator": (255, 255, 0),  # 提升机颜色-黄色
    "target": (0, 0, 0)  # 目标点颜色
}

# 初始化pygame
pygame.init()

# 设置窗口大小
width, height = 1200, 200
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("颜色展示")

# 设置字体
font = pygame.font.SysFont('Arial', 16)


# 主循环
def main():
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((240, 240, 240))  # 浅灰色背景

        # 绘制颜色块
        x_pos = 20
        block_size = 100
        margin = 20

        # 绘制基本颜色
        for name, color in COLORS.items():
            if name == "shuttle_colors":  # 修正拼写错误
                continue
            if name == "shuttle_colors":  # 修正拼写错误
                continue
            if name == "shuttle_colors":
                continue

            # 绘制颜色块
            pygame.draw.rect(screen, color, (x_pos, 50, block_size, block_size))

            # 绘制颜色名称
            text = font.render(name, True, (0, 0, 0))
            screen.blit(text, (x_pos, 30))

            x_pos += block_size + margin

        # 绘制穿梭车颜色
        for i, color in enumerate(COLORS["shuttle_colors"], 1):
            # 绘制颜色块
            pygame.draw.rect(screen, color, (x_pos, 50, block_size, block_size))

            # 绘制颜色名称
            text = font.render(f"shuttle{i}", True, (0, 0, 0))
            screen.blit(text, (x_pos, 30))

            x_pos += block_size + margin

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()