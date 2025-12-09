# simple_test.py
import pygame
from config import ROWS, COLS, CELL_SIZE


def simple_draw():
    pygame.init()
    screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        # 简单绘制网格
        for row in range(ROWS):
            for col in range(COLS):
                color = (100, 150, 100) if (row + col) % 2 == 0 else (150, 100, 100)
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    simple_draw()