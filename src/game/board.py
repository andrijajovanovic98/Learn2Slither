import random
from typing import List, Tuple
import pygame

GREEN = (76, 175, 80)
RED = (244, 67, 54)
BLUE = (33, 150, 243)
GREY = (30, 30, 30)
WHITE = (220, 220, 220)

APPLE_GREEN = "G"
APPLE_RED = "R"


class Board:
    def __init__(self, size: int, cell_px: int = 40):
        self.size = size
        self.cell_px = cell_px
        self.win_px = 600
        self.steps = 0

    def init_pygame(self):
        pygame.init()
        self.cell_px = self.win_px // self.size
        w = self.win_px + 200
        h = self.cell_px * self.size
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"Learn2Slither {self.size}x{self.size}")

        self.stats_panel_width = 200
        self.panel_rect = pygame.Rect(w - self.stats_panel_width, 0, self.stats_panel_width, h)

    def empty_cells(self, snake_cells: List[Tuple[int, int]], apples: List[Tuple[int, int]]):
        occ = set(snake_cells) | set(apples)
        return [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in occ]

    def spawn_apples(self, snake_cells, want_green=2, want_red=1):
        apples = []

        def pick_empty():
            empties = self.empty_cells(snake_cells, apples)
            return random.choice(empties) if empties else None

        for _ in range(want_green):
            pos = pick_empty()
            if pos:
                apples.append(pos)
        for _ in range(want_red):
            pos = pick_empty()
            if pos:
                apples.append(pos)
        return apples

    def draw(self, snake_cells, apples_kinds, fps=6.0, steps=0, max_len=0):
        if not self.screen:
            return
        self.screen.fill((30, 30, 30))

        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * self.cell_px, y * self.cell_px, self.cell_px - 1, self.cell_px - 1)
                pygame.draw.rect(self.screen, (45, 45, 45), rect, 1)

        # Apples drawing
        for (x, y), kind in apples_kinds:
            color = (76, 175, 80) if kind == "G" else (244, 67, 54)
            rect = pygame.Rect(x * self.cell_px, y * self.cell_px, self.cell_px - 2, self.cell_px - 2)
            pygame.draw.rect(self.screen, color, rect)

        # Snake drawing
        for i, (x, y) in enumerate(snake_cells):
            rect = pygame.Rect(x * self.cell_px, y * self.cell_px, self.cell_px - 2, self.cell_px - 2)
            pygame.draw.rect(self.screen, (33, 150, 243), rect)

        pygame.draw.rect(self.screen, (0, 0, 0), self.panel_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.panel_rect, 3)

        font = pygame.font.Font(None, 36)
        length_text = font.render(f"Length: {len(snake_cells)}", True, (255, 255, 255))
        max_len_text = font.render(f"Max Length: {max_len}", True, (255, 255, 255))
        steps_text = font.render(f"Steps: {steps}", True, (255, 255, 255))

        self.screen.blit(length_text, (self.panel_rect.x + 10, self.panel_rect.y + 10))
        self.screen.blit(max_len_text, (self.panel_rect.x + 10, self.panel_rect.y + 50))
        self.screen.blit(steps_text, (self.panel_rect.x + 10, self.panel_rect.y + 90))

        pygame.display.update()
