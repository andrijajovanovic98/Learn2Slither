from typing import List, Tuple

W, H, S, G, R, Z = "W", "H", "S", "G", "R", "0"

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT


def encode_state(head: Tuple[int, int], snake_cells: List[Tuple[int, int]], apples_kinds, size: int):
    """
    Encodes the snake's perception by raycasting in four directions from its head
    and returning what it sees first in each direction (wall, body, head, or apple).
    """
    sx, sy = head
    vision = []
    snake_set = set(snake_cells)
    apples_map = {(x, y): k for (x, y), k in apples_kinds}
    for dx, dy in DIRS:
        x, y = sx, sy
        sym = Z
        while True:
            x += dx
            y += dy
            if x < 0 or y < 0 or x >= size or y >= size:
                sym = W
                break
            if (x, y) in snake_set:
                sym = S if (x, y) != (sx, sy) else H
                break
            if (x, y) in apples_map:
                sym = apples_map[(x, y)]  # "G" or "R"
                break
        vision.append(sym)
    return tuple(vision)
