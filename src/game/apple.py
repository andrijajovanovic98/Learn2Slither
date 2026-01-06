from typing import List, Tuple
from .board import APPLE_GREEN, APPLE_RED


def classify_apples(snake_cells: List[Tuple[int, int]], apples: List[Tuple[int, int]]):
    """Labels the first two apples green, the last one red."""
    kinds = []
    for i, pos in enumerate(apples):
        kinds.append((pos, APPLE_GREEN if i < 2 else APPLE_RED))
    return kinds


def eat_effect(pos, apples_kinds):
    """Returns grow_delta (+1 for green, -1 for red) and the eaten apple index."""
    for idx, ((ax, ay), kind) in enumerate(apples_kinds):
        if (ax, ay) == pos:
            return (1 if kind == APPLE_GREEN else -1), idx
    return 0, None
