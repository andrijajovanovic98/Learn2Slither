from collections import deque


UP, RIGHT, DOWN, LEFT = (0, 1, 2, 3)

DIR_VECT = {
    UP:    (0, -1),
    RIGHT: (1,  0),
    DOWN:  (0,  1),
    LEFT:  (-1, 0),
}


class Snake:
    def __init__(self, board_size: int, start_cells):
        self.size = board_size
        self.body = deque(start_cells)
        self.alive = True
        self.pending_growth = 0

    @property
    def head(self):
        return self.body[0]

    def move(self, direction):
        if not self.alive:
            return
        dx, dy = DIR_VECT[direction]
        hx, hy = self.head
        nx, ny = hx + dx, hy + dy

        # wall collision
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            self.alive = False
            return
        # self-collision
        if (nx, ny) in self.body:
            self.alive = False
            return

        self.body.appendleft((nx, ny))

        if self.pending_growth > 0:
            self.pending_growth -= 1
        else:
            self.body.pop()

    def apply_growth(self, delta: int):
        """
        delta > 0: do not cut the tail for this many steps (effectively grow)
        delta < 0: immediate shortening (red apple)
        """
        if delta > 0:
            self.pending_growth += delta
        elif delta < 0:
            cuts = -delta
            for _ in range(cuts):
                if len(self.body) > 1:
                    self.body.pop()
                else:
                    self.alive = False
                    break

    def length(self):
        return len(self.body)
