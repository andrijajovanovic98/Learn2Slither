import random
import pygame
from game.board import Board
from game.snake import Snake
from game.apple import classify_apples, eat_effect
from agent.state_encoder import encode_state
from agent.q_agent import QAgent
from .metrics import EpisodeStats

REWARD_GREEN = +1.0
REWARD_RED = -1.0
REWARD_STEP = -0.01
REWARD_DEAD = -10.0


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.board = Board(cfg.board_size)
        if cfg.visual:
            self.board.init_pygame()
            self.clock = pygame.time.Clock()
        else:
            self.clock = None
        self.agent = QAgent()
        if cfg.load_path:
            self.agent.load(cfg.load_path)

        self.max_len = 0
        self.paused = bool(self.cfg.step)
        self.step_once = False

    def _random_snake3(self, size: int):
        """Generates a horizontal 3-length snake not near walls, with the head facing right."""
        x = random.randrange(2, size - 2)
        y = random.randrange(2, size - 2)
        return [(x+2, y), (x+1, y), (x, y)]

    def _handle_events(self):
        """Handles quit events and pause/step keys (SPACE/P/ESC)."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise SystemExit
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    if not self.paused:
                        self.paused = True
                    else:
                        self.step_once = True
                elif ev.key == pygame.K_p:
                    self.paused = not self.paused
                elif ev.key == pygame.K_ESCAPE:
                    raise SystemExit

    def _safe_action(self, snake, vision, a):
        """
        Ensures the chosen move is safe; selects the best legal alternative if needed.
        """
        from game.snake import DIR_VECT, UP, RIGHT, DOWN, LEFT
        size = self.cfg.board_size
        hx, hy = snake.head

        # Check if the chosen action 'a' is legal
        dx, dy = DIR_VECT[a]
        nx, ny = hx + dx, hy + dy
        if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in snake.body:
            return a

        # Alternatives ordered: prefer G, then 0, then R; avoid W/S/H
        order = {"G": 0, "0": 1, "R": 2, "W": 3, "S": 4, "H": 4}
        candidates = [UP, RIGHT, DOWN, LEFT]
        viable = []
        for cand in candidates:
            cdx, cdy = DIR_VECT[cand]
            tx, ty = hx + cdx, hy + cdy
            legal = (0 <= tx < size and 0 <= ty < size and (tx, ty) not in snake.body)
            if legal:
                sym = vision[cand]
                viable.append((order.get(sym, 5), cand))

        if viable:
            viable.sort(key=lambda x: x[0])
            return viable[0][1]

        return a

    def run(self):
        if self.cfg.visual:
            self.show_lobby()

        while True:
            for ep in range(self.cfg.episodes):
                stats = EpisodeStats()
                self.board.steps = 0

                snake = Snake(self.cfg.board_size, self._random_snake3(self.cfg.board_size))

                apples = self.board.spawn_apples(list(snake.body))
                apples_kinds = classify_apples(list(snake.body), apples)
                s = encode_state(snake.head, list(snake.body), apples_kinds, self.cfg.board_size)

                while snake.alive:
                    if self.cfg.visual:
                        self._handle_events()
                        self.board.draw(list(snake.body), apples_kinds, self.cfg.fps, self.board.steps, self.max_len)
                        if self.paused and not self.step_once:
                            if self.clock:
                                self.clock.tick(self.cfg.fps)
                            continue

                    a = self.agent.act(s)

                    if self.cfg.step:
                        dir_names = ["UP", "RIGHT", "DOWN", "LEFT"]
                        print(f"vision={s} pick={dir_names[a]}")

                    if len(snake.body) >= 2:
                        hx, hy = snake.head
                        nx, ny = snake.body[1]
                        cur_dx, cur_dy = hx - nx, hy - ny
                        from game.snake import DIR_VECT
                        cand_dx, cand_dy = DIR_VECT[a]
                        if (cand_dx, cand_dy) == (-cur_dx, -cur_dy):
                            dir_map = {v: k for k, v in DIR_VECT.items()}
                            a = dir_map[(cur_dx, cur_dy)]

                    a = self._safe_action(snake, s, a)

                    greens = [i for i, sym in enumerate(s) if sym == "G"]
                    if greens and random.random() < 0.80:
                        a = random.choice(greens)

                    if self.cfg.step:
                        print(f"vision={s} action={a} len={snake.length()}")

                    snake.move(a)
                    self.board.steps += 1
                    reward = REWARD_STEP

                    if not snake.alive:
                        reward = REWARD_DEAD
                        self.agent.learn(s, a, reward, None, dontlearn=self.cfg.dontlearn)
                        stats.log_step(reward, snake.length())
                        break

                    grow, eaten_idx = eat_effect(snake.head, apples_kinds)
                    if eaten_idx is not None:
                        snake.apply_growth(grow)
                        reward = REWARD_GREEN if grow > 0 else REWARD_RED

                        if not snake.alive or snake.length() <= 0:
                            reward = REWARD_DEAD
                            self.agent.learn(s, a, reward, None, dontlearn=self.cfg.dontlearn)
                            stats.log_step(reward, snake.length())
                            break

                        apples = self.board.spawn_apples(list(snake.body))
                        apples_kinds = classify_apples(list(snake.body), apples)

                    s_next = encode_state(snake.head, list(snake.body), apples_kinds, self.cfg.board_size)
                    self.agent.learn(s, a, reward, s_next, dontlearn=self.cfg.dontlearn)
                    stats.log_step(reward, snake.length())
                    s = s_next

                    if self.step_once:
                        self.step_once = False

                    self.max_len = max(self.max_len, snake.length())

                    if self.cfg.visual and self.clock:
                        self.clock.tick(self.cfg.fps)

                print(f"[Episode {ep+1}/{self.cfg.episodes}] reward={stats.total_reward:.1f} "
                      f"steps={stats.steps} max_len={self.max_len}")

                if self.cfg.visual:
                    self.show_game_over(self.max_len, stats.steps, stats.total_reward)

                if self.cfg.save_path:
                    self.agent.save(self.cfg.save_path)
                    print(f"Saved model to {self.cfg.save_path}")
                    if ep + 1 >= self.cfg.episodes:
                        print("Training complete. Exiting...")
                        return

            if self.cfg.visual:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.reset_game()
                        break
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return

    def show_lobby(self):
        """
        Displays a pre-game lobby screen where the user can adjust settings (board size, FPS, step mode)
        using keyboard input before starting the game.
        """
        pygame.font.init()
        font = pygame.font.Font(None, 36)
        screen = self.board.screen
        clock = pygame.time.Clock()

        title_text = font.render("Game Lobby, Press Enter to start", True, (255, 255, 255))
        board_size_text = font.render(f"Board Size: {self.cfg.board_size}", True, (255, 255, 255))
        fps_text = font.render(f"FPS: {self.cfg.fps}", True, (255, 255, 255))
        step_text = font.render(f"Step Mode: {'ON' if self.cfg.step else 'OFF'}", True, (255, 255, 255))

        running = True
        while running:
            screen.fill((0, 0, 255))
            screen.blit(title_text, (screen.get_width() // 2 - title_text.get_width() // 2, 50))
            screen.blit(board_size_text, (screen.get_width() // 2 - board_size_text.get_width() // 2, 150))
            screen.blit(fps_text, (screen.get_width() // 2 - fps_text.get_width() // 2, 200))
            screen.blit(step_text, (screen.get_width() // 2 - step_text.get_width() // 2, 250))

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        self.cfg.board_size += 1
                        if self.cfg.board_size > 15:
                            self.cfg.board_size = 15
                    elif event.key == pygame.K_DOWN:
                        self.cfg.board_size -= 1
                        if self.cfg.board_size < 5:
                            self.cfg.board_size = 5
                    elif event.key == pygame.K_LEFT:
                        self.cfg.fps -= 1
                        if self.cfg.fps < 3:
                            self.cfg.fps = 3
                    elif event.key == pygame.K_RIGHT:
                        self.cfg.fps += 1
                        if self.cfg.fps > 10:
                            self.cfg.fps = 10
                        elif event.key == pygame.K_s:
                            self.cfg.step = not self.cfg.step
                            self.paused = bool(self.cfg.step)
                    elif event.key == pygame.K_RETURN:
                        running = False

            clock.tick(30)

    def reset_game(self):
        """Resets the game state to start a new game."""
        self.board = Board(self.cfg.board_size)
        if self.cfg.visual:
            self.board.init_pygame()
        self.agent = QAgent()
        if self.cfg.load_path:
            self.agent.load(self.cfg.load_path)
        self.max_len = 0
        self.paused = bool(self.cfg.step)
        self.step_once = False

    def show_game_over(self, max_len, steps, total_reward):
        """
        Displays the Game Over screen showing stats (max length, steps, total reward)
        and waits for the user to either restart (R) or quit (ESC).
        """
        pygame.font.init()
        font = pygame.font.Font(None, 36)
        screen = self.board.screen
        # clock = pygame.time.Clock()

        game_over_text = font.render("Game Over", True, (255, 0, 0))
        stats_text = font.render(f"Max Length: {max_len} | Steps: {steps} | Reward: {total_reward}", True, (255, 255, 255))

        screen.fill((0, 0, 0))
        screen.blit(game_over_text, (screen.get_width() // 2 - game_over_text.get_width() // 2, screen.get_height() // 4))
        screen.blit(stats_text, (screen.get_width() // 2 - stats_text.get_width() // 2, screen.get_height() // 2))

        restart_text = font.render("Press R to Restart or ESC to Quit", True, (255, 255, 255))
        screen.blit(restart_text, (screen.get_width() // 2 - restart_text.get_width() // 2, screen.get_height() // 1.5))

        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_r:
                        self.reset_game()
                        waiting = False
