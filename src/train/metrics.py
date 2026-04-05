class EpisodeStats:
    def __init__(self):
        self.total_reward = 0.0
        self.steps = 0
        self.max_len = 0

    def log_step(self, reward, snake_len):
        self.total_reward += reward
        self.steps += 1
        self.max_len = max(self.max_len, snake_len)
