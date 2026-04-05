import random
import pickle
from collections import defaultdict
import logging


ACTIONS = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.Q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_start = 0.6
        self.eps_end = 0.05
        self.eps_decay_episodes = 1500

    def set_episode(self, ep_index: int):
        """Sets epsilon based on episode index using linear decay."""

        t = min(1.0, ep_index / max(1, self.eps_decay_episodes))
        self.epsilon = self.eps_start + (self.eps_end - self.eps_start) * t

    def act(self, state):
        """Chooses an action using epsilon-greedy policy based on the current state."""
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
            logging.info(f"[RANDOM] State={state} → Action={action}")
            return action

        q = self.Q[state]
        action = max(range(4), key=lambda a: q[a])
        logging.info(f"[Q-TABLE] State={state} → Q-values={q} → Action={action}")
        return action

    def learn(self, s, a, r, s_next, dontlearn=False):
        """Updates the Q-value for the given state-action pair using the Q-learning formula."""

        if dontlearn:
            return
        qsa = self.Q[s][a]
        max_next = max(self.Q[s_next]) if s_next is not None else 0.0
        target = r + self.gamma * max_next
        self.Q[s][a] = qsa + self.alpha * (target - qsa)

    def save(self, path):
        """Saves the current Q-table to a file using pickle."""

        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        """Loads a previously saved Q-table from a file into memory."""

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.Q.clear()
            for k, v in data.items():
                self.Q[k] = v
