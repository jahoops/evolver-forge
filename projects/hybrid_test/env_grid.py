import random
import numpy as np

class GridEnvironment:
    """Simple 2D grid world with goal and obstacles."""

    def __init__(self, size=10, goal=(9, 9), obstacles=4):
        self.size = size
        self.goal = np.array(goal)
        self.obstacles = {tuple(np.random.randint(0, size, 2)) for _ in range(obstacles)}

    def reset(self):
        self.pos = np.array([0, 0])
        self.energy_used = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.pos / self.size, self.goal / self.size])

    def step(self, action):
        """Move in one of 4 directions; return (state, reward, done)."""
        moves = {0: np.array([1, 0]), 1: np.array([-1, 0]), 2: np.array([0, 1]), 3: np.array([0, -1])}
        self.pos = np.clip(self.pos + moves[action], 0, self.size - 1)
        self.energy_used += 1

        if tuple(self.pos) in self.obstacles:
            reward = -5.0
        elif np.all(self.pos == self.goal):
            reward = 10.0
        else:
            # negative distance penalty
            dist = np.linalg.norm(self.goal - self.pos)
            reward = -dist / self.size

        done = np.all(self.pos == self.goal) or self.energy_used >= self.size * 2
        return self._get_state(), reward, done
