import numpy as np
import random

class RuleEngine:
    """Fast symbolic approximator; moves toward goal greedily."""

    def decide(self, state):
        # state = [px, py, gx, gy] normalized
        px, py, gx, gy = state
        dx, dy = gx - px, gy - py

        if abs(dx) > abs(dy):
            return 0 if dx > 0 else 1  # right or left
        else:
            return 2 if dy > 0 else 3  # up or down

    def mutate(self):
        # later: evolve thresholds, add randomization
        return self
