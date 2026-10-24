import numpy as np
from rule_engine import RuleEngine
from neural_core import NeuralCore

class HybridAgent:
    """Combines a symbolic rule engine and neural evaluator."""

    def __init__(self, nn_weight=0.5):
        self.rule = RuleEngine()
        self.nn = NeuralCore()
        self.nn_weight = nn_weight

    def act(self, state):
        rule_action = self.rule.decide(state)
        nn_action = self.nn.choose_action(state)

        # Blend or probabilistic choice
        if np.random.rand() < self.nn_weight:
            return nn_action
        return rule_action

    def mutate(self):
        self.nn.mutate(0.1)
        self.rule.mutate()
        self.nn_weight = np.clip(self.nn_weight + np.random.randn() * 0.05, 0, 1)
