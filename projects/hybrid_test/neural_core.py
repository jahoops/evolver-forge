import torch
import torch.nn as nn
import numpy as np

class NeuralCore(nn.Module):
    """Tiny NN evaluator to refine rule output."""

    def __init__(self, input_dim=4, hidden=8, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def choose_action(self, state):
        with torch.no_grad():
            s = torch.tensor(np.array(state), dtype=torch.float32)
            probs = self.forward(s)
            return int(torch.argmax(probs).item())

    def mutate(self, scale=0.1):
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * scale)
