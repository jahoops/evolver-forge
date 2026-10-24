"""
rule_feedback.py — Generate new synthetic training samples from rule logic
and use them to fine-tune the neural network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .tasks import TaskSpec


def generate_rule_based_samples(n_points=3000, seed=123):
    """Generate synthetic data that reflect the current symbolic rules."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_points, 2))
    x, y = X[:, 0], X[:, 1]

    # Same class boundaries used by the rule system
    c0 = (x**2 + y**2 < 0.4**2)
    c1 = (~c0) & (np.abs(x - y) < 0.2)
    c2 = (~c0) & (~c1) & (y < 0.3 * np.sin(3 * x) + 0.1)

    y_labels = np.zeros(n_points, dtype=int)
    y_labels[c1] = 1
    y_labels[c2] = 2

    return X, y_labels


def fine_tune_nn(model, task: TaskSpec, epochs: int = 30, lr: float = 0.005):
    """Fine-tune the NN using synthetic rule-consistent data when available."""

    if task.name != "shape":
        # Only the classic shape world has bespoke rule-derived samples today.
        return model, {}, 0.0

    X, y = generate_rule_based_samples()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    model.train()

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X_t).argmax(dim=1)
        fine_tune_acc = (preds == y_t).float().mean().item()

    print(f"[green]NN fine-tuned on {len(X)} rule-based samples.[/green]")
    return model, opt.state_dict(), fine_tune_acc
