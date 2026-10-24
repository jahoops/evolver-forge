"""visualize_rules.py — visualize neural and rule decision regions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import load
from dataclasses import asdict

from .main import generate_points
from .tasks import TaskSpec
from .nn_agent import FlexibleNN, _logits_to_probs
from .architectures import ArchGenome, NNArch
from projects.coevolve_base.utils.checkpoint import safe_load_checkpoint


def visualize_boundaries(project_dir: Path, gen_idx: int, task: TaskSpec):
    """Plot model and rule decision regions for the active task."""

    X, _ = generate_points()
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    checkpoint_base = Path(__file__).parent
    arch_path = project_dir / "arch_current.json"
    nn_arch = NNArch()
    if arch_path.exists():
        try:
            arch_data = json.loads(arch_path.read_text(encoding="utf-8"))
            payload = arch_data.get("genome", arch_data) if isinstance(arch_data, dict) else arch_data
            nn_arch = ArchGenome.from_dict(payload).nn
        except Exception:
            pass

    if task.loss_type == "bce":
        output_dim = 1
    else:
        output_dim = task.output_dim or task.num_classes or 3

    model = FlexibleNN(X.shape[1], int(output_dim), nn_arch)

    if not safe_load_checkpoint(
        model,
        checkpoint_base / "nn_model.pt",
        model_name="TaskNN",
        expected_type="FlexibleNN",
        expected_arch=asdict(nn_arch),
    ):
        raise FileNotFoundError("Neural network checkpoint nn_model.pt not found or incompatible.")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(grid, dtype=torch.float32))
        probs = _logits_to_probs(logits, task).cpu().numpy()

    if probs.ndim == 1 or probs.shape[1] == 1:
        preds_nn = (probs.reshape(-1) >= 0.5).astype(int)
    else:
        preds_nn = probs.argmax(axis=1)

    tree_path = project_dir / "rules_model.joblib"
    tree = load(tree_path)
    preds_rule = tree.predict(grid)

    mutated_file = project_dir / "mutated_rules.txt"
    if mutated_file.exists():
        mutated_txt = mutated_file.read_text(encoding="utf-8")
        preds_mut = np.array([eval_mutated(mutated_txt, point) for point in grid])
    else:
        preds_mut = preds_rule

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds_nn.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
    plt.contour(xx, yy, preds_rule.reshape(xx.shape), colors="green", linewidths=1, alpha=0.6)
    plt.contour(xx, yy, preds_mut.reshape(xx.shape), colors="orange", linewidths=1, alpha=0.8)

    plt.title(
        "Co-evolution Gen {gen}: NN vs Rules ({task})".format(
            gen=gen_idx,
            task=task.name,
        )
    )
    plt.tight_layout()
    out_path = project_dir / f"gen_{gen_idx:03d}_visual.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def eval_mutated(rules_txt: str, x: np.ndarray) -> int:
    """
    A minimal evaluator that checks mutated rule text like:
    'if x0 <= 0.3 and x1 > -0.5: return 1 else: return 0'
    """
    # This is just a placeholder — not full Python parsing.
    if "x0" not in rules_txt:
        return 0
    # Use simple heuristics: compare first variable threshold
    try:
        import re
        match = re.search(r"x0\s*(<=|>)\s*([-\d\.]+)", rules_txt)
        if match:
            op, val = match.groups()
            val = float(val)
            return int(x[0] <= val if op == "<=" else x[0] > val)
    except Exception:
        pass
    return 0
