"""
coevolve_base/main.py
Step 1 – Synthetic data generator for the NN–Rule coevolution project.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# --- Config -----------------------------------------------------------------

cfg_path = Path(__file__).parent / "config.yaml"
if cfg_path.exists():
    cfg = yaml.safe_load(open(cfg_path))
else:
    cfg = {"shape": "circle", "radius": 0.6, "points": 1000, "seed": 42}

np.random.seed(cfg.get("seed", 42))

# --- Synthetic world --------------------------------------------------------

import numpy as np

def generate_points(n_points=1000, seed=42):
    """Generate 2D points with 3 nonlinear classes."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_points, 2))
    x, y = X[:, 0], X[:, 1]

    # --- Class 0: circle ---
    c0 = (x**2 + y**2 < 0.4**2)

    # --- Class 1: diagonal band ---
    c1 = (~c0) & (np.abs(x - y) < 0.2)

    # --- Class 2: sinusoidal lower curve ---
    c2 = (~c0) & (~c1) & (y < 0.3 * np.sin(3 * x) + 0.1)

    labels = np.zeros(n_points, dtype=int)
    labels[c1] = 1
    labels[c2] = 2

    return X, labels

def label_points(points, shape="circle", radius=0.6):
    """Assign class 1 if inside the target shape, else 0."""
    x, y = points[:, 0], points[:, 1]
    if shape == "circle":
        inside = (x**2 + y**2) < radius**2
    elif shape == "ellipse":
        inside = (x**2 / radius**2 + y**2 / (radius * 1.5) ** 2) < 1
    elif shape == "square":
        inside = (abs(x) < radius) & (abs(y) < radius)
    else:
        inside = np.zeros_like(x, dtype=bool)
    return inside.astype(int)

# --- Run demo ---------------------------------------------------------------

if __name__ == "__main__":
    pts, _ = generate_points(cfg["points"])
    labels = label_points(pts, cfg["shape"], cfg["radius"])

    # Visualization
    plt.figure(figsize=(4, 4))
    plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="coolwarm", s=8)
    plt.title(f"Shape World – {cfg['shape']}")
    plt.axis("equal")
    plt.tight_layout()

    out_path = Path(__file__).parent / "world_preview.png"
    plt.savefig(out_path)
    print(f"Preview saved to {out_path}")
