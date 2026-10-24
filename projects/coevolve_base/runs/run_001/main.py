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

def generate_points(n=1000):
    """Return random (x, y) pairs uniformly sampled in [-1, 1]²."""
    pts = np.random.uniform(-1, 1, size=(n, 2))
    return pts

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
    pts = generate_points(cfg["points"])
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
