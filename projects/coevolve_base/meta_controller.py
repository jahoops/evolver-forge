"""Meta-controller that adapts coevolution hyperparameters from recent metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

try:  # Optional dependency; controller still works without pandas
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

HAS_PANDAS = pd is not None


@dataclass
class MetaController:
    metrics_path: Path
    params: Dict[str, float] = field(default_factory=lambda: {
        "learning_rate": 0.01,
        "mutation_rate": 0.1,
        "lambda_consistency": 0.2,
    })
    last_reward: float = 0.0

    def _load_metrics(self):
        if pd is None:
            return None
        if not self.metrics_path.exists():
            return None
        try:
            return pd.read_csv(self.metrics_path)
        except Exception:
            return None

    def update(self) -> Dict[str, float]:
        history = self._load_metrics()
        if history is None or history.empty:
            self.last_reward = 0.0
            return self.params

        recent = history.tail(5)
        acc_series = recent.get("nn_accuracy")
        fid_series = recent.get("rule_fidelity")
        if acc_series is None or fid_series is None:
            self.last_reward = 0.0
            return self.params

        acc_grad = np.gradient(acc_series.to_numpy(dtype=float)) if len(acc_series) else np.array([0.0])
        fid_grad = np.gradient(fid_series.to_numpy(dtype=float)) if len(fid_series) else np.array([0.0])
        avg_gain = float(np.mean(acc_grad + fid_grad))

        if avg_gain > 0.01:
            self.params["learning_rate"] *= 1.05
            self.params["mutation_rate"] *= 0.95
        elif avg_gain < -0.01:
            self.params["learning_rate"] *= 0.9
            self.params["mutation_rate"] *= 1.1

        recent_consistency = recent.get("consistency_loss")
        consistency_term = float(np.mean(np.abs(recent_consistency.to_numpy(dtype=float)))) if recent_consistency is not None else 0.0
        self.last_reward = avg_gain - consistency_term

        if self.last_reward > 0.0:
            self.params["lambda_consistency"] *= 1.02
        else:
            self.params["lambda_consistency"] *= 0.98

        self.params["learning_rate"] = float(np.clip(self.params["learning_rate"], 1e-4, 0.1))
        self.params["mutation_rate"] = float(np.clip(self.params["mutation_rate"], 0.01, 0.5))
        self.params["lambda_consistency"] = float(np.clip(self.params["lambda_consistency"], 0.05, 0.5))

        return self.params

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learning_rate": self.params["learning_rate"],
            "mutation_rate": self.params["mutation_rate"],
            "lambda_consistency": self.params["lambda_consistency"],
            "last_reward": self.last_reward,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
