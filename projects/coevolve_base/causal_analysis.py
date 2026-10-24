"""Simple causal attribution utilities for coevolution runs."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Iterable, List

import math

import pandas as pd


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for candidate in paths:
        if candidate.exists():
            return candidate
    return None


def load_interventions(path: Path) -> List[Dict[str, Any]]:
    """Load intervention records from a JSONL log."""
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def diff_in_diff(df: pd.DataFrame, iv_row: Dict[str, Any], metric: str, window: int = 2) -> float:
    """Compute a crude difference-in-differences estimate around an intervention."""
    if metric not in df.columns:
        return 0.0
    g = int(iv_row.get("gen", 0))
    if df.empty:
        return 0.0
    gen_col = df["generation"]
    min_gen = int(gen_col.min())
    max_gen = int(gen_col.max())
    pre_gen = max(g - 1, min_gen)
    post_gen = min(g + 1, max_gen)
    pre = df.loc[gen_col == pre_gen, metric].mean()
    post = df.loc[gen_col == post_gen, metric].mean()
    mask_before = gen_col < g
    if mask_before.any():
        drift_series = df.loc[mask_before, metric].diff().dropna().tail(window)
        drift = drift_series.mean() if not drift_series.empty else 0.0
    else:
        drift = 0.0
    pre = 0.0 if math.isnan(pre) else pre
    post = 0.0 if math.isnan(post) else post
    drift = 0.0 if math.isnan(drift) else drift
    return float((post - pre) - drift)


def lag_regression(df: pd.DataFrame, iv_gens: List[int], metric: str, lags: int = 2) -> List[float]:
    """Regress the metric on lagged intervention indicators."""
    if metric not in df.columns or df.empty:
        return [0.0] * lags
    import numpy as np

    y = df[metric].fillna(0.0).to_numpy(dtype=float)
    gen_vals = df["generation"].astype(int).to_numpy()
    X_rows: List[List[float]] = []
    for current_gen in gen_vals:
        row: List[float] = []
        for k in range(1, lags + 1):
            row.append(1.0 if (current_gen - k) in iv_gens else 0.0)
        X_rows.append(row)
    X = np.asarray(X_rows, dtype=float)
    if X.size == 0:
        return [0.0] * lags
    y_vec = y.reshape(-1, 1)
    try:
        beta = np.linalg.pinv(X) @ y_vec
    except Exception:
        return [0.0] * lags
    return [float(val) for val in beta.ravel()]


def granger_like(df: pd.DataFrame, iv_gens: List[int], metric: str, lags: int = 2) -> tuple[float, List[float]]:
    """Return a simple Granger-style association score and coefficients."""
    betas = lag_regression(df, iv_gens, metric, lags)
    score = sum(abs(b) for b in betas)
    return float(score), betas


def attribute(project_path: Path, window: int = 2, lags: int = 2) -> List[Dict[str, Any]]:
    """Aggregate causal signals for interventions recorded in a project."""
    metrics_path = _first_existing(
        [
            project_path / "metrics.csv",
            project_path / "coevolve_metrics.csv",
        ]
    )
    if metrics_path is None:
        return []
    df = pd.read_csv(metrics_path)
    if "generation" not in df.columns:
        return []
    interventions = load_interventions(project_path / "causal_log.jsonl")
    if not interventions:
        return []

    available_metrics = [m for m in ("nn_accuracy", "rule_fidelity", "combined_fitness") if m in df.columns]
    if not available_metrics:
        return []

    results: List[Dict[str, Any]] = []
    kinds = sorted({iv.get("kind", "") for iv in interventions if isinstance(iv, dict)})
    for kind in kinds:
        subset = [iv for iv in interventions if iv.get("kind") == kind]
        if not subset:
            continue
        gens = [int(iv.get("gen", 0)) for iv in subset]
        for metric in available_metrics:
            did_estimates = [diff_in_diff(df, iv, metric, window) for iv in subset]
            if did_estimates:
                did_mean = float(pd.Series(did_estimates).mean(skipna=True))
            else:
                did_mean = 0.0
            granger_score, betas = granger_like(df, gens, metric, lags)
            results.append({
                "kind": kind,
                "metric": metric,
                "did_mean": float(did_mean),
                "granger_score": float(granger_score),
                "betas": [float(b) for b in betas],
                "count": len(subset),
            })
    return results
