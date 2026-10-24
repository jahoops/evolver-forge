"""Robustness analytics helpers for coevolution runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List
import math


def _to_records(data: Any) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore

        if isinstance(data, pd.DataFrame):
            return data.to_dict("records")
    except ImportError:  # pragma: no cover - optional dependency
        pass

    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        return [dict(data)]
    raise TypeError("Unsupported data format for robustness computation")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((val - mean_val) ** 2 for val in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_robustness(data: Any, *, noise_window: int = 5) -> Dict[str, float]:
    """Return aggregate robustness indicators from metrics-style records."""

    records = _to_records(data)
    if not records:
        return {
            "acc_std_under_noise": 0.0,
            "fid_std_under_noise": 0.0,
            "drop_at_noise_5pct": 0.0,
            "delta_with_no_rule": 0.0,
            "delta_with_no_nn": 0.0,
            "delta_with_no_explore": 0.0,
            "stability_score": 1.0,
        }

    def _is_probe(row: Dict[str, Any]) -> bool:
        probe_val = _safe_float(row.get("is_probe"))
        return bool(probe_val and probe_val > 0)

    signal_rows = [row for row in records if not _is_probe(row)]

    noise_rows = [
        row
        for row in signal_rows
        if any((_safe_float(row.get(col)) or 0.0) > 0.0 for col in ("label_noise", "feature_noise", "target_noise"))
    ]
    noise_tail = noise_rows[-noise_window:] if noise_window > 0 else noise_rows

    acc_std = _std([
        _safe_float(row.get("nn_accuracy")) or 0.0
        for row in noise_tail
    ]) if noise_tail else 0.0
    fid_std = _std([
        _safe_float(row.get("rule_fidelity")) or 0.0
        for row in noise_tail
    ]) if noise_tail else 0.0

    noisy_rows = [
        row for row in signal_rows if abs((_safe_float(row.get("label_noise")) or 0.0) - 0.05) <= 0.01
    ]
    clean_rows = [
        row for row in signal_rows if (_safe_float(row.get("label_noise")) or 0.0) == 0.0
    ]
    drop_at_noise = (
        _mean([_safe_float(row.get("nn_accuracy")) or 0.0 for row in noisy_rows])
        - _mean([_safe_float(row.get("nn_accuracy")) or 0.0 for row in clean_rows])
    ) if noisy_rows and clean_rows else 0.0

    def _last_row_with_flag(flag: str, value: float) -> Dict[str, Any] | None:
        for row in reversed(signal_rows):
            flag_val = _safe_float(row.get(flag))
            if flag_val is not None and abs(flag_val - value) < 1e-6:
                return row
        return None

    baseline_row = _last_row_with_flag("abl_no_rule", 0.0)
    if baseline_row is not None:
        # Ensure baseline does not include other ablations
        if any((_safe_float(baseline_row.get(flag)) or 0.0) > 0.0 for flag in ("abl_no_rule", "abl_no_nn", "abl_no_explore", "abl_no_consistency")):
            baseline_row = None
    if baseline_row is None:
        for row in reversed(signal_rows):
            if not any(((_safe_float(row.get(flag)) or 0.0) > 0.0) for flag in ("abl_no_rule", "abl_no_nn", "abl_no_explore", "abl_no_consistency")):
                baseline_row = row
                break

    baseline_acc = _safe_float(baseline_row.get("nn_accuracy")) if baseline_row else None

    def _delta(flag: str) -> float:
        row = _last_row_with_flag(flag, 1.0)
        if row is None or baseline_acc is None:
            return 0.0
        ablation_acc = _safe_float(row.get("nn_accuracy"))
        if ablation_acc is None:
            return 0.0
        return ablation_acc - baseline_acc

    delta_no_rule = _delta("abl_no_rule")
    delta_no_nn = _delta("abl_no_nn")
    delta_no_explore = _delta("abl_no_explore")

    stability_score = 1.0 / (1.0 + acc_std + fid_std)

    return {
        "acc_std_under_noise": acc_std,
        "fid_std_under_noise": fid_std,
        "drop_at_noise_5pct": drop_at_noise,
        "delta_with_no_rule": delta_no_rule,
        "delta_with_no_nn": delta_no_nn,
        "delta_with_no_explore": delta_no_explore,
        "stability_score": stability_score,
    }
