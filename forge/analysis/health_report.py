"""Forge coevolution health analyzer.

Usage
-----
poetry run python -m forge.analysis.health_report projects/coevolve_base/coevolve_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


REQUIRED_COLUMNS = [
    "generation",
    "nn_accuracy",
    "rule_fidelity",
    "combined_fitness",
    "nn_learning_gain",
    "rule_adaptation_gain",
    "rule_mutations",
    "consistency_loss",
    "avg_disagreement",
    "n_new_samples",
    "rule_agreement_rate",
    "lambda_consistency",
]


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_metrics(csv_path: Path) -> Dict[str, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        rows: List[Dict[str, float]] = []
        for row in reader:
            rows.append({key: _parse_float(row.get(key, "")) for key in REQUIRED_COLUMNS})

    if not rows:
        raise ValueError("No metric rows found in CSV.")

    data = {key: np.array([row[key] for row in rows], dtype=float) for key in REQUIRED_COLUMNS}
    return data


def compute_slopes(series: np.ndarray) -> np.ndarray:
    if series.size < 2:
        return np.zeros_like(series)
    return np.gradient(series)


def evaluate_stats(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    acc = data["nn_accuracy"]
    fid = data["rule_fidelity"]
    dis = data["avg_disagreement"]
    mutations = data["rule_mutations"]
    nn_gain = data["nn_learning_gain"]

    acc_slope = compute_slopes(acc)
    fid_slope = compute_slopes(fid)
    dis_slope = compute_slopes(dis)

    def nanstd(values: np.ndarray) -> float:
        return float(np.nanstd(values))

    def nanmean(values: np.ndarray) -> float:
        return float(np.nanmean(values))

    def nanvar(values: np.ndarray) -> float:
        return float(np.nanvar(values))

    def nancorr(x: np.ndarray, y: np.ndarray) -> float:
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            return float("nan")
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    return {
        "acc_slope": acc_slope,
        "fid_slope": fid_slope,
        "dis_slope": dis_slope,
        "mean_dis": nanmean(dis),
        "stability_std": nanstd(acc_slope),
        "causality_corr": nancorr(acc, fid),
        "energy_mean": nanmean(mutations),
        "energy_std": nanstd(mutations),
        "dis_slope_mean": nanmean(dis_slope),
        "generalization": nanmean(nn_gain),
        "interpretability": nanvar(mutations),
    }


def _badge(condition: bool, warning: bool = False) -> str:
    if condition:
        return "🟢"
    if warning:
        return "🟡"
    return "🔴"


def assess_health(stats: Dict[str, float]) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    std = stats["stability_std"]
    stability_good = std < 0.03
    stability_warn = std < 0.05
    results["stability"] = {
        "badge": _badge(stability_good, stability_warn),
        "detail": f"σ={std:.3f}",
        "score": 1 if stability_good else 0.5 if stability_warn else 0,
    }

    corr = stats["causality_corr"]
    causality_good = corr > 0.4
    causality_warn = corr > 0.25
    results["causality"] = {
        "badge": _badge(causality_good, causality_warn),
        "detail": f"r={corr:.2f}",
        "score": 1 if causality_good else 0.5 if causality_warn else 0,
    }

    mut_mean = stats["energy_mean"]
    mut_std = stats["energy_std"]
    energy_good = mut_mean < 8 and mut_std < 2
    energy_warn = mut_mean < 10 and mut_std < 3
    results["energy"] = {
        "badge": _badge(energy_good, energy_warn),
        "detail": f"mut={mut_mean:.1f}±{mut_std:.1f}",
        "score": 1 if energy_good else 0.5 if energy_warn else 0,
    }

    dis_slope_mean = stats["dis_slope_mean"]
    convergence_good = dis_slope_mean < 0
    convergence_warn = dis_slope_mean < 0.01
    arrow = "↓" if dis_slope_mean < 0 else "→" if math.isclose(dis_slope_mean, 0.0, abs_tol=0.005) else "↑"
    results["convergence"] = {
        "badge": _badge(convergence_good, convergence_warn),
        "detail": arrow,
        "score": 1 if convergence_good else 0.5 if convergence_warn else 0,
    }

    generalization_mean = stats["generalization"]
    generalization_good = abs(generalization_mean) < 0.1
    generalization_warn = abs(generalization_mean) < 0.2
    results["generalization"] = {
        "badge": _badge(generalization_good, generalization_warn),
        "detail": f"Δ={generalization_mean:+.3f}",
        "score": 1 if generalization_good else 0.5 if generalization_warn else 0,
    }

    interpretability_var = stats["interpretability"]
    interpretability_good = interpretability_var < 3
    interpretability_warn = interpretability_var < 5
    results["interpretability"] = {
        "badge": _badge(interpretability_good, interpretability_warn),
        "detail": f"var={interpretability_var:.2f}",
        "score": 1 if interpretability_good else 0.5 if interpretability_warn else 0,
    }

    return results


def summarize_categories(results: Dict[str, Dict[str, object]]) -> tuple[int, int, str, str]:
    total = len(results)
    greens = sum(1 for entry in results.values() if entry["badge"] == "🟢")
    avg_score = sum(float(entry["score"]) for entry in results.values()) / total
    if avg_score >= 0.85:
        verdict = "Stable learning"
        overall_badge = "🟢"
    elif avg_score >= 0.6:
        verdict = "Needs tuning"
        overall_badge = "🟡"
    else:
        verdict = "Drifting"
        overall_badge = "🔴"
    return total, greens, overall_badge, verdict


def format_summary(results: Dict[str, Dict[str, object]]) -> str:
    order = ["stability", "causality", "energy", "convergence", "generalization", "interpretability"]
    lines = [
        "┌────────────────────────────┐",
        "│ Forge Coevolution Health   │",
        "├────────────────────────────┤",
    ]

    for key in order:
        entry = results[key]
        label = key.capitalize()
        lines.append(f"│ {label:13s}: {entry['badge']} ({entry['detail']}) │")

    lines.append("├────────────────────────────┤")

    total, greens, overall_badge, verdict = summarize_categories({key: results[key] for key in order})

    lines.append(f"│ Overall: {greens}/{total} {overall_badge} {verdict:15s} │")
    lines.append("└────────────────────────────┘")
    return "\n".join(lines)


def _arrow_for_delta(delta: float, *, invert: bool = False, threshold: float = 0.005) -> tuple[str, float]:
    adj = -delta if invert else delta
    if adj > threshold:
        return "↑", delta
    if adj < -threshold:
        return "↓", delta
    return "→", delta


def _slice_recent(data: Dict[str, np.ndarray], window: int | None) -> Dict[str, np.ndarray]:
    if window is None:
        return data
    series = data.get("generation")
    if series is None or series.size <= window:
        return data
    start = max(series.size - window, 0)
    return {key: values[start:] for key, values in data.items()}


def evaluate_health_metrics(csv_path: Path, window: int | None = None) -> Dict[str, object]:
    metrics = load_metrics(csv_path)
    metrics = _slice_recent(metrics, window)

    stats = evaluate_stats(metrics)
    results = assess_health(stats)
    total, greens, badge, verdict = summarize_categories(results)

    acc = metrics["nn_accuracy"]
    fid = metrics["rule_fidelity"]
    dis = metrics["avg_disagreement"]
    combined = metrics["combined_fitness"]

    acc_arrow, acc_delta = _arrow_for_delta(float(acc[-1] - acc[0]))
    fid_arrow, fid_delta = _arrow_for_delta(float(fid[-1] - fid[0]))
    dis_arrow, dis_delta = _arrow_for_delta(float(dis[-1] - dis[0]), invert=True)
    comb_arrow, comb_delta = _arrow_for_delta(float(combined[-1] - combined[0]))

    summary_lines = [
        f"{badge} {verdict}",
        f"NN accuracy: {acc[0]:.3f} → {acc[-1]:.3f} ({acc_arrow})",
        f"Rule fidelity: {fid[0]:.3f} → {fid[-1]:.3f} ({fid_arrow})",
        f"Disagreement: {dis[0]:.3f} → {dis[-1]:.3f} ({dis_arrow})",
        f"Combined fitness: {combined[0]:+.3f} → {combined[-1]:+.3f} ({comb_arrow})",
    ]

    console_line = (
        f"🧭 Health: {badge} {verdict} | "
        f"Acc {acc_arrow}{acc_delta:+.3f} | Fid {fid_arrow}{fid_delta:+.3f} | Dis {dis_arrow}{dis_delta:+.3f}"
    )

    return {
        "summary": "\n".join(summary_lines),
        "console_line": console_line,
        "badge": badge,
        "verdict": verdict,
        "accuracy_delta": float(acc_delta),
        "fidelity_delta": float(fid_delta),
        "disagreement_delta": float(dis_delta),
        "combined_delta": float(comb_delta),
        "results": results,
        "stats": stats,
        "total": total,
        "greens": greens,
    }


def save_json(csv_path: Path, stats: Dict[str, float], results: Dict[str, Dict[str, object]]) -> Path:
    def _to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, (int, float, str)) or value is None:
            return value
        return value

    serializable_stats = {key: _to_serializable(value) for key, value in stats.items()}
    payload = {
        "stats": serializable_stats,
        "categories": results,
    }
    output_path = csv_path.with_name("coevolve_health.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forge coevolution health analyzer")
    parser.add_argument("csv_path", type=Path, help="Path to coevolution metrics CSV")
    parser.add_argument("--no-json", action="store_true", help="Skip writing JSON summary")
    args = parser.parse_args(argv)

    data = load_metrics(args.csv_path)
    stats = evaluate_stats(data)
    results = assess_health(stats)

    print(format_summary(results))

    if not args.no_json:
        output_path = save_json(args.csv_path, stats, results)
        print(f"Saved details to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
