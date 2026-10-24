"""Visual health dashboard for Forge coevolution runs.

Usage
-----
poetry run python -m forge.analysis.health_dashboard projects/coevolve_base/coevolve_metrics.csv [--radar] [--json PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from .health_report import (
    load_metrics,
    evaluate_stats,
    assess_health,
    summarize_categories,
)

COLOR_MAP = {
    "🟢": "#2ca02c",
    "🟡": "#ffbf00",
    "🔴": "#d62728",
}

BADGE_MARK = {
    "🟢": "●",
    "🟡": "●",
    "🔴": "●",
}

BADGE_NAMES = {
    "🟢": "Green",
    "🟡": "Yellow",
    "🔴": "Red",
}


def _maybe_trim(metrics: Dict[str, np.ndarray], limit: int = 20, threshold: int = 100) -> Dict[str, np.ndarray]:
    generations = metrics.get("generation")
    if generations is None:
        return metrics
    if limit <= 0:
        return metrics
    if generations.size <= threshold:
        return metrics
    start = max(generations.size - limit, 0)
    return {key: value[start:] for key, value in metrics.items()}

console = Console()


AXES_INFO: List[Tuple[str, str]] = [
    ("Structural Stability", "stability"),
    ("Bidirectional Causality", "causality"),
    ("Energy Use", "energy"),
    ("Convergence", "convergence"),
    ("Generalization Proxy", "generalization"),
    ("Interpretability Proxy", "interpretability"),
]

def _trend_arrow(series: np.ndarray, invert: bool = False) -> str:
    if series.size < 2 or np.all(np.isnan(series)):
        return "→"
    delta = series[-1] - series[0]
    if invert:
        delta *= -1
    if delta > 0.01:
        return "↑"
    if delta < -0.01:
        return "↓"
    return "→"


STATUS_ARROW = {
    "🟢": "↑",
    "🟡": "↓",
    "🔴": "↑",
}


def _status_line(results: Dict[str, Dict[str, object]]) -> str:
    segments = []
    for label, key in [("Stability", "stability"), ("Energy", "energy"), ("Convergence", "convergence")]:
        entry = results.get(key)
        if not entry:
            continue
        badge = entry["badge"]
        arrow = STATUS_ARROW.get(badge, "→")
        segments.append(f"{badge} {label} {arrow}")
    return " | ".join(segments)


def _bar_dashboard(metrics: Dict[str, np.ndarray], results: Dict[str, Dict[str, object]], output_path: Path, footer: str, trends_axes: Dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    y_positions = np.arange(len(AXES_INFO))

    widths = [float(results[key]["score"]) for _, key in AXES_INFO]
    colors = [COLOR_MAP.get(results[key]["badge"], "#999999") for _, key in AXES_INFO]

    ax.barh(y_positions, widths, color=colors, alpha=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in AXES_INFO], fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Health Score", fontsize=11)
    ax.set_title("Forge Health Overview", fontsize=14, fontweight="bold")

    for idx, width in enumerate(widths):
        badge = results[AXES_INFO[idx][1]]["badge"]
        marker = BADGE_MARK.get(badge, "●")
        ax.text(width + 0.02, idx, marker, color=COLOR_MAP.get(badge, "#444444"), va="center", fontsize=12)

    fig.text(0.02, 0.02, footer, fontsize=10, ha="left", va="bottom")

    if trends_axes:
        trend_ax = fig.add_axes([0.6, 0.1, 0.35, 0.18])
        trend_ax.set_title("Mini Trends", fontsize=9)
        if "nn_accuracy" in trends_axes:
            trend_ax.plot(trends_axes["generation"], trends_axes["nn_accuracy"], color="tab:blue", alpha=0.4, label="Accuracy")
        if "rule_fidelity" in trends_axes:
            trend_ax.plot(trends_axes["generation"], trends_axes["rule_fidelity"], color="tab:red", alpha=0.4, label="Fidelity")
        if "avg_disagreement" in trends_axes:
            trend_ax.plot(trends_axes["generation"], trends_axes["avg_disagreement"], color="tab:orange", alpha=0.4, label="Disagreement")
        trend_ax.set_xticks([])
        trend_ax.set_yticks([])
        trend_ax.legend(loc="upper right", fontsize=7)
        trend_ax.spines["top"].set_visible(False)
        trend_ax.spines["right"].set_visible(False)
        trend_ax.spines["bottom"].set_visible(False)
        trend_ax.spines["left"].set_visible(False)

    fig.subplots_adjust(bottom=0.18)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _radar_dashboard(metrics: Dict[str, np.ndarray], results: Dict[str, Dict[str, object]], output_path: Path, footer: str):
    labels = [label for label, _ in AXES_INFO]
    values = [float(results[key]["score"]) for _, key in AXES_INFO]
    colors = [COLOR_MAP.get(results[key]["badge"], "#999999") for _, key in AXES_INFO]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_loop = values + values[:1]
    angles_loop = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.set_title("Forge Health Overview", fontsize=14, fontweight="bold")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.plot(angles_loop, values_loop, color="#1f77b4", linewidth=2)
    ax.fill(angles_loop, values_loop, color="#1f77b4", alpha=0.25)

    for angle, value, color, badge in zip(angles, values, colors, [results[key]["badge"] for _, key in AXES_INFO]):
        marker = BADGE_MARK.get(badge, "●")
        ax.scatter([angle], [value], color=color, s=80)
        ax.text(angle, value + 0.05, marker, color=color, fontsize=12, ha="center")

    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_footer(total: int, greens: int, badge: str, verdict: str, metrics: Dict[str, np.ndarray]) -> str:
    badge_label = BADGE_NAMES.get(badge, badge)
    return (
        f"Forge Coevolution Health — last gen: {greens}/{total} {badge_label} ({verdict}) | "
        f"Accuracy trend: {_trend_arrow(metrics['nn_accuracy'])} | "
        f"Fidelity: {_trend_arrow(metrics['rule_fidelity'])} | "
        f"Disagreement: {_trend_arrow(metrics['avg_disagreement'], invert=True)}"
    )


def _collect_trend_data(metrics: Dict[str, np.ndarray], json_path: Path | None) -> Dict[str, np.ndarray]:
    if json_path and json_path.exists():
        return {
            "generation": metrics["generation"],
            "nn_accuracy": metrics["nn_accuracy"],
            "rule_fidelity": metrics["rule_fidelity"],
            "avg_disagreement": metrics["avg_disagreement"],
        }
    return {}


def update_live_health(
    csv_path: Path,
    out_path: Path,
    refresh_interval: float = 5.0,
    *,
    radar: bool = False,
    json_path: Path | None = None,
    trim_limit: int = 20,
    trim_threshold: int = 100,
) -> None:
    sentinel = csv_path.parent / "stop_health_watch.txt"
    console.print(f"🔄 Updating Forge Health every {refresh_interval}s — Ctrl-C to stop")

    last_status = ""
    try:
        while True:
            if sentinel.exists():
                break

            try:
                metrics = load_metrics(csv_path)
            except Exception as exc:
                console.print(f"[red]Failed to load metrics: {exc}[/red]")
                time.sleep(refresh_interval)
                continue

            metrics = _maybe_trim(metrics, limit=trim_limit, threshold=trim_threshold)
            try:
                stats = evaluate_stats(metrics)
                results = assess_health(stats)
            except Exception as exc:
                console.print(f"[red]Failed to evaluate health: {exc}[/red]")
                time.sleep(refresh_interval)
                continue
            total, greens, badge, verdict = summarize_categories(results)
            footer = _build_footer(total, greens, badge, verdict, metrics)
            trend_data = _collect_trend_data(metrics, json_path)

            if radar:
                _radar_dashboard(metrics, results, out_path, footer)
            else:
                _bar_dashboard(metrics, results, out_path, footer, trend_data)

            status_line = _status_line(results)
            console.print(status_line)
            last_status = status_line

            if sentinel.exists():
                break
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        console.print("[yellow]KeyboardInterrupt received. Exiting watch mode.[/yellow]")
    finally:
        if sentinel.exists():
            console.print("[bold yellow]Sentinel detected; stopping health monitor.[/bold yellow]")
        if last_status:
            console.print(f"Last status: {last_status}")
        console.print("✅ Health monitor stopped — last status saved as coevolve_health.png")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forge coevolution health dashboard")
    parser.add_argument("csv_path", type=Path, help="Path to coevolution metrics CSV")
    parser.add_argument("--json", type=Path, help="Optional health JSON produced by health_report.py")
    parser.add_argument("--radar", action="store_true", help="Render radar plot instead of bar chart")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh the dashboard")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds for watch mode")
    args = parser.parse_args(argv)

    output_path = args.csv_path.with_name("coevolve_health.png")
    json_path = args.json or args.csv_path.with_name("coevolve_health.json")

    if args.watch:
        update_live_health(
            args.csv_path,
            output_path,
            refresh_interval=args.interval,
            radar=args.radar,
            json_path=json_path,
        )
        return 0

    metrics = load_metrics(args.csv_path)
    metrics = _maybe_trim(metrics)
    stats = evaluate_stats(metrics)
    results = assess_health(stats)

    total, greens, badge, verdict = summarize_categories(results)
    summary_footer = _build_footer(total, greens, badge, verdict, metrics)
    trend_data = _collect_trend_data(metrics, json_path)

    if args.radar:
        _radar_dashboard(metrics, results, output_path, summary_footer)
    else:
        _bar_dashboard(metrics, results, output_path, summary_footer, trend_data)

    print(f"Saved health dashboard to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
