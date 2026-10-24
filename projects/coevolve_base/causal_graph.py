"""Render a simple causal graph from attribution results."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable, List


def _flatten_results(payload: Any) -> List[dict[str, Any]]:
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "results" in payload[0]:
            rows: List[dict[str, Any]] = []
            for item in payload:
                rows.extend(item.get("results", []))
            return rows
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "runs" in payload and isinstance(payload["runs"], Iterable):
            rows: List[dict[str, Any]] = []
            for run in payload["runs"]:
                if isinstance(run, dict):
                    rows.extend(run.get("results", []))
            return rows
        if "results" in payload and isinstance(payload["results"], Iterable):
            return [item for item in payload["results"] if isinstance(item, dict)]
    return []


def build_graph(report_path: Path, out_path: Path, threshold: float = 0.02):
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return None

    if not report_path.exists():
        return None

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    rows = _flatten_results(payload)
    if not rows:
        return None

    graph = nx.DiGraph()
    for row in rows:
        kind = row.get("kind")
        metric = row.get("metric")
        weight = row.get("did_mean")
        if not isinstance(kind, str) or not isinstance(metric, str):
            continue
        if not isinstance(weight, (int, float)):
            continue
        weight_val = float(weight)
        if abs(weight_val) < threshold:
            continue
        graph.add_edge(kind, metric, weight=weight_val)

    if not graph.edges:
        return None

    pos = nx.spring_layout(graph, seed=42)
    widths = [max(1.0, 6.0 * abs(graph.edges[edge]["weight"])) for edge in graph.edges]
    colors = ["tab:green" if graph.edges[edge]["weight"] > 0 else "tab:red" for edge in graph.edges]

    plt.figure(figsize=(8, 5))
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="lightgray")
    nx.draw_networkx_labels(graph, pos, font_size=8)
    nx.draw_networkx_edges(graph, pos, width=widths, edge_color=colors, arrows=True, arrowsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
