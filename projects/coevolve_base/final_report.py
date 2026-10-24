"""Generate a consolidated report for Forge coevolution runs."""

from __future__ import annotations

from pathlib import Path
import base64
import datetime as _dt
import json
from typing import Any, Dict, List

try:  # Optional dependency
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - fallback without pandas
    pd = None  # type: ignore

import csv
import shutil
from html import escape


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<title>Forge Final Report</title>
<style>
body {
    font-family: Segoe UI, Roboto, sans-serif;
    margin: 2rem;
    background: #f8f9fa;
    color: #222;
}
h1,h2 { color: #004080; }
summary { cursor: pointer; font-weight: bold; margin-top: 1em; }
details { background:#fff; border:1px solid #ccc; border-radius:8px; padding:0.5em 1em; margin-bottom:1em; }
img { max-width:100%; border:1px solid #ddd; border-radius:6px; }
code, pre { background:#f0f0f0; border-radius:4px; padding:0.3em 0.6em; }
pre { white-space: pre-wrap; word-break: break-word; overflow-x: auto; }
code { white-space: pre-wrap; word-break: break-word; }
.toggle { float:right; font-size:0.9em; cursor:pointer; color:#555; }
.dark body { background:#111; color:#ddd; }
.dark h1,h2 { color:#8cf; }
.dark details { background:#222; border-color:#444; }
nav a { margin-right: 0.5rem; text-decoration: none; color: #004080; }
.dark nav a { color: #8cf; }
hr { margin: 1.5rem 0; }
</style>
<script>
function toggleDark() {
    document.body.classList.toggle('dark');
    localStorage.setItem('forge_theme', document.body.classList.contains('dark') ? 'dark' : 'light');
}
window.onload = () => {
    if (localStorage.getItem('forge_theme') === 'dark') {
        document.body.classList.add('dark');
    }
};
</script>
</head>
<body>
<button class=\"toggle\" onclick=\"toggleDark()\">🌗 Toggle Dark Mode</button>
CONTENT_PLACEHOLDER
</body>
</html>"""


def section_html(title: str, inner: str, *, section_id: str | None = None) -> str:
        anchor = f" id=\"{section_id}\"" if section_id else ""
        return f"<details open{anchor}><summary>{title}</summary>\n{inner}\n</details>"


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


def encode_png(path: Path) -> str:
    try:
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"![{path.name}](data:image/png;base64,{encoded})"
    except Exception:
        return f"![missing]({path.name})"


def encode_png_html(path: Path) -> str:
    try:
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"<img src=\"data:image/png;base64,{encoded}\" alt=\"{escape(path.name, quote=True)}\"/>"
    except Exception:
        return f"<div>missing image: {escape(path.name, quote=True)}</div>"


def _coerce_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2)
    except Exception:
        return str(value)


def _load_meta_eval(meta_eval_path: Path) -> Dict[str, Any] | None:
    try:
        lines = [line for line in meta_eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    for payload in reversed(lines):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None


def _metrics_tail(metrics_path: Path) -> Dict[str, Any] | None:
    if pd is not None:
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            return _metrics_tail_csv(metrics_path)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()
    return _metrics_tail_csv(metrics_path)


def _metrics_tail_csv(metrics_path: Path) -> Dict[str, Any] | None:
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            reader = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    if not reader:
        return None
    return reader[-1]


def build_final_report(project_root: Path, out_name: str = "Final_Report.md") -> Path:
    """Merge metrics, visuals, and textual analyses into one report."""

    project_root = Path(project_root)
    out_path = project_root / out_name
    text: List[str] = [f"# Forge Final Report\n\nDate: {_dt.datetime.now():%Y-%m-%d %H:%M}"]
    html_sections: List[str] = []
    nav_links: List[str] = []

    metrics_path = project_root / "metrics.csv"
    chart_path = project_root / "co_evolve_metrics_split.png"
    if metrics_path.exists():
        tail = _metrics_tail(metrics_path)
        if tail:
            text.append(_section("Metrics Summary", _coerce_json(tail)))
            chart_html = encode_png_html(chart_path) if chart_path.exists() else ""
            metrics_html = section_html(
                "Metrics Summary",
                f"<pre>{escape(_coerce_json(tail))}</pre>" + chart_html,
                section_id="metrics",
            )
            html_sections.append(metrics_html)
            nav_links.append('<a href="#metrics">Metrics</a>')
        if chart_path.exists():
            text.append(encode_png(chart_path))

    robustness_path = project_root / "robustness_report.json"
    if robustness_path.exists():
        try:
            robustness_payload = json.loads(robustness_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            robustness_payload = robustness_path.read_text(encoding="utf-8")
            text.append(_section("Robustness Report", _coerce_json(robustness_payload)))
            html_sections.append(
                section_html(
                    "Robustness Report",
                    f"<pre>{escape(_coerce_json(robustness_payload))}</pre>",
                    section_id="robustness",
                )
            )
            nav_links.append('<a href="#robustness">Robustness</a>')

    causal_path = project_root / "causal_report.json"
    if causal_path.exists():
        try:
            causal_payload = json.loads(causal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            causal_payload = causal_path.read_text(encoding="utf-8")
        text.append(_section("Causal Attribution", _coerce_json(causal_payload)))
        graph_markdown = ""
        graph_html = ""
        causal_graph_path = project_root / "causal_graph.png"
        if causal_graph_path.exists():
            graph_markdown = encode_png(causal_graph_path)
            graph_html = encode_png_html(causal_graph_path)
            text.append(graph_markdown)
        html_sections.append(
            section_html(
                "Causal Attribution",
                f"<pre>{escape(_coerce_json(causal_payload))}</pre>" + graph_html,
                section_id="causal",
            )
        )
        nav_links.append('<a href="#causal">Causal</a>')

    meta_eval_path = project_root / "meta_eval.jsonl"
    if meta_eval_path.exists():
        latest_meta = _load_meta_eval(meta_eval_path)
        if latest_meta:
            text.append(_section("Meta-Evaluation", _coerce_json(latest_meta)))
            html_sections.append(
                section_html(
                    "Meta-Evaluation",
                    f"<pre>{escape(_coerce_json(latest_meta))}</pre>",
                    section_id="meta",
                )
            )
            nav_links.append('<a href="#meta">Meta-Eval</a>')

    canon_path = project_root / "canon.json"
    if canon_path.exists():
        try:
            canon_payload = json.loads(canon_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            canon_payload = {"rules": []}
        rules_section = "\n".join(
            f"- {entry.get('text', '')} (w={float(entry.get('weight', 0.0)):.2f})"
            for entry in canon_payload.get("rules", [])
            if entry.get("text")
        ) or "(no canon rules recorded)"
        text.append(_section("Canon Rules", rules_section))
        html_sections.append(
            section_html(
                "Canon Rules",
                f"<div>{escape(rules_section)}</div>",
                section_id="canon",
            )
        )
        nav_links.append('<a href="#canon">Canon</a>')

    log_path = project_root / "log.md"
    if log_path.exists():
        try:
            log_tail = log_path.read_text(encoding="utf-8")[-15000:]
        except OSError:
            log_tail = log_path.read_text(errors="ignore")[-15000:]
        text.append(_section("Recent Log", f"```\n{log_tail}\n```"))
        html_sections.append(
            section_html(
                "Recent Log",
                f"<pre>{escape(log_tail)}</pre>",
                section_id="log",
            )
        )
        nav_links.append('<a href="#log">Log</a>')

    out_path.write_text("\n".join(text), encoding="utf-8")
    print(f"[Forge] Final report written → {out_path}")

    html_path = out_path.with_suffix(".html")
    try:
        nav_html = " | ".join(nav_links)
        content_html = (
            f"<h1>Forge Final Report</h1><p>Date: {_dt.datetime.now():%Y-%m-%d %H:%M}</p>"
            + (f"<nav>{nav_html}</nav><hr/>" if nav_links else "")
            + "\n".join(html_sections)
        )
        html_payload = HTML_TEMPLATE.replace("CONTENT_PLACEHOLDER", content_html)
        html_path.write_text(html_payload, encoding="utf-8")
        print(f"[Forge] HTML version → {html_path}")
    except Exception as exc:
        print(f"[Forge] HTML generation skipped: {exc}")

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a consolidated Forge final report")
    parser.add_argument(
        "--project",
        default="projects/coevolve_base",
        help="Path to a Forge project directory",
    )
    parser.add_argument(
        "--output",
        default="Final_Report.md",
        help="Output filename for the Markdown report",
    )
    args = parser.parse_args()
    build_final_report(Path(args.project), out_name=args.output)
