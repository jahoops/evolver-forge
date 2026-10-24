"""HTML health report generation for Forge projects."""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import List

import pandas as pd
from rich.console import Console

try:
    import markdown
except ImportError:  # pragma: no cover - dependency warning
    markdown = None

console = Console()


def _read_metrics_table(metrics_csv: Path) -> str:
    if not metrics_csv.exists():
        return "<p>No metrics CSV found.</p>"

    try:
        df = pd.read_csv(metrics_csv)
    except Exception as exc:
        return f"<p>Failed to read metrics CSV: {exc}</p>"

    if df.empty:
        return "<p>Metrics CSV is empty.</p>"

    last_rows = df.tail(10)
    return last_rows.to_html(index=False, float_format="%.4f", classes="metrics-table")


def _encode_dashboard_image(health_png: Path) -> str:
    if not health_png.exists():
        return "<p>No health dashboard image found.</p>"

    try:
        with health_png.open("rb") as fh:
            img_base64 = base64.b64encode(fh.read()).decode("ascii")
    except Exception as exc:
        return f"<p>Failed to load dashboard image: {exc}</p>"

    return (
        f'<img src="data:image/png;base64,{img_base64}" '
        f'style="width:100%;border-radius:12px;" alt="Health dashboard" />'
    )


def _extract_health_summaries(log_md: Path) -> str:
    if not log_md.exists():
        return "<p>No log file found.</p>"

    text = log_md.read_text(encoding="utf-8")
    pattern = r"(?ms)^## Health Summary — .*?(?=^## |^### |\Z)"
    sections: List[str] = [section.strip() for section in re.findall(pattern, text)]

    if not sections:
        return "<p>No health summaries recorded yet.</p>"

    combined = "\n\n".join(sections)
    try:
        if markdown is None:
            raise RuntimeError("markdown package not available")
        return markdown.markdown(combined, extensions=["tables"])
    except Exception as exc:
        console.print(f"[yellow]Failed to render markdown summaries: {exc}[/yellow]")
        escaped = combined.replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{escaped}</pre>"


def generate_health_report(project_path: Path):
    """
    Combines metrics, plots, and log summaries into a single HTML report.
    Saves to <project_path>/health_report.html
    """
    if not project_path.exists():
        raise FileNotFoundError(f"Project path not found: {project_path}")

    project_name = project_path.name
    metrics_csv = project_path / "coevolve_metrics.csv"
    health_png = project_path / "coevolve_health.png"
    log_md = project_path / "log.md"

    table_html = _read_metrics_table(metrics_csv)
    image_html = _encode_dashboard_image(health_png)
    summaries_html = _extract_health_summaries(log_md)

    full_html = f"""
<html>
  <head>
    <meta charset="utf-8">
    <title>Forge Health Report — {project_name}</title>
    <style>
      body {{ font-family: Segoe UI, sans-serif; margin: 30px; background: #fafafa; color: #222; }}
      h1, h2 {{ color: #333; }}
      .metrics-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
      .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; }}
      .summary {{ background: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 6px #ccc; }}
      img {{ border-radius: 12px; }}
    </style>
  </head>
  <body>
    <h1>Forge Health Report — {project_name}</h1>
    <h2>📊 Latest Metrics</h2>
    {table_html}
    <h2>🧭 Health Dashboard</h2>
    {image_html}
    <h2>🗒️ Summaries</h2>
    <div class="summary">{summaries_html}</div>
  </body>
</html>
""".strip()

    report_path = project_path / "health_report.html"
    report_path.write_text(full_html, encoding="utf-8")
    print(f"✅ Health report saved to {report_path}")
