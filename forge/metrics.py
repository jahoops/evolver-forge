import csv
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from .ai import summarize_text
from .constants import PROJECTS_DIR
#import datetime
from datetime import datetime

console = Console()

import matplotlib.pyplot as plt

def live_plot_update(df, proj_path, title="Forge Live Metrics"):
    """Update a live plot of metrics after each auto-run."""
    plt.ion()
    plt.clf()

    # --- detect correct run column ---
    run_col = "run"
    if "run" not in df.columns and "run_id" in df.columns:
        run_col = "run_id"
    elif not any(c.startswith("run") for c in df.columns):
        # fallback if weird schema
        run_col = df.columns[0]

    runs = df[run_col].astype(str).tolist()
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    param_cols = [c for c in df.columns if c.startswith("param_")]

    if param_cols:
        ax1 = plt.gca()
        for col in param_cols:
            ax1.plot(runs, df[col], "o--", label=f"param: {col[6:]}")
        ax1.set_xlabel("Run")
        ax1.set_ylabel("Parameters")

    if metric_cols:
        ax2 = ax1.twinx() if param_cols else plt.gca()
        for col in metric_cols:
            ax2.plot(runs, df[col], "o-", label=f"metric: {col[7:]}")
        ax2.set_ylabel("Metrics")

    plt.title(title)
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.pause(0.1)  # brief pause to refresh UI

    live_path = proj_path / "metrics_live.png"
    plt.savefig(live_path)
    return live_path


def extract_metrics(output_text: str) -> dict:
    """Parse numeric metrics from experiment output lines."""
    metrics = {}
    for line in output_text.splitlines():
        if "score=" in line:
            try:
                value = float(line.split("score=")[-1])
                metrics["score"] = value
            except ValueError:
                pass
    return metrics

def plot_metrics(proj_name: str, proj_path: Path):
    csv_path = proj_path / "results.csv"
    if not csv_path.exists():
        console.print(f"[red]No results.csv found for {proj_name}.[/red]")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if len(reader) < 2:
        console.print("[yellow]Not enough runs to plot trends.[/yellow]")
        return

    runs = [r["run_id"] for r in reader]
    columns = [c for c in reader[0].keys() if c != "run_id"]

    # Separate parameters (left Y) and metrics (right Y)
    param_cols = [c for c in columns if c.startswith("param_")]
    metric_cols = [c for c in columns if c.startswith("metric_")]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx() if param_cols and metric_cols else None

    legend_labels = []

    # --- Plot parameters on left axis ---
    for col in param_cols:
        try:
            values = [float(r[col]) for r in reader if r.get(col) not in (None, "", "None")]
            if len(values) == 0:
                continue
            ax1.plot(runs[:len(values)], values, marker="o", linestyle="--", label=col.replace("param_", "param: "))
            legend_labels.append(col.replace("param_", "param: "))
        except Exception as e:
            console.print(f"[yellow]Skipping param column {col}: {e}[/yellow]")

    # --- Plot metrics on right axis ---
    if ax2:
        for col in metric_cols:
            try:
                values = [float(r[col]) for r in reader if r.get(col) not in (None, "", "None")]
                if len(values) == 0:
                    continue
                ax2.plot(runs[:len(values)], values, marker="o", label=col.replace("metric_", "metric: "))
                legend_labels.append(col.replace("metric_", "metric: "))
            except Exception as e:
                console.print(f"[yellow]Skipping metric column {col}: {e}[/yellow]")
    else:
        # If no dual axis needed
        for col in metric_cols:
            try:
                values = [float(r[col]) for r in reader if r.get(col) not in (None, "", "None")]
                if len(values) == 0:
                    continue
                ax1.plot(runs[:len(values)], values, marker="o", label=col.replace("metric_", "metric: "))
                legend_labels.append(col.replace("metric_", "metric: "))
            except Exception as e:
                console.print(f"[yellow]Skipping metric column {col}: {e}[/yellow]")

    ax1.set_title(f"Forge Metrics — {proj_name}")
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Parameters" if param_cols else "Values")
    if ax2:
        ax2.set_ylabel("Metrics")

    if legend_labels:
        ax1.legend(legend_labels, loc="best")
    else:
        console.print("[yellow]No labeled metrics found for legend.[/yellow]")

    plt.grid(True)
    plt.tight_layout()

# --- AI Annotation of Metric Trends ---
    try:
        if metric_cols:
            console.print("\n[cyan]Requesting AI insight annotations for key metric points...[/cyan]")

            for col in metric_cols:
                values = [float(r[col]) for r in reader if r.get(col) not in (None, "", "None")]
                if len(values) < 3:
                    continue

                deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
                trend_summary = f"{col.replace('metric_', '')}: " + ", ".join(f"{d:+.3f}" for d in deltas)

                prompt = (
                    f"You are an analytical assistant summarizing experiment trends for metric '{col}'.\n"
                    "For each change between runs, provide one short descriptive phrase (5–10 words max), "
                    "reflecting the shape of the curve — for example, 'sharp drop then recovery', 'steady improvement', or 'plateau near optimum'.\n"
                    "Be concise, factual, and use plain English.\n\n"
                    f"{trend_summary}"
                )

                # Ask the model for per-segment labels
                ai_text = summarize_text(prompt)
                console.print(f"\n[magenta]{col} insight:[/magenta] {ai_text}")

                # Split into segment-level lines or phrases
                phrases = [p.strip(" .") for p in ai_text.replace("\n", ",").split(",") if p.strip()]
                while len(phrases) < len(values):
                    phrases.append("")  # pad if needed

                # Add annotation text for each value
                for i, phrase in enumerate(phrases[:len(values)]):
                    if phrase:
                        ax2.text(
                            i,
                            values[i],
                            phrase,
                            fontsize=8,
                            color="darkmagenta",
                            rotation=15,
                            ha="left",
                            va="bottom",
                            alpha=0.8,
                        )

    except Exception as e:
        console.print(f"[yellow]AI annotation skipped: {e}[/yellow]")

    plot_path = proj_path / "metrics_plot.png"
    plt.savefig(plot_path)

    # --- Markdown report generation ---
    try:
        report_path = proj_path / "metrics_report.md"
        console.print(f"\n[cyan]Saving Markdown summary to {report_path}...[/cyan]")

        # Build Markdown content
        md = []
        md.append(f"# Forge Metrics Report — {proj_path.name}\n")
        md.append(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Numeric summary table
        if metric_cols:
            md.append("## Numeric Summary\n")
            md.append("| Metric | Last | Previous | Δ | Average |")
            md.append("|---------|-------|-----------|-----------|----------|")
            for col in metric_cols:
                values = [float(r[col]) for r in reader if r.get(col) not in (None, "", "None")]
                if len(values) >= 2:
                    diff = values[-1] - values[-2]
                    avg = sum(values) / len(values)
                    md.append(
                        f"| {col.replace('metric_', '')} | {values[-1]:.4f} | {values[-2]:.4f} | {diff:+.4f} | {avg:.4f} |"
                    )

        # AI insights
        md.append("\n## AI Insights\n")
        if "ai_text" in locals():
            md.append(ai_text)
        else:
            md.append("_No AI commentary available._")

        # Embed image reference
        md.append("\n## Plot\n")
        md.append(f"![Metrics Plot](./{plot_path.name})\n")

        # Write the file
        report_path.write_text("\n".join(md), encoding="utf-8")
        console.print(f"[green]Markdown report saved: {report_path}[/green]")

    except Exception as e:
        console.print(f"[yellow]Markdown report generation failed: {e}[/yellow]")

    plt.close()
    console.print(f"[green]Metrics plot saved as {plot_path}[/green]")

    # --- Numeric summary ---
    summary_lines = []
    for col in metric_cols:
        values = []
        for r in reader:
            val = r.get(col)
            if val in (None, "", "None"):
                continue
            try:
                values.append(float(val))
            except ValueError:
                continue
        if len(values) >= 2:
            diff = values[-1] - values[-2]
            avg = sum(values) / len(values)
            summary_lines.append(f"{col.replace('metric_', '')}: last={values[-1]:.4f}, prev={values[-2]:.4f}, Δ={diff:+.4f}, avg={avg:.4f}")

    if summary_lines:
        console.print("\n[bold cyan]Numeric Trend Summary:[/bold cyan]")
        console.print("\n".join(summary_lines))
    else:
        console.print("[yellow]No numeric data available for summary.[/yellow]")
