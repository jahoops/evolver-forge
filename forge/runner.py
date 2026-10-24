import sys
import csv
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import yaml
from rich.console import Console

from .ai import summarize_text
from .metrics import extract_metrics
from .constants import PROJECTS_DIR

import pandas as pd
from forge.metrics import live_plot_update

console = Console()

# -----------------------------------------------------
# Utility: run folder creation
# -----------------------------------------------------
def create_run_folder(proj_path: Path) -> Path:
    """Create a sequentially numbered run folder."""
    runs_dir = proj_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    run_id = f"run_{len(list(runs_dir.iterdir()))+1:03d}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(exist_ok=True)
    return run_dir


# -----------------------------------------------------
# Core: run a project's main.py and log results
# -----------------------------------------------------
def run_project(proj_name: str):
    """Run main.py for a project, log output, extract metrics, summarize."""
    proj_path = PROJECTS_DIR / proj_name
    main_path = proj_path / "main.py"
    log_path = proj_path / "log.md"

    if not main_path.exists():
        console.print(f"[red]{main_path} not found![/red]")
        return

    # --- Execute main.py ---
    console.print(f"[cyan]Running {proj_name}...[/cyan]")

    process = subprocess.Popen(
        [sys.executable, str(main_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    for line in process.stdout:
        console.print(line.rstrip())
        output_lines.append(line)
    process.wait()

    result_stdout = "".join(output_lines)

    # --- Archive run ---
    run_dir = create_run_folder(proj_path)
    (run_dir / "main.py").write_text(main_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "config.yaml").write_text((proj_path / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    run_log_path = run_dir / "log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_log_path.write_text(
        f"### Run at {timestamp}\n#### Output\n```\n{result_stdout}\n```\n#### End of Output\n",
        encoding="utf-8"
    )

    console.print(f"[green]Run output saved to log.md and archived in {run_dir.name}[/green]")

    # --- Extract metrics ---
    metrics = extract_metrics(result_stdout)
    if metrics:
        results_csv = proj_path / "results.csv"
        write_header = not results_csv.exists()

        # ✅ Always write results first
        with results_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", *metrics.keys()])
            if write_header:
                writer.writeheader()
            row = {"run_id": run_dir.name}
            row.update(metrics)
            writer.writerow(row)

        console.print(f"[green]Metrics extracted: {metrics}[/green]")

        # --- Live plotting update ---
        try:
            df = pd.read_csv(proj_path / "results.csv")
            live_path = live_plot_update(df, proj_path, f"Forge Live — {proj_name}")
            console.print(f"[gray]Live plot updated: {live_path.name}[/gray]")
        except Exception as e:
            import traceback
            console.print(f"[red]Live plot update failed:[/red] {e}")
            traceback.print_exc()
    else:
        console.print("[yellow]No metrics found in output.[/yellow]")


    # --- AI Summary ---
    cfg = yaml.safe_load((proj_path / "config.yaml").read_text(encoding="utf-8"))
    context = cfg.get("context", "")
    console.print("\n[yellow]Generating AI summary...[/yellow]")

    spinner_active = True
    def spinner():
        while spinner_active:
            console.print(".", end="", style="yellow", soft_wrap=False)
            time.sleep(0.5)

    thread = threading.Thread(target=spinner)
    thread.start()
    summary = summarize_text(result_stdout, context=context)
    spinner_active = False
    thread.join()

    console.print("\n[green]Summary appended to log.md[/green]")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n\n---\n")
        f.write(f"### Run at {timestamp}\n")
        f.write(f"## AI Summary\n{summary}\n")


# -----------------------------------------------------
# Auto-run multiple experiments with parameter mutation
# -----------------------------------------------------
def auto_run(proj_name: str, count: int = 5):
    """Perform multiple runs with simple parameter mutation."""
    proj_path = PROJECTS_DIR / proj_name
    config_file = proj_path / "config.yaml"

    if not config_file.exists():
        console.print(f"[red]No config.yaml found for {proj_name}[/red]")
        return

    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    base_step = cfg.get("param_step_size", 0.5)

    for i in range(count):
        console.print(f"\n[bold cyan]=== Auto Run {i+1}/{count} ===[/bold cyan]")
        mutation = (1 + (0.05 - 0.1 * (i % 2)))  # ±5%
        new_step = round(base_step * mutation, 5)
        cfg["param_step_size"] = new_step
        config_file.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")

        console.print(f"[gray]Mutated step_size: {base_step} → {new_step}[/gray]")
        base_step = new_step
        run_project(proj_name)

    console.print(f"\n[green]Auto-run sequence complete for {proj_name}.[/green]")
