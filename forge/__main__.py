import argparse
import csv
import random
import re
import shutil
import sqlite3
import statistics
import time
from datetime import datetime
from email import parser
from pathlib import Path

import matplotlib.pyplot as plt
import ollama
import pandas as pd
import yaml
from rich.console import Console

from config_util import load_forge_config, save_forge_config

from .ai import summarize_text
from .constants import (
    CURRENT_AI_MODEL,
    DB_FILE,
    DEFAULT_AI_MODEL,
    DEEP_ANALYSIS_MODEL,
    PROJECTS_DIR,
)
from .metrics import extract_metrics, plot_metrics
from .projects import create_project, list_projects, list_runs
from .runner import auto_run, run_project
from projects.coevolve_base.utils.checkpoint import plot_checkpoint_lineage

# --- Ensure local imports work regardless of how Forge is run ---
ROOT_DIR = Path(__file__).parent.resolve()

console = Console()

CONFIG_PATH = Path(__file__).parent / "forge_config.yaml"

def ai_call(
    prompt: str,
    model: str,
    role: str = "user",
    system_instruction: str = None,
    num_predict: int = 1024,
) -> str:
    """
    Unified AI call for Forge (Ollama backend).
    Handles spinners, timing, and consistent error capture.
    The caller must specify the model explicitly.
    """
    import time, threading
    stop = False

    def spinner():
        while not stop:
            console.print(".", end="", style="yellow", soft_wrap=False)
            time.sleep(0.4)

    thread = threading.Thread(target=spinner)
    thread.start()

    start_time = time.time()
    try:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": role, "content": prompt})

        response = ollama.chat(
            model=model,
            options={"num_predict": num_predict},
            messages=messages,
        )
        result = response["message"]["content"].strip()
    except Exception as e:
        result = f"[Error calling {model}: {e}]"
    finally:
        stop = True
        thread.join()

    console.print(f"\n[gray]Model {model} completed in {time.time() - start_time:.1f}s[/gray]")
    return result


def record_run_results(run_dir: Path, cfg_path: Path, metrics: dict):
    """Append parameter values and metrics to results.csv for long-term analysis."""
    results_path = run_dir.parent.parent / "results.csv"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    params = data.get("params", {})
    row = {"run_id": run_dir.name}

    # include parameters
    for k, v in params.items():
        row[f"param_{k}"] = v

    # include all numeric metrics
    for k, v in metrics.items():
        row[f"metric_{k}"] = v

    # append header if needed
    new_file = not results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    console.print(f"[cyan]Logged results for {run_dir.name} → results.csv[/cyan]")

def mutate_config(cfg_path: Path, factor: float = 0.2) -> float:
    """Mutate one numeric param in config.yaml by ±factor and return new value."""
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    params = data.get("params", {})
    if not params:
        console.print("[yellow]No params to mutate in config.yaml.[/yellow]")
        return 0

    # Pick one numeric key
    key = random.choice(list(params.keys()))
    val = params[key]
    if not isinstance(val, (int, float)):
        console.print(f"[yellow]Skipping non-numeric param {key}.[/yellow]")
        return 0

    delta = val * random.uniform(-factor, factor)
    new_val = round(val + delta, 5)
    params[key] = new_val
    data["params"] = params

    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    console.print(f"[cyan]Mutated {key}: {val} → {new_val}[/cyan]")
    return abs(delta)

def parse_metrics(log_text: str) -> dict:
    """Extract numeric metrics from a run's log text."""
    metrics = {}
    # Common patterns: score=, accuracy=, loss=, etc.
    patterns = [
        r"score\s*=\s*(-?\d+\.\d+)",
        r"accuracy\s*=\s*(\d+\.\d+)",
        r"loss\s*=\s*(\d+\.\d+)"
    ]
    for pat in patterns:
        for match in re.findall(pat, log_text):
            key = pat.split("\\")[0].strip(":=")
            key = re.sub(r"[^a-zA-Z0-9_]", "", key)
            metrics.setdefault(key, []).append(float(match))
    # Compute means
    summary = {k: round(sum(v)/len(v), 4) for k, v in metrics.items() if v}
    return summary



def summarize_metric_trends(csv_path: Path) -> str:
    """Read metrics.csv and summarize basic trends numerically."""
    if not csv_path.exists():
        return "No metrics.csv file yet."
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if len(reader) < 2:
        return "Only one run recorded; not enough data for trend analysis."

    # Compute per-metric trends
    metrics = reader[0].keys()
    metrics = [m for m in metrics if m != "run"]
    lines = []
    for m in metrics:
        values = []
        for row in reader:
            try:
                values.append(float(row[m]))
            except Exception:
                continue
        if len(values) > 1:
            diff = values[-1] - values[-2]
            avg = round(statistics.mean(values), 4)
            lines.append(f"{m}: last={values[-1]:.4f}, prev={values[-2]:.4f}, Δ={diff:+.4f}, avg={avg}")
    if not lines:
        return "No numeric trend data found."
    return "\n".join(lines)

def create_run_folder(proj_path: Path) -> Path:
    """Create a uniquely numbered run folder inside <project>/runs/ without overwriting old runs."""
    runs_dir = proj_path / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Collect all existing run folders safely
    existing = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )

    # Find the highest numeric suffix
    next_num = 1
    if existing:
        try:
            nums = [int(d.name.replace("run_", "")) for d in existing if d.name[4:].isdigit()]
            next_num = max(nums) + 1 if nums else 1
        except Exception:
            next_num = len(existing) + 1

    # Create the new folder (never reuse an old one)
    run_dir = runs_dir / f"run_{next_num:03d}"
    if run_dir.exists():
        # Extremely rare — only if duplicate race
        timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / f"run_{next_num:03d}_{timestamp_tag}"

    run_dir.mkdir()
    return run_dir


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects
                 (name TEXT PRIMARY KEY, description TEXT, created TEXT)''')
    conn.commit()
    conn.close()

def main():
    # --- Argument parser ---
    parser = argparse.ArgumentParser(description="The Forge - Project Manager")

    # Global options
    parser.add_argument("--fast", action="store_true", help="Use fast local model")
    parser.add_argument("--deep", action="store_true", help="Use deep analysis model")
    parser.add_argument("--model", type=str, help="Manually specify model name")

    # --- Subcommands ---
    sub = parser.add_subparsers(dest="command")

    # project management
    new_cmd = sub.add_parser("new", help="Create a new project")
    new_cmd.add_argument("name")
    new_cmd.add_argument("--desc", default="")

    sub.add_parser("list", help="List all projects")

    runs_cmd = sub.add_parser("runs", help="List versioned runs for a project")
    runs_cmd.add_argument("name")

    # experiment commands
    run_cmd = sub.add_parser("run", help="Run a project's main.py and log output")
    run_cmd.add_argument("name")

    sub.add_parser("summarize", help="Summarize the project log")

    diff_cmd = sub.add_parser("diff", help="Compare AI summaries between runs")
    diff_cmd.add_argument("name")

    plot_cmd = sub.add_parser("plot", help="Generate metric plots for a project")
    plot_cmd.add_argument("name")

    analyze_cmd = sub.add_parser("analyze", help="Analyze metrics and correlations")
    analyze_cmd.add_argument("name")

    auto_cmd = sub.add_parser("auto", help="Run multiple experiments automatically")
    auto_cmd.add_argument("name")
    auto_cmd.add_argument("--count", type=int, default=5, help="Number of auto-runs to perform")

    lineage_cmd = sub.add_parser("lineage", help="Plot checkpoint lineage for a project")
    lineage_cmd.add_argument("name")

    # --- Parse arguments now ---
    args = parser.parse_args()
    console = Console()

    # --- Config and model selection ---
    config = load_forge_config()
    model_to_use = config.get("last_model", DEFAULT_AI_MODEL)

    if args.model:
        model_to_use = args.model
    elif args.deep:
        model_to_use = DEEP_ANALYSIS_MODEL
    elif args.fast:
        model_to_use = DEFAULT_AI_MODEL

    config["last_model"] = model_to_use
    save_forge_config(config)

    console.print(f"[gray]Using model: {model_to_use}[/gray]")

    # --- command dispatch ---
    if args.command == "new":
        create_project(args.name, args.desc)
    elif args.command == "list":
        list_projects()
    elif args.command == "runs":
        list_runs(args.name)
    elif args.command == "plot":
        proj_name = args.name
        proj_path = PROJECTS_DIR / proj_name
        plot_metrics(proj_name, proj_path)
    elif args.command == "run":
        run_project(args.name)
    elif args.command == "auto":
        auto_run(args.name, args.count)
    elif args.command == "lineage":
        plot_checkpoint_lineage(PROJECTS_DIR / args.name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
