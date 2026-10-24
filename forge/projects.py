from pathlib import Path
import shutil
import yaml
from rich.console import Console
from .constants import PROJECTS_DIR

console = Console()

def create_project(name: str, desc: str = ""):
    """Create a new project folder with config and main.py template."""
    proj_path = PROJECTS_DIR / name
    if proj_path.exists():
        console.print(f"[yellow]Project '{name}' already exists.[/yellow]")
        return proj_path

    proj_path.mkdir(parents=True, exist_ok=True)

    # Default files
    (proj_path / "runs").mkdir(exist_ok=True)
    (proj_path / "log.md").write_text("# Log for " + name + "\n", encoding="utf-8")
    config = {
        "description": desc or "No description provided.",
        "model": "codellama:13b",
        "context": f"This project ({name}) performs evolutionary or optimization experiments."
    }
    (proj_path / "config.yaml").write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")

    main_py = proj_path / "main.py"
    main_py.write_text(
        """print("Starting simple Forge test...")
for i in range(1, 6):
    x = i * 1.234
    score = x - 3.14159
    print(f"Generation {i}: x={x:.3f}, score={score:.3f}")
print("Finished experiment.")""",
        encoding="utf-8"
    )

    console.print(f"[green]Created new project '{name}'[/green]")
    return proj_path


def list_projects():
    """List all existing projects under /projects."""
    if not PROJECTS_DIR.exists():
        console.print("[red]No projects directory found.[/red]")
        return
    projects = [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]
    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        return
    console.print("[bold cyan]Available Projects:[/bold cyan]")
    for p in projects:
        console.print(f" - {p}")


def list_runs(project_name: str):
    """List versioned runs for a specific project."""
    runs_dir = PROJECTS_DIR / project_name / "runs"
    if not runs_dir.exists():
        console.print(f"[red]No runs directory for project '{project_name}'[/red]")
        return
    runs = sorted(runs_dir.iterdir(), key=lambda p: p.name)
    if not runs:
        console.print(f"[yellow]No runs recorded yet for '{project_name}'[/yellow]")
        return
    console.print(f"[bold cyan]Runs for {project_name}:[/bold cyan]")
    for r in runs:
        console.print(f" - {r.name}")
