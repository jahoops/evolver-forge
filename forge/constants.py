from pathlib import Path

# Always anchor everything to the root of the repo (where this file lives)
ROOT_DIR = Path(__file__).resolve().parent.parent

PROJECTS_DIR = ROOT_DIR / "projects"
DB_FILE = ROOT_DIR / "forge_memory.db"
TEMPLATES_DIR = ROOT_DIR / "templates"

# Default models
DEFAULT_AI_MODEL = "codellama:13b"
DEEP_ANALYSIS_MODEL = "deepseek33b:latest"
CURRENT_AI_MODEL = DEFAULT_AI_MODEL
