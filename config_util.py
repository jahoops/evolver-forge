# config_util.py
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "forge_config.yaml"

def load_forge_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}

def save_forge_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.safe_dump(cfg), encoding="utf-8")
