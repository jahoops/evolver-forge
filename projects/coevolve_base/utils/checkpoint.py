# projects/coevolve_base/utils/checkpoint.py

import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from rich.console import Console

console = Console()


def _prune_unscored_entries(base_path: Path, manifest_path: Path, entries: list[dict]) -> list[dict]:
    """Remove manifest entries without numeric scores and delete orphaned files."""
    scored = []
    unscored = []
    for entry in entries:
        score = entry.get("score")
        if isinstance(score, (int, float)):
            scored.append(entry)
        else:
            unscored.append(entry)

    if not unscored:
        return scored

    scored_filenames = {e.get("filename") for e in scored if e.get("filename")}
    deleted_files = 0
    for entry in unscored:
        filename = entry.get("filename")
        if not filename or filename in scored_filenames:
            continue
        file_path = base_path / filename
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_files += 1
                console.print(f"[gray]Removed legacy checkpoint file {filename}[/gray]")
            except Exception as exc:
                console.print(f"[yellow]Could not remove legacy checkpoint {filename}: {exc}[/yellow]")

    try:
        manifest_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
        console.print(f"[gray]Pruned {len(unscored)} manifest entries without scores.[/gray]")
        if deleted_files:
            console.print(f"[gray]Deleted {deleted_files} legacy checkpoint files.[/gray]")
    except Exception as exc:
        console.print(f"[yellow]Failed to rewrite manifest after pruning: {exc}[/yellow]")
        return entries

    return scored

def _find_latest_checkpoint(base_path: Path, prefix="nn_model", suffix=".pt"):
    """Return the latest matching checkpoint path by version suffix."""
    candidates = sorted(base_path.glob(f"{prefix}_v*.pt"), key=lambda p: p.name)
    if candidates:
        return candidates[-1]
    default = base_path / f"{prefix}{suffix}"
    return default if default.exists() else None


def _update_manifest(base_path: Path, entry: dict):
    """Append checkpoint metadata to checkpoints.json in same directory."""
    manifest_path = base_path / "checkpoints.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            console.print("[yellow]Warning: corrupt checkpoints.json, recreating...[/yellow]")

    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    console.print(f"[gray]Manifest updated ({len(manifest)} total checkpoints).[/gray]")


def _normalize_arch_payload(payload: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            normalized[key] = round(value, 6)
        else:
            normalized[key] = value
    return normalized


def safe_load_checkpoint(
    model,
    path: str | Path = None,
    model_name: str = "model",
    expected_type: str = None,
    expected_version: str = None,
    *,
    expected_arch: Dict[str, Any] | None = None,
    strict: bool = True,
) -> bool:
    """
    Safely load the most recent compatible checkpoint.
    Automatically falls back to versioned files like nn_model_v1.1.pt
    """
    base_dir = Path(path).parent if path else Path.cwd()
    prefix = Path(path).stem.split("_v")[0] if path else "nn_model"
    path = Path(path) if path else _find_latest_checkpoint(base_dir, prefix)

    if not path or not path.exists():
        console.print(f"[yellow]No checkpoint found for {model_name} ({prefix}).[/yellow]")
        return False

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt.get("__meta__", {})
        state = ckpt.get("state_dict", ckpt)

        expected_arch_norm = _normalize_arch_payload(expected_arch)
        stored_arch_norm = _normalize_arch_payload(meta.get("nn_arch"))
        if expected_arch_norm and stored_arch_norm and expected_arch_norm != stored_arch_norm:
            console.print(
                f"[yellow]{model_name}: checkpoint architecture mismatch; "
                "skipping weight restore.[/yellow]"
            )
            console.print(f"Expected arch: {expected_arch_norm}")
            console.print(f"Checkpoint arch: {stored_arch_norm}")
            return False

        model_state = model.state_dict()
        incompat_keys: list[str] = []
        for key, tensor in state.items():
            if key not in model_state:
                incompat_keys.append(key)
                continue
            if model_state[key].shape != tensor.shape:
                incompat_keys.append(key)

        missing_keys = [key for key in model_state.keys() if key not in state]

        if incompat_keys or missing_keys:
            console.print(
                f"[yellow]{model_name}: checkpoint tensor mismatch; skipping load.[/yellow]"
            )
            if incompat_keys:
                console.print(f"Mismatched tensors: {incompat_keys}")
            if missing_keys:
                console.print(f"Missing tensors: {missing_keys}")
            return False

        # --- Validate ---
        if expected_type and meta.get("model_type") != expected_type:
            console.print(f"[red]{model_name}: model type mismatch![/red]")
            console.print(f"Expected {expected_type}, found {meta.get('model_type')}")
            return False
        if expected_version and meta.get("version") != expected_version:
            console.print(f"[yellow]{model_name}: version differs ({meta.get('version')} != {expected_version})[/yellow]")

        model.load_state_dict(state, strict=strict)
        console.print(f"[green]Loaded checkpoint for {model_name} from {path.name} ({meta.get('model_type','?')})[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Failed to load checkpoint {path}: {e}[/red]")
        return False


def save_checkpoint(
    model,
    path: str | Path,
    model_type: str,
    version: str = "1.0.0",
    score: float | None = None,
    add_version_to_name: bool = True,
    extra_meta: dict | None = None,
):
    """
    Save model weights with metadata. Creates a versioned copy automatically
    and logs it in checkpoints.json.
    """
    path = Path(path)
    version_clean = version.replace('.', '_')
    target = path.with_stem(f"{path.stem}_v{version_clean}") if add_version_to_name else path

    score_value = None
    if score is not None:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            console.print(f"[yellow]Warning: could not record score {score!r}; storing as null.[/yellow]")

    meta = {
        "model_type": model_type,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "score": score_value,
    }
    if extra_meta:
        try:
            meta.update(extra_meta)
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[yellow]Warning: could not merge extra metadata: {exc}[/yellow]")

    payload = {
        "__meta__": meta,
        "state_dict": model.state_dict(),
    }

    # --- Save model ---
    torch.save(payload, target)
    if add_version_to_name:
        # Keep latest alias without version for easy loading
        torch.save(payload, path)

    console.print(f"[gray]Saved checkpoint for {model_type} -> {target.name}[/gray]")

    # --- Log manifest entry ---
    _update_manifest(target.parent, {
        "filename": target.name,
        "model_type": model_type,
        "version": version,
        "score": score_value,
        "timestamp": datetime.now().isoformat(),
    })


def plot_checkpoint_lineage(base_dir: str | Path | None = None):
    """Plot score vs version from checkpoints.json manifest."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping checkpoint lineage plot.[/yellow]")
        return

    base_path = Path(base_dir) if base_dir else Path.cwd()
    manifest_path = base_path / "checkpoints.json"

    if not manifest_path.exists():
        console.print(f"[yellow]No checkpoint manifest found at {manifest_path}.[/yellow]")
        return

    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Failed to read {manifest_path}: {exc}[/red]")
        return

    entries = _prune_unscored_entries(base_path, manifest_path, entries)

    def parse_version(ver: str) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in ver.replace('_', '.').split('.') if part != "")
        except Exception:
            return ()

    # Filter entries with numeric scores
    scored = [e for e in entries if isinstance(e.get("score"), (int, float))]
    if not scored:
        console.print("[yellow]No scored checkpoints available for plotting.[/yellow]")
        return

    def sort_key(entry: dict):
        ts = entry.get("timestamp")
        if ts:
            try:
                return (0, datetime.fromisoformat(ts))
            except Exception:
                pass
        return (1, parse_version(str(entry.get("version", ""))))

    scored.sort(key=sort_key)

    def format_timestamp(entry: dict) -> str:
        ts = entry.get("timestamp")
        if not ts:
            return str(entry.get("version", "?"))
        try:
            dt = datetime.fromisoformat(ts)
            return dt.strftime("%m-%d %H:%M")
        except Exception:
            return ts

    tick_labels = [format_timestamp(e) for e in scored]
    scores = [float(e["score"]) for e in scored]
    x_vals = list(range(len(scored)))

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, scores, marker="o", linestyle="-", color="tab:blue")

    # Trend line via linear regression
    try:
        import numpy as np
        coeffs = np.polyfit(x_vals, scores, 1)
        trend = np.polyval(coeffs, x_vals)
        plt.plot(x_vals, trend, linestyle="--", color="gray", label="trend")
    except Exception as exc:
        console.print(f"[yellow]Trend line skipped: {exc}[/yellow]")

    for idx, entry in enumerate(scored):
        legend_label = format_timestamp(entry)
        plt.scatter([idx], [scores[idx]], label=legend_label)

    plt.xticks(x_vals, tick_labels, rotation=45)
    plt.xlabel("Checkpoint Time")
    plt.ylabel("Score")
    plt.title("Checkpoint Lineage")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    output_path = base_path / "checkpoint_lineage.png"
    try:
        plt.savefig(output_path)
        console.print(f"[green]Checkpoint lineage plot saved to {output_path}[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to save plot: {exc}[/red]")
    finally:
        plt.close()
