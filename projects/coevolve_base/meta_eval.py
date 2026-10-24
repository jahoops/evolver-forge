"""Meta-evaluation helper for post-run self-critique."""

from __future__ import annotations

from pathlib import Path
import json
import re
import time
from typing import Dict, List

try:  # Prefer dedicated forge_ai helper when available
    from forge_ai import ai_call  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    try:
        from forge.ai import ai_call  # type: ignore
    except ImportError:  # pragma: no cover - fallback path
        ai_call = None  # type: ignore[assignment]

META_SYSTEM = (
    "You are a rigorous research editor. "
    "You evaluate clarity, falsifiability, evidence, and actionability. "
    "Be concise, numeric when possible, and avoid inventing facts."
)


def _last_block(path: Path, header: str, tail_chars: int = 20000) -> str:
    """Grab the last section beginning with header from a text file."""
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")[-tail_chars:]
    except OSError:
        return ""
    match = re.search(rf"(?s){re.escape(header)}.*?(?=^#|\Z)", text, flags=re.MULTILINE)
    return match.group(0) if match else ""


def evaluate_latest(project_root: Path, model: str = "codellama:13b") -> Dict:
    """Evaluate latest research artifacts via LLM prompt."""
    log_path = project_root / "log.md"
    dialogue_path = project_root / "dialogue_log.md"
    causal_path = project_root / "causal_report.json"
    robustness_path = project_root / "robustness_report.json"

    prompt_parts: List[str] = [
        "Evaluate the latest research artifacts. Return JSON with keys: "
        "`scores` (0-1 for clarity,falsifiability,evidence,actionability), "
        "`issues` (bullet list), `improvements` (bullet list), "
        "`promote_rules` (list of 1-sentence rules worth keeping), "
        "`archive_rules` (rules to demote or merge), and `summary` (≤4 sentences).",
        "\n--- Latest AI Plot Insight ---\n",
        _last_block(log_path, "## AI Plot Insight") or "(none)",
        "\n--- Latest Robustness Report ---\n",
        robustness_path.read_text(encoding="utf-8") if robustness_path.exists() else "(none)",
        "\n--- Latest Causal Attribution ---\n",
        _last_block(log_path, "== Causal Attribution ==") or "(none)",
        "\n--- Recent Dialogue (hypotheses) ---\n",
        dialogue_path.read_text(encoding="utf-8")[-4000:] if dialogue_path.exists() else "(none)",
    ]

    prompt = "".join(prompt_parts)
    if ai_call is None:
        return {"error": "forge_ai.ai_call unavailable", "t": time.time()}
    try:
        response = ai_call(prompt, model=model, system_instruction=META_SYSTEM, num_predict=800)
    except Exception as exc:  # pragma: no cover - fallback path
        return {"error": str(exc), "t": time.time()}

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        data = {"raw": response.strip()}
    data["t"] = time.time()
    return data
