"""Self-dialogue synthesis over recent reflections."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

try:  # Prefer forge.ai utilities when available
    from forge.ai import ai_call  # type: ignore
except ImportError:  # pragma: no cover
    from forge import ai_call  # type: ignore


class DialogEngine:
    """Generate autonomous dialogue from recent reflection entries."""

    def __init__(self, reflections_path: Path, model: str = "codellama:13b") -> None:
        self.reflections_path = Path(reflections_path)
        self.model = model

    def _load_reflections(self) -> List[str]:
        if not self.reflections_path.exists():
            return []
        text = self.reflections_path.read_text(encoding="utf-8")
        blocks = re.findall(r"### Reflection.*?\n(.*?)(?=\n###|\Z)", text, flags=re.S)
        trimmed = [block.strip() for block in blocks if block.strip()]
        return trimmed[-5:]

    def conduct_dialogue(self) -> str:
        reflections = self._load_reflections()
        if len(reflections) < 2:
            return "Not enough reflections for dialogue."

        joined = "\n---\n".join(reflections)
        prompt = f"""
        You are the Forge's research analyst conducting a self-dialogue across its last {len(reflections)} reflections.
        Each reflection describes co-evolution runs between a neural network and a rule engine.
        Synthesize the conversation:
        1. Identify recurring themes or trends.
        2. Debate potential causes or mechanisms.
        3. End with 2–3 clear experimental hypotheses the Forge could test next.
        Keep tone analytical, not poetic.
        ---
        {joined}
        """

        reply = ai_call(
            prompt.strip(),
            model=self.model,
            system_instruction="You are two collaborating AI researchers discussing experimental progress.",
        )
        self._save_dialogue(reply)
        return reply

    def _save_dialogue(self, text: str) -> None:
        out_path = self.reflections_path.parent / "dialogue_log.md"
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n### Dialogue\n")
            handle.write(text.strip() + "\n")
