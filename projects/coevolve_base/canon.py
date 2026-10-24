"""Persistent canon of curated design rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class Canon:
    """Maintain a lightweight canon of reusable rules."""

    def __init__(self, root: Path):
        self.path = Path(root) / "canon.json"
        self.data: Dict[str, List[Dict[str, object]]] = {"rules": []}
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict) and "rules" in payload:
                    self.data = payload
            except (json.JSONDecodeError, OSError):  # pragma: no cover - fallback
                self.data = {"rules": []}

    def add(self, rules: List[str], source: str) -> None:
        for rule in rules:
            text = rule.strip()
            if not text:
                continue
            existing = next((item for item in self.data["rules"] if item.get("text") == text), None)
            if existing is None:
                self.data["rules"].append({"text": text, "source": source, "weight": 1.0})
            else:
                existing["weight"] = min(5.0, float(existing.get("weight", 1.0)) + 0.25)

    def remove(self, rules: List[str]) -> None:
        if not rules:
            return
        to_remove = {rule.strip() for rule in rules if rule.strip()}
        if not to_remove:
            return
        self.data["rules"] = [item for item in self.data["rules"] if item.get("text") not in to_remove]

    def top(self, k: int = 5) -> List[str]:
        sorted_rules = sorted(
            self.data["rules"],
            key=lambda entry: -float(entry.get("weight", 1.0)),
        )
        return [entry.get("text", "") for entry in sorted_rules[:k] if entry.get("text")]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)
