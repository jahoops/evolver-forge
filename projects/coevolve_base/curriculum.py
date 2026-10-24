"""Curriculum manager tracking cross-task transfer gains."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class CurriculumRecord:
    from_task: str
    to_task: str
    gain: float


@dataclass
class CurriculumManager:
    """Persisted curriculum history that drives automatic task selection."""

    path: Path
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.history = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, list):
            return data
        return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def record(self, from_task: str, to_task: str, gain: float) -> None:
        self.history.append({"from": from_task, "to": to_task, "gain": float(gain)})
        self._save()

    def suggest_next(self, available_tasks: List[str]) -> str:
        """Pick the next task with the highest expected transfer gain."""

        if not available_tasks:
            raise ValueError("No available tasks supplied to CurriculumManager")

        if not self.history:
            return random.choice(available_tasks)

        avg_gain: Dict[str, float] = {}
        for task in available_tasks:
            gains = [entry["gain"] for entry in self.history if entry.get("to") == task]
            if gains:
                avg_gain[task] = float(np.mean(gains))
            else:
                avg_gain[task] = 0.0

        return max(avg_gain, key=avg_gain.get)

    def last_gains(self, n: int) -> List[float]:
        if n <= 0:
            return []
        return [entry["gain"] for entry in self.history[-n:]]
