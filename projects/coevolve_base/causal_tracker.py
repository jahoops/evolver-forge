"""Lightweight intervention logging for causal attribution."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from typing import Any, Dict


@dataclass
class Intervention:
    """Represents a single intervention applied during a run."""

    t: float
    gen: int
    kind: str
    old: float | str | None
    new: float | str | None
    context: Dict[str, Any]


class CausalTracker:
    """Append-only tracker that records interventions to JSONL."""

    def __init__(self, proj_path: Path):
        self.proj_path = proj_path
        self.log_path = proj_path / "causal_log.jsonl"

    def record(self, iv: Intervention) -> None:
        """Persist a single intervention entry."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(iv)) + "\n")

    def stamp(self, *, gen: int, kind: str, old: float | str | None, new: float | str | None, context: Dict[str, Any]) -> None:
        """Convenience helper to capture an intervention with the current time."""
        self.record(
            Intervention(
                t=time.time(),
                gen=gen,
                kind=kind,
                old=old,
                new=new,
                context=context,
            )
        )
