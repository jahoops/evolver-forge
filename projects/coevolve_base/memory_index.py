"""Persistent cross-experiment knowledge base for the Forge."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:  # Prefer forge.ai utilities when available
    from forge.ai import ai_call  # type: ignore
except ImportError:  # pragma: no cover
    from forge import ai_call  # type: ignore

from .canon import Canon


class MemoryIndex:
    """Persistent cross-experiment knowledge base.

    Learns "rules about rules" by analyzing experiment history and dialogue logs.
    """

    def __init__(self, project_root: Path, model: str = "codellama:13b") -> None:
        self.root = Path(project_root)
        self.history_path = self.root / "experiment_history.json"
        self.memory_path = self.root / "memory_bank.json"
        self.dialogue_path = self.root / "dialogue_log.md"
        self.model = model
        self.canon = Canon(self.root)
        self.mem = self._load()
        self.mem.setdefault("lessons", [])
        self.mem.setdefault("clusters", [])
        self.mem.setdefault("processed_runs", [])
        canon_rules = self.canon.top(5)
        if canon_rules:
            self.mem.setdefault("canon_rules", canon_rules)

    def _load(self) -> Dict[str, Any]:
        if self.memory_path.exists():
            try:
                with self.memory_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (json.JSONDecodeError, OSError):
                pass
        return {"lessons": [], "clusters": [], "processed_runs": []}

    def save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("w", encoding="utf-8") as handle:
            json.dump(self.mem, handle, indent=2)

    def extract_lessons(self) -> List[Dict[str, str]]:
        """Summarize outcomes of recent experiments into reusable lessons."""
        if not self.history_path.exists():
            return []

        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []

        experiments = history.get("experiments", [])
        if not experiments:
            return []

        processed_runs = set(self.mem.get("processed_runs", []))
        existing_rules = {lesson.get("rule") for lesson in self.mem.get("lessons", [])}
        lessons: List[Dict[str, str]] = []
        dialogue_context = self._recent_dialogue()
        canon_snapshot = self.canon.top(5)
        if canon_snapshot:
            self.mem["canon_rules"] = canon_snapshot
        elif "canon_rules" in self.mem:
            self.mem.pop("canon_rules")
        for exp in experiments[-5:]:
            hypothesis = exp.get("hypothesis", "unknown")
            config = exp.get("config", {})
            run_id = exp.get("run_dir") or exp.get("timestamp") or hypothesis
            if run_id in processed_runs:
                continue
            prompt = (
                "Summarize the key insight from this experiment in one compact rule:\n"
                f"Hypothesis: {hypothesis}\n"
                f"Config: {json.dumps(config)}\n"
                f"Prior outcomes: {json.dumps(exp, indent=2)}\n"
                f"Dialogue context: {dialogue_context or 'None'}\n"
                "Express the rule as a general principle (cause -> effect) that could guide future runs."
            )
            try:
                insight = ai_call(prompt, model=self.model)
            except Exception:
                continue
            rule = insight.strip()
            if not rule or rule in existing_rules:
                continue
            lessons.append({"hypothesis": hypothesis, "rule": rule})
            existing_rules.add(rule)
            processed_runs.add(run_id)

        if lessons:
            self.mem.setdefault("lessons", []).extend(lessons)
            self.mem["processed_runs"] = list(processed_runs)
            self.save()
        return lessons

    def cluster_lessons(self) -> None:
        """Group lessons by simple semantic similarity."""
        clusters: Dict[str, Dict[str, List[str]]] = {}
        for lesson in self.mem.get("lessons", []):
            rule = lesson.get("rule", "")
            if not rule:
                continue
            fingerprint = hashlib.md5(rule.encode("utf-8")).hexdigest()[:6]
            bucket = clusters.setdefault(fingerprint, {"examples": []})
            bucket["examples"].append(rule)
        self.mem["clusters"] = list(clusters.values())
        self.save()

    def retrieve_relevant(self, query: str, limit: int = 3) -> List[str]:
        """Return rules most relevant to the supplied query."""
        scores = []
        for lesson in self.mem.get("lessons", []):
            rule = lesson.get("rule", "")
            if not rule:
                continue
            score = self._similarity(rule, query)
            scores.append((score, rule))
        canon_rules = self.canon.top(max(limit, 3))
        for rule in canon_rules:
            score = self._similarity(rule, query)
            scores.append((score + 0.05, rule))  # slight boost for canon entries
        scores.sort(key=lambda item: item[0], reverse=True)
        seen = set()
        results: List[str] = []
        for score, rule in scores:
            if score <= 0:
                continue
            if rule in seen:
                continue
            seen.add(rule)
            results.append(rule)
            if len(results) >= limit:
                break
        return results

    def _similarity(self, text_a: str, text_b: str) -> float:
        """Compute a basic cosine similarity between two strings."""
        tokens_a = [token for token in self._tokenize(text_a)]
        tokens_b = [token for token in self._tokenize(text_b)]
        if not tokens_a or not tokens_b:
            return 0.0
        vocab = sorted(set(tokens_a) | set(tokens_b))
        counts_a = Counter(tokens_a)
        counts_b = Counter(tokens_b)
        vec_a = np.array([counts_a.get(token, 0) for token in vocab], dtype=float)
        vec_b = np.array([counts_b.get(token, 0) for token in vocab], dtype=float)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        sanitized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return [tok for tok in sanitized.split() if tok]

    def _recent_dialogue(self, lines: int = 20) -> str:
        if not self.dialogue_path.exists():
            return ""
        try:
            text = self.dialogue_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        parts = text.splitlines()
        tail = parts[-lines:]
        return "\n".join(tail)
