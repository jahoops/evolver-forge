"""LLM-assisted reflection summaries for coevolution runs."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:  # Prefer forge.ai if available
    from forge.ai import ai_call  # type: ignore
except ImportError:  # pragma: no cover - fallback for legacy layout
    from forge import ai_call  # type: ignore


class ReflectionEngine:
    """Generate concise natural-language reflections from recent metrics."""

    def __init__(self, metrics_path: Path, memory_path: Path, model: str = "codellama:13b") -> None:
        self.metrics_path = Path(metrics_path)
        self.memory_path = Path(memory_path)
        self.model = model

    def summarize(self) -> str:
        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover - pandas optional
            return "Reflection unavailable: pandas is not installed."

        if not self.metrics_path.exists():
            return "Reflection unavailable: metrics not found."

        try:
            df = pd.read_csv(self.metrics_path)
        except Exception as exc:  # pragma: no cover - corrupted file guard
            return f"Reflection unavailable: failed to read metrics ({exc})."

        if len(df) < 2:
            return "Insufficient data for reflection."

        recent = df.tail(min(5, len(df)))

        if "nn_accuracy" not in recent or "rule_fidelity" not in recent:
            return "Reflection unavailable: required metrics missing."

        try:
            acc_trend = float(np.gradient(recent["nn_accuracy"].astype(float)).mean())
            fid_trend = float(np.gradient(recent["rule_fidelity"].astype(float)).mean())
        except Exception:  # pragma: no cover - gradient edge cases
            acc_trend = 0.0
            fid_trend = 0.0

        combined = recent.get("combined_fitness")
        if combined is not None and len(combined) >= 2:
            try:
                meta_reward = float(combined.iloc[-1] - combined.iloc[0])
            except Exception:
                meta_reward = 0.0
        else:
            meta_reward = 0.0

        try:
            avg_accuracy = statistics.mean(recent["nn_accuracy"].astype(float))
            avg_fidelity = statistics.mean(recent["rule_fidelity"].astype(float))
        except statistics.StatisticsError:
            avg_accuracy = float(recent["nn_accuracy"].iloc[-1])
            avg_fidelity = float(recent["rule_fidelity"].iloc[-1])

        summary_data = {
            "accuracy": round(float(recent["nn_accuracy"].iloc[-1]), 4),
            "fidelity": round(float(recent["rule_fidelity"].iloc[-1]), 4),
            "avg_accuracy": round(float(avg_accuracy), 4),
            "avg_fidelity": round(float(avg_fidelity), 4),
            "meta_reward": round(meta_reward, 4),
            "acc_trend": round(acc_trend, 4),
            "fid_trend": round(fid_trend, 4),
        }

        prompt = f"""
        The Forge co-evolution system has completed several generations.
        Metrics summary: {json.dumps(summary_data, indent=2)}

        Please produce a short technical narrative (3–6 sentences)
        explaining:
        1. What the system seems to have learned.
        2. How the rules and neural network are influencing each other.
        3. Any apparent emergent behavior or instability.
        Use concise, analytical language.
        """

        try:
            text = ai_call(
                prompt.strip(),
                model=self.model,
                role="user",
                system_instruction="You are an AI research analyst summarizing experiment logs.",
            )
        except Exception as exc:  # pragma: no cover - inference failure guard
            text = f"[Reflection engine error: {exc}]"

        self._save_reflection(text)
        return text

    def _save_reflection(self, text: str) -> None:
        output_path = self.metrics_path.parent / "reflection_log.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        heading = f"\n\n### Reflection — {timestamp}\n"
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(heading)
            handle.write(text.strip() + "\n")

    def latest_reflection(self) -> Optional[str]:
        log_path = self.metrics_path.parent / "reflection_log.md"
        if not log_path.exists():
            return None
        content = log_path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        sections = [section.strip() for section in content.split("### Reflection") if section.strip()]
        if not sections:
            return None
        latest = sections[-1]
        return latest.splitlines()[-1] if latest else None
