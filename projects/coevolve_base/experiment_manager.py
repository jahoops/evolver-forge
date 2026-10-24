"""Autonomous experiment orchestration based on dialogue hypotheses."""

from __future__ import annotations

import json
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from .canon import Canon
from .memory_index import MemoryIndex
from .final_report import build_final_report

console = Console()


class ExperimentManager:
    """Selects hypotheses from dialogue logs and launches follow-up experiments."""

    def __init__(self, project_root: Path, use_memory: bool = False) -> None:
        self.project_root = Path(project_root)
        self.repo_root = self._resolve_repo_root(self.project_root)
        self.dialog_path = self.project_root / "dialogue_log.md"
        self.history_path = self.project_root / "experiment_history.json"
        self.runs_dir = self.project_root / "runs"
        self.use_memory = use_memory
        self.history = self._load_history()

    @staticmethod
    def _resolve_repo_root(project_root: Path) -> Path:
        candidate = project_root.resolve()
        while candidate != candidate.parent:
            if (candidate / "pyproject.toml").exists():
                return candidate
            candidate = candidate.parent
        return project_root.resolve()

    def _load_history(self) -> Dict[str, Any]:
        if self.history_path.exists():
            try:
                with self.history_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (json.JSONDecodeError, OSError):
                pass
        return {"experiments": []}

    def parse_hypotheses(self) -> List[str]:
        if not self.dialog_path.exists():
            return []
        text = self.dialog_path.read_text(encoding="utf-8")
        matches = re.findall(r"Hypothesis\s*\d+[:\-]\s*(.+)", text)
        return [match.strip() for match in matches[-10:]]

    def select_hypothesis(self) -> Optional[str]:
        hypotheses = self.parse_hypotheses()
        if not hypotheses:
            return None
        tried = {entry.get("hypothesis") for entry in self.history.get("experiments", [])}
        candidates = [hyp for hyp in hypotheses if hyp not in tried]
        if not candidates:
            candidates = hypotheses
        return random.choice(candidates)

    def configure_experiment(self, hypothesis: str) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "generations": 5,
            "task": "xor",
            "meta": True,
            "dialogue_every": None,
            "auto_experiment": True,
            "extra_cli": [],
        }

        def _set_flag(flag: str, value: float) -> None:
            cli = cfg.get("extra_cli", [])
            normalized: List[str] = []
            i = 0
            while i < len(cli):
                token = cli[i]
                if token == flag:
                    i += 2
                    continue
                normalized.append(token)
                if i + 1 < len(cli):
                    normalized.append(cli[i + 1])
                    i += 2
                else:
                    i += 1
            normalized.extend([flag, str(value)])
            cfg["extra_cli"] = normalized

        lowered = hypothesis.lower()
        if "sine" in lowered:
            cfg["task"] = "sine"
        elif "pid" in lowered:
            cfg["task"] = "pid"

        mutation_match = re.search(r"mutation[^\d]*(\d+\.\d+)", lowered)
        if mutation_match:
            cfg["mutation_rate"] = float(mutation_match.group(1))
            _set_flag("--mutation-rate", cfg["mutation_rate"])

        lambda_match = re.search(r"lambda[^\d]*(\d+\.\d+)", lowered)
        if lambda_match:
            cfg["lambda_consistency"] = float(lambda_match.group(1))
            _set_flag("--lambda-consistency", cfg["lambda_consistency"])

        lr_match = re.search(r"learning\s*rate[^\d]*(\d+\.\d+)", lowered)
        if lr_match:
            cfg["learning_rate"] = float(lr_match.group(1))
            _set_flag("--learning-rate", cfg["learning_rate"])

        canon = Canon(self.project_root)
        canon_rules = canon.top(3)
        if canon_rules:
            cfg["canon_rules"] = canon_rules
            for rule in canon_rules:
                rule_lower = rule.lower()
                if "mutation rate" in rule_lower and "instability" in rule_lower:
                    capped_rate = min(float(cfg.get("mutation_rate", 0.3)), 0.25)
                    cfg["mutation_rate"] = capped_rate
                    _set_flag("--mutation-rate", capped_rate)

        return cfg

    def run_experiment(
        self,
        hypothesis: str,
        cfg: Dict[str, Any],
        *,
        meta_enabled: bool,
        dialogue_every: Optional[int],
        auto_curriculum: bool,
        auto_stop: Optional[int],
    ) -> int:
        self._announce_relevant_lessons(hypothesis)
        cmd = [
            "poetry",
            "run",
            "python",
            "-m",
            "projects.coevolve_base.coevolve",
            "--task",
            cfg["task"],
            "--generations",
            str(cfg["generations"]),
            "--auto-experiment",
            "--no-watch",
        ]

        if meta_enabled or cfg.get("meta"):
            cmd.append("--meta")
        if dialogue_every:
            cmd.extend(["--dialogue-every", str(dialogue_every)])
        if auto_curriculum:
            cmd.append("--auto-curriculum")
        if auto_stop:
            cmd.extend(["--auto-stop", str(auto_stop)])
        canon_rules = cfg.get("canon_rules")
        if canon_rules:
            console.print("[green][Forge] Canon guidance:[/green]")
            for rule in canon_rules:
                console.print(f"  • {rule}")
        if cfg.get("extra_cli"):
            cmd.extend(cfg["extra_cli"])

        console.print(f"[yellow][Forge] Launching autonomous experiment:[/yellow] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.repo_root, check=False)
        return result.returncode

    def record_result(self, hypothesis: str, cfg: Dict[str, Any], exit_code: int) -> None:
        run_idx = len(self.history["experiments"]) + 1
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self.runs_dir / f"run_{run_idx:03d}"
        run_dir.mkdir(exist_ok=True)

        metadata = {
            "hypothesis": hypothesis,
            "config": cfg,
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir.relative_to(self.project_root)),
        }
        self.history["experiments"].append(metadata)
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

        summary_path = run_dir / "experiment.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        status = "recovered" if exit_code == 0 else f"exit={exit_code}"
        console.print(
            f"[green][Forge] Completed hypothesis test #{run_idx}: {hypothesis} ({status}).[/green]"
        )
        self._update_memory()

    def _announce_relevant_lessons(self, hypothesis: str) -> None:
        if not self.use_memory:
            return
        try:
            memory = MemoryIndex(self.project_root)
            relevant = memory.retrieve_relevant(hypothesis)
        except Exception as exc:  # pragma: no cover
            console.print(f"[yellow]Memory retrieval failed: {exc}[/yellow]")
            return
        if not relevant:
            return
        console.print("[green][Forge] Relevant prior lessons:[/green]")
        for rule in relevant:
            console.print(f"- {rule}")

    def _update_memory(self) -> None:
        if not self.use_memory:
            return
        try:
            memory = MemoryIndex(self.project_root)
            new_lessons = memory.extract_lessons()
            memory.cluster_lessons()
            cluster_count = len(memory.mem.get("clusters", []))
        except Exception as exc:  # pragma: no cover
            console.print(f"[yellow]Memory indexing failed: {exc}[/yellow]")
            return

        if not new_lessons:
            if cluster_count:
                console.print(f"[green][Forge] Memory index refreshed; {cluster_count} clusters available.[/green]")
            return

        console.print(
            f"[green][Forge] Indexed {len(new_lessons)} new lessons; {cluster_count} clusters updated.[/green]"
        )

    def rebuild_report(self):
        """Refresh the consolidated final report for the project."""
        try:
            report_path = build_final_report(self.project_root)
        except Exception as exc:  # pragma: no cover
            console.print(f"[yellow]Final report generation failed: {exc}[/yellow]")
            return None
        console.print(f"[green][Forge] Final report refreshed → {report_path.name}[/green]")
        return report_path
