"""
coevolve.py — orchestrates co-evolution between neural and symbolic agents.
"""

from pathlib import Path
import yaml
import csv
import json
import random
import copy
import math
import numpy as np
import torch
from datetime import datetime
import threading
import subprocess
from dataclasses import asdict
from .curriculum import CurriculumManager
from .meta_controller import MetaController, HAS_PANDAS as META_HAS_PANDAS
from .reflection import ReflectionEngine
from .dialog_engine import DialogEngine
from .experiment_manager import ExperimentManager
from .architectures import ArchGenome, ExploreArch, NNArch, RuleArch, mutate_arch
from .nn_agent import train_nn, _logits_to_probs, _build_model, _compute_primary_metric
from .rule_agent import extract_rules, _tree_for_task, _nn_targets_from_logits
from .rule_mutation import save_mutated_rules
from .memory_bank import MemoryBank
from .rule_feedback import fine_tune_nn
from .tasks import apply_task, available_tasks, TaskSpec, get_task_spec
from .robustness import compute_robustness
from .causal_tracker import CausalTracker
from .meta_eval import evaluate_latest
from .canon import Canon
try:
    from .causal_analysis import attribute
except ImportError:  # pragma: no cover - optional dependency
    attribute = None  # type: ignore[assignment]
try:
    from .causal_graph import build_graph
except ImportError:  # pragma: no cover - optional dependency
    build_graph = None  # type: ignore[assignment]
from rich.console import Console
from joblib import load
from .utils.checkpoint import safe_load_checkpoint

try:
    from forge.analysis.health_dashboard import update_live_health
except ImportError:  # pragma: no cover - optional dependency
    update_live_health = None

try:
    from forge.analysis.health_report import evaluate_health_metrics
except ImportError:  # pragma: no cover - optional dependency
    evaluate_health_metrics = None

try:
    from forge.analysis.health_report_html import generate_health_report
except ImportError:  # pragma: no cover - optional dependency
    generate_health_report = None

console = Console()


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and Torch for deterministic runs."""
    if seed is None:
        seed = int(datetime.now().timestamp()) % 10_000_000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    console.print(f"[gray]Using global seed {seed}[/gray]")
    return seed


def mutate_params(params: dict, bounds: dict, sigma: float = 0.1) -> dict:
    """Apply a Gaussian perturbation to evolvable parameters within their bounds."""
    new_params: dict = {}
    for key, value in params.items():
        if key not in bounds:
            new_params[key] = value
            continue
        low, high = bounds[key]
        perturb = random.gauss(0, sigma * (high - low))
        new_val = max(low, min(high, value + perturb))
        new_params[key] = round(new_val, 6)
    return new_params


def _load_arch_file(path: Path) -> tuple[ArchGenome | None, float | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    if not isinstance(payload, dict):
        return None, None

    genome_data = payload.get("genome") if "genome" in payload else payload
    score = payload.get("composite_score")
    try:
        genome = ArchGenome.from_dict(genome_data or {})
    except Exception:
        return None, None
    return genome, score if isinstance(score, (int, float)) else None


def _write_arch_file(path: Path, genome: ArchGenome, *, task: str, composite_score: float | None = None) -> None:
    payload = {
        "task": task,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "genome": genome.to_dict(),
    }
    if composite_score is not None:
        payload["composite_score"] = round(float(composite_score), 6)
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not persist architecture genome to {path.name}: {exc}[/yellow]")


def _describe_genome(genome: ArchGenome) -> str:
    nn = genome.nn
    rules = genome.rules
    explore = genome.explore
    return (
        f"NN: {nn.hidden_layers}x{nn.hidden_width} {nn.activation} norm={nn.normalization} drop={nn.dropout:.2f} | "
        f"Rules: depth={rules.max_depth} leaf={rules.min_samples_leaf} α={rules.prune_ccp_alpha:.3f} | "
        f"Explore: k={explore.uncertainty_k} μ={explore.mutation_rate:.2f} λ={explore.lambda_consistency:.2f}"
    )


def _normalize_primary(metric: float | None, higher_is_better: bool) -> float:
    if metric is None or not isinstance(metric, (int, float)) or math.isnan(metric):
        return 0.0
    value = float(metric)
    if higher_is_better:
        return float(np.clip(value, 0.0, 1.0))
    return 1.0 / (1.0 + abs(value))


def _normalize_secondary(metric: float | None) -> float:
    if metric is None or not isinstance(metric, (int, float)) or math.isnan(metric):
        return 0.0
    value = float(metric)
    return float(np.clip(value, -1.0, 1.0))


def _model_penalty(param_count: int, baseline: float = 50000.0) -> float:
    if baseline <= 0:
        return 0.0
    return float(np.clip(param_count / baseline, 0.0, 1.5))


def _composite_score(
    *,
    primary_metric: float | None,
    task: TaskSpec,
    transfer_norm: float,
    memory_norm: float,
    param_count: int,
) -> float:
    primary_norm = _normalize_primary(primary_metric, task.primary_higher_is_better)
    score = primary_norm + 0.5 * float(np.clip(transfer_norm, 0.0, 1.0))
    score += 0.25 * _normalize_secondary(memory_norm)
    score -= 0.15 * _model_penalty(param_count)
    return float(score)


def _probe_architecture(
    genome: ArchGenome,
    *,
    cfg_path: Path,
    task: TaskSpec,
    data: tuple[np.ndarray, np.ndarray],
    baseline_memory: float | None,
    transfer_norm: float,
    rng: random.Random,
    learning_rate: float,
) -> dict[str, float | ArchGenome | int] | None:
    X_probe, y_probe = data
    try:
        probe_seed = rng.randint(0, 1_000_000)
        (  # noqa: E241 - tuple unpack readability
            model,
            nn_metric,
            _sup,
            _cons,
            _opt_state,
            param_count,
        ) = train_nn(
            cfg_path,
            task,
            generation=0,
            rule_model=None,
            lambda_consistency=genome.explore.lambda_consistency,
            learning_rate=learning_rate,
            extra_data=None,
            initial_state=None,
            arch=genome.nn,
            data=(X_probe, y_probe),
            persist=False,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        console.print(f"[yellow]Architecture probe failed: {exc}[/yellow]")
        return None

    try:
        with torch.no_grad():
            logits = model(torch.tensor(X_probe, dtype=torch.float32))
        nn_targets = _nn_targets_from_logits(logits, task)
        tree = _tree_for_task(task, genome.rules, probe_seed)
        tree.fit(X_probe, nn_targets)
        rule_metric = float(tree.score(X_probe, nn_targets)) if hasattr(tree, "score") else float("nan")
    except Exception:
        rule_metric = float("nan")

    memory_norm = 0.0
    if baseline_memory is not None and isinstance(nn_metric, (int, float)):
        memory_norm = nn_metric - baseline_memory

    composite = _composite_score(
        primary_metric=nn_metric,
        task=task,
        transfer_norm=transfer_norm,
        memory_norm=memory_norm,
        param_count=param_count,
    )

    return {
        "genome": genome,
        "score": composite,
        "nn_metric": float(nn_metric) if isinstance(nn_metric, (int, float)) else float("nan"),
        "rule_metric": rule_metric,
        "param_count": int(param_count),
        "memory_norm": float(memory_norm),
    }


def _initial_architecture(
    project_dir: Path,
    task_name: str,
    *,
    reset: bool,
    override_path: Path | None,
) -> tuple[ArchGenome, float | None, float | None]:
    if override_path is not None:
        try:
            override_payload = json.loads(Path(override_path).read_text(encoding="utf-8"))
            genome = ArchGenome.from_dict(override_payload.get("genome", override_payload))
            return genome, None, None
        except Exception as exc:
            console.print(f"[yellow]Failed to load architecture from {override_path}: {exc}[/yellow]")

    current_path = project_dir / "arch_current.json"
    best_path = project_dir / f"arch_best_{task_name}.json"

    if reset:
        return ArchGenome(), None, None

    genome, score = _load_arch_file(current_path)
    if genome is not None:
        best_genome, best_score = _load_arch_file(best_path)
        if best_genome is None:
            best_score = None
        return genome, score, best_score

    genome, score = _load_arch_file(best_path)
    if genome is not None:
        return genome, None, score

    return ArchGenome(), None, None


def _align_rule_probabilities(rule_model, probs: np.ndarray, task: TaskSpec) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    if task.loss_type == "bce":
        if probs.ndim == 2:
            if probs.shape[1] == 1:
                return probs
            if probs.shape[1] >= 2:
                return probs[:, 1:2]
        return np.zeros((len(probs), 1), dtype=np.float32)

    num_classes = task.num_classes or probs.shape[1]
    aligned = np.zeros((len(probs), num_classes), dtype=np.float32)
    classes = getattr(rule_model, "classes_", [])
    if probs.shape[1] == num_classes and len(classes) == num_classes:
        return probs.astype(np.float32)
    for idx, cls in enumerate(classes):
        cls_idx = int(cls)
        if 0 <= cls_idx < num_classes:
            aligned[:, cls_idx] = probs[:, idx]
    return aligned


def evaluate_transfer(
    project_dir: Path,
    current_task: TaskSpec,
    target_task_name: str,
    seed: int | None,
    extra_memory: tuple[np.ndarray | None, np.ndarray | None] | None,
    *,
    arch: NNArch | None = None,
) -> dict | None:
    """Evaluate how the current model/rule set performs on a new task."""

    target_spec = get_task_spec(target_task_name)
    if target_spec is None:
        console.print(f"[yellow]Unknown transfer target task: {target_task_name}[/yellow]")
        return None

    eval_seed = (seed or 0) + 101
    try:
        X_target, y_target = target_spec.generator(512, eval_seed)
    except Exception as exc:
        console.print(f"[yellow]Transfer evaluation skipped (dataset error): {exc}[/yellow]")
        return None

    if X_target is None or len(X_target) == 0:
        console.print("[yellow]Transfer evaluation skipped (empty dataset).[/yellow]")
        return None

    input_dim = X_target.shape[1]
    model = _build_model(current_task, input_dim, arch)
    checkpoint_path = project_dir / "nn_model.pt"
    expected_arch_dict = asdict(arch) if arch is not None else None
    if not safe_load_checkpoint(
        model,
        checkpoint_path,
        model_name="TaskNN",
        expected_type=model.__class__.__name__,
        expected_arch=expected_arch_dict,
    ):
        console.print("[yellow]Transfer evaluation skipped (no NN checkpoint).[/yellow]")
        return None

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_target, dtype=torch.float32))
        nn_probs = _logits_to_probs(logits, current_task).cpu().numpy()

    if current_task.loss_type == "bce":
        nn_prob_vector = nn_probs.reshape(-1)
        nn_preds = (nn_prob_vector >= 0.5).astype(int)
    else:
        nn_preds = nn_probs.argmax(axis=1)

    y_target_arr = np.asarray(y_target)
    if y_target_arr.ndim > 1:
        y_target_arr = y_target_arr.reshape(-1)

    metric_label = "accuracy"
    boot_metric = 0.0

    if target_spec.kind == "regression":
        metric_label = "mse"
        y_true = y_target_arr.astype(np.float32)
        preds = nn_probs.reshape(-1) if nn_probs.ndim > 1 else nn_probs
        preds = preds.astype(np.float32)
        min_len = min(len(preds), len(y_true))
        if min_len == 0:
            boot_metric = float("nan")
        else:
            boot_metric = float(np.mean((preds[:min_len] - y_true[:min_len]) ** 2))
    elif target_spec.kind == "control":
        metric_label = "reward"
        y_comp = y_target_arr.astype(int)
        if current_task.loss_type == "bce":
            y_comp = np.clip(y_comp, 0, 1)
        elif nn_probs.ndim == 2:
            y_comp = np.clip(y_comp, 0, nn_probs.shape[1] - 1)
        boot_metric = float(np.mean(nn_preds == y_comp))
    else:
        y_comp = y_target_arr.astype(int)
        if current_task.loss_type == "bce":
            y_comp = np.clip(y_comp, 0, 1)
        elif nn_probs.ndim == 2:
            y_comp = np.clip(y_comp, 0, nn_probs.shape[1] - 1)
        boot_metric = float(np.mean(nn_preds == y_comp)) if len(y_comp) else float("nan")

    fidelity_value = None
    rule_model_path = project_dir / "rules_model.joblib"
    if rule_model_path.exists():
        try:
            rule_model = load(rule_model_path)
            if hasattr(rule_model, "predict_proba"):
                rule_probs = rule_model.predict_proba(X_target)
                if current_task.loss_type == "bce":
                    if rule_probs.shape[1] == 1:
                        rule_preds = (rule_probs[:, 0] >= 0.5).astype(int)
                    else:
                        rule_preds = (rule_probs[:, 1] >= 0.5).astype(int)
                else:
                    rule_preds = np.argmax(rule_probs, axis=1)
            else:
                rule_preds = rule_model.predict(X_target)
                if rule_preds.ndim > 1:
                    rule_preds = np.argmax(rule_preds, axis=1)
            fidelity_value = float(np.mean(rule_preds == nn_preds))
        except Exception as exc:
            console.print(f"[yellow]Transfer fidelity unavailable: {exc}[/yellow]")
            fidelity_value = None

    memory_points = 0
    if extra_memory and extra_memory[0] is not None:
        memory_points = int(extra_memory[0].shape[0])

    return {
        "from_task": current_task.name,
        "to_task": target_spec.name,
        "boot_metric": boot_metric,
        "boot_metric_label": metric_label,
        "boot_fidelity": fidelity_value,
        "memory_points": memory_points,
        "target_kind": target_spec.kind,
    }


def append_transfer_metrics(
    metrics_csv: Path,
    headers: list[str],
    current_task: TaskSpec,
    transfer_result: dict,
    generation: int,
    transfer_gain: float | None,
    next_task: str | None,
) -> None:
    """Append a transfer evaluation row to the metrics CSV."""

    row_template = {field: "" for field in headers}
    row_template.update({
        "generation": generation,
        "task_name": current_task.name,
        "from_task": transfer_result.get("from_task", ""),
        "to_task": transfer_result.get("to_task", ""),
        "boot_metric": round(transfer_result.get("boot_metric", float("nan")), 6)
        if transfer_result.get("boot_metric") is not None
        else "",
        "boot_metric_label": transfer_result.get("boot_metric_label", ""),
        "transfer_gain": round(transfer_gain, 6) if transfer_gain is not None else "",
        "next_task": next_task or "",
    })

    fidelity_val = transfer_result.get("boot_fidelity")
    if fidelity_val is not None:
        row_template["boot_fidelity"] = round(fidelity_val, 6)

    need_header = not metrics_csv.exists() or metrics_csv.stat().st_size == 0
    with metrics_csv.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=headers)
        if need_header:
            writer.writeheader()
        writer.writerow({field: row_template.get(field, "") for field in headers})


def coevolve(
    cfg_path: Path,
    generations: int = 5,
    watch: bool = True,
    seed: int | None = None,
    resume: bool = False,
    task_name: str | None = None,
    transfer_to: str | None = None,
    auto_curriculum: bool = False,
    auto_stop: int | None = None,
    meta_enabled: bool = False,
    dialogue_every: int | None = None,
    auto_experiment: bool = False,
    use_memory: bool = False,
    override_learning_rate: float | None = None,
    override_mutation_rate: float | None = None,
    override_lambda_consistency: float | None = None,
    arch_auto: bool = False,
    arch_population: int = 3,
    arch_interval: int = 3,
    arch_reset: bool = False,
    arch_load: Path | None = None,
    arch_locks: tuple[str, ...] | None = None,
    label_noise: float = 0.0,
    feature_noise: float = 0.0,
    target_noise: float = 0.0,
    label_smoothing: float = 0.0,
    eval_with_dropout: bool = False,
    no_rule_updates: bool = False,
    no_nn_train: bool = False,
    no_explore: bool = False,
    no_consistency: bool = False,
    robustness_sweep: bool = False,
    meta_eval_enabled: bool = False,
    meta_model: str | None = None,
    promote_threshold: float = 0.7,
):
    """Alternates between NN and rule training, recording metrics each generation."""

    if dialogue_every is not None and dialogue_every <= 0:
        dialogue_every = None

    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    project_dir = Path(__file__).parent
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    results_csv = project_dir / "results.csv"
    metrics_csv = project_dir / "coevolve_metrics.csv"
    log_path = project_dir / "log.md"
    run_meta_path = project_dir / "run_meta.json"
    tracker = CausalTracker(project_dir)
    run_meta: dict[str, object] = {}
    try:
        previous_run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}
    except Exception:
        previous_run_meta = {}
    if isinstance(previous_run_meta, dict):
        run_meta.update(previous_run_meta)

    default_params = {
        "learning_rate": 0.001,
        "rule_mutation_rate": 0.2,
        "lambda_consistency": cfg.get("lambda_consistency", 0.2),
    }
    cfg_params = cfg.get("evolvable_params", {}) or {}
    evolvable_params = {**default_params, **cfg_params}

    if override_learning_rate is not None:
        evolvable_params["learning_rate"] = float(override_learning_rate)
    if override_mutation_rate is not None:
        evolvable_params["rule_mutation_rate"] = float(override_mutation_rate)
    if override_lambda_consistency is not None:
        evolvable_params["lambda_consistency"] = float(override_lambda_consistency)
    param_bounds = {
        "learning_rate": (1e-5, 0.01),
        "rule_mutation_rate": (0.05, 0.5),
        "lambda_consistency": (0.05, 0.5),
    }
    sigma = 0.1
    prev_meta_fitness: float | None = None
    lambda_consistency = evolvable_params.get("lambda_consistency", 0.2)
    params_path = project_dir / "params.yaml"
    write_header = not results_csv.exists()

    metrics_headers = [
        "generation",
        "task_name",
        "nn_accuracy",
        "rule_fidelity",
        "nn_primary",
        "rule_primary",
        "combined_fitness",
        "nn_learning_gain",
        "rule_adaptation_gain",
        "nn_primary_delta",
        "rule_primary_delta",
        "rule_mutations",
        "consistency_loss",
        "avg_disagreement",
        "n_new_samples",
        "rule_agreement_rate",
        "lambda_consistency",
        "meta_fitness",
        "learning_rate",
        "rule_mutation_rate",
        "nn_primary_label",
        "rule_primary_label",
        "memory_embeddings_used",
        "memory_rules_reused",
        "memory_boost",
        "meta_learning_rate",
        "meta_mutation_rate",
        "meta_lambda_consistency",
        "meta_reward",
    "label_noise",
    "feature_noise",
    "target_noise",
    "label_smoothing",
    "eval_with_dropout",
    "abl_no_rule",
    "abl_no_nn",
    "abl_no_explore",
    "abl_no_consistency",
    "is_probe",
        "arch_hidden_layers",
        "arch_hidden_width",
        "arch_activation",
        "arch_normalization",
        "arch_dropout",
        "arch_max_depth",
        "arch_min_samples_leaf",
        "arch_prune_alpha",
        "arch_uncertainty_k",
        "arch_mutation_rate",
        "arch_lambda_consistency",
        "model_param_count",
        "composite_score",
        "transfer_gain",
        "next_task",
        "from_task",
        "to_task",
        "boot_metric",
        "boot_metric_label",
        "boot_fidelity",
    ]

    metrics_exists = False
    existing_metric_rows: list[dict] = []
    if metrics_csv.exists():
        try:
            with metrics_csv.open("r", encoding="utf-8") as metrics_in:
                existing_metric_rows = list(csv.DictReader(metrics_in))
        except Exception:
            existing_metric_rows = []

        needs_rewrite = True
        if existing_metric_rows:
            first_row = existing_metric_rows[0]
            if all(field in first_row for field in metrics_headers):
                needs_rewrite = False

        if needs_rewrite:
            with metrics_csv.open("w", newline="", encoding="utf-8") as metrics_out:
                writer = csv.DictWriter(metrics_out, fieldnames=metrics_headers)
                writer.writeheader()
                for row in existing_metric_rows:
                    writer.writerow({field: row.get(field, "") for field in metrics_headers})
            metrics_exists = True
        else:
            metrics_exists = True

    memory_bank_path = project_dir / "memory_bank.json"
    memory_bank = MemoryBank(memory_bank_path)
    curriculum = CurriculumManager(project_dir / "curriculum.json")
    meta_params_path = project_dir / "meta_params.json"
    meta_controller = MetaController(metrics_csv) if meta_enabled else None
    if meta_controller is not None and not META_HAS_PANDAS:
        console.print("[yellow]Meta-controller disabled: pandas is not installed.[/yellow]")
        meta_controller = None
        meta_enabled = False

    extra_X = None
    extra_y = None
    history: list[dict] = []
    prev_nn_primary = None
    prev_rule_primary = None
    current_model_state = None
    last_transfer_norm = 0.0

    start_gen = 1
    restored_rng = False
    latest_checkpoint_name = None
    checkpoint_task = None

    if resume:
        ckpts = sorted(checkpoint_dir.glob("checkpoint_gen_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            latest_checkpoint_name = latest.name
            # Torch 2.6 tightened defaults; our checkpoints include trusted numpy payloads.
            data = torch.load(latest, map_location="cpu", weights_only=False)
            current_model_state = data.get("model_state")
            extra_X = data.get("extra_X")
            extra_y = data.get("extra_y")
            history = data.get("history", []) or []
            lambda_consistency = data.get("lambda_consistency", lambda_consistency)
            evolvable_params = data.get("evolvable_params", evolvable_params)
            sigma = data.get("sigma", sigma)
            prev_meta_fitness = data.get("prev_meta_fitness")
            lambda_consistency = evolvable_params.get("lambda_consistency", lambda_consistency)
            prev_nn_primary = data.get("prev_nn_primary", data.get("prev_nn_acc"))
            prev_rule_primary = data.get("prev_rule_primary", data.get("prev_rule_fidelity"))
            start_gen = data.get("generation", 0) + 1
            stored_seed = data.get("seed")
            checkpoint_task = data.get("task")
            if seed is None and stored_seed is not None:
                seed = stored_seed

            random_state = data.get("random_state")
            numpy_state = data.get("numpy_state")
            torch_state = data.get("torch_state")
            if random_state is not None and numpy_state is not None and torch_state is not None:
                try:
                    random.setstate(random_state)
                    np.random.set_state(numpy_state)
                    torch.set_rng_state(torch_state)
                    restored_rng = True
                except Exception:
                    console.print("[yellow]Could not restore RNG state from checkpoint.[/yellow]")

            console.print(
                f"[cyan]Resuming from {latest.name} (gen={start_gen}, seed={seed})[/cyan]"
            )
        else:
            console.print("[yellow]No checkpoints found; starting fresh.[/yellow]")
            resume = False

    if restored_rng:
        if seed is None:
            seed = int(datetime.now().timestamp()) % 10_000_000
            console.print(
                f"[yellow]Checkpoint missing seed; continuing with derived seed {seed}[/yellow]"
            )
        else:
            console.print(f"[gray]Using global seed {seed}[/gray]")
    else:
        seed = set_global_seed(seed)

    if history and (prev_nn_primary is None or prev_rule_primary is None):
        last_entry = history[-1]
        prev_nn_primary = last_entry.get("nn_primary", last_entry.get("acc"))
        prev_rule_primary = last_entry.get("rule_primary", last_entry.get("fidelity"))

    effective_task_name = task_name or checkpoint_task
    if checkpoint_task and task_name and checkpoint_task != task_name:
        console.print(
            f"[yellow]Checkpoint task '{checkpoint_task}' differs from requested '{task_name}'. Using requested task.[/yellow]"
        )
    task_spec = apply_task(effective_task_name)
    console.print(f"[gray]Task dataset selected: {task_spec.name}[/gray]")

    arch_current_path = project_dir / "arch_current.json"
    arch_best_path = project_dir / f"arch_best_{task_spec.name}.json"
    current_genome, current_arch_score, best_arch_score = _initial_architecture(
        project_dir,
        task_spec.name,
        reset=arch_reset,
        override_path=arch_load,
    )
    if arch_load is not None:
        console.print(f"[cyan]Loaded architecture genome from {arch_load}[/cyan]")
    elif arch_reset:
        console.print("[cyan]Architecture genome reset to defaults.[/cyan]")
    elif current_arch_score is not None:
        console.print(
            f"[gray]Restored previous architecture (composite={current_arch_score:.3f}).[/gray]"
        )
    elif best_arch_score is not None:
        console.print(
            f"[gray]Starting from best recorded architecture (composite={best_arch_score:.3f}).[/gray]"
        )

    if arch_auto or arch_load is not None or arch_reset:
        _write_arch_file(arch_current_path, current_genome, task=task_spec.name, composite_score=current_arch_score)

    noise_config = {
        "label_noise": max(0.0, float(label_noise)),
        "feature_noise": max(0.0, float(feature_noise)),
        "target_noise": max(0.0, float(target_noise)),
    }

    ablation_flags = {
        "no_rule": bool(no_rule_updates),
        "no_nn": bool(no_nn_train),
        "no_explore": bool(no_explore),
        "no_consistency": bool(no_consistency),
    }
    ablation_names = [name for name, active in ablation_flags.items() if active]
    if ablation_names:
        ablation_banner = "RUN_ABLATIONS: " + ", ".join(ablation_names)
        console.print(f"[yellow]{ablation_banner}[/yellow]")
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("\n" + ablation_banner + "\n")
        except Exception:
            pass

    run_meta.update({
        "label_noise": noise_config["label_noise"],
        "feature_noise": noise_config["feature_noise"],
        "target_noise": noise_config["target_noise"],
        "label_smoothing": float(label_smoothing),
        "eval_with_dropout": bool(eval_with_dropout),
        "ablations": ablation_names,
    })

    previous_ablations = set(previous_run_meta.get("ablations", [])) if isinstance(previous_run_meta, dict) else set()

    def _coerce_iv_value(value: object) -> float | str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return str(value)

    def _ablation_bits() -> dict[str, int]:
        return {name: int(bool(ablation_flags.get(name, False))) for name in ablation_flags}

    def _intervention_context(gen_idx: int) -> dict[str, object]:
        return {
            "task": task_spec.name,
            "seed": seed,
            "abl": _ablation_bits(),
            "arch": current_genome.to_dict() if current_genome else None,
            "generation": gen_idx,
        }

    def _record_intervention(kind: str, old_val: object, new_val: object, gen_idx: int) -> None:
        coerced_old = _coerce_iv_value(old_val)
        coerced_new = _coerce_iv_value(new_val)
        if coerced_old == coerced_new:
            return
        tracker.stamp(
            gen=gen_idx,
            kind=kind,
            old=coerced_old,
            new=coerced_new,
            context=_intervention_context(gen_idx),
        )

    pending_pre_run_interventions: list[tuple[str, object, object]] = []
    noise_kind_map = {
        "label_noise": "noise.label",
        "feature_noise": "noise.feature",
        "target_noise": "noise.target",
    }
    for key, kind in noise_kind_map.items():
        new_val = noise_config.get(key)
        old_val = previous_run_meta.get(key) if isinstance(previous_run_meta, dict) else None
        if _coerce_iv_value(old_val) != _coerce_iv_value(new_val):
            pending_pre_run_interventions.append((kind, old_val, new_val))

    flag_kind_map = {
        "no_rule": "flag.no_rule_updates",
        "no_nn": "flag.no_nn_train",
        "no_explore": "flag.no_explore",
        "no_consistency": "flag.no_consistency",
    }
    for name, kind in flag_kind_map.items():
        new_state = bool(ablation_flags.get(name, False))
        old_state = name in previous_ablations
        if _coerce_iv_value(old_state) != _coerce_iv_value(new_state):
            pending_pre_run_interventions.append((kind, old_state, new_state))

    for kind, old_val, new_val in pending_pre_run_interventions:
        _record_intervention(kind, old_val, new_val, start_gen)

    lock_candidates = arch_locks or ()
    arch_lock_set = {
        lock.strip().lower()
        for lock in lock_candidates
        if isinstance(lock, str)
    }
    valid_lock_domains = {"nn", "rules", "explore"}
    arch_lock_set = {lock for lock in arch_lock_set if lock in valid_lock_domains}
    if arch_lock_set:
        console.print(
            "[gray]Architecture mutation lock active for: {domains}[/gray]".format(
                domains=", ".join(sorted(arch_lock_set))
            )
        )

    arch_auto_effective = arch_auto and not ablation_names
    if arch_auto and ablation_names:
        console.print(
            "[yellow]Architecture evolution paused during ablation run; selection will resume when ablations are disabled.[/yellow]"
        )

    apply_arch_hparams = any([
        arch_auto_effective,
        arch_load is not None,
        arch_reset,
        current_arch_score is not None,
        best_arch_score is not None,
        bool(arch_lock_set),
        arch_current_path.exists(),
    ])

    def data_generator(
        n_points: int = 1000,
        seed_value: int | None = None,
        *,
        return_clean: bool = False,
        **overrides,
    ):
        overrides.pop("return_clean", None)
        params = {**noise_config, **overrides}
        return task_spec.generator(n_points, seed_value, return_clean=return_clean, **params)

    def _coerce_dataset(payload: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(payload) == 4:
            return (
                np.asarray(payload[0], dtype=np.float32),
                np.asarray(payload[1]),
                np.asarray(payload[2], dtype=np.float32),
                np.asarray(payload[3]),
            )
        X_raw = np.asarray(payload[0], dtype=np.float32)
        y_raw = np.asarray(payload[1])
        return X_raw, y_raw, X_raw.copy(), y_raw.copy()

    def _evaluate_rule_fidelity(rule_estimator, model, *, seed_offset: int = 0) -> float:
        if rule_estimator is None or model is None:
            return float("nan")
        eval_payload = data_generator(
            int(cfg.get("points", 1000)),
            seed_value=(seed or 0) + seed_offset,
            return_clean=True,
            label_noise=0.0,
            target_noise=0.0,
        )
        _, _, X_clean, _ = _coerce_dataset(eval_payload)
        if X_clean.size == 0:
            return float("nan")
        with torch.no_grad():
            logits_eval = model(torch.tensor(X_clean, dtype=torch.float32))
            nn_eval_targets = _nn_targets_from_logits(logits_eval, task_spec)
        try:
            fidelity_val = float(rule_estimator.score(X_clean, nn_eval_targets))
        except Exception:
            try:
                preds = rule_estimator.predict(X_clean)
                preds = np.asarray(preds)
                if preds.ndim > 1:
                    preds = preds.argmax(axis=1)
                fidelity_val = float(np.mean(preds == nn_eval_targets))
            except Exception:
                fidelity_val = float("nan")
        return fidelity_val

    memory_pool_size = int(cfg.get("memory_pool_size", 256))
    memory_rules_top_n = int(cfg.get("memory_rules_top_n", 5))
    memory_dimension_warning_logged = False

    probe_points = max(16, min(memory_pool_size, int(cfg.get("points", 1000))))
    try:
        probe_X, _ = data_generator(probe_points, seed_value=(seed or 0) + 17)
    except Exception:
        probe_X = None

    feature_dim = probe_X.shape[1] if probe_X is not None and len(probe_X) else None

    memory_X: np.ndarray | None = None
    memory_y: np.ndarray | None = None
    if feature_dim is not None:
        sampled_embeddings, sampled_labels = memory_bank.sample_pairs(
            k=memory_pool_size,
            feature_dim=feature_dim,
        )
        if sampled_embeddings.size:
            memory_X = sampled_embeddings
            if sampled_labels is not None and sampled_labels.size:
                labels_arr = np.asarray(sampled_labels)
                labels_arr = labels_arr.reshape(-1)
                if np.allclose(labels_arr, np.round(labels_arr)):
                    labels_arr = np.round(labels_arr).astype(np.int64)
                else:
                    labels_arr = labels_arr.astype(np.float32)
                if labels_arr.shape[0] != memory_X.shape[0]:
                    labels_arr = labels_arr[: memory_X.shape[0]]
                memory_y = labels_arr
    memory_rules = memory_bank.get_rules(top_n=memory_rules_top_n)

    health_summary_available = evaluate_health_metrics is not None
    health_summary_warned = False

    run_meta.update({
        "seed": seed,
        "resume": resume,
        "start_generation": start_gen,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task_spec.name,
        "transfer_to": transfer_to,
        "meta_enabled": meta_enabled,
        "dialogue_every": dialogue_every,
        "auto_experiment": auto_experiment,
        "override_learning_rate": override_learning_rate,
        "override_mutation_rate": override_mutation_rate,
        "override_lambda_consistency": override_lambda_consistency,
        "arch_auto": arch_auto,
        "arch_population": arch_population,
        "arch_interval": arch_interval,
        "arch_locks": sorted(arch_lock_set),
    })
    if latest_checkpoint_name:
        run_meta["resumed_from"] = latest_checkpoint_name
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    monitor_thread = None
    stop_file = project_dir / "stop_health_watch.txt"
    dialogue_counter_path = project_dir / "dialogue_count.json"
    try:
        if dialogue_counter_path.exists():
            dialogue_counter = json.loads(dialogue_counter_path.read_text(encoding="utf-8")).get("count", 0)
        else:
            dialogue_counter = 0
    except Exception:
        dialogue_counter = 0

    if watch and update_live_health is None:
        console.print("[yellow]Health dashboard module not found — skipping live monitoring[/yellow]")
        watch = False

    robustness_summary = None
    robustness_report_path = project_dir / "robustness_report.json"
    if metrics_csv.exists():
        try:
            try:
                import pandas as pd  # type: ignore

                metrics_source = pd.read_csv(metrics_csv)
            except ImportError:
                with metrics_csv.open("r", encoding="utf-8") as metrics_in:
                    metrics_source = list(csv.DictReader(metrics_in))
            robustness_summary = compute_robustness(metrics_source)
        except Exception as exc:
            console.print(f"[yellow]Robustness report unavailable: {exc}[/yellow]")
            robustness_summary = None

    if robustness_summary:
        report_text = (
            "== Robustness Report ==\n"
            f"Noise: acc_std={robustness_summary['acc_std_under_noise']:.3f}, "
            f"fid_std={robustness_summary['fid_std_under_noise']:.3f}, "
            f"drop@5%LN={robustness_summary['drop_at_noise_5pct']:.3f}\n"
            f"Ablations: Δ(no_rule)={robustness_summary['delta_with_no_rule']:.3f}, "
            f"Δ(no_nn)={robustness_summary['delta_with_no_nn']:.3f}, "
            f"Δ(no_explore)={robustness_summary['delta_with_no_explore']:.3f}\n"
            f"Stability score: {robustness_summary['stability_score']:.3f}"
        )
        console.print("\n" + report_text)
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("\n" + report_text + "\n")
        except Exception:
            pass

        report_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": task_spec.name,
            "generation": history[-1]["gen"] if history else generations,
            "label_noise": noise_config["label_noise"],
            "feature_noise": noise_config["feature_noise"],
            "target_noise": noise_config["target_noise"],
            "label_smoothing": label_smoothing,
            "eval_with_dropout": bool(eval_with_dropout),
            "ablations": ablation_names,
            **robustness_summary,
        }

        try:
            existing_entries: list[dict]
            if robustness_report_path.exists():
                existing_payload = json.loads(robustness_report_path.read_text(encoding="utf-8"))
                if isinstance(existing_payload, dict) and "reports" in existing_payload:
                    existing_entries = list(existing_payload.get("reports", []))
                elif isinstance(existing_payload, list):
                    existing_entries = list(existing_payload)
                else:
                    existing_entries = []
            else:
                existing_entries = []
        except Exception:
            existing_entries = []

        existing_entries.append(report_entry)
        robustness_payload = {"reports": existing_entries}
        try:
            robustness_report_path.write_text(json.dumps(robustness_payload, indent=2), encoding="utf-8")
        except Exception as exc:
            console.print(f"[yellow]Failed to persist robustness report: {exc}[/yellow]")

        run_meta.update({
            "robustness_summary": robustness_summary,
            "robustness_timestamp": report_entry["timestamp"],
        })
        run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if watch:
        if stop_file.exists():
            try:
                stop_file.unlink()
            except Exception:
                pass
        try:
            monitor_thread = threading.Thread(
                target=update_live_health,
                args=(metrics_csv, project_dir / "coevolve_health.png"),
                kwargs={"refresh_interval": 10},
                daemon=True,
            )
            monitor_thread.start()
            console.print("[cyan]🔍 Live health monitor started (refresh 10s)[/cyan]")
        except Exception as exc:
            console.print(f"[yellow]Failed to start live health monitor: {exc}[/yellow]")
            monitor_thread = None
            watch = False

    pending_generations = list(range(start_gen, generations + 1))
    if not pending_generations:
        console.print("[yellow]All requested generations already completed; nothing to do.[/yellow]")

    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["generation", "nn_accuracy", "rule_fidelity", "timestamp"],
        )
        if write_header:
            writer.writeheader()

        for gen in pending_generations:
            console.print(
                f"\n[bold cyan]=== Coevolution Step {gen}/{generations} ===[/bold cyan]"
            )

            if apply_arch_hparams:
                if override_mutation_rate is None:
                    evolvable_params["rule_mutation_rate"] = current_genome.explore.mutation_rate
                if override_lambda_consistency is None:
                    evolvable_params["lambda_consistency"] = current_genome.explore.lambda_consistency

            learning_rate = evolvable_params.get("learning_rate", 0.001)
            rule_mutation_rate = evolvable_params.get("rule_mutation_rate", 0.2)
            lambda_consistency = evolvable_params.get("lambda_consistency", lambda_consistency)
            current_lambda = lambda_consistency
            base_learning_rate = float(learning_rate)
            base_rule_mut_rate = float(rule_mutation_rate)
            base_lambda_consistency = float(lambda_consistency)

            meta_learning_rate_val: float | None = None
            meta_mutation_rate_val: float | None = None
            meta_lambda_val: float | None = None
            meta_reward_val: float | None = None

            if meta_controller is not None:
                meta_params = meta_controller.update()
                learning_rate = meta_params.get("learning_rate", learning_rate)
                rule_mutation_rate = meta_params.get("mutation_rate", rule_mutation_rate)
                lambda_consistency = meta_params.get("lambda_consistency", lambda_consistency)
                current_lambda = lambda_consistency

                meta_learning_rate_val = float(learning_rate)
                meta_mutation_rate_val = float(rule_mutation_rate)
                meta_lambda_val = float(lambda_consistency)

                console.print(
                    f"[cyan]Meta-adjusted: lr={learning_rate:.4f}, mut={rule_mutation_rate:.3f}, λ={lambda_consistency:.2f}[/cyan]"
                )

            evolvable_params["learning_rate"] = learning_rate
            evolvable_params["rule_mutation_rate"] = rule_mutation_rate
            evolvable_params["lambda_consistency"] = lambda_consistency

            if learning_rate != base_learning_rate:
                _record_intervention("meta.learning_rate", base_learning_rate, learning_rate, gen)
            if rule_mutation_rate != base_rule_mut_rate:
                _record_intervention("meta.rule_mutation_rate", base_rule_mut_rate, rule_mutation_rate, gen)
            if lambda_consistency != base_lambda_consistency:
                _record_intervention("meta.lambda_consistency", base_lambda_consistency, lambda_consistency, gen)

            if meta_learning_rate_val is None:
                meta_learning_rate_val = float(learning_rate)
            if meta_mutation_rate_val is None:
                meta_mutation_rate_val = float(rule_mutation_rate)
            if meta_lambda_val is None:
                meta_lambda_val = float(lambda_consistency)

            rule_model = None
            rule_model_path = project_dir / "rules_model.joblib"
            if rule_model_path.exists():
                try:
                    rule_model = load(rule_model_path)
                    console.print("[gray]Loaded rule engine for consistency loss.[/gray]")
                except Exception as exc:
                    console.print(f"[yellow]Failed to load rule engine: {exc}[/yellow]")

            combined_extra_X = extra_X
            combined_extra_y = extra_y
            memory_embeddings_used = 0

            if (
                memory_X is not None
                and memory_y is not None
                and getattr(memory_X, "size", 0)
                and getattr(memory_y, "size", 0)
            ):
                if combined_extra_X is None or combined_extra_y is None:
                    combined_extra_X = memory_X
                    combined_extra_y = memory_y
                    memory_embeddings_used = memory_X.shape[0]
                elif combined_extra_X.shape[1] == memory_X.shape[1]:
                    combined_extra_X = np.concatenate([memory_X, combined_extra_X], axis=0)
                    combined_extra_y = np.concatenate([memory_y, combined_extra_y], axis=0)
                    memory_embeddings_used = memory_X.shape[0]
                else:
                    if not memory_dimension_warning_logged:
                        console.print("[yellow]Memory embeddings skipped due to dimensionality mismatch.[/yellow]")
                        memory_dimension_warning_logged = True
            else:
                memory_embeddings_used = 0

            extra_data = (
                (combined_extra_X, combined_extra_y)
                if combined_extra_X is not None and combined_extra_y is not None
                else None
            )

            optimizer_state: dict = {}
            fine_tune_state: dict = {}
            model_param_count = 0
            supervised_loss = 0.0
            consistency_loss = 0.0

            if no_nn_train:
                eval_seed = (seed or 0) + gen * 37
                eval_payload = data_generator(
                    int(cfg.get("points", 1000)),
                    seed_value=eval_seed,
                    return_clean=True,
                    label_noise=0.0,
                    target_noise=0.0,
                )
                _, _, X_eval, y_eval = _coerce_dataset(eval_payload)
                input_dim_eval = X_eval.shape[1]
                model = _build_model(task_spec, input_dim_eval, current_genome.nn)
                expected_arch = asdict(current_genome.nn)
                if not safe_load_checkpoint(
                    model,
                    project_dir / "nn_model.pt",
                    model_name="TaskNN",
                    expected_type=model.__class__.__name__,
                    expected_arch=expected_arch,
                ):
                    raise FileNotFoundError("Neural network checkpoint nn_model.pt not found or incompatible.")
                model_param_count = model.parameter_count
                prev_state = model.training
                if eval_with_dropout:
                    model.train()
                else:
                    model.eval()
                with torch.no_grad():
                    logits_eval = model(torch.tensor(X_eval, dtype=torch.float32))
                    if task_spec.loss_type == "bce":
                        y_eval_t = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1)
                    else:
                        y_eval_t = torch.tensor(y_eval, dtype=torch.long)
                    nn_primary_metric = _compute_primary_metric(logits_eval, y_eval_t, task_spec)
                model.train(prev_state)
                console.print("[yellow]NN training skipped (no-nn ablation active).[/yellow]")
            else:
                (
                    model,
                    nn_primary_metric,
                    supervised_loss,
                    consistency_loss,
                    optimizer_state,
                    model_param_count,
                ) = train_nn(
                    cfg_path,
                    task_spec,
                    generation=gen,
                    rule_model=rule_model,
                    lambda_consistency=current_lambda,
                    learning_rate=learning_rate,
                    extra_data=extra_data,
                    initial_state=current_model_state,
                    arch=current_genome.nn,
                    label_noise=noise_config["label_noise"],
                    feature_noise=noise_config["feature_noise"],
                    target_noise=noise_config["target_noise"],
                    label_smoothing=label_smoothing,
                    eval_with_dropout=eval_with_dropout,
                )
                if no_consistency:
                    consistency_loss = 0.0
                model, fine_tune_state, _ = fine_tune_nn(model, task_spec, epochs=20, lr=0.003)

            if no_consistency:
                current_lambda = 0.0
                lambda_consistency = 0.0

            if no_rule_updates:
                rule_tree = rule_model
                rule_primary_metric = _evaluate_rule_fidelity(rule_tree, model, seed_offset=gen * 41)
                if math.isnan(rule_primary_metric) and rule_model is None:
                    rule_primary_metric = float("nan")
                rule_lines = memory_rules if memory_rules else []
                rules = "(rule updates disabled)"
                mutation_count = 0
                mutated_path = None
                console.print("[yellow]Rule updates skipped (no-rule ablation active).[/yellow]")
            else:
                rules, rule_primary_metric, rule_tree = extract_rules(
                    cfg_path,
                    task_spec,
                    nn_arch=current_genome.nn,
                    rule_arch=current_genome.rules,
                    label_noise=noise_config["label_noise"],
                    feature_noise=noise_config["feature_noise"],
                    target_noise=noise_config["target_noise"],
                )
                rule_lines = [line.strip() for line in rules.splitlines() if line.strip()]
                mutated_path, mutation_count = save_mutated_rules(
                    rules,
                    project_dir,
                    mutation_rate=rule_mutation_rate,
                    seed_rules=memory_rules,
                )
                print(f"[cyan]Mutated rule set saved to {mutated_path.name}[/cyan]")

            memory_rules_reused = len(memory_rules)

            writer.writerow({
                "generation": gen,
                "nn_accuracy": round(nn_primary_metric, 4),
                "rule_fidelity": round(rule_primary_metric, 4),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            f.flush()

            console.print(
                "[green]Logged:[/green] Gen={gen}, NN={nn:.3f} ({nn_lbl}), Rules={rule:.3f} ({rule_lbl})".format(
                    gen=gen,
                    nn=nn_primary_metric,
                    nn_lbl=task_spec.primary_metric_label,
                    rule=rule_primary_metric,
                    rule_lbl=task_spec.secondary_metric_label,
                )
            )

            effective_rule_metric = rule_primary_metric
            if isinstance(effective_rule_metric, float) and math.isnan(effective_rule_metric):
                effective_rule_metric = nn_primary_metric
            combined_fitness = (nn_primary_metric + effective_rule_metric) / 2
            nn_gain = (
                nn_primary_metric - prev_nn_primary if prev_nn_primary is not None else 0.0
            )
            if isinstance(rule_primary_metric, float) and math.isnan(rule_primary_metric):
                rule_gain = 0.0
            else:
                rule_gain = (
                    rule_primary_metric - prev_rule_primary if prev_rule_primary is not None else 0.0
                )

            if meta_controller is not None:
                meta_reward_val = (nn_gain + rule_gain) - abs(consistency_loss)
                meta_controller.last_reward = float(meta_reward_val)
                meta_controller.save(meta_params_path)

            baseline_nn_avg = memory_bank.average_metric(task_spec.name, "nn_primary")
            memory_boost = (
                nn_primary_metric - baseline_nn_avg
                if baseline_nn_avg is not None and memory_embeddings_used > 0
                else 0.0
            )

            memory_norm = memory_boost if isinstance(memory_boost, (int, float)) else 0.0
            current_composite = _composite_score(
                primary_metric=nn_primary_metric,
                task=task_spec,
                transfer_norm=last_transfer_norm,
                memory_norm=memory_norm,
                param_count=model_param_count,
            )
            current_arch_score = current_composite

            _write_arch_file(
                arch_current_path,
                current_genome,
                task=task_spec.name,
                composite_score=current_arch_score,
            )
            if best_arch_score is None or current_arch_score > best_arch_score:
                _write_arch_file(
                    arch_best_path,
                    current_genome,
                    task=task_spec.name,
                    composite_score=current_arch_score,
                )
                best_arch_score = current_arch_score

            pool_multiplier = int(cfg.get("uncertainty_pool_multiplier", 2))
            base_points = int(cfg.get("points", 1000))
            pool_size = max(base_points * pool_multiplier, base_points)
            X_pool, _ = data_generator(pool_size, seed_value=cfg.get("seed", 42) + gen)

            previous_training_state = model.training
            model.eval()
            with torch.no_grad():
                nn_logits = model(torch.tensor(X_pool, dtype=torch.float32))
                nn_probs = _logits_to_probs(nn_logits, task_spec).cpu().numpy()
            model.train(previous_training_state)

            if rule_tree is not None and hasattr(rule_tree, "predict_proba"):
                rule_probs_raw = rule_tree.predict_proba(X_pool)
                rule_probs = _align_rule_probabilities(rule_tree, rule_probs_raw, task_spec)
            elif rule_tree is not None and hasattr(rule_tree, "predict"):
                preds = np.asarray(rule_tree.predict(X_pool))
                if preds.ndim == 1 and nn_probs.ndim == 2:
                    rule_probs = np.zeros_like(nn_probs)
                    for cls_idx in range(rule_probs.shape[1]):
                        rule_probs[:, cls_idx] = (preds == cls_idx).astype(np.float32)
                else:
                    rule_probs = preds.astype(np.float32)
                    if rule_probs.ndim == 1:
                        rule_probs = rule_probs.reshape(-1, 1)
            else:
                rule_probs = np.zeros_like(nn_probs)

            disagreement = np.abs(rule_probs - nn_probs).mean(axis=1)

            avg_disagreement = float(np.mean(disagreement)) if disagreement.size else 0.0
            rule_agreement_rate = 1.0 - avg_disagreement

            effective_no_explore = no_explore or rule_tree is None
            if effective_no_explore:
                n_new_samples = 0
                X_new = np.empty((0, X_pool.shape[1]), dtype=np.float32)
                y_new = np.empty((0,), dtype=np.float32)
            else:
                if apply_arch_hparams:
                    n_new = max(1, min(len(X_pool), int(current_genome.explore.uncertainty_k)))
                else:
                    n_new = cfg.get("uncertainty_batch", max(1, base_points // 5))
                    n_new = max(1, min(len(X_pool), n_new))
                uncertain_idxs = np.argsort(disagreement)[-n_new:]
                n_new_samples = int(len(uncertain_idxs))

                X_new = X_pool[uncertain_idxs]
                y_new = np.asarray(rule_tree.predict(X_new))
                if y_new.ndim > 1:
                    y_new = y_new.reshape(-1)

                if extra_X is None:
                    extra_X = X_new
                    extra_y = y_new
                else:
                    extra_X = np.concatenate([extra_X, X_new], axis=0)
                    extra_y = np.concatenate([extra_y, y_new], axis=0)

            prev_nn_primary = nn_primary_metric
            prev_rule_primary = rule_primary_metric

            current_fitness = round(
                0.5 * nn_primary_metric + 0.5 * rule_primary_metric - consistency_loss,
                6,
            )
            evolvable_params["fitness"] = current_fitness

            if prev_meta_fitness is not None:
                if current_fitness > prev_meta_fitness:
                    sigma *= 0.9
                else:
                    sigma *= 1.1
                sigma = min(max(sigma, 0.02), 0.3)
            prev_meta_fitness = current_fitness

            with metrics_csv.open("a", newline="", encoding="utf-8") as metrics_file:
                metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_headers)
                if not metrics_exists:
                    metrics_writer.writeheader()
                    metrics_exists = True
                metrics_writer.writerow({
                    "generation": gen,
                    "task_name": task_spec.name,
                    "nn_accuracy": round(nn_primary_metric, 6),
                    "rule_fidelity": round(rule_primary_metric, 6),
                    "nn_primary": round(nn_primary_metric, 6),
                    "rule_primary": round(rule_primary_metric, 6),
                    "combined_fitness": round(combined_fitness, 6),
                    "nn_learning_gain": round(nn_gain, 6),
                    "rule_adaptation_gain": round(rule_gain, 6),
                    "nn_primary_delta": round(nn_gain, 6),
                    "rule_primary_delta": round(rule_gain, 6),
                    "rule_mutations": mutation_count,
                    "consistency_loss": round(consistency_loss, 6),
                    "avg_disagreement": round(avg_disagreement, 6),
                    "n_new_samples": n_new_samples,
                    "rule_agreement_rate": round(rule_agreement_rate, 6),
                    "lambda_consistency": round(current_lambda, 6),
                    "meta_fitness": current_fitness,
                    "learning_rate": learning_rate,
                    "rule_mutation_rate": rule_mutation_rate,
                    "nn_primary_label": task_spec.primary_metric_label,
                    "rule_primary_label": task_spec.secondary_metric_label,
                    "memory_embeddings_used": memory_embeddings_used,
                    "memory_rules_reused": memory_rules_reused,
                    "memory_boost": round(memory_boost, 6),
                    "meta_learning_rate": round(meta_learning_rate_val, 6) if meta_learning_rate_val is not None else "",
                    "meta_mutation_rate": round(meta_mutation_rate_val, 6) if meta_mutation_rate_val is not None else "",
                    "meta_lambda_consistency": round(meta_lambda_val, 6) if meta_lambda_val is not None else "",
                    "meta_reward": round(meta_reward_val, 6) if meta_reward_val is not None else "",
                    "label_noise": round(noise_config["label_noise"], 6),
                    "feature_noise": round(noise_config["feature_noise"], 6),
                    "target_noise": round(noise_config["target_noise"], 6),
                    "label_smoothing": round(label_smoothing, 6),
                    "eval_with_dropout": int(bool(eval_with_dropout)),
                    "abl_no_rule": int(no_rule_updates),
                    "abl_no_nn": int(no_nn_train),
                    "abl_no_explore": int(no_explore),
                    "abl_no_consistency": int(no_consistency),
                    "is_probe": 0,
                    "arch_hidden_layers": current_genome.nn.hidden_layers,
                    "arch_hidden_width": current_genome.nn.hidden_width,
                    "arch_activation": current_genome.nn.activation,
                    "arch_normalization": current_genome.nn.normalization,
                    "arch_dropout": round(current_genome.nn.dropout, 4),
                    "arch_max_depth": current_genome.rules.max_depth,
                    "arch_min_samples_leaf": current_genome.rules.min_samples_leaf,
                    "arch_prune_alpha": round(current_genome.rules.prune_ccp_alpha, 6),
                    "arch_uncertainty_k": current_genome.explore.uncertainty_k,
                    "arch_mutation_rate": round(current_genome.explore.mutation_rate, 6),
                    "arch_lambda_consistency": round(current_genome.explore.lambda_consistency, 6),
                    "model_param_count": model_param_count,
                    "composite_score": round(current_composite, 6),
                    "transfer_gain": "",
                    "next_task": "",
                })

            if not effective_no_explore and len(X_new):
                try:
                    memory_bank.add(
                        task_spec.name,
                        embeddings=X_new,
                        rules=rule_lines,
                        meta={
                            "generation": gen,
                            "nn_primary": float(nn_primary_metric),
                            "rule_primary": float(rule_primary_metric),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        },
                        labels=y_new,
                    )
                except Exception as exc:
                    console.print(f"[yellow]Failed to update memory bank: {exc}[/yellow]")
                else:
                    if memory_X is None:
                        memory_X = X_new.copy()
                        memory_y = y_new.copy() if y_new is not None else None
                    else:
                        memory_X = np.concatenate([memory_X, X_new], axis=0)
                        if memory_y is not None and y_new is not None:
                            memory_y = np.concatenate([memory_y, y_new], axis=0)
                        elif y_new is not None:
                            memory_y = y_new.copy()

                    if memory_X is not None and memory_pool_size and memory_X.shape[0] > memory_pool_size:
                        kept_idx = np.random.choice(memory_X.shape[0], memory_pool_size, replace=False)
                        memory_X = memory_X[kept_idx]
                        if memory_y is not None and memory_y.shape[0] >= len(kept_idx):
                            memory_y = memory_y[kept_idx]

                    memory_rules = memory_bank.get_rules(top_n=memory_rules_top_n)

            if arch_auto_effective and arch_population > 0 and (arch_interval > 0 and gen % arch_interval == 0):
                try:
                    probe_points = int(cfg.get("points", 1000))
                    probe_seed = (seed or 0) + gen * 37
                    probe_data = data_generator(probe_points, seed_value=probe_seed)
                    probe_X = np.asarray(probe_data[0], dtype=np.float32)
                    probe_y = np.asarray(probe_data[1])
                    if probe_y.ndim > 1:
                        probe_y = probe_y.reshape(-1)
                except Exception as exc:
                    console.print(f"[yellow]Architecture probe skipped: {exc}[/yellow]")
                else:
                    probe_rng = random.Random((seed or 0) + gen * 101)
                    candidates: list[dict] = []
                    for idx in range(max(arch_population, 0)):
                        candidate_changes: list[tuple[str, object, object]] = []
                        candidate_genome = mutate_arch(
                            current_genome,
                            rng=probe_rng,
                            locks=arch_lock_set,
                            change_log=candidate_changes,
                        )
                        result = _probe_architecture(
                            candidate_genome,
                            cfg_path=cfg_path,
                            task=task_spec,
                            data=(probe_X, probe_y),
                            baseline_memory=baseline_nn_avg,
                            transfer_norm=last_transfer_norm,
                            rng=probe_rng,
                            learning_rate=learning_rate,
                        )
                        if result is not None:
                            result["change_log"] = candidate_changes
                            candidates.append(result)

                    if candidates:
                        best_candidate = max(
                            candidates,
                            key=lambda entry: (
                                entry.get("score", float("-inf")),
                                -entry.get("param_count", 0),
                            ),
                        )
                        candidate_score = best_candidate.get("score", float("-inf"))
                        if candidate_score > current_composite + 1e-4:
                            adopted_genome: ArchGenome = best_candidate["genome"]  # type: ignore[index]
                            change_log = best_candidate.get("change_log", []) or []
                            current_genome = adopted_genome
                            current_arch_score = candidate_score
                            _write_arch_file(
                                arch_current_path,
                                current_genome,
                                task=task_spec.name,
                                composite_score=current_arch_score,
                            )
                            for kind, old_val, new_val in change_log:
                                _record_intervention(str(kind), old_val, new_val, gen)
                            if best_arch_score is None or current_arch_score > best_arch_score:
                                _write_arch_file(
                                    arch_best_path,
                                    current_genome,
                                    task=task_spec.name,
                                    composite_score=current_arch_score,
                                )
                                best_arch_score = current_arch_score
                            console.print(
                                "[blue][arch] Adopted {desc} (composite={score:.3f}).[/blue]".format(
                                    desc=_describe_genome(current_genome),
                                    score=current_arch_score,
                                )
                            )
                            current_composite = current_arch_score
                        else:
                            console.print(
                                "[gray][arch] Baseline genome retained (best probe={best:.3f}, current={curr:.3f}).[/gray]".format(
                                    best=candidate_score,
                                    curr=current_composite,
                                )
                            )

            if health_summary_available:
                try:
                    health_snapshot = evaluate_health_metrics(metrics_csv, window=20)
                    summary_text = health_snapshot.get("summary", "No summary available")
                    console_line = health_snapshot.get("console_line")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with log_path.open("a", encoding="utf-8") as log_file:
                        log_file.write(f"\n## Health Summary — {timestamp}\n")
                        log_file.write(summary_text + "\n")
                    if console_line:
                        console.print(f"[magenta]{console_line}[/magenta]")
                except Exception as exc:
                    console.print(f"[yellow]Failed to record health summary: {exc}[/yellow]")
            elif not health_summary_warned:
                console.print("[yellow]Health summary unavailable; skipping log update.[/yellow]")
                health_summary_warned = True

            if not no_consistency and len(history) >= 3:
                old_lambda_local = float(lambda_consistency)
                recent_disagreements = [entry["disagreement"] for entry in history[-3:]]
                recent_mean = float(np.mean(recent_disagreements))
                if avg_disagreement < recent_mean:
                    updated_lambda = max(0.05, lambda_consistency * 0.9)
                    if updated_lambda != lambda_consistency:
                        console.print(
                            f"[blue]Consistency weight decreased: {lambda_consistency:.3f} → {updated_lambda:.3f}[/blue]"
                        )
                        _record_intervention(
                            "meta.lambda_consistency",
                            old_lambda_local,
                            updated_lambda,
                            gen,
                        )
                    lambda_consistency = updated_lambda
                else:
                    updated_lambda = min(0.5, lambda_consistency * 1.1)
                    if updated_lambda != lambda_consistency:
                        console.print(
                            f"[red]Consistency weight increased: {lambda_consistency:.3f} → {updated_lambda:.3f}[/red]"
                        )
                        _record_intervention(
                            "meta.lambda_consistency",
                            old_lambda_local,
                            updated_lambda,
                            gen,
                        )
                    lambda_consistency = updated_lambda

            lambda_consistency = round(lambda_consistency, 6)
            evolvable_params["lambda_consistency"] = lambda_consistency

            history.append({
                "gen": gen,
                "nn_primary": float(nn_primary_metric),
                "rule_primary": float(rule_primary_metric),
                "acc": float(nn_primary_metric),
                "fidelity": float(rule_primary_metric),
                "disagreement": avg_disagreement,
                "lambda": float(lambda_consistency),
            })

            print(
                "[Gen {gen}] NN={nn:.3f} ({nn_lbl}), Rule={rule:.3f} ({rule_lbl}), "
                "Disagree={disagree:.3f}, λ={lam:.2f}".format(
                    gen=gen,
                    nn=nn_primary_metric,
                    nn_lbl=task_spec.primary_metric_label,
                    rule=rule_primary_metric,
                    rule_lbl=task_spec.secondary_metric_label,
                    disagree=avg_disagreement,
                    lam=lambda_consistency,
                )
            )

            from .visualize_rules import visualize_boundaries
            try:
                vis_path = visualize_boundaries(project_dir, gen, task_spec)
            except FileNotFoundError as exc:
                console.print(f"[yellow]Visualization skipped: {exc}[/yellow]")
                vis_path = None
            if vis_path is not None:
                print(f"[green]Preview saved to {vis_path.name}[/green]")

            params_subset = {key: evolvable_params[key] for key in param_bounds if key in evolvable_params}
            mutated_params = mutate_params(params_subset, param_bounds, sigma=sigma)
            evolvable_params.update(mutated_params)
            for key, new_val in mutated_params.items():
                old_val = params_subset.get(key)
                if key == "learning_rate":
                    _record_intervention("meta.learning_rate", old_val, new_val, gen)
                elif key == "rule_mutation_rate":
                    _record_intervention("meta.rule_mutation_rate", old_val, new_val, gen)
                elif key == "lambda_consistency":
                    _record_intervention("meta.lambda_consistency", old_val, new_val, gen)
            lambda_consistency = evolvable_params["lambda_consistency"]

            with params_path.open("w", encoding="utf-8") as params_file:
                yaml.safe_dump(evolvable_params, params_file, sort_keys=True)

            console.print(
                "[blue]Generation {gen} complete. Meta-params → lr = {lr:.4f}  "
                "rule_mut = {rm:.2f}  λ = {lc:.2f}  (meta_fitness = {mf:.2f})[/blue]".format(
                    gen=gen,
                    lr=evolvable_params["learning_rate"],
                    rm=evolvable_params["rule_mutation_rate"],
                    lc=evolvable_params["lambda_consistency"],
                    mf=current_fitness,
                )
            )

            current_model_state = copy.deepcopy(model.state_dict())

            checkpoint_payload = {
                "generation": gen,
                "model_state": current_model_state,
                "optimizer_state": optimizer_state,
                "fine_tune_state": fine_tune_state,
                "seed": seed,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "lambda_consistency": lambda_consistency,
                "extra_X": extra_X,
                "extra_y": extra_y,
                "history": history,
                "prev_nn_primary": prev_nn_primary,
                "prev_rule_primary": prev_rule_primary,
                "prev_nn_acc": prev_nn_primary,
                "prev_rule_fidelity": prev_rule_primary,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "evolvable_params": copy.deepcopy(evolvable_params),
                "sigma": sigma,
                "prev_meta_fitness": prev_meta_fitness,
                "task": task_spec.name,
                "architecture": current_genome.to_dict() if current_genome else None,
                "architecture_score": current_arch_score,
                "arch_locks": sorted(arch_lock_set),
            }

            checkpoint_path = checkpoint_dir / f"checkpoint_gen_{gen:03d}.pt"
            torch.save(checkpoint_payload, checkpoint_path)
            console.print(f"[gray]Checkpoint saved → {checkpoint_path.name}[/gray]")

            run_meta.update({
                "last_generation": gen,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "last_checkpoint": checkpoint_path.name,
                "meta_learning_rate": meta_learning_rate_val,
                "meta_mutation_rate": meta_mutation_rate_val,
                "meta_lambda_consistency": meta_lambda_val,
                "meta_reward": meta_reward_val,
                "architecture": current_genome.to_dict(),
                "architecture_score": current_arch_score,
                "arch_locks": sorted(arch_lock_set),
            })
            run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    final_model = model if "model" in locals() else None  # type: ignore[name-defined]
    final_rule_model = rule_tree if "rule_tree" in locals() else None  # type: ignore[name-defined]

    base_metric = prev_nn_primary if prev_nn_primary is not None else 0.0

    transfer_extra_X = extra_X
    transfer_extra_y = extra_y
    if (
        memory_X is not None
        and memory_y is not None
        and getattr(memory_X, "size", 0)
        and getattr(memory_y, "size", 0)
    ):
        if transfer_extra_X is None or transfer_extra_y is None:
            transfer_extra_X = memory_X
            transfer_extra_y = memory_y
        elif transfer_extra_X.shape[1] == memory_X.shape[1]:
            transfer_extra_X = np.concatenate([memory_X, transfer_extra_X], axis=0)
            transfer_extra_y = np.concatenate([memory_y, transfer_extra_y], axis=0)
        elif not memory_dimension_warning_logged:
            console.print("[yellow]Transfer memory skipped due to dimensionality mismatch.[/yellow]")
            memory_dimension_warning_logged = True

    transfer_memory_payload = None
    if transfer_extra_X is not None and transfer_extra_y is not None:
        transfer_memory_payload = (transfer_extra_X, transfer_extra_y)

    transfer_candidates = []
    if transfer_to:
        transfer_candidates = [transfer_to]
    else:
        transfer_candidates = [task for task in available_tasks() if task != task_spec.name]

    transfer_results: list[dict] = []
    best_summary: dict | None = None
    best_gain: float | None = None

    for candidate in transfer_candidates:
        if candidate == task_spec.name:
            continue

        summary = evaluate_transfer(
            project_dir,
            task_spec,
            candidate,
            seed,
            transfer_memory_payload,
            arch=current_genome.nn,
        )

        if not summary:
            continue

        boot_metric = summary.get("boot_metric")
        gain = None
        if isinstance(boot_metric, (int, float)):
            boot_val = float(boot_metric)
            if not math.isnan(boot_val):
                gain = boot_val - base_metric

        summary["transfer_gain"] = gain
        transfer_results.append(summary)

        metric_label = summary.get("boot_metric_label", "metric")
        fidelity_val = summary.get("boot_fidelity")
        metric_str = "n/a"
        fidelity_str = "n/a"
        if isinstance(boot_metric, (int, float)) and not math.isnan(float(boot_metric)):
            metric_str = f"{float(boot_metric):.3f}"
        if isinstance(fidelity_val, (int, float)) and not math.isnan(float(fidelity_val)):
            fidelity_str = f"{float(fidelity_val):.3f}"
        gain_str = f"{gain:.3f}" if gain is not None else "n/a"

        console.print(
            "[cyan]Transfer test:[/cyan] {src} → {dst} | boot_{label}={metric} | gain={gain_val} | boot_fidelity={fid}".format(
                src=summary.get("from_task"),
                dst=summary.get("to_task"),
                label=metric_label,
                metric=metric_str,
                gain_val=gain_str,
                fid=fidelity_str,
            )
        )

        metric_json = None
        fidelity_json = None
        if isinstance(boot_metric, (int, float)) and not math.isnan(float(boot_metric)):
            metric_json = float(boot_metric)
        if isinstance(fidelity_val, (int, float)) and not math.isnan(float(fidelity_val)):
            fidelity_json = float(fidelity_val)

        log_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "from_task": summary.get("from_task"),
            "to_task": summary.get("to_task"),
            "boot_metric": metric_json,
            "boot_metric_label": metric_label,
            "boot_fidelity": fidelity_json,
            "memory_points": summary.get("memory_points", 0),
            "target_kind": summary.get("target_kind"),
            "transfer_gain": gain,
        }
        transfer_log_path = project_dir / "transfer_log.json"
        with transfer_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

        if gain is not None:
            if best_gain is None or gain > best_gain:
                best_summary = summary
                best_gain = gain
        elif best_summary is None:
            best_summary = summary

    suggested_task: str | None = None
    auto_stop_triggered = False

    if best_summary:
        record_gain = best_summary.get("transfer_gain")
        if record_gain is None:
            record_gain = 0.0
        curriculum.record(
            task_spec.name,
            best_summary.get("to_task", ""),
            float(record_gain),
        )
        boot_metric_val = best_summary.get("boot_metric")
        if isinstance(boot_metric_val, (int, float)) and not math.isnan(float(boot_metric_val)):
            last_transfer_norm = float(np.clip(float(boot_metric_val), 0.0, 1.0))
        if isinstance(record_gain, (int, float)) and not math.isnan(float(record_gain)):
            last_transfer_norm = max(
                last_transfer_norm,
                float(np.clip(float(record_gain), -1.0, 1.0)),
            )

    candidate_pool = [task for task in ["xor", "sine", "pid"] if task in available_tasks()]
    if not candidate_pool:
        candidate_pool = [task for task in available_tasks() if task != task_spec.name]
    if candidate_pool:
        suggested_task = curriculum.suggest_next(candidate_pool)
        console.print(f"[cyan]Suggested next task:[/cyan] {suggested_task}")

    if best_summary:
        transfer_generation = history[-1]["gen"] if history else generations
        append_transfer_metrics(
            metrics_csv,
            metrics_headers,
            task_spec,
            best_summary,
            transfer_generation,
            best_gain,
            suggested_task,
        )

        metric_json = best_summary.get("boot_metric")
        metric_json = float(metric_json) if isinstance(metric_json, (int, float)) and not math.isnan(float(metric_json)) else None
        fidelity_json = best_summary.get("boot_fidelity")
        fidelity_json = float(fidelity_json) if isinstance(fidelity_json, (int, float)) and not math.isnan(float(fidelity_json)) else None

        run_meta.update({
            "transfer_candidates": [res.get("to_task") for res in transfer_results],
            "transfer_best_to": best_summary.get("to_task"),
            "transfer_boot_metric": metric_json,
            "transfer_boot_metric_label": best_summary.get("boot_metric_label"),
            "transfer_boot_fidelity": fidelity_json,
            "transfer_gain": best_gain,
            "next_task": suggested_task,
        })
        run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

        if auto_stop and auto_stop > 0:
            recent_gains = curriculum.last_gains(auto_stop)
            if len(recent_gains) >= auto_stop and all(gain < 0 for gain in recent_gains):
                console.print(
                    f"[green]Curriculum stabilized after {auto_stop} steps — entering consolidation phase.[/green]"
                )
                auto_stop_triggered = True

    if auto_curriculum and suggested_task and suggested_task != task_spec.name and not auto_stop_triggered:
        command = [
            "poetry",
            "run",
            "python",
            "-m",
            "projects.coevolve_base.coevolve",
            "--task",
            suggested_task,
            "--generations",
            str(generations),
        ]
        if not watch:
            command.append("--no-watch")
        if auto_curriculum:
            command.append("--auto-curriculum")
        if auto_stop:
            command.extend(["--auto-stop", str(auto_stop)])
        if meta_enabled:
            command.append("--meta")
        if dialogue_every:
            command.extend(["--dialogue-every", str(dialogue_every)])
        if auto_experiment:
            command.append("--auto-experiment")
        if override_learning_rate is not None:
            command.extend(["--learning-rate", str(override_learning_rate)])
        if override_mutation_rate is not None:
            command.extend(["--mutation-rate", str(override_mutation_rate)])
        if override_lambda_consistency is not None:
            command.extend(["--lambda-consistency", str(override_lambda_consistency)])
        if cfg_path:
            command.extend(["--config", str(cfg_path)])

        console.print(f"[blue]Launching next curriculum task → {suggested_task}[/blue]")
        subprocess.run(command, check=False)

    causal_report_path = project_dir / "causal_report.json"
    causal_graph_path = project_dir / "causal_graph.png"
    causal_attrs: list[dict] | None = None
    if attribute is not None:
        try:
            causal_attrs = attribute(project_dir, window=2, lags=2)
        except Exception as exc:
            console.print(f"[yellow]Causal attribution unavailable: {exc}[/yellow]")
            causal_attrs = None
    if causal_attrs:
        timestamp = datetime.now().isoformat(timespec="seconds")
        report_entry = {
            "timestamp": timestamp,
            "task": task_spec.name,
            "seed": seed,
            "results": causal_attrs,
        }
        existing_runs: list[dict] = []
        if causal_report_path.exists():
            try:
                previous_payload = json.loads(causal_report_path.read_text(encoding="utf-8"))
                if isinstance(previous_payload, dict) and "runs" in previous_payload:
                    existing_runs = [run for run in previous_payload.get("runs", []) if isinstance(run, dict)]
                elif isinstance(previous_payload, list):
                    existing_runs = [entry for entry in previous_payload if isinstance(entry, dict)]
            except Exception:
                existing_runs = []
        existing_runs.append(report_entry)
        try:
            causal_report_path.write_text(json.dumps({"runs": existing_runs}, indent=2), encoding="utf-8")
        except Exception as exc:
            console.print(f"[yellow]Failed to persist causal report: {exc}[/yellow]")

        lines = ["== Causal Attribution =="]
        for row in causal_attrs:
            count = int(row.get("count", 0)) if isinstance(row.get("count"), (int, float)) else 0
            if count <= 0:
                continue
            did_val = float(row.get("did_mean", 0.0)) if isinstance(row.get("did_mean"), (int, float)) else 0.0
            granger_val = float(row.get("granger_score", 0.0)) if isinstance(row.get("granger_score"), (int, float)) else 0.0
            mark = "↑" if did_val > 0 else ("↓" if did_val < 0 else "·")
            suffix = " (low evidence)" if count < 2 else ""
            try:
                kind = str(row.get("kind"))
                metric = str(row.get("metric"))
            except Exception:
                continue
            lines.append(
                f"{kind} ⇒ {metric}: DiD={did_val:+.3f} {mark} | Granger={granger_val:.3f} (n={count}){suffix}"
            )
        if len(lines) > 1:
            block = "\n".join(lines)
            console.print(block)
            try:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("\n" + block + "\n")
            except Exception:
                pass

        if build_graph is not None:
            try:
                graph_result = build_graph(causal_report_path, causal_graph_path, threshold=0.01)
                if graph_result is not None:
                    console.print(f"[gray]Causal graph saved: {causal_graph_path.name}[/gray]")
            except Exception as exc:
                console.print(f"[yellow]Causal graph generation failed: {exc}[/yellow]")

    meta_eval_path = project_dir / "meta_eval.jsonl"
    if meta_eval_enabled:
        try:
            meta_result = evaluate_latest(project_dir, model=meta_model or "codellama:13b")
        except Exception as exc:
            console.print(f"[yellow]Meta-evaluation failed: {exc}[/yellow]")
            meta_result = None
        if meta_result:
            try:
                with meta_eval_path.open("a", encoding="utf-8") as meta_file:
                    meta_file.write(json.dumps(meta_result) + "\n")
            except Exception as exc:
                console.print(f"[yellow]Could not write meta-eval log: {exc}[/yellow]")

            scores = meta_result.get("scores") if isinstance(meta_result.get("scores"), dict) else {}
            numeric_scores = [float(v) for v in scores.values() if isinstance(v, (int, float))]
            avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
            issues = meta_result.get("issues")
            improvements = meta_result.get("improvements")
            summary_text = meta_result.get("summary") if isinstance(meta_result.get("summary"), str) else ""
            raw_text = meta_result.get("raw") if isinstance(meta_result.get("raw"), str) else ""
            error_text = meta_result.get("error") if isinstance(meta_result.get("error"), str) else ""

            log_lines = ["== Meta-Evaluation ==", f"scores: {scores}"]
            if summary_text:
                log_lines.append(summary_text)
            if isinstance(issues, list) and issues:
                log_lines.append("issues: " + "; ".join(str(item) for item in issues))
            if isinstance(improvements, list) and improvements:
                log_lines.append("improvements: " + "; ".join(str(item) for item in improvements))
            if raw_text:
                log_lines.append("raw: " + raw_text)
            if error_text:
                log_lines.append("error: " + error_text)

            log_block = "\n".join(log_lines)
            console.print(log_block)
            try:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("\n" + log_block + "\n")
            except Exception:
                pass

            canon = Canon(project_dir)
            threshold_val = promote_threshold if isinstance(promote_threshold, (int, float)) else 0.7
            promote_rules = meta_result.get("promote_rules") if isinstance(meta_result.get("promote_rules"), list) else []
            archive_rules = meta_result.get("archive_rules") if isinstance(meta_result.get("archive_rules"), list) else []
            if numeric_scores and avg_score >= threshold_val and promote_rules:
                canon.add([str(rule) for rule in promote_rules], source="meta_eval")
            if archive_rules:
                canon.remove([str(rule) for rule in archive_rules])
            canon.save()

            run_meta.update({
                "meta_eval_scores": scores,
                "meta_eval_average": avg_score,
                "meta_eval_timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if watch:
        try:
            stop_file.touch()
            console.print("[gray]Stopping live health monitor...[/gray]")
        except Exception:
            console.print("[yellow]Could not signal health monitor to stop.[/yellow]")
        if monitor_thread:
            monitor_thread.join(timeout=5)

    if generate_health_report is None:
        console.print("[yellow]Health report module not found; skipping report generation.[/yellow]")
    else:
        try:
            generate_health_report(project_dir)
            console.print("[green]📄 Health report generated successfully.[/green]")
            report_path = project_dir / "health_report.html"
            console.print(f"[cyan]Open in browser:[/cyan] file:///{report_path.resolve()}")
        except Exception as exc:
            console.print(f"[yellow]⚠️ Could not generate health report: {exc}[/yellow]")

    reflection_output: str | None = None
    try:
        reflector = ReflectionEngine(metrics_csv, memory_bank_path)
        reflection_output = reflector.summarize()
    except Exception as exc:
        console.print(f"[yellow]Reflection generation failed: {exc}[/yellow]")
    else:
        if reflection_output:
            console.print(f"[magenta]{reflection_output}[/magenta]")

    dialogue_counter += 1
    try:
        dialogue_counter_path.write_text(json.dumps({"count": dialogue_counter}), encoding="utf-8")
    except Exception:
        pass
    if dialogue_every and dialogue_every > 0:
        if dialogue_counter % dialogue_every == 0:
            try:
                dialogue_engine = DialogEngine(project_dir / "reflection_log.md")
                dialogue_result = dialogue_engine.conduct_dialogue()
            except Exception as exc:
                console.print(f"[yellow]Dialogue generation failed: {exc}[/yellow]")
            else:
                console.print(f"[blue]{dialogue_result}[/blue]")

    if auto_experiment:
        try:
            manager = ExperimentManager(project_dir, use_memory=use_memory)
            hypothesis = manager.select_hypothesis()
        except Exception as exc:
            console.print(f"[yellow]Experiment manager initialization failed: {exc}[/yellow]")
        else:
            if hypothesis:
                console.print(f"[yellow]Testing new hypothesis:[/yellow] {hypothesis}")
                cfg = manager.configure_experiment(hypothesis)
                if dialogue_every:
                    cfg["dialogue_every"] = dialogue_every
                elif cfg.get("dialogue_every") is None:
                    cfg["dialogue_every"] = 1
                cfg["auto_curriculum"] = auto_curriculum
                cfg["auto_stop"] = auto_stop
                cfg["meta"] = meta_enabled or cfg.get("meta", False)
                cfg["auto_experiment"] = True
                experiment_cli_dialogue = cfg.get("dialogue_every") or dialogue_every
                exit_code = manager.run_experiment(
                    hypothesis,
                    cfg,
                    meta_enabled=cfg.get("meta", meta_enabled),
                    dialogue_every=experiment_cli_dialogue,
                    auto_curriculum=auto_curriculum,
                    auto_stop=auto_stop,
                )
                manager.record_result(hypothesis, cfg, exit_code)
            else:
                console.print("[gray]No new hypotheses to test.[/gray]")

    if robustness_sweep:
        if final_model is None:
            console.print("[yellow]Robustness sweep skipped: no trained model available.[/yellow]")
        else:
            console.print("[cyan]Running robustness sweep probes...[/cyan]")
            sweep_rows: list[tuple[float, float, str, float, float]] = []
            sweep_generation = history[-1]["gen"] if history else generations
            sweep_learning_rate = evolvable_params.get("learning_rate", 0.001)
            sweep_rule_rate = evolvable_params.get("rule_mutation_rate", 0.2)
            sweep_lambda = evolvable_params.get("lambda_consistency", 0.0 if no_consistency else current_lambda)

            def _probe_disagreement(model_ref, rule_ref, X_eval: np.ndarray) -> tuple[float, float]:
                if X_eval.size == 0:
                    return 0.0, 1.0
                with torch.no_grad():
                    logits = model_ref(torch.tensor(X_eval, dtype=torch.float32))
                    nn_probs = _logits_to_probs(logits, task_spec).cpu().numpy()
                if rule_ref is not None and hasattr(rule_ref, "predict_proba"):
                    rule_probs_raw = rule_ref.predict_proba(X_eval)
                    rule_probs = _align_rule_probabilities(rule_ref, rule_probs_raw, task_spec)
                elif rule_ref is not None and hasattr(rule_ref, "predict"):
                    preds = np.asarray(rule_ref.predict(X_eval))
                    if preds.ndim == 1 and nn_probs.ndim == 2:
                        rule_probs = np.zeros_like(nn_probs)
                        for cls_idx in range(rule_probs.shape[1]):
                            rule_probs[:, cls_idx] = (preds == cls_idx).astype(np.float32)
                    else:
                        rule_probs = preds.astype(np.float32)
                        if rule_probs.ndim == 1:
                            rule_probs = rule_probs.reshape(-1, 1)
                else:
                    rule_probs = np.zeros_like(nn_probs)
                disagreement_vals = np.abs(rule_probs - nn_probs).mean(axis=1)
                avg_disagreement = float(np.mean(disagreement_vals)) if disagreement_vals.size else 0.0
                return avg_disagreement, 1.0 - avg_disagreement

            with metrics_csv.open("a", newline="", encoding="utf-8") as metrics_file:
                metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_headers)
                if not metrics_exists:
                    metrics_writer.writeheader()
                    metrics_exists = True

                combo_seed = 0
                for ln_val in (0.0, 0.02, 0.05):
                    for fn_val in (0.0, 0.02):
                        for mode in ("none", "no_rule", "no_nn"):
                            combo_seed += 1
                            dataset = data_generator(
                                int(cfg.get("points", 1000)),
                                seed_value=(seed or 0) + 900 + combo_seed,
                                return_clean=True,
                                label_noise=ln_val,
                                feature_noise=fn_val,
                            )
                            X_noisy_probe, y_noisy_probe, X_clean_probe, y_clean_probe = _coerce_dataset(dataset)

                            abl_flags = {
                                "abl_no_rule": 1 if mode == "no_rule" else 0,
                                "abl_no_nn": 1 if mode == "no_nn" else 0,
                                "abl_no_explore": 0,
                                "abl_no_consistency": int(no_consistency),
                            }

                            if mode == "no_nn":
                                probe_model = final_model
                                with torch.no_grad():
                                    logits_probe = probe_model(torch.tensor(X_clean_probe, dtype=torch.float32))
                                    if task_spec.loss_type == "bce":
                                        y_eval_t = torch.tensor(y_clean_probe, dtype=torch.float32).unsqueeze(1)
                                    else:
                                        y_eval_t = torch.tensor(y_clean_probe, dtype=torch.long)
                                    nn_metric = _compute_primary_metric(logits_probe, y_eval_t, task_spec)
                                model_param_count_probe = probe_model.parameter_count
                            else:
                                (
                                    probe_model,
                                    nn_metric,
                                    _sup,
                                    _cons,
                                    _opt,
                                    model_param_count_probe,
                                ) = train_nn(
                                    cfg_path,
                                    task_spec,
                                    generation=0,
                                    rule_model=final_rule_model,
                                    lambda_consistency=0.0 if no_consistency else sweep_lambda,
                                    learning_rate=sweep_learning_rate,
                                    extra_data=None,
                                    initial_state=current_model_state,
                                    arch=current_genome.nn,
                                    data=(X_noisy_probe, y_noisy_probe, X_clean_probe, y_clean_probe),
                                    persist=False,
                                    verbose=False,
                                    label_noise=ln_val,
                                    feature_noise=fn_val,
                                    target_noise=noise_config["target_noise"],
                                    label_smoothing=label_smoothing,
                                    eval_with_dropout=eval_with_dropout,
                                )

                            rule_metric = _evaluate_rule_fidelity(final_rule_model, probe_model, seed_offset=combo_seed)
                            avg_disagreement, rule_agreement_rate = _probe_disagreement(probe_model, final_rule_model, X_clean_probe)

                            combined_probe = (
                                nn_metric + (rule_metric if not math.isnan(rule_metric) else nn_metric)
                            ) / 2

                            metrics_writer.writerow({
                                "generation": sweep_generation,
                                "task_name": task_spec.name,
                                "nn_accuracy": round(nn_metric, 6),
                                "rule_fidelity": round(rule_metric, 6) if not math.isnan(rule_metric) else math.nan,
                                "nn_primary": round(nn_metric, 6),
                                "rule_primary": round(rule_metric, 6) if not math.isnan(rule_metric) else math.nan,
                                "combined_fitness": round(combined_probe, 6),
                                "nn_learning_gain": 0.0,
                                "rule_adaptation_gain": 0.0,
                                "nn_primary_delta": 0.0,
                                "rule_primary_delta": 0.0,
                                "rule_mutations": 0,
                                "consistency_loss": 0.0,
                                "avg_disagreement": round(avg_disagreement, 6),
                                "n_new_samples": 0,
                                "rule_agreement_rate": round(rule_agreement_rate, 6),
                                "lambda_consistency": 0.0 if no_consistency else sweep_lambda,
                                "meta_fitness": round(combined_probe, 6),
                                "learning_rate": sweep_learning_rate,
                                "rule_mutation_rate": sweep_rule_rate,
                                "nn_primary_label": task_spec.primary_metric_label,
                                "rule_primary_label": task_spec.secondary_metric_label,
                                "memory_embeddings_used": 0,
                                "memory_rules_reused": 0,
                                "memory_boost": 0.0,
                                "meta_learning_rate": math.nan,
                                "meta_mutation_rate": math.nan,
                                "meta_lambda_consistency": math.nan,
                                "meta_reward": math.nan,
                                "label_noise": round(ln_val, 6),
                                "feature_noise": round(fn_val, 6),
                                "target_noise": round(noise_config["target_noise"], 6),
                                "label_smoothing": round(label_smoothing, 6),
                                "eval_with_dropout": int(bool(eval_with_dropout)),
                                "abl_no_rule": abl_flags["abl_no_rule"],
                                "abl_no_nn": abl_flags["abl_no_nn"],
                                "abl_no_explore": abl_flags["abl_no_explore"],
                                "abl_no_consistency": abl_flags["abl_no_consistency"],
                                "is_probe": 1,
                                "arch_hidden_layers": current_genome.nn.hidden_layers,
                                "arch_hidden_width": current_genome.nn.hidden_width,
                                "arch_activation": current_genome.nn.activation,
                                "arch_normalization": current_genome.nn.normalization,
                                "arch_dropout": round(current_genome.nn.dropout, 4),
                                "arch_max_depth": current_genome.rules.max_depth,
                                "arch_min_samples_leaf": current_genome.rules.min_samples_leaf,
                                "arch_prune_alpha": round(current_genome.rules.prune_ccp_alpha, 6),
                                "arch_uncertainty_k": current_genome.explore.uncertainty_k,
                                "arch_mutation_rate": round(current_genome.explore.mutation_rate, 6),
                                "arch_lambda_consistency": round(current_genome.explore.lambda_consistency, 6),
                                "model_param_count": model_param_count_probe,
                                "composite_score": round(combined_probe, 6),
                                "transfer_gain": math.nan,
                                "next_task": "",
                                "from_task": "",
                                "to_task": "",
                                "boot_metric": math.nan,
                                "boot_metric_label": "",
                                "boot_fidelity": math.nan,
                            })

                            sweep_rows.append((ln_val, fn_val, mode, nn_metric, rule_metric))

            if sweep_rows:
                console.print("[gray]Robustness sweep probes (nn / rule):[/gray]")
                for ln_val, fn_val, mode, nn_metric, rule_metric in sweep_rows:
                    console.print(
                        f"  ln={ln_val:.2f} fn={fn_val:.2f} mode={mode:>7} → nn={nn_metric:.3f} rule={rule_metric:.3f}"
                    )

    console.print("\n[bold green]Coevolution complete![/bold green]")
    console.print(f"Results saved to {results_csv}")

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forge NN-Rule coevolution")
    parser.add_argument("--generations", type=int, default=5, help="Number of coevolution steps")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml", help="Config file path")
    parser.add_argument("--no-watch", action="store_true", help="Disable live health monitoring")
    parser.add_argument("--seed", type=int, help="Global random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--task", choices=available_tasks(), default="shape", help="Dataset to use")
    parser.add_argument("--transfer-to", choices=available_tasks(), help="Evaluate transfer boot performance to another task")
    parser.add_argument("--auto-curriculum", action="store_true", help="Enable autonomous curriculum scheduling")
    parser.add_argument(
        "--auto-stop",
        type=int,
        metavar="N",
        help="Stop auto curriculum when last N transfer gains are negative",
    )
    parser.add_argument("--meta", action="store_true", help="Activate meta-controller self-tuning")
    parser.add_argument(
        "--dialogue-every",
        type=int,
        metavar="K",
        help="Run autonomous dialogue every K runs",
    )
    parser.add_argument("--auto-experiment", action="store_true", help="Enable autonomous experiment launching")
    parser.add_argument("--learning-rate", type=float, help="Override initial learning rate")
    parser.add_argument("--mutation-rate", type=float, help="Override initial mutation rate")
    parser.add_argument("--lambda-consistency", type=float, help="Override initial lambda consistency")
    parser.add_argument("--with-memory", action="store_true", help="Enable cross-experiment memory indexing")
    parser.add_argument("--arch-auto", action="store_true", help="Enable periodic architecture genome evolution")
    parser.add_argument("--arch-pop", type=int, default=3, metavar="B", help="Number of probe candidates when evolving architectures")
    parser.add_argument("--arch-interval", type=int, default=3, metavar="A", help="Generations between architecture selection rounds")
    parser.add_argument("--arch-reset", action="store_true", help="Reset architecture genome to defaults for this run")
    parser.add_argument("--arch-load", type=Path, help="Load architecture genome from JSON path")
    parser.add_argument(
        "--arch-lock",
        choices=["nn", "rules", "explore"],
        action="append",
        help="Lock specific genome domains during mutation",
    )
    parser.add_argument("--label-noise", type=float, default=0.0, help="Fraction of labels to flip for classification tasks")
    parser.add_argument("--feature-noise", type=float, default=0.0, help="Standard deviation of Gaussian noise added to features")
    parser.add_argument("--target-noise", type=float, default=0.0, help="Gaussian noise applied to regression targets")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing applied during NN training")
    parser.add_argument("--eval-with-dropout", action="store_true", help="Keep dropout active during evaluation for MC-dropout style probes")
    parser.add_argument("--no-rule-updates", action="store_true", help="Disable rule extraction and mutations (rules remain frozen)")
    parser.add_argument("--no-nn-train", action="store_true", help="Disable NN training (reuse frozen weights)")
    parser.add_argument("--no-explore", action="store_true", help="Disable uncertainty-guided exploration and new sample acquisition")
    parser.add_argument("--no-consistency", action="store_true", help="Disable rule-to-NN consistency loss (lambda=0)")
    parser.add_argument("--robustness-sweep", action="store_true", help="Run quick what-if probes across noise/ablation grid")
    parser.add_argument("--meta-eval", action="store_true", help="Run meta-evaluation after the run")
    parser.add_argument("--meta-model", type=str, help="Model to use for meta-evaluation prompts")
    parser.add_argument(
        "--promote-threshold",
        type=float,
        default=0.70,
        help="Average meta-eval score required to promote rules to canon",
    )
    args = parser.parse_args()

    coevolve(
        args.config,
        generations=args.generations,
        watch=not args.no_watch,
        seed=args.seed,
        resume=args.resume,
        task_name=args.task,
        transfer_to=args.transfer_to,
        auto_curriculum=args.auto_curriculum,
        auto_stop=args.auto_stop,
        meta_enabled=args.meta,
        dialogue_every=args.dialogue_every,
        auto_experiment=args.auto_experiment,
        use_memory=args.with_memory,
        override_learning_rate=args.learning_rate,
        override_mutation_rate=args.mutation_rate,
        override_lambda_consistency=args.lambda_consistency,
        arch_auto=args.arch_auto,
        arch_population=args.arch_pop,
        arch_interval=args.arch_interval,
        arch_reset=args.arch_reset,
        arch_load=args.arch_load,
        arch_locks=tuple(args.arch_lock) if args.arch_lock else None,
        label_noise=args.label_noise,
        feature_noise=args.feature_noise,
        target_noise=args.target_noise,
        label_smoothing=args.label_smoothing,
        eval_with_dropout=args.eval_with_dropout,
        no_rule_updates=args.no_rule_updates,
        no_nn_train=args.no_nn_train,
        no_explore=args.no_explore,
        no_consistency=args.no_consistency,
        robustness_sweep=args.robustness_sweep,
        meta_eval_enabled=args.meta_eval,
        meta_model=args.meta_model,
        promote_threshold=args.promote_threshold,
    )
