from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from joblib import dump
import numpy as np
import torch
import yaml
from pathlib import Path
from dataclasses import asdict

from .main import generate_points
from .nn_agent import FlexibleNN
from .tasks import TaskSpec
from .architectures import NNArch, RuleArch
from projects.coevolve_base.utils.checkpoint import safe_load_checkpoint


def _load_trained_model(
    task: TaskSpec,
    input_dim: int,
    checkpoint_dir: Path,
    arch: NNArch | None,
) -> FlexibleNN:
    arch = arch or NNArch()
    if task.loss_type == "bce":
        output_dim = 1
    else:
        output_dim = task.output_dim or task.num_classes or 3

    model = FlexibleNN(input_dim, int(output_dim), arch)

    if not safe_load_checkpoint(
        model,
        checkpoint_dir / "nn_model.pt",
        model_name="TaskNN",
        expected_type="FlexibleNN",
        expected_arch=asdict(arch),
    ):
        raise FileNotFoundError("Neural network checkpoint nn_model.pt not found or incompatible.")
    model.eval()
    return model


def _tree_for_task(task: TaskSpec, arch: RuleArch, seed: int):
    tree_kwargs = {
        "max_depth": int(arch.max_depth),
        "min_samples_leaf": int(arch.min_samples_leaf),
        "ccp_alpha": float(arch.prune_ccp_alpha),
        "random_state": seed,
    }
    if task.loss_type == "bce" or task.kind == "classification":
        return DecisionTreeClassifier(**tree_kwargs)
    return DecisionTreeRegressor(**tree_kwargs)


def _nn_targets_from_logits(logits: torch.Tensor, task: TaskSpec) -> np.ndarray:
    if task.loss_type == "bce":
        probs = torch.sigmoid(logits).squeeze(-1)
        return (probs >= 0.5).long().cpu().numpy()
    return logits.argmax(dim=1).cpu().numpy()


def extract_rules(
    cfg_path: Path,
    task: TaskSpec,
    nn_arch: NNArch | None = None,
    rule_arch: RuleArch | None = None,
    *,
    label_noise: float = 0.0,
    feature_noise: float = 0.0,
    target_noise: float = 0.0,
):
    """Learn a symbolic model approximating the active NN for the task."""

    cfg = yaml.safe_load(open(cfg_path))
    generated = generate_points(
        cfg.get("points", 1000),
        label_noise=label_noise,
        feature_noise=feature_noise,
        target_noise=target_noise,
        return_clean=True,
    )

    if len(generated) == 4:
        X_noisy, _, X_clean, _ = generated
    else:
        X_noisy, _ = generated
        X_clean = np.asarray(X_noisy, dtype=np.float32).copy()

    input_dim = X_noisy.shape[1]

    checkpoint_dir = Path(__file__).parent
    nn_arch = nn_arch or NNArch()
    rule_arch = rule_arch or RuleArch()

    model = _load_trained_model(task, input_dim, checkpoint_dir, nn_arch)

    with torch.no_grad():
        logits_train = model(torch.tensor(X_noisy, dtype=torch.float32))
        nn_targets = _nn_targets_from_logits(logits_train, task)

    tree = _tree_for_task(task, rule_arch, cfg.get("seed", 42))
    tree.fit(X_noisy, nn_targets)

    try:
        rules = export_text(tree, feature_names=[f"x{i}" for i in range(input_dim)])
    except Exception:
        rules = "(rule export unavailable)"

    print("=== Derived Symbolic Rules ===")
    print(rules)

    if hasattr(tree, "score"):
        with torch.no_grad():
            logits_eval = model(torch.tensor(X_clean, dtype=torch.float32))
            nn_eval_targets = _nn_targets_from_logits(logits_eval, task)
        fidelity = tree.score(X_clean, nn_eval_targets)
    else:
        fidelity = float("nan")

    print(f"Tree fidelity to NN: {fidelity:.3f}")

    tree_path = checkpoint_dir / "rules_model.joblib"
    dump(tree, tree_path)

    return rules, fidelity, tree

if __name__ == "__main__":
    from .tasks import apply_task

    cfg_path = Path(__file__).parent / "config.yaml"
    task_spec = apply_task("shape")
    extract_rules(cfg_path, task_spec)
