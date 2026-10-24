import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import asdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

from .main import generate_points
from .tasks import TaskSpec
from .architectures import NNArch
from projects.coevolve_base.utils.checkpoint import save_checkpoint


def _activation_layer(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


class FlexibleNN(nn.Module):
    """Configurable MLP whose shape is defined by an NNArch."""

    def __init__(self, input_dim: int, output_dim: int, arch: NNArch):
        super().__init__()
        self.arch = arch

        layers: list[nn.Module] = []
        hidden_layers = max(1, int(arch.hidden_layers))
        hidden_width = max(1, int(arch.hidden_width))
        prev_dim = input_dim

        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_width))
            if arch.normalization == "batch":
                layers.append(nn.BatchNorm1d(hidden_width))
            layers.append(_activation_layer(arch.activation))
            if arch.dropout > 0:
                layers.append(nn.Dropout(p=float(arch.dropout)))
            prev_dim = hidden_width

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())


def _logits_to_probs(logits: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    if task.loss_type == "bce":
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


def _align_rule_probabilities(rule_model, X: np.ndarray, task: TaskSpec) -> Optional[np.ndarray]:
    try:
        probs = rule_model.predict_proba(X)
    except Exception:
        return None

    probs = np.asarray(probs, dtype=np.float32)
    if task.loss_type == "bce":
        if probs.ndim == 2:
            if probs.shape[1] == 1:
                return probs
            if probs.shape[1] >= 2:
                return probs[:, 1:2]
        return None

    num_classes = task.num_classes or probs.shape[1]
    aligned = np.zeros((len(X), num_classes), dtype=np.float32)
    classes = getattr(rule_model, "classes_", [])
    for idx, cls in enumerate(classes):
        cls_idx = int(cls)
        if 0 <= cls_idx < num_classes:
            aligned[:, cls_idx] = probs[:, idx]
    return aligned


def _build_model(task: TaskSpec, input_dim: int, arch: NNArch | None = None) -> FlexibleNN:
    arch = arch or NNArch()
    if task.loss_type == "bce":
        output_dim = 1
    else:
        if task.output_dim:
            output_dim = task.output_dim
        elif task.num_classes:
            output_dim = int(task.num_classes)
        else:
            output_dim = 3
    return FlexibleNN(input_dim, output_dim, arch)


def _compute_primary_metric(logits: torch.Tensor, targets: torch.Tensor, task: TaskSpec) -> float:
    if task.loss_type == "bce":
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= 0.5).long()
        return (preds == targets.long()).float().mean().item()

    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_nn(
    cfg_path: Path,
    task: TaskSpec,
    generation: int = 0,
    rule_model=None,
    lambda_consistency: float = 0.2,
    learning_rate: float = 0.01,
    extra_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    initial_state: Optional[dict] = None,
    arch: NNArch | None = None,
    *,
    data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    persist: bool = True,
    verbose: bool = True,
    label_noise: float = 0.0,
    feature_noise: float = 0.0,
    target_noise: float = 0.0,
    label_smoothing: float = 0.0,
    eval_with_dropout: bool = False,
):
    """Train an NN tailored to the active task specification."""

    label_smoothing = float(max(0.0, min(label_smoothing, 0.49)))

    cfg = {}
    points = cfg.get("points", 1000)

    def _coerce_dataset(payload: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    if data is not None:
        X_noisy, y_noisy, X_clean, y_clean = _coerce_dataset(data)
    else:
        generated = generate_points(
            points,
            label_noise=label_noise,
            feature_noise=feature_noise,
            target_noise=target_noise,
            return_clean=True,
        )
        X_noisy, y_noisy, X_clean, y_clean = _coerce_dataset(generated)  # type: ignore[arg-type]

    if extra_data is not None:
        extra_X, extra_y = extra_data
        if len(extra_X):
            extra_X = np.asarray(extra_X, dtype=np.float32)
            extra_y = np.asarray(extra_y)
            X_noisy = np.concatenate([X_noisy, extra_X], axis=0)
            y_noisy = np.concatenate([y_noisy, extra_y], axis=0)
            X_clean = np.concatenate([X_clean, extra_X], axis=0)
            y_clean = np.concatenate([y_clean, extra_y], axis=0)

    n_samples = X_noisy.shape[0]
    if n_samples == 0:
        raise ValueError("Empty dataset passed to train_nn")

    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.2)
    X_train = X_noisy[train_idx]
    y_train = y_noisy[train_idx]
    X_val = X_clean[val_idx]
    y_val = y_clean[val_idx]

    input_dim = X_train.shape[1]

    arch = arch or NNArch()
    model = _build_model(task, input_dim, arch)
    if initial_state:
        try:
            model.load_state_dict(initial_state)
        except RuntimeError as exc:
            print(
                "[arch] Skipping checkpoint restore due to architecture mismatch:"
                f" {exc}"
            )
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"[arch] Could not restore checkpoint state ({exc}); continuing from scratch")

    opt = optim.Adam(model.parameters(), lr=learning_rate)

    if task.loss_type == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        pos_target = 1.0 - label_smoothing
        neg_target = label_smoothing
        y_train_targets = y_train_tensor * (pos_target - neg_target) + neg_target
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_train_targets = y_train_tensor
        y_val_t = torch.tensor(y_val, dtype=torch.long)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    rule_probs_t = None
    if rule_model is not None:
        aligned = _align_rule_probabilities(rule_model, X_train, task)
        if aligned is not None and np.any(aligned):
            rule_probs_t = torch.tensor(aligned, dtype=torch.float32, device=X_train_t.device)

    last_supervised = 0.0
    last_consistency = 0.0

    for _ in range(80):
        opt.zero_grad()
        logits = model(X_train_t)
        supervised_loss = loss_fn(logits, y_train_targets)

        consistency_loss = torch.tensor(0.0, dtype=torch.float32, device=X_train_t.device)
        if rule_probs_t is not None:
            nn_probs = _logits_to_probs(logits, task)
            consistency_loss = F.mse_loss(nn_probs, rule_probs_t)
            total_loss = supervised_loss + lambda_consistency * consistency_loss
        else:
            total_loss = supervised_loss

        total_loss.backward()
        opt.step()

        last_supervised = supervised_loss.detach().item()
        last_consistency = consistency_loss.detach().item()

    prev_training_state = model.training
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    if eval_with_dropout:
        model.train()
    else:
        model.eval()

    with torch.no_grad():
        val_logits = model(X_val_t)
        primary_metric = _compute_primary_metric(val_logits, y_val_t, task)

    model.train(prev_training_state)
    torch.set_grad_enabled(grad_enabled)

    if persist:
        noise_meta = {
            "label_noise": float(label_noise),
            "feature_noise": float(feature_noise),
            "target_noise": float(target_noise),
            "label_smoothing": float(label_smoothing),
            "eval_with_dropout": bool(eval_with_dropout),
        }
        save_checkpoint(
            model,
            Path(__file__).parent / "nn_model.pt",
            model_type=model.__class__.__name__,
            score=primary_metric,
            extra_meta={"nn_arch": asdict(arch), **noise_meta},
        )

    if verbose:
        print(f"Final {task.primary_metric_label.split()[0]}: {primary_metric:.3f}")
        if generation:
            print(
                f"Gen {generation}: supervised={last_supervised:.3f}, consistency={last_consistency:.3f}"
            )
        else:
            print(
                f"Supervised loss={last_supervised:.3f}, consistency={last_consistency:.3f}"
            )

    param_count = model.parameter_count

    return (
        model,
        primary_metric,
        last_supervised,
        last_consistency,
        opt.state_dict(),
        param_count,
    )


if __name__ == "__main__":
    cfg_path = Path(__file__).parent / "config.yaml"
    from .tasks import apply_task

    task_spec = apply_task("shape")
    train_nn(cfg_path, task_spec)
