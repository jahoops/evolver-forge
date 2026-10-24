"""Task adapters that expose dataset generators and metric metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

import numpy as np


ArrayTuple = Tuple[np.ndarray, np.ndarray]


def _apply_noise(
    X: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    kind: str,
    num_classes: int | None,
    label_noise: float = 0.0,
    feature_noise: float = 0.0,
    target_noise: float = 0.0,
    return_clean: bool = False,
) -> ArrayTuple | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply configurable noise to dataset copies while preserving the clean view."""

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    X_clean = X.astype(np.float32, copy=True)
    y_clean = y.copy()

    X_noisy = X_clean.copy()
    if feature_noise and feature_noise > 0.0:
        X_noisy = X_noisy + rng.normal(loc=0.0, scale=feature_noise, size=X_noisy.shape).astype(np.float32)

    y_noisy = y_clean.copy()
    if kind == "classification" and label_noise and label_noise > 0.0:
        mask = rng.random(size=y_noisy.shape[0]) < float(label_noise)
        if mask.any():
            if num_classes == 2:
                y_noisy[mask] = 1 - y_noisy[mask]
            else:
                classes = np.arange(num_classes if num_classes is not None else int(y_noisy.max()) + 1)
                for idx in np.where(mask)[0]:
                    current = y_noisy[idx]
                    candidates = classes[classes != current]
                    if len(candidates):
                        y_noisy[idx] = rng.choice(candidates)
    elif kind != "classification" and target_noise and target_noise > 0.0:
        noise = rng.normal(loc=0.0, scale=target_noise, size=y_noisy.shape)
        y_noisy = y_noisy.astype(np.float32, copy=True)
        y_noisy += noise.astype(np.float32)

    if return_clean:
        return X_noisy, y_noisy, X_clean, y_clean
    return X_noisy, y_noisy


@dataclass
class TaskSpec:
    name: str
    kind: str  # classification | regression | control
    loss_type: str
    num_classes: int | None
    output_dim: int
    primary_metric_label: str
    primary_higher_is_better: bool
    secondary_metric_label: str
    secondary_higher_is_better: bool
    generator: Callable[..., ArrayTuple]

    @property
    def is_binary(self) -> bool:
        return self.kind == "classification" and self.num_classes == 2


class BaseTask:
    """Common task metadata with dataset generation hook."""

    name = "base"
    kind = "classification"
    loss_type = "cross_entropy"
    num_classes: int | None = None
    output_dim: int | None = None
    primary_metric_label = "Accuracy ↑"
    primary_higher_is_better = True
    secondary_metric_label = "Rule Accuracy ↑"
    secondary_higher_is_better = True

    def spec(self) -> TaskSpec:
        if self.output_dim is None:
            raise ValueError("Task must define output_dim")
        return TaskSpec(
            name=self.name,
            kind=self.kind,
            loss_type=self.loss_type,
            num_classes=self.num_classes,
            output_dim=self.output_dim,
            primary_metric_label=self.primary_metric_label,
            primary_higher_is_better=self.primary_higher_is_better,
            secondary_metric_label=self.secondary_metric_label,
            secondary_higher_is_better=self.secondary_higher_is_better,
            generator=self.make_data,
        )

    def make_data(
        self,
        n: int = 1000,
        seed: int | None = None,
        *,
        label_noise: float = 0.0,
        feature_noise: float = 0.0,
        target_noise: float = 0.0,
        return_clean: bool = False,
    ) -> ArrayTuple:  # pragma: no cover - interface only
        raise NotImplementedError


class XORTask(BaseTask):
    name = "xor"
    kind = "classification"
    loss_type = "bce"
    num_classes = 2
    output_dim = 1

    def make_data(
        self,
        n: int = 1000,
        seed: int | None = None,
        *,
        label_noise: float = 0.0,
        feature_noise: float = 0.0,
        target_noise: float = 0.0,
        return_clean: bool = False,
    ) -> ArrayTuple:
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
        y = np.bitwise_xor(X[:, 0].astype(int), X[:, 1].astype(int)).astype(np.int64)
        return _apply_noise(
            X,
            y,
            rng=rng,
            kind=self.kind,
            num_classes=self.num_classes,
            label_noise=label_noise,
            feature_noise=feature_noise,
            target_noise=target_noise,
            return_clean=return_clean,
        )


class SineTask(BaseTask):
    name = "sine"
    kind = "classification"
    loss_type = "cross_entropy"
    num_classes = 3
    output_dim = 3
    primary_metric_label = "Bucket Accuracy ↑"
    primary_higher_is_better = True
    secondary_metric_label = "Rule Bucket Acc ↑"
    secondary_higher_is_better = True

    def make_data(
        self,
        n: int = 1000,
        seed: int | None = None,
        *,
        label_noise: float = 0.0,
        feature_noise: float = 0.0,
        target_noise: float = 0.0,
        return_clean: bool = False,
    ) -> ArrayTuple:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-np.pi, np.pi, size=n)
        sinx = np.sin(x)
        X = np.stack([x, sinx], axis=1).astype(np.float32)
        bins = np.array([-0.33, 0.33])
        y = np.digitize(sinx, bins, right=False).astype(np.int64)
        return _apply_noise(
            X,
            y,
            rng=rng,
            kind=self.kind,
            num_classes=self.num_classes,
            label_noise=label_noise,
            feature_noise=feature_noise,
            target_noise=target_noise,
            return_clean=return_clean,
        )


class PIDTask(BaseTask):
    name = "pid"
    kind = "classification"
    loss_type = "cross_entropy"
    num_classes = 3
    output_dim = 3
    primary_metric_label = "Stability Accuracy ↑"
    primary_higher_is_better = True
    secondary_metric_label = "Rule Stability Acc ↑"
    secondary_higher_is_better = True

    def make_data(
        self,
        n: int = 1000,
        seed: int | None = None,
        *,
        label_noise: float = 0.0,
        feature_noise: float = 0.0,
        target_noise: float = 0.0,
        return_clean: bool = False,
    ) -> ArrayTuple:
        rng = np.random.default_rng(seed)
        states = rng.uniform(-2.0, 2.0, size=n)
        integrals = rng.uniform(-1.0, 1.0, size=n)
        X = np.stack([states, integrals], axis=1).astype(np.float32)
        control = -0.6 * states - 0.1 * integrals + rng.normal(scale=0.05, size=n)
        control = np.clip(control, -1.0, 1.0)
        y = np.digitize(control, [-0.25, 0.25]).astype(np.int64)
        return _apply_noise(
            X,
            y,
            rng=rng,
            kind=self.kind,
            num_classes=self.num_classes,
            label_noise=label_noise,
            feature_noise=feature_noise,
            target_noise=target_noise,
            return_clean=return_clean,
        )


TASK_REGISTRY: Dict[str, Type[BaseTask] | None] = {
    "shape": None,
    "xor": XORTask,
    "sine": SineTask,
    "pid": PIDTask,
}


_ORIGINAL_GENERATOR: Callable[[int, int | None], ArrayTuple] | None = None
_SHAPE_SPEC: TaskSpec | None = None


def available_tasks() -> Tuple[str, ...]:
    return tuple(TASK_REGISTRY.keys())


def _ensure_shape_spec() -> TaskSpec:
    """Return the cached TaskSpec for the original shape task."""

    from . import main as main_module

    global _ORIGINAL_GENERATOR, _SHAPE_SPEC
    if _ORIGINAL_GENERATOR is None:
        _ORIGINAL_GENERATOR = main_module.generate_points
    
    def shape_generator(
        n: int = 1000,
        seed: int | None = None,
        *,
        label_noise: float = 0.0,
        feature_noise: float = 0.0,
        target_noise: float = 0.0,
        return_clean: bool = False,
    ) -> ArrayTuple:
        rng = np.random.default_rng(seed)
        generator = _ORIGINAL_GENERATOR or main_module.generate_points
        X, y = generator(n_points=n, seed=seed)
        return _apply_noise(
            X,
            y,
            rng=rng,
            kind="classification",
            num_classes=3,
            label_noise=label_noise,
            feature_noise=feature_noise,
            target_noise=target_noise,
            return_clean=return_clean,
        )
    if _SHAPE_SPEC is None:
        _SHAPE_SPEC = TaskSpec(
            name="shape",
            kind="classification",
            loss_type="cross_entropy",
            num_classes=3,
            output_dim=3,
            primary_metric_label="Accuracy ↑",
            primary_higher_is_better=True,
            secondary_metric_label="Rule Accuracy ↑",
            secondary_higher_is_better=True,
            generator=shape_generator,
        )
    return _SHAPE_SPEC


def get_task_spec(task_name: str | None) -> TaskSpec:
    """Return a TaskSpec without mutating global generators."""

    normalized = (task_name or "shape").lower()
    task_cls = TASK_REGISTRY.get(normalized)
    if normalized == "shape" or task_cls is None:
        return _ensure_shape_spec()
    task = task_cls()
    return task.spec()


def apply_task(task_name: str | None) -> TaskSpec:
    """Return the active task spec and propagate its generator."""

    from . import main as main_module
    from . import nn_agent as nn_agent_module
    from . import rule_agent as rule_agent_module
    from . import visualize_rules as viz_module

    global _ORIGINAL_GENERATOR

    if _ORIGINAL_GENERATOR is None:
        _ORIGINAL_GENERATOR = main_module.generate_points

    spec = get_task_spec(task_name)
    generator = spec.generator
    main_module.generate_points = generator
    _propagate_generator(generator, nn_agent_module, rule_agent_module, viz_module)
    return spec


def _propagate_generator(generator: Callable[[int, int | None], ArrayTuple], *modules) -> None:
    for module in modules:
        setattr(module, "generate_points", generator)
