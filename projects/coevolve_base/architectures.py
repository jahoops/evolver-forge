"""Architecture genome definitions and mutation utilities."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Literal, Sequence

Activation = Literal["relu", "tanh", "gelu", "leaky_relu", "sigmoid"]
Norm = Literal["none", "batch"]

ACTIVATIONS: Sequence[Activation] = ("relu", "tanh", "gelu", "leaky_relu", "sigmoid")
NORM_OPTIONS: Sequence[Norm] = ("none", "batch")


@dataclass
class NNArch:
    hidden_layers: int = 2
    hidden_width: int = 16
    activation: Activation = "relu"
    normalization: Norm = "none"
    dropout: float = 0.0


@dataclass
class RuleArch:
    max_depth: int = 4
    min_samples_leaf: int = 2
    prune_ccp_alpha: float = 0.0


@dataclass
class ExploreArch:
    uncertainty_k: int = 64
    mutation_rate: float = 0.1
    lambda_consistency: float = 0.2


@dataclass
class ArchGenome:
    nn: NNArch = field(default_factory=NNArch)
    rules: RuleArch = field(default_factory=RuleArch)
    explore: ExploreArch = field(default_factory=ExploreArch)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nn": asdict(self.nn),
            "rules": asdict(self.rules),
            "explore": asdict(self.explore),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchGenome":
        if not isinstance(data, dict):
            raise TypeError("Genome payload must be a dictionary")

        def _construct(dc_cls, payload: Dict[str, Any]):
            base = dc_cls().__dict__.copy()
            base.update(payload or {})
            return dc_cls(**base)

        return cls(
            nn=_construct(NNArch, data.get("nn", {})),
            rules=_construct(RuleArch, data.get("rules", {})),
            explore=_construct(ExploreArch, data.get("explore", {})),
        )


def clip_bounds(value: float, low: float, high: float) -> float:
    """Clamp a numeric value into the provided bounds."""
    return max(low, min(high, value))


def mutate_arch(
    genome: ArchGenome,
    rng: random.Random | None = None,
    *,
    jitter: float = 0.2,
    locks: set[str] | None = None,
    change_log: list[tuple[str, Any, Any]] | None = None,
) -> ArchGenome:
    """Return a mutated copy of the supplied genome using bounded jitter."""

    rng = rng or random.Random()
    locks = locks or set()
    change_log = change_log if change_log is not None else None

    def _log_change(kind: str, old: Any, new: Any) -> None:
        if change_log is None:
            return
        if old == new:
            return
        change_log.append((kind, old, new))

    mutated = ArchGenome(
        nn=replace(genome.nn),
        rules=replace(genome.rules),
        explore=replace(genome.explore),
    )

    if "nn" not in locks:
        old_layers = mutated.nn.hidden_layers
        mutated.nn.hidden_layers = int(
            clip_bounds(old_layers + rng.choice([-1, 0, 1]), 1, 4)
        )
        _log_change("arch.nn.hidden_layers", old_layers, mutated.nn.hidden_layers)
        width_scale = 1.0 + rng.uniform(-jitter, jitter)
        old_width = mutated.nn.hidden_width
        new_width = int(round(old_width * width_scale / 8) * 8)
        mutated.nn.hidden_width = int(clip_bounds(new_width, 8, 128))
        _log_change("arch.nn.hidden_width", old_width, mutated.nn.hidden_width)
        if rng.random() < 0.25:
            choices = [act for act in ACTIVATIONS if act != mutated.nn.activation]
            old_activation = mutated.nn.activation
            mutated.nn.activation = rng.choice(choices)
            _log_change("arch.nn.activation", old_activation, mutated.nn.activation)
        if rng.random() < 0.2:
            choices = [norm for norm in NORM_OPTIONS if norm != mutated.nn.normalization]
            old_norm = mutated.nn.normalization
            mutated.nn.normalization = rng.choice(choices)
            _log_change("arch.nn.normalization", old_norm, mutated.nn.normalization)
        old_dropout = mutated.nn.dropout
        mutated.nn.dropout = float(
            clip_bounds(old_dropout + rng.uniform(-0.1, 0.1), 0.0, 0.5)
        )
        _log_change("arch.nn.dropout", old_dropout, mutated.nn.dropout)

    if "rules" not in locks:
        old_depth = mutated.rules.max_depth
        mutated.rules.max_depth = int(
            clip_bounds(old_depth + rng.choice([-1, 0, 1]), 2, 8)
        )
        _log_change("arch.rules.max_depth", old_depth, mutated.rules.max_depth)
        old_leaf = mutated.rules.min_samples_leaf
        mutated.rules.min_samples_leaf = int(
            clip_bounds(old_leaf + rng.choice([-1, 0, 1]), 1, 20)
        )
        _log_change("arch.rules.min_samples_leaf", old_leaf, mutated.rules.min_samples_leaf)
        old_alpha = mutated.rules.prune_ccp_alpha
        mutated.rules.prune_ccp_alpha = float(
            clip_bounds(old_alpha + rng.uniform(-0.01, 0.01), 0.0, 0.05)
        )
        _log_change("arch.rules.prune_ccp_alpha", old_alpha, mutated.rules.prune_ccp_alpha)

    if "explore" not in locks:
        old_k = mutated.explore.uncertainty_k
        factor = 1.0 + rng.uniform(-0.3, 0.3)
        new_k = int(round(old_k * factor / 8) * 8)
        mutated.explore.uncertainty_k = int(clip_bounds(new_k, 16, 512))
        _log_change("arch.explore.uncertainty_k", old_k, mutated.explore.uncertainty_k)
        old_rate = mutated.explore.mutation_rate
        mutated.explore.mutation_rate = float(
            clip_bounds(
                old_rate * (1.0 + rng.uniform(-jitter, jitter)),
                0.01,
                0.5,
            )
        )
        _log_change("arch.explore.mutation_rate", old_rate, mutated.explore.mutation_rate)
        old_lambda = mutated.explore.lambda_consistency
        mutated.explore.lambda_consistency = float(
            clip_bounds(
                old_lambda + rng.uniform(-0.05, 0.05),
                0.0,
                0.7,
            )
        )
        _log_change("arch.explore.lambda_consistency", old_lambda, mutated.explore.lambda_consistency)

    return mutated
