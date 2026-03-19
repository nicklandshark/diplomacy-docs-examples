from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from data_generator import EnvironmentKind

HYBRID_ENVIRONMENTS: tuple[EnvironmentKind, ...] = (
    "tool_accuracy",
    "target_execution",
    "supported_target",
    "cooperative_press",
    "full_press",
)

HYBRID_MAX_TURNS_BY_ENVIRONMENT: dict[EnvironmentKind, int] = {
    "tool_accuracy": 14,
    "target_execution": 12,
    "supported_target": 14,
    "cooperative_press": 16,
    "full_press": 20,
}

HYBRID_MAX_TRAJECTORY_TOKENS_BY_ENVIRONMENT: dict[EnvironmentKind, int] = {
    "tool_accuracy": 8_192,
    "target_execution": 8_192,
    "supported_target": 8_192,
    "cooperative_press": 12_288,
    "full_press": 12_288,
}

HYBRID_TOTAL_BATCHES_DEFAULT = 80
HYBRID_UI_TOTAL_BATCHES_DEFAULT = 16
HYBRID_BATCH_SIZE = 16
HYBRID_GROUP_SIZE = 4
HYBRID_MAX_TOKENS = 384
HYBRID_EVAL_EXAMPLES_PER_ENVIRONMENT = 4
HYBRID_TRAIN_SEED = 21
HYBRID_EVAL_SEED_BASE = 10_021
HYBRID_TRAIN_SEED_OFFSETS: dict[EnvironmentKind, int] = {
    "tool_accuracy": 0,
    "target_execution": 1_000,
    "supported_target": 2_000,
    "cooperative_press": 3_000,
    "full_press": 4_000,
}
HYBRID_EVAL_SEED_OFFSETS: dict[EnvironmentKind, int] = {
    "tool_accuracy": 0,
    "target_execution": 100,
    "supported_target": 200,
    "cooperative_press": 300,
    "full_press": 400,
}


@dataclass(frozen=True)
class HybridPhaseSpec:
    index: int
    name: str
    start_batch: int
    end_batch_exclusive: int | None
    environment_weights: dict[EnvironmentKind, float]

    def contains(self, batch_index: int) -> bool:
        if batch_index < self.start_batch:
            return False
        if self.end_batch_exclusive is None:
            return True
        return batch_index < self.end_batch_exclusive

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "start_batch": self.start_batch,
            "end_batch_exclusive": self.end_batch_exclusive,
            "environment_weights": {
                environment: float(weight)
                for environment, weight in self.environment_weights.items()
            },
        }


HYBRID_PHASE_SCHEDULE: tuple[HybridPhaseSpec, ...] = (
    HybridPhaseSpec(
        index=1,
        name="phase_1_warmup_tool_accuracy",
        start_batch=0,
        end_batch_exclusive=2,
        environment_weights={
            "tool_accuracy": 1.0,
            "target_execution": 0.0,
            "supported_target": 0.0,
            "cooperative_press": 0.0,
            "full_press": 0.0,
        },
    ),
    HybridPhaseSpec(
        index=2,
        name="phase_2_easy_to_hard_bridge",
        start_batch=2,
        end_batch_exclusive=6,
        environment_weights={
            "tool_accuracy": 0.40,
            "target_execution": 0.30,
            "supported_target": 0.15,
            "cooperative_press": 0.10,
            "full_press": 0.05,
        },
    ),
    HybridPhaseSpec(
        index=3,
        name="phase_3_balanced_mix",
        start_batch=6,
        end_batch_exclusive=10,
        environment_weights={
            "tool_accuracy": 0.20,
            "target_execution": 0.20,
            "supported_target": 0.20,
            "cooperative_press": 0.20,
            "full_press": 0.20,
        },
    ),
    HybridPhaseSpec(
        index=4,
        name="phase_4_full_press_ramp",
        start_batch=10,
        end_batch_exclusive=14,
        environment_weights={
            "tool_accuracy": 0.10,
            "target_execution": 0.10,
            "supported_target": 0.15,
            "cooperative_press": 0.20,
            "full_press": 0.45,
        },
    ),
    HybridPhaseSpec(
        index=5,
        name="phase_5_full_press_tail",
        start_batch=14,
        end_batch_exclusive=None,
        environment_weights={
            "tool_accuracy": 0.05,
            "target_execution": 0.05,
            "supported_target": 0.10,
            "cooperative_press": 0.15,
            "full_press": 0.65,
        },
    ),
)


def phase_for_batch(batch_index: int) -> HybridPhaseSpec:
    normalized_batch = max(0, int(batch_index))
    for phase in HYBRID_PHASE_SCHEDULE:
        if phase.contains(normalized_batch):
            return phase
    return HYBRID_PHASE_SCHEDULE[-1]


def phase_schedule_payload() -> list[dict[str, Any]]:
    return [phase.to_manifest_dict() for phase in HYBRID_PHASE_SCHEDULE]


def environment_weights_for_batch(batch_index: int) -> dict[EnvironmentKind, float]:
    phase = phase_for_batch(batch_index)
    return dict(phase.environment_weights)


def sample_environment_plan(
    *,
    total_batches: int,
    batch_size: int,
    seed: int,
) -> tuple[tuple[EnvironmentKind, ...], ...]:
    rng = random.Random(seed)
    batches: list[tuple[EnvironmentKind, ...]] = []
    for batch_index in range(max(0, total_batches)):
        weights = environment_weights_for_batch(batch_index)
        environments = list(weights.keys())
        probabilities = [weights[environment] for environment in environments]
        chosen = tuple(
            rng.choices(environments, weights=probabilities, k=max(0, batch_size))
        )
        batches.append(chosen)
    return tuple(batches)


def batch_fraction_metrics(
    environment_batch: tuple[EnvironmentKind, ...],
    *,
    batch_size: int,
) -> dict[str, float]:
    counts = Counter(environment_batch)
    denominator = float(max(1, batch_size))
    return {
        f"mix/train_batch_fraction/{environment}": counts.get(environment, 0) / denominator
        for environment in HYBRID_ENVIRONMENTS
    }
