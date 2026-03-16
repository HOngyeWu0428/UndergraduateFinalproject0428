from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int
    seed: int
    batch_size: int


@dataclass(frozen=True)
class EnvConfig:
    env_id: str
    frame_stack: int
    image_size: int
    action_repeat: int


@dataclass(frozen=True)
class AugmentationConfig:
    name: Literal["none", "random_shift"]
    pad: int = 4


@dataclass(frozen=True)
class EvalConfig:
    eval_interval: int
    eval_episodes: int
    survival_threshold: float


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    algorithm: Literal["sac"]
    train: TrainConfig
    env: EnvConfig
    augmentation: AugmentationConfig
    evaluation: EvalConfig


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    cfg = ExperimentConfig(
        experiment_name=raw["experiment_name"],
        algorithm=raw["algorithm"],
        train=TrainConfig(**raw["train"]),
        env=EnvConfig(**raw["env"]),
        augmentation=AugmentationConfig(**raw["augmentation"]),
        evaluation=EvalConfig(**raw["evaluation"]),
    )

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    _require_positive("train.total_steps", cfg.train.total_steps)
    _require_positive("train.batch_size", cfg.train.batch_size)
    _require_positive("env.frame_stack", cfg.env.frame_stack)
    _require_positive("env.image_size", cfg.env.image_size)
    _require_positive("env.action_repeat", cfg.env.action_repeat)
    _require_positive("evaluation.eval_interval", cfg.evaluation.eval_interval)
    _require_positive("evaluation.eval_episodes", cfg.evaluation.eval_episodes)

    if not (0.0 <= cfg.evaluation.survival_threshold <= 1.0):
        raise ValueError(
            "evaluation.survival_threshold must be in [0, 1], got "
            f"{cfg.evaluation.survival_threshold}"
        )

    if cfg.augmentation.name == "none" and cfg.augmentation.pad != 0:
        raise ValueError("augmentation.pad must be 0 when augmentation.name='none'")

    if cfg.augmentation.name == "random_shift" and cfg.augmentation.pad <= 0:
        raise ValueError("augmentation.pad must be > 0 for random_shift")
