from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class EnvConfig:
    env_id: str
    image_size: int
    frame_stack: int


@dataclass(frozen=True)
class AugmentationConfig:
    name: Literal["none", "random_shift"]
    pad: int = 4


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int
    seed: int
    batch_size: int
    start_steps: int
    update_after: int
    update_every: int
    eval_interval: int
    eval_episodes: int
    replay_capacity: int
    gamma: float
    tau: float
    lr: float
    alpha_lr: float
    init_temperature: float


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int
    hidden_dim: int
    log_std_min: float
    log_std_max: float


@dataclass(frozen=True)
class OutputConfig:
    run_dir: str
    checkpoint_interval: int
    log_interval: int
    use_tensorboard: bool


@dataclass(frozen=True)
class BaselineExperimentConfig:
    experiment_name: str
    env: EnvConfig
    augmentation: AugmentationConfig
    train: TrainConfig
    model: ModelConfig
    output: OutputConfig


def load_baseline_config(path: str | Path) -> BaselineExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = BaselineExperimentConfig(
        experiment_name=raw["experiment_name"],
        env=EnvConfig(**raw["env"]),
        augmentation=AugmentationConfig(**raw.get("augmentation", {"name": "none", "pad": 0})),
        train=TrainConfig(**raw["train"]),
        model=ModelConfig(**raw["model"]),
        output=OutputConfig(**raw["output"]),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: BaselineExperimentConfig) -> None:
    if cfg.train.total_steps <= 0:
        raise ValueError("train.total_steps must be > 0")
    if cfg.train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0")
    if cfg.train.replay_capacity < cfg.train.batch_size:
        raise ValueError("train.replay_capacity must be >= train.batch_size")
    if not (0.0 < cfg.train.gamma <= 1.0):
        raise ValueError("train.gamma must be in (0,1]")
    if not (0.0 < cfg.train.tau <= 1.0):
        raise ValueError("train.tau must be in (0,1]")

    if cfg.augmentation.name == "none" and cfg.augmentation.pad != 0:
        raise ValueError("augmentation.pad must be 0 when augmentation.name='none'")
    if cfg.augmentation.name == "random_shift" and cfg.augmentation.pad <= 0:
        raise ValueError("augmentation.pad must be > 0 when augmentation.name='random_shift'")
