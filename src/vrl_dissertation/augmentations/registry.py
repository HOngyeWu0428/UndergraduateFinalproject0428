from __future__ import annotations

from vrl_dissertation.augmentations.base import Augmentation, IdentityAugmentation
from vrl_dissertation.augmentations.random_shift import RandomShiftAugmentation
from vrl_dissertation.baseline_config import AugmentationConfig


def build_augmentation(cfg: AugmentationConfig) -> Augmentation:
    if cfg.name == "none":
        return IdentityAugmentation()
    if cfg.name == "random_shift":
        return RandomShiftAugmentation(pad=cfg.pad)
    raise ValueError(f"Unknown augmentation: {cfg.name}")
