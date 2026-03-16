from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Augmentation(nn.Module, ABC):
    """Base interface for observation augmentations used in training updates."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a batch of observations (N, C, H, W)."""
        raise NotImplementedError


class IdentityAugmentation(Augmentation):
    """No-op augmentation for baseline runs."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs
