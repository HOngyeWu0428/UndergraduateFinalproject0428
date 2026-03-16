from __future__ import annotations

from collections.abc import Callable

import numpy as np


def add_gaussian_noise(obs: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = obs.astype(np.float32) + rng.normal(loc=0.0, scale=sigma, size=obs.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def color_distortion(obs: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    x = obs.astype(np.float32)
    x = x * contrast + brightness
    return np.clip(x, 0, 255).astype(np.uint8)


def texture_overlay(obs: np.ndarray, alpha: float, block_size: int = 6) -> np.ndarray:
    h, w = obs.shape[:2]
    yy, xx = np.indices((h, w))
    checker = ((yy // block_size + xx // block_size) % 2).astype(np.float32)
    checker = checker[..., None] * 255.0
    mixed = (1 - alpha) * obs.astype(np.float32) + alpha * checker
    return np.clip(mixed, 0, 255).astype(np.uint8)


def build_spatial_transform(name: str, params: dict, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
    if name == "gaussian_noise":
        sigma = float(params["sigma"])
        return lambda obs: add_gaussian_noise(obs, sigma=sigma, rng=rng)
    if name == "color_distortion":
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 1.0))
        return lambda obs: color_distortion(obs, brightness=brightness, contrast=contrast)
    if name == "texture_variation":
        alpha = float(params["alpha"])
        block_size = int(params.get("block_size", 6))
        return lambda obs: texture_overlay(obs, alpha=alpha, block_size=block_size)

    raise ValueError(f"Unknown spatial shift name: {name}")
