from __future__ import annotations

import gymnasium as gym
import numpy as np

from vrl_dissertation.envs.shifts.base import ShiftSpec
from vrl_dissertation.envs.shifts.physics import apply_physics_shift
from vrl_dissertation.envs.shifts.spatial import build_spatial_transform
from vrl_dissertation.envs.shifts.temporal import TemporalInterruption


class ShiftedEnvWrapper(gym.Wrapper):
    """Applies a controlled shift only for evaluation environment."""

    def __init__(self, env: gym.Env, spec: ShiftSpec, seed: int) -> None:
        super().__init__(env)
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.temporal: TemporalInterruption | None = None
        self.spatial_fn = None

        if spec.category == "spatial":
            self.spatial_fn = build_spatial_transform(spec.name, spec.params, self.rng)
        elif spec.category == "temporal":
            self.temporal = TemporalInterruption(spec.name, spec.params, self.rng)
        elif spec.category == "physics":
            apply_physics_shift(self.env, spec.name, spec.params)
        else:
            raise ValueError(f"Unsupported category: {spec.category}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.temporal is not None:
            self.temporal.reset()
            obs = self.temporal.apply(obs)
        elif self.spatial_fn is not None:
            obs = self.spatial_fn(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.temporal is not None:
            obs = self.temporal.apply(obs)
        elif self.spatial_fn is not None:
            obs = self.spatial_fn(obs)

        info = dict(info)
        info["shift_scenario_id"] = self.spec.scenario_id
        info["shift_category"] = self.spec.category
        info["shift_name"] = self.spec.name
        info["shift_severity"] = self.spec.severity
        return obs, reward, terminated, truncated, info
