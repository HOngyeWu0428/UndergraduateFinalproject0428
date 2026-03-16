from __future__ import annotations

from vrl_dissertation.envs.shifts.base import ShiftSpec


def minimal_dissertation_scenarios() -> list[ShiftSpec]:
    """First-pass scenarios for a minimal dissertation experiment set.

    2 severities per category for quick but meaningful robustness comparison.
    """

    return [
        # Spatial corruption
        ShiftSpec("spatial_gauss_s1", "spatial", "gaussian_noise", 1, {"sigma": 8.0}),
        ShiftSpec("spatial_gauss_s2", "spatial", "gaussian_noise", 2, {"sigma": 16.0}),
        ShiftSpec("spatial_color_s1", "spatial", "color_distortion", 1, {"brightness": 8.0, "contrast": 0.9}),
        ShiftSpec("spatial_color_s2", "spatial", "color_distortion", 2, {"brightness": 16.0, "contrast": 0.8}),
        # Temporal interruption
        ShiftSpec("temporal_freeze_s1", "temporal", "frame_freeze", 1, {"freeze_prob": 0.10}),
        ShiftSpec("temporal_freeze_s2", "temporal", "frame_freeze", 2, {"freeze_prob": 0.20}),
        ShiftSpec("temporal_delay_s1", "temporal", "frame_delay", 1, {"delay_k": 1}),
        ShiftSpec("temporal_delay_s2", "temporal", "frame_delay", 2, {"delay_k": 2}),
        # Physics shift (env-adapter dependent)
        ShiftSpec("physics_friction_s1", "physics", "friction_scale", 1, {"scale": 1.2}),
        ShiftSpec("physics_friction_s2", "physics", "friction_scale", 2, {"scale": 1.5}),
        ShiftSpec("physics_mass_s1", "physics", "mass_scale", 1, {"scale": 1.2}),
        ShiftSpec("physics_mass_s2", "physics", "mass_scale", 2, {"scale": 1.5}),
    ]
