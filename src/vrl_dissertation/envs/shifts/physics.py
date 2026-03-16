from __future__ import annotations


def apply_physics_shift(env, name: str, params: dict) -> None:
    """Best-effort physics parameter application.

    This function is intentionally backend-agnostic. For a specific environment family
    (e.g., DMControl), replace with exact API calls in an environment-specific adapter.
    """

    unwrapped = getattr(env, "unwrapped", env)

    if name == "friction_scale":
        scale = float(params["scale"])
        if hasattr(unwrapped, "friction"):
            unwrapped.friction *= scale
        else:
            setattr(unwrapped, "friction_scale", scale)
        return

    if name == "mass_scale":
        scale = float(params["scale"])
        if hasattr(unwrapped, "mass"):
            unwrapped.mass *= scale
        else:
            setattr(unwrapped, "mass_scale", scale)
        return

    if name == "action_limit_scale":
        scale = float(params["scale"])
        setattr(unwrapped, "action_limit_scale", scale)
        return

    raise ValueError(f"Unknown physics shift: {name}")
