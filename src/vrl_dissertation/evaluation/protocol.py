from __future__ import annotations

from dataclasses import dataclass

from vrl_dissertation.envs.shifts.base import ShiftSpec


@dataclass(frozen=True)
class EvalProtocol:
    """Evaluation-only protocol; no training-time corruption here."""

    clean_eval_episodes: int
    shifted_eval_episodes: int
    base_seed: int

    def scenario_seed(self, scenario_index: int, seed_offset: int = 10000) -> int:
        return self.base_seed + seed_offset + scenario_index


def protocol_for_seed(seed: int) -> EvalProtocol:
    return EvalProtocol(clean_eval_episodes=10, shifted_eval_episodes=10, base_seed=seed)


def scenario_to_run_name(spec: ShiftSpec, seed: int) -> str:
    return f"{spec.scenario_id}/seed_{seed}"
