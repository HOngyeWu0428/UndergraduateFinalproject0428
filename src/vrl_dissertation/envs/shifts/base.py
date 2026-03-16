from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ShiftCategory = Literal["spatial", "temporal", "physics"]


@dataclass(frozen=True)
class ShiftSpec:
    """Single controlled domain-shift setting used for evaluation."""

    scenario_id: str
    category: ShiftCategory
    name: str
    severity: int
    params: dict[str, Any]
