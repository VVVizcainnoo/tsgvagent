from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


StateValue = str | int | float | bool | None
StateDict = dict[str, StateValue]


@dataclass(frozen=True)
class Delta:
    """A required state transition, such as fridge.state: closed -> open."""

    var: str
    before: StateValue | None
    after: StateValue
    required: bool = True


@dataclass(frozen=True)
class Constraint:
    """A relation between two deltas."""

    left: str
    relation: str
    right: str


@dataclass(frozen=True)
class StateProgram:
    query: str
    variables: dict[str, list[StateValue]]
    deltas: list[Delta]
    constraints: list[Constraint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "S": self.variables,
            "Delta": [delta.__dict__ for delta in self.deltas],
            "C": [constraint.__dict__ for constraint in self.constraints],
        }


@dataclass(frozen=True)
class TraceState:
    t: float
    state: StateDict
    frame: str | None = None


@dataclass(frozen=True)
class Trace:
    video_id: str
    states: list[TraceState]


@dataclass(frozen=True)
class MatchedDelta:
    delta: Delta
    start_t: float
    end_t: float
    before_value: StateValue
    after_value: StateValue


@dataclass(frozen=True)
class Verification:
    ok: bool
    completeness: bool
    order: bool
    single_event: bool
    reasons: list[str]


@dataclass(frozen=True)
class LocateResult:
    start_t: float
    end_t: float
    score: float
    state_score: float
    transition_score: float
    constraint_score: float
    matched_deltas: list[MatchedDelta]
    verification: Verification
