from __future__ import annotations

import re

from .models import Constraint, Delta, StateProgram


TAKE_FROM_RE = re.compile(
    r"\b(?:take|remove|get|grab|pick up)\s+(?P<object>[\w-]+)\s+from\s+(?P<container>[\w-]+)\b",
    re.IGNORECASE,
)
PUT_IN_RE = re.compile(
    r"\b(?:put|place|insert)\s+(?P<object>[\w-]+)\s+(?:in|into|inside)\s+(?P<container>[\w-]+)\b",
    re.IGNORECASE,
)
OPEN_RE = re.compile(r"\bopen\s+(?P<object>[\w-]+)\b", re.IGNORECASE)
CLOSE_RE = re.compile(r"\bclose\s+(?P<object>[\w-]+)\b", re.IGNORECASE)


def induce_state_program(query: str) -> StateProgram:
    """Induce a structured state-change program from a natural-language query.

    This is intentionally deterministic for the MVP. It covers common physical
    manipulation patterns and can later be replaced by an LLM-backed inducer.
    """

    normalized = query.strip()

    match = TAKE_FROM_RE.search(normalized)
    if match:
        obj = match.group("object").lower()
        container = match.group("container").lower()
        return StateProgram(
            query=normalized,
            variables={
                f"{container}.state": ["open", "closed"],
                f"{obj}.loc": ["inside", "outside"],
            },
            deltas=[
                Delta(f"{container}.state", "closed", "open"),
                Delta(f"{obj}.loc", "inside", "outside"),
            ],
            constraints=[Constraint(f"{container}.state", "BEFORE_OR_AT", f"{obj}.loc")],
        )

    match = PUT_IN_RE.search(normalized)
    if match:
        obj = match.group("object").lower()
        container = match.group("container").lower()
        return StateProgram(
            query=normalized,
            variables={
                f"{container}.state": ["open", "closed"],
                f"{obj}.loc": ["inside", "outside"],
            },
            deltas=[
                Delta(f"{container}.state", "closed", "open"),
                Delta(f"{obj}.loc", "outside", "inside"),
            ],
            constraints=[Constraint(f"{container}.state", "BEFORE_OR_AT", f"{obj}.loc")],
        )

    match = OPEN_RE.search(normalized)
    if match:
        obj = match.group("object").lower()
        return StateProgram(
            query=normalized,
            variables={f"{obj}.state": ["open", "closed"]},
            deltas=[Delta(f"{obj}.state", "closed", "open")],
        )

    match = CLOSE_RE.search(normalized)
    if match:
        obj = match.group("object").lower()
        return StateProgram(
            query=normalized,
            variables={f"{obj}.state": ["open", "closed"]},
            deltas=[Delta(f"{obj}.state", "open", "closed")],
        )

    words = [word.lower() for word in re.findall(r"[\w-]+", normalized)]
    variables = {f"{word}.visible": [True, False] for word in words[-2:]}
    deltas = [Delta(var, False, True, required=False) for var in variables]
    return StateProgram(query=normalized, variables=variables, deltas=deltas)
