from __future__ import annotations

from .models import MatchedDelta, StateProgram, TraceState, Verification


def verify_segment(
    segment: list[TraceState],
    program: StateProgram,
    matched_deltas: list[MatchedDelta],
) -> Verification:
    reasons: list[str] = []

    required = [delta for delta in program.deltas if delta.required]
    matched_vars = {match.delta.var for match in matched_deltas}
    completeness = all(delta.var in matched_vars for delta in required)
    if not completeness:
        missing = [delta.var for delta in required if delta.var not in matched_vars]
        reasons.append(f"missing required transitions: {', '.join(missing)}")

    order = _check_constraints(program, matched_deltas)
    if not order:
        reasons.append("constraint order is not satisfied")

    single_event = _check_single_event(segment, matched_deltas)
    if not single_event:
        reasons.append("matched transitions are too far apart for one event")

    ok = completeness and order and single_event
    if ok:
        reasons.append("all required transitions and constraints are satisfied")

    return Verification(
        ok=ok,
        completeness=completeness,
        order=order,
        single_event=single_event,
        reasons=reasons,
    )


def _check_constraints(
    program: StateProgram, matched_deltas: list[MatchedDelta]
) -> bool:
    by_var = {match.delta.var: match for match in matched_deltas}
    for constraint in program.constraints:
        left = by_var.get(constraint.left)
        right = by_var.get(constraint.right)
        if left is None or right is None:
            continue
        if constraint.relation == "BEFORE":
            if not left.end_t < right.start_t:
                return False
        elif constraint.relation == "BEFORE_OR_AT":
            if not left.end_t <= right.end_t:
                return False
        elif constraint.relation == "AFTER":
            if not left.start_t > right.end_t:
                return False
    return True


def _check_single_event(
    segment: list[TraceState], matched_deltas: list[MatchedDelta]
) -> bool:
    if len(matched_deltas) <= 1:
        return True
    first = min(match.start_t for match in matched_deltas)
    last = max(match.end_t for match in matched_deltas)
    span = max(0.0, segment[-1].t - segment[0].t)
    return (last - first) <= max(2.0, span * 0.75)
