from __future__ import annotations

from .models import Delta, LocateResult, MatchedDelta, StateProgram, Trace, TraceState
from .verifier import verify_segment


def locate_moments(
    trace: Trace,
    program: StateProgram,
    *,
    window_seconds: float = 8.0,
    top_k: int = 3,
) -> list[LocateResult]:
    if len(trace.states) < 2:
        return []

    candidates: list[LocateResult] = []
    for start_idx, start in enumerate(trace.states[:-1]):
        for end_idx in range(start_idx + 1, len(trace.states)):
            end = trace.states[end_idx]
            if end.t - start.t > window_seconds:
                break
            segment = trace.states[start_idx : end_idx + 1]
            matched = _match_deltas(segment, program.deltas)
            result = _score_segment(segment, program, matched)
            if result.score > 0:
                candidates.append(result)

    candidates.sort(key=lambda item: (item.verification.ok, item.score), reverse=True)
    return _dedupe(candidates)[:top_k]


def _match_deltas(segment: list[TraceState], deltas: list[Delta]) -> list[MatchedDelta]:
    matched: list[MatchedDelta] = []
    for delta in deltas:
        for prev, cur in zip(segment, segment[1:]):
            before = prev.state.get(delta.var)
            after = cur.state.get(delta.var)
            before_ok = delta.before is None or before == delta.before
            after_ok = after == delta.after
            changed = before != after
            if before_ok and after_ok and changed:
                matched.append(
                    MatchedDelta(
                        delta=delta,
                        start_t=prev.t,
                        end_t=cur.t,
                        before_value=before,
                        after_value=after,
                    )
                )
                break
    return matched


def _score_segment(
    segment: list[TraceState],
    program: StateProgram,
    matched: list[MatchedDelta],
) -> LocateResult:
    required = [delta for delta in program.deltas if delta.required]
    required_count = max(1, len(required))
    matched_required = [
        item for item in matched if any(item.delta == delta for delta in required)
    ]
    transition_score = len(matched_required) / required_count

    wanted_values = {(delta.var, delta.after) for delta in program.deltas}
    state_hits = 0
    for state in segment:
        state_hits += sum(state.state.get(var) == value for var, value in wanted_values)
    state_score = state_hits / max(1, len(segment) * len(wanted_values))

    verification = verify_segment(segment, program, matched)
    constraint_score = 1.0 if verification.order else 0.0
    score = 0.35 * state_score + 0.5 * transition_score + 0.15 * constraint_score

    return LocateResult(
        start_t=segment[0].t,
        end_t=segment[-1].t,
        score=round(score, 4),
        state_score=round(state_score, 4),
        transition_score=round(transition_score, 4),
        constraint_score=round(constraint_score, 4),
        matched_deltas=matched,
        verification=verification,
    )


def _dedupe(results: list[LocateResult]) -> list[LocateResult]:
    kept: list[LocateResult] = []
    for result in results:
        overlaps = any(
            result.start_t <= item.end_t and item.start_t <= result.end_t
            for item in kept
        )
        if not overlaps:
            kept.append(result)
    return kept
