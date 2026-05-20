from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouterDecision:
    selected: str
    normal_score: float
    hard_score: float | None = None
    margin: float = 0.0
    auto_hard_triggered: bool = False
    trigger_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "selected": self.selected,
            "normal_score": round(self.normal_score, 4),
            "margin": self.margin,
            "auto_hard_triggered": self.auto_hard_triggered,
            "trigger_reasons": list(self.trigger_reasons),
        }
        if self.hard_score is not None:
            result["hard_score"] = round(self.hard_score, 4)
        return result


class RouterSkill:
    """Route examples between normal sparse grounding and hard dense search."""

    def __init__(self, hard_margin: float = 0.2, trigger_quality_threshold: float = 5.0):
        self.hard_margin = hard_margin
        self.trigger_quality_threshold = trigger_quality_threshold

    @staticmethod
    def row_duration(row: dict[str, Any]) -> float:
        try:
            return float(row["pred_end"]) - float(row["pred_start"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def verifier_checks(row: dict[str, Any]) -> dict[str, Any]:
        verifier = row.get("verifier") or {}
        fine = row.get("fine") or {}
        return verifier.get("checks") or (fine.get("verifier") or {})

    @staticmethod
    def requirement_types(program: dict[str, Any] | None) -> set[str]:
        if not program:
            return set()
        return {
            str(req.get("type"))
            for req in program.get("requirements", [])
            if isinstance(req, dict) and req.get("type")
        }

    def routing_quality(self, row: dict[str, Any]) -> float:
        if row.get("error"):
            return -100.0
        duration = self.row_duration(row)
        score = 0.0
        if duration <= 0:
            score -= 5.0
        elif duration < 3:
            score -= 1.5
        elif duration > 30:
            score -= 0.8
        else:
            score += 0.5

        fine = row.get("fine") or {}
        try:
            score += float(fine.get("confidence") or 0.0) * 2.0
        except (TypeError, ValueError):
            pass
        evidence = fine.get("matched_evidence") or []
        score += min(len(evidence), 5) * 0.15

        checks = self.verifier_checks(row)
        if checks.get("complete") is True:
            score += 0.8
        elif checks.get("complete") is False:
            score -= 0.8
        if checks.get("temporal_order_ok") is True:
            score += 0.4
        elif checks.get("temporal_order_ok") is False:
            score -= 0.8
        if checks.get("same_event") is True:
            score += 0.6
        elif checks.get("same_event") is False:
            score -= 1.0

        boundary_quality = checks.get("boundary_quality")
        if boundary_quality == "tight":
            score += 0.5
        elif boundary_quality in {"too_short", "too_long"}:
            score -= 0.4
        elif boundary_quality == "uncertain":
            score -= 0.8

        if fine.get("top_candidates"):
            score += 0.2
        return score

    def should_hard(self, row: dict[str, Any], program: dict[str, Any] | None) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        types = self.requirement_types(program)
        duration = self.row_duration(row)
        checks = self.verifier_checks(row)
        fine = row.get("fine") or {}
        query = str(row.get("query", "")).lower()

        if row.get("error"):
            reasons.append("normal_error")
        if duration <= 0:
            reasons.append("degenerate_interval")
        elif duration < 4:
            reasons.append("very_short_interval")
        if self.routing_quality(row) < self.trigger_quality_threshold:
            reasons.append("low_routing_quality")
        if checks.get("boundary_quality") in {"too_short", "too_long", "uncertain"}:
            reasons.append(f"boundary_{checks.get('boundary_quality')}")
        if checks.get("complete") is False or checks.get("same_event") is False:
            reasons.append("verifier_failed")
        if (row.get("coarse") or {}).get("relevant") is False:
            reasons.append("coarse_uncertain")
        if len(fine.get("matched_evidence") or []) <= 1:
            reasons.append("sparse_evidence")
        if types & {
            "state_hold",
            "state_transition",
            "interaction_transition",
            "relation_transition",
            "unknown_event",
        }:
            reasons.append("risk_requirement_type")
        if any(
            word in query
            for word in ["put", "puts", "take", "takes", "open", "close", "closes", "smil", "laugh", "drink"]
        ):
            reasons.append("risk_query_verb")

        transition_types = {
            "state_transition",
            "interaction_transition",
            "relation_transition",
            "action_event",
            "unknown_event",
        }
        try:
            pred_start = float(row["pred_start"])
        except (KeyError, TypeError, ValueError):
            pred_start = 0.0
        weak_reasons = {
            "normal_error",
            "degenerate_interval",
            "very_short_interval",
            "low_routing_quality",
            "boundary_too_short",
            "boundary_too_long",
            "boundary_uncertain",
            "verifier_failed",
            "coarse_uncertain",
            "sparse_evidence",
        }
        risk_late_transition = (
            {"risk_requirement_type", "risk_query_verb"}.issubset(reasons)
            and bool(types & transition_types)
            and pred_start >= 12.0
            and duration <= 16.0
        )
        return bool((set(reasons) & weak_reasons) or risk_late_transition), reasons

    def choose(self, normal: dict[str, Any], hard: dict[str, Any] | None = None) -> tuple[dict[str, Any], RouterDecision]:
        normal_score = self.routing_quality(normal)
        if not hard or hard.get("error"):
            return normal, RouterDecision(selected="normal", normal_score=normal_score, margin=self.hard_margin)

        hard_score = self.routing_quality(hard)
        try:
            hard_starts_earlier = float(hard["pred_start"]) + 1.0 <= float(normal["pred_start"])
        except (KeyError, TypeError, ValueError):
            hard_starts_earlier = False
        choose_hard = hard_score >= normal_score + self.hard_margin
        if hard_starts_earlier and hard_score >= normal_score - 0.2:
            choose_hard = True
        selected = hard if choose_hard else normal
        decision = RouterDecision(
            selected="hard" if choose_hard else "normal",
            normal_score=normal_score,
            hard_score=hard_score,
            margin=self.hard_margin,
        )
        return selected, decision
