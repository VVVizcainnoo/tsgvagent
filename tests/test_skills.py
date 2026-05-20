import tempfile
import unittest
from pathlib import Path

from video_moment_agent.skills import RouterSkill, RunnerSkill


class SkillTest(unittest.TestCase):
    def test_runner_merge_overlays_successful_rerun(self):
        runner = RunnerSkill()
        annotations = [{"id": "q1", "start": 1.0, "end": 5.0}, {"id": "q2", "start": 0.0, "end": 2.0}]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.jsonl"
            overlay = tmp_path / "overlay.jsonl"
            runner.write_jsonl(
                base,
                [
                    {"id": "q1", "pred_start": 1.0, "pred_end": 4.0},
                    {"id": "q2", "error": "timeout", "pred_start": None, "pred_end": None},
                ],
            )
            runner.write_jsonl(overlay, [{"id": "q2", "pred_start": 0.0, "pred_end": 2.0}])

            merged = runner.merge(annotations, [base], [overlay])

        self.assertEqual(2, merged.metrics["rows"])
        self.assertEqual(0, merged.metrics["errors"])
        self.assertEqual(1, merged.overlaid_rows)
        self.assertGreater(merged.metrics["mIoU"], 0.8)

    def test_router_margin_prefers_normal_when_hard_gain_is_small(self):
        router = RouterSkill(hard_margin=0.5)
        normal = {
            "pred_start": 1.0,
            "pred_end": 8.0,
            "fine": {"confidence": 0.8, "matched_evidence": ["#1 t=1.0s", "#2 t=3.0s"]},
            "verifier": {"checks": {"complete": True, "temporal_order_ok": True, "same_event": True, "boundary_quality": "tight"}},
        }
        hard = {
            "pred_start": 1.0,
            "pred_end": 8.0,
            "fine": {"confidence": 0.9, "matched_evidence": ["#1 t=1.0s", "#2 t=3.0s"]},
            "verifier": {"checks": {"complete": True, "temporal_order_ok": True, "same_event": True, "boundary_quality": "tight"}},
        }

        _, decision = router.choose(normal, hard)

        self.assertEqual("normal", decision.selected)
        self.assertLess(decision.hard_score - decision.normal_score, 0.5)

    def test_router_allows_earlier_hard_when_scores_are_close(self):
        router = RouterSkill(hard_margin=0.5)
        normal = {
            "pred_start": 16.0,
            "pred_end": 24.0,
            "fine": {"confidence": 0.8, "matched_evidence": ["#1 t=16.0s", "#2 t=20.0s"]},
            "verifier": {"checks": {"complete": True, "temporal_order_ok": True, "same_event": True, "boundary_quality": "tight"}},
        }
        hard = {
            "pred_start": 12.0,
            "pred_end": 17.0,
            "fine": {"confidence": 0.75, "matched_evidence": ["#1 t=12.0s", "#2 t=15.0s"]},
            "verifier": {"checks": {"complete": True, "temporal_order_ok": True, "same_event": True, "boundary_quality": "tight"}},
        }

        _, decision = router.choose(normal, hard)

        self.assertEqual("hard", decision.selected)

    def test_router_does_not_trigger_hard_on_risk_tags_only(self):
        router = RouterSkill()
        row = {
            "query": "person takes a cup.",
            "pred_start": 1.0,
            "pred_end": 8.0,
            "fine": {"confidence": 1.0, "matched_evidence": ["#1 t=1.0s", "#2 t=3.0s", "#3 t=5.0s"]},
            "coarse": {"relevant": True},
            "verifier": {"checks": {"complete": True, "temporal_order_ok": True, "same_event": True, "boundary_quality": "tight"}},
        }
        program = {"requirements": [{"type": "interaction_transition"}]}

        triggered, reasons = router.should_hard(row, program)

        self.assertFalse(triggered)
        self.assertIn("risk_requirement_type", reasons)
        self.assertIn("risk_query_verb", reasons)

    def test_router_triggers_hard_for_late_risky_transition(self):
        router = RouterSkill()
        row = {
            "query": "person takes a cup.",
            "pred_start": 16.0,
            "pred_end": 24.0,
            "fine": {"confidence": 1.0, "matched_evidence": ["#1 t=16.0s", "#2 t=20.0s", "#3 t=23.0s"]},
            "coarse": {"relevant": True},
            "verifier": {"checks": {"complete": True, "temporal_order_ok": True, "same_event": True, "boundary_quality": "tight"}},
        }
        program = {"requirements": [{"type": "interaction_transition"}]}

        triggered, reasons = router.should_hard(row, program)

        self.assertTrue(triggered)
        self.assertIn("risk_requirement_type", reasons)
        self.assertIn("risk_query_verb", reasons)


if __name__ == "__main__":
    unittest.main()
