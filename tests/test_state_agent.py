import unittest

from video_moment_agent.io import load_trace
from video_moment_agent.search import locate_moments
from video_moment_agent.state_program import induce_state_program


class StateAgentTest(unittest.TestCase):
    def test_induce_take_from(self):
        program = induce_state_program("take milk from fridge")
        self.assertEqual(["fridge.state", "milk.loc"], [delta.var for delta in program.deltas])
        self.assertEqual("closed", program.deltas[0].before)
        self.assertEqual("open", program.deltas[0].after)
        self.assertEqual("inside", program.deltas[1].before)
        self.assertEqual("outside", program.deltas[1].after)

    def test_locate_fridge_trace(self):
        trace = load_trace("examples/fridge_trace.json")
        program = induce_state_program("take milk from fridge")
        results = locate_moments(trace, program)
        self.assertTrue(results)
        best = results[0]
        self.assertTrue(best.verification.ok)
        self.assertEqual(1.0, best.transition_score)
        self.assertLessEqual(best.start_t, 2.0)
        self.assertGreaterEqual(best.end_t, 3.0)


if __name__ == "__main__":
    unittest.main()
