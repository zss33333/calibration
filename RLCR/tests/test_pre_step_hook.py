"""Verify that the gradient observer runs before the optimizer clears it."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pre_step_hook import install_pre_step_hook


class PreStepHookTest(unittest.TestCase):
    def test_observer_reads_gradient_before_step_clears_it(self):
        class Engine:
            gradient = 3.0

            def step(self, multiplier=1):
                result = self.gradient * multiplier
                self.gradient = None
                return result

        engine = Engine()
        observed = []
        install_pre_step_hook(engine, lambda: observed.append(engine.gradient))
        self.assertEqual(engine.step(multiplier=2), 6.0)
        self.assertEqual(observed, [3.0])
        self.assertIsNone(engine.gradient)


if __name__ == "__main__":
    unittest.main()
