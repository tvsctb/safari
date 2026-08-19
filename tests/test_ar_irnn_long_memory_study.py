import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ar_irnn_long_memory_study.py"
SPEC = importlib.util.spec_from_file_location("irnn_long_memory_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IRNNLongMemoryStudyTest(unittest.TestCase):
    def test_exact_four_by_four_matrix(self):
        trials = MODULE.study_trials()
        self.assertEqual(len(trials), 16)
        self.assertEqual(
            {(trial.track, trial.seed) for trial in trials},
            {
                (f"irnn-{label}", seed)
                for label in ("memory2x", "memory4x", "boundary", "chunk8")
                for seed in (204, 205, 206, 207)
            },
        )

    def test_each_condition_changes_exactly_one_axis(self):
        for trial in MODULE.study_trials():
            self.assertEqual(trial.activation, "relu")
            self.assertEqual(trial.recurrent_init, "identity")
            self.assertEqual(trial.aux_weight, 0.1)
            self.assertEqual(trial.tau, MODULE.IRNN_TAU)
            if trial.track == "irnn-memory2x":
                self.assertAlmostEqual(trial.rho, 16.0 / math.sqrt(2.0))
                self.assertEqual(trial.chunk_size, 4)
                self.assertFalse(trial.condition_memory_reconstruction_on_boundary)
            elif trial.track == "irnn-memory4x":
                self.assertEqual(trial.rho, 8.0)
                self.assertEqual(trial.chunk_size, 4)
                self.assertFalse(trial.condition_memory_reconstruction_on_boundary)
            elif trial.track == "irnn-boundary":
                self.assertEqual(trial.rho, 16.0)
                self.assertEqual(trial.chunk_size, 4)
                self.assertTrue(trial.condition_memory_reconstruction_on_boundary)
            else:
                self.assertEqual(trial.rho, 16.0)
                self.assertEqual(trial.chunk_size, 8)
                self.assertFalse(trial.condition_memory_reconstruction_on_boundary)


if __name__ == "__main__":
    unittest.main()
