import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ar_rnn_controlled_lag_study.py"
SPEC = importlib.util.spec_from_file_location("controlled_lag_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControlledLagStudyTest(unittest.TestCase):
    def test_exact_selected_endpoint_matrix(self):
        trials = MODULE.selected_trials()
        self.assertEqual(len(trials), 16)
        cells = {(MODULE.condition(t), t.seed) for t in trials}
        expected = {
            (condition, seed)
            for condition in ("tanh-aux", "tanh-noaux", "irnn-aux", "irnn-noaux")
            for seed in (204, 205, 206, 207)
        }
        self.assertEqual(cells, expected)

    def test_selected_scales_are_exact(self):
        for trial in MODULE.selected_trials():
            self.assertEqual(trial.chunk_size, 4)
            self.assertFalse(trial.condition_memory_reconstruction_on_boundary)
            if trial.track == "tanh":
                self.assertEqual(trial.aux_weight, 0.05)
                self.assertEqual(trial.tau, MODULE.TANH_TAU)
                self.assertEqual(trial.activation, "tanh")
                self.assertEqual(trial.recurrent_init, "orthogonal")
            else:
                self.assertEqual(trial.aux_weight, 0.1)
                self.assertEqual(trial.tau, MODULE.IRNN_TAU)
                self.assertEqual(trial.activation, "relu")
                self.assertEqual(trial.recurrent_init, "identity")


if __name__ == "__main__":
    unittest.main()
