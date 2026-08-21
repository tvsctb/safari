import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ar_irnn_scale_lr_crossfade_study.py"
SPEC = importlib.util.spec_from_file_location("irnn_scale_crossfade", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


class IRNNScaleLRCrossfadeStudyTest(unittest.TestCase):
    def test_exact_factorial_and_lr_semantics(self):
        trials = study.tuning_trials()
        self.assertEqual(len(trials), 8)
        self.assertEqual(len({trial.track for trial in trials}), 4)
        for trial in trials:
            self.assertEqual(trial.max_epochs, 400)
            self.assertEqual(trial.rho, 8.0)
            self.assertEqual(trial.tau, 243.242356581615)
            self.assertEqual(trial.memory_scale_target_mode, "learned")
            self.assertIsNone(trial.memory_scale_target)
            self.assertTrue(trial.scale_crossfade_enabled)
            self.assertEqual(trial.gaussian_lr_initial, 0.0)
            self.assertEqual(trial.target_lr_final, 0.0)
            self.assertEqual(trial.scale_crossfade_steps, 12560)

    def test_command_contains_crossfade_without_a_fixed_goal(self):
        trial = study.tuning_trials()[0]
        with tempfile.TemporaryDirectory() as directory:
            command = study.build_rnn_command(
                trial, Path(directory), "project", "entity", "group"
            )
        self.assertIn("model.memory_scale_target=null", command)
        self.assertIn("model.memory_scale_target_mode=learned", command)
        self.assertIn("model.gaussian_scale_mode=learned", command)
        self.assertIn("callbacks.rnn_scale_crossfade.enabled=true", command)
        self.assertIn(
            "callbacks.rnn_scale_crossfade.gaussian_lr_initial=0.0", command
        )
        self.assertIn(
            "callbacks.rnn_scale_crossfade.target_lr_final=0.0", command
        )


if __name__ == "__main__":
    unittest.main()
