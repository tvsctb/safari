import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ar_irnn_warmup_scale_study.py"
SPEC = importlib.util.spec_from_file_location("irnn_warmup_scale", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


class IRNNWarmupScaleStudyTest(unittest.TestCase):
    def test_common_warmup_and_tuning_branch_cardinality(self):
        warmups = study.warmup_trials(study.TUNING_SEEDS)
        self.assertEqual(len(warmups), 2)
        self.assertTrue(all(trial.max_epochs == 80 for trial in warmups))
        self.assertTrue(
            all(trial.gaussian_scale_mode == "learned" for trial in warmups)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibrations = {
                seed: {"target_rms": 2.0 + seed / 1000}
                for seed in study.TUNING_SEEDS
            }
            trials = study.branch_trials(
                study.TUNING_SEEDS,
                root,
                calibrations,
                0.25,
                ("fixed", "learnable-free", "warmup-anchored"),
            )
        self.assertEqual(len(trials), 6)
        anchored = [trial for trial in trials if trial.track == "warmup-anchored"]
        self.assertEqual(len(anchored), 2)
        self.assertTrue(
            all(trial.memory_scale_constraint_weight == 0.25 for trial in anchored)
        )
        self.assertTrue(
            all(
                trial.gaussian_scale_learning_start_step
                == study.FULL_WARMUP_STEPS
                for trial in anchored
            )
        )

    def test_branch_commands_share_checkpoint_and_continuation_loader_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = study.branch_trials(
                (202,),
                root / "warmup",
                {202: {"target_rms": 3.5}},
                0.2,
                ("fixed", "warmup-anchored"),
            )
            commands = [
                study.build_rnn_command(
                    trial, root / "branch", "project", "entity", "group"
                )
                for trial in trials
            ]
        checkpoints = [
            next(value for value in command if value.startswith("train.ckpt="))
            for command in commands
        ]
        self.assertEqual(checkpoints[0], checkpoints[1])
        for command in commands:
            self.assertIn("dataset.loader_seed=10000202", command)
            self.assertIn("model.gaussian_scale_mode=learned", command)
        self.assertIn("model.memory_scale_target=3.5", commands[1])
        self.assertIn("model.memory_scale_constraint_weight=0.2", commands[1])

    def test_selection_never_selects_fixed_control(self):
        ranking = [
            {"condition": "fixed"},
            {"condition": "warmup-anchored"},
            {"condition": "learnable-free"},
        ]
        self.assertEqual(study.select_learnable(ranking), "warmup-anchored")


if __name__ == "__main__":
    unittest.main()
