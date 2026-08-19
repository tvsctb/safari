import argparse
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ar_irnn_aux_pressure_factorial_study.py"
SPEC = importlib.util.spec_from_file_location("irnn_aux_pressure_factorial", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class IRNNAuxPressureFactorialStudyTest(unittest.TestCase):
    def test_exact_two_by_two_design(self):
        conditions = MODULE.conditions()
        self.assertEqual(len(conditions), 4)
        self.assertEqual(
            {(condition.lambda_high, condition.memory_high) for condition in conditions},
            {(False, False), (False, True), (True, False), (True, True)},
        )
        by_label = {condition.label: condition for condition in conditions}
        self.assertAlmostEqual(by_label["control"].memory_pressure, 1.0)
        self.assertAlmostEqual(by_label["memory2x"].memory_pressure, 2.0)
        self.assertAlmostEqual(by_label["lambda2x"].memory_pressure, 2.0)
        self.assertAlmostEqual(by_label["combined"].memory_pressure, 4.0)
        self.assertTrue(
            all(math.isclose(condition.terminal_pressure, 1.0) for condition in conditions)
        )

    def test_exact_tuning_and_heldout_matrices(self):
        tuning = MODULE.tuning_trials()
        self.assertEqual(len(tuning), 8)
        self.assertEqual(
            {(trial.track, trial.seed) for trial in tuning},
            {
                (f"factorial-{label}", seed)
                for label in ("control", "memory2x", "lambda2x", "combined")
                for seed in (202, 203)
            },
        )
        self.assertTrue(all(trial.max_epochs == 400 for trial in tuning))
        self.assertTrue(all(trial.num_active_associations == 5 for trial in tuning))

        combined = next(c for c in MODULE.conditions() if c.label == "combined")
        heldout = MODULE.heldout_trials(combined)
        self.assertEqual([trial.seed for trial in heldout], [204, 205, 206, 207])
        self.assertTrue(all(trial.track == "factorial-combined" for trial in heldout))

    def test_selection_excludes_control_and_uses_long_lag(self):
        trials = MODULE.tuning_trials()
        values = {
            "control": 0.60,
            "memory2x": 0.62,
            "lambda2x": 0.61,
            "combined": 0.65,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for trial in trials:
                label = trial.track.removeprefix("factorial-")
                path = root / "controlled_lag" / f"{trial.trial_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "overall": {"accuracy": values[label], "nll": 1.0},
                            "by_lag": [
                                {"lag": lag, "accuracy": values[label], "nll": 1.0}
                                for lag in range(2, 41, 2)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            selection = MODULE.select_treatment(MODULE.rank_conditions(trials, root))
        self.assertEqual(selection["selected_condition"]["label"], "combined")
        self.assertAlmostEqual(selection["selected_minus_control_lag_22_40"], 0.05)

    def test_evaluation_receives_k5_and_scaled_tau(self):
        condition = next(c for c in MODULE.conditions() if c.label == "lambda2x")
        trial = MODULE.make_trial(condition, 204, "heldout")
        args = argparse.Namespace(
            examples_per_lag=10,
            base_batch_size=2,
            dataset_seed=7,
            wandb_project="p",
            wandb_entity="e",
            eval_group="g",
        )
        from run_ar_rnn_controlled_lag_study import evaluation_command

        command = evaluation_command(trial, Path("/tmp/output"), 0, args)
        self.assertIn("--num-active-associations", command)
        self.assertEqual(command[command.index("--num-active-associations") + 1], "5")
        self.assertEqual(command[command.index("--tau") + 1], str(MODULE.HIGH_LAMBDA_TAU))

    def test_study_controller_interleaves_gpu_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = MODULE.StudyController(
                Path(directory),
                [0, 1],
                4,
                "project",
                "entity",
                "group",
                dry_run=True,
            )
            allocation = [controller.slots.get_nowait() for _ in range(8)]
        self.assertEqual(allocation, [0, 1, 0, 1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
