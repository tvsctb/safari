import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_ar_irnn_inverse_capacity_gradmatch_study.py"
SPEC = importlib.util.spec_from_file_location("inverse_capacity_gradmatch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ARIRNNInverseCapacityGradmatchStudyTest(unittest.TestCase):
    def test_exact_fixed_gradient_matched_matrix(self):
        trials = MODULE.tuning_trials()
        self.assertEqual(len(trials), 6)
        self.assertEqual(len({trial.trial_id for trial in trials}), 6)
        self.assertEqual({trial.seed for trial in trials}, {202, 203})
        self.assertEqual(
            {
                (
                    trial.inverse_capacity_multiplier,
                    trial.aux_weight,
                    MODULE.condition_for_trial(trial)["inverse_parameters"],
                )
                for trial in trials
            },
            {
                (0.125, 0.030, 5197),
                (0.25, 0.035, 10462),
                (0.5, 0.040, 20734),
            },
        )
        self.assertTrue(all(trial.rho == 8.0 for trial in trials))
        self.assertTrue(all(trial.tau == MODULE.TAU for trial in trials))
        self.assertTrue(
            all(trial.num_active_associations == 5 for trial in trials)
        )

    def test_build_command_contains_fixed_capacity_and_lambda(self):
        trial = MODULE.tuning_trials()[0]
        command = MODULE.build_rnn_command(
            trial, Path("/tmp/gradmatch"), "p", "e", "g"
        )
        self.assertIn("model.inverse_capacity_multiplier=0.125", command)
        self.assertIn("task.aux_weight=0.03", command)
        self.assertIn("task.aux_weight_final=0.03", command)
        self.assertIn("task.aux_weight_schedule=fixed", command)

    def test_ranking_and_heldout_keep_selected_fixed_lambda(self):
        trials = MODULE.tuning_trials()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for trial in trials:
                label = MODULE.condition_for_trial(trial)["label"]
                accuracy = {
                    "one-eighth": 0.31,
                    "quarter": 0.44,
                    "half": 0.40,
                }[label]
                report = {
                    "overall": {"accuracy": accuracy, "nll": 1 - accuracy},
                    "by_lag": [
                        {"lag": lag, "accuracy": accuracy, "nll": 1 - accuracy}
                        for lag in range(2, 41, 2)
                    ],
                }
                path = root / "controlled_lag" / f"{trial.trial_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report))
            ranking = MODULE.rank_conditions(trials, root)
        self.assertEqual(ranking[0]["label"], "quarter")
        heldout = MODULE.heldout_trials(ranking[0])
        self.assertEqual([trial.seed for trial in heldout], [204, 205, 206, 207])
        self.assertTrue(
            all(
                trial.inverse_capacity_multiplier == 0.25
                for trial in heldout
            )
        )
        self.assertTrue(all(trial.aux_weight == 0.035 for trial in heldout))

    def test_common_probe_protocol_is_still_dense_one_x(self):
        trial = MODULE.tuning_trials()[0]
        args = argparse.Namespace(
            probe_dataset_seed=20260824,
            probe_forward_batch_size=512,
            probe_batch_size=4096,
            probe_train_base_examples=10000,
            probe_val_base_examples=2000,
            probe_test_base_examples=5000,
            probe_max_epochs=120,
            probe_min_epochs=30,
            probe_patience=20,
            wandb_project="p",
            wandb_entity="e",
        )
        from run_ar_irnn_inverse_capacity_study import probe_command

        command = probe_command(
            trial, Path("/tmp/gradmatch"), "full", args, "probe-group"
        )
        self.assertNotIn("--inverse-capacity-multiplier", command)
        self.assertIn("probe-group", command)


if __name__ == "__main__":
    unittest.main()
