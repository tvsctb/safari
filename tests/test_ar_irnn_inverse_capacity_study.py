import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_ar_irnn_inverse_capacity_study.py"
SPEC = importlib.util.spec_from_file_location("inverse_capacity_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ARIRNNInverseCapacityStudyTest(unittest.TestCase):
    def test_exact_tuning_matrix_and_parameter_counts(self):
        trials = MODULE.tuning_trials()
        self.assertEqual(len(trials), 10)
        self.assertEqual(len({trial.trial_id for trial in trials}), 10)
        self.assertEqual({trial.seed for trial in trials}, {202, 203})
        self.assertEqual(
            {
                (
                    MODULE.capacity_for_trial(trial)["multiplier"],
                    MODULE.capacity_for_trial(trial)["inverse_parameters"],
                )
                for trial in trials
            },
            {
                (0.125, 5197),
                (0.25, 10462),
                (0.5, 20734),
                (1.0, 41536),
                (2.0, 83011),
            },
        )
        self.assertTrue(all(trial.num_active_associations == 5 for trial in trials))
        self.assertTrue(all(trial.rho == 8.0 for trial in trials))

    def test_build_command_contains_capacity(self):
        trial = MODULE.tuning_trials()[0]
        command = MODULE.build_rnn_command(
            trial, Path("/tmp/capacity"), "p", "e", "g"
        )
        self.assertIn("model.inverse_capacity_multiplier=0.125", command)

    def test_ranking_and_heldout_selection(self):
        trials = MODULE.tuning_trials()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for trial in trials:
                label = MODULE.capacity_for_trial(trial)["label"]
                accuracy = {
                    "one-eighth": 0.30,
                    "quarter": 0.35,
                    "half": 0.45,
                    "full": 0.40,
                    "double": 0.39,
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
            ranking = MODULE.rank_capacities(trials, root)
        self.assertEqual(ranking[0]["label"], "half")
        heldout = MODULE.heldout_trials(ranking[0])
        self.assertEqual([trial.seed for trial in heldout], [204, 205, 206, 207])
        self.assertTrue(all(trial.inverse_capacity_multiplier == 0.5 for trial in heldout))

    def test_probe_command_uses_fixed_common_dense_probe_protocol(self):
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
        command = MODULE.probe_command(
            trial, Path("/tmp/capacity"), "global_unit", args, "probe-group"
        )
        self.assertIn("scripts/train_ar_rnn_past_info_probe.py", command)
        self.assertIn("global_unit", command)
        self.assertIn("probe-group", command)
        self.assertNotIn("--inverse-capacity-multiplier", command)


if __name__ == "__main__":
    unittest.main()
