import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ar_irnn_difficulty_ladder_study.py"
SPEC = importlib.util.spec_from_file_location("irnn_difficulty_ladder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IRNNDifficultyLadderStudyTest(unittest.TestCase):
    def test_exact_screen_and_heldout_matrices(self):
        screen = MODULE.screen_trials()
        self.assertEqual(len(screen), 8)
        self.assertEqual(
            {(trial.num_active_associations, trial.seed) for trial in screen},
            {(k, seed) for k in (3, 5, 7, 9) for seed in (202, 203)},
        )
        self.assertTrue(all(trial.probe_only for trial in screen))
        self.assertTrue(all(trial.max_epochs == 400 for trial in screen))

        heldout = MODULE.heldout_trials(7)
        self.assertEqual(len(heldout), 12)
        self.assertEqual(
            {(trial.track.split("-")[-1], trial.seed) for trial in heldout},
            {
                (condition, seed)
                for condition in ("noaux", "aux", "memory4x")
                for seed in (204, 205, 206, 207)
            },
        )
        self.assertTrue(all(trial.num_active_associations == 7 for trial in heldout))

    def test_selection_uses_only_noaux_long_lag_and_prefers_hardest_in_range(self):
        trials = MODULE.screen_trials()
        values = {3: 0.60, 5: 0.45, 7: 0.30, 9: 0.20}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for trial in trials:
                path = root / "controlled_lag" / f"{trial.trial_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "by_lag": [
                                {"lag": lag, "accuracy": values[trial.num_active_associations]}
                                for lag in range(2, 41, 2)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            selection = MODULE.select_difficulty(trials, root)
        self.assertEqual(selection["selected_num_active_associations"], 7)
        self.assertEqual(
            selection["selection_rule"], "largest_k_inside_predeclared_interval"
        )

    def test_evaluation_command_receives_selected_k(self):
        trial = MODULE.heldout_trials(5)[0]
        args = argparse.Namespace(
            examples_per_lag=10,
            base_batch_size=2,
            dataset_seed=7,
            wandb_project="p",
            wandb_entity="e",
            eval_group="g",
        )
        command = MODULE.evaluation_command(
            trial, Path("/tmp/output"), 0, args
        )
        self.assertIn("--num-active-associations", command)
        index = command.index("--num-active-associations")
        self.assertEqual(command[index + 1], "5")


if __name__ == "__main__":
    unittest.main()
