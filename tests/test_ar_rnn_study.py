import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ar_rnn_study.py"
SPEC = importlib.util.spec_from_file_location("run_ar_rnn_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


class ArRNNStudyTest(unittest.TestCase):
    def test_failed_preflight_prints_the_captured_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fail(command, **kwargs):
                kwargs["stdout"].write("exact gpu failure\n")
                return mock.Mock(returncode=7)

            stderr = StringIO()
            with mock.patch.object(study.subprocess, "run", side_effect=fail):
                with redirect_stderr(stderr):
                    with self.assertRaisesRegex(RuntimeError, "returncode=7"):
                        study.run_preflight(root, 0)
        self.assertIn("exact gpu failure", stderr.getvalue())

    def test_baseline_and_screen_cardinality(self):
        baselines = study.make_baseline_trials()
        screens = study.make_screen_trials(study.RHO_GRID)
        self.assertEqual(
            [trial.seed for trial in baselines], [202, 203, 204, 205, 206, 207]
        )
        self.assertEqual(len(screens), 14)
        self.assertTrue(all(trial.max_epochs == 150 for trial in screens))

    def test_command_matches_rmt_protocol_and_enables_expected_aux(self):
        trial = study.make_screen_trials([1.0])[0]
        with tempfile.TemporaryDirectory() as directory:
            command = study.build_rnn_command(
                trial, Path(directory), "project", "entity", "group"
            )
        self.assertIn(
            "experiment=synthetics/associative_recall/rnn_aux", command
        )
        self.assertIn("scheduler.num_training_steps=62800", command)
        self.assertIn("scheduler.num_warmup_steps=12560", command)
        self.assertIn("optimizer.lr=0.001", command)
        self.assertIn("optimizer.weight_decay=0.1", command)
        self.assertIn("model.d_model=64", command)
        self.assertIn("model.n_layer=3", command)
        self.assertIn("model.chunk_offset=random", command)
        self.assertIn("model.rho=1.0", command)
        self.assertIn("model.stop_gradient_memory_target=false", command)
        self.assertIn("model.use_chunk_loss=true", command)
        self.assertIn("model.use_discrete_loss=true", command)
        self.assertIn("model.use_memory_loss=true", command)
        self.assertIn("task.aux_weight=0.1", command)
        self.assertIn("task.aux_gradient_norm_interval=1570", command)
        self.assertNotIn("terminal", "\n".join(command))

    def test_noaux_command_disables_aux_execution_and_diagnostics(self):
        trial = study.make_baseline_trials()[0]
        with tempfile.TemporaryDirectory() as directory:
            command = study.build_rnn_command(
                trial, Path(directory), "project", "entity", "group"
            )
        self.assertIn("task.aux_weight=0.0", command)
        self.assertIn("task.aux_gradient_norm_interval=0", command)
        self.assertIn("task.aux_diagnostic_interval=0", command)

    def test_ranking_uses_mean_accuracy_then_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                1.0: [(0.8, 0.4), (0.82, 0.5)],
                2.0: [(0.79, 0.2), (0.80, 0.2)],
            }
            for rho, records in values.items():
                for seed, (accuracy, loss) in zip(study.TUNING_SEEDS, records):
                    trial_id = f"rnn-aux-rho{study.rho_label(rho)}-s{seed}"
                    result = root / "trials" / trial_id / "result.json"
                    result.parent.mkdir(parents=True, exist_ok=True)
                    result.write_text(
                        json.dumps(
                            {
                                "last": {
                                    "metrics": {
                                        "val/accuracy_ignore_index": accuracy,
                                        "val/loss": loss,
                                    }
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
            ranking = study.rank_rhos(root, values)
        self.assertEqual(ranking[0]["rho"], 1.0)

    def test_report_records_parameter_match_and_no_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            report = study.build_report(
                Path(directory),
                study.make_baseline_trials(),
                study.make_heldout_trials(1.0),
                1.0,
            )
        self.assertEqual(report["condition"]["forward_parameters"], 26432)
        self.assertFalse(report["condition"]["terminal_loss"])
        self.assertFalse(report["condition"]["target_sg"])


if __name__ == "__main__":
    unittest.main()
