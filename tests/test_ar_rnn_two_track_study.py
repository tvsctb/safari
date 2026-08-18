import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_ar_rnn_two_track_study.py"
SPEC = importlib.util.spec_from_file_location("run_ar_rnn_two_track_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def fake_calibration():
    return {
        core.label: {
            "median_unit_tau_ratio": 1.0,
            "reference_aux_weight": study.REFERENCE_AUX_WEIGHT,
        }
        for core in study.CORE_SPECS
    }


class ArRNNTwoTrackStudyTest(unittest.TestCase):
    def test_two_tracks_have_expected_cardinality_and_stronger_lambdas(self):
        calibration = fake_calibration()
        tanh = study.make_tanh_screen(calibration)
        cores = study.make_core_screen(calibration)
        self.assertEqual(len(tanh), 15)
        self.assertEqual(len(study.flatten_candidates(tanh)), 30)
        self.assertEqual(len(cores), 4)
        self.assertEqual(len(study.flatten_candidates(cores)), 8)
        self.assertIn(0.2, study.LAMBDA_GRID)
        self.assertIn(0.4, study.LAMBDA_GRID)
        self.assertLess(max(study.TERMINAL_RATIO_TARGETS), 0.001)
        self.assertEqual(study.EXPECTED_UNIQUE_RUNS, 64)

    def test_terminal_tau_is_lambda_aware_and_hits_requested_ratio(self):
        calibration = fake_calibration()
        core = study.CORE_SPECS[0]
        target = study.TERMINAL_RATIO_TARGETS[-1]
        low = study.calibrated_tau(calibration, core, target, 0.025)
        high = study.calibrated_tau(calibration, core, target, 0.4)
        self.assertAlmostEqual(high / low, 4.0)
        for aux_weight in study.LAMBDA_GRID:
            tau = study.calibrated_tau(
                calibration, core, target, aux_weight
            )
            realized = (
                calibration[core.label]["median_unit_tau_ratio"]
                * aux_weight
                / study.REFERENCE_AUX_WEIGHT
                / tau**2
            )
            self.assertAlmostEqual(realized, target)

    def test_main_conditions_default_to_no_m0_and_terminal_on(self):
        trial = next(
            iter(study.make_tanh_screen(fake_calibration()).values())
        )[0]
        self.assertTrue(trial.exclude_initial_memory_reconstruction)
        self.assertTrue(trial.use_terminal_loss)
        self.assertFalse(trial.probe_only)

    def test_core_screen_is_independent_detached_noaux(self):
        candidates = study.make_core_screen(fake_calibration())
        self.assertEqual(set(candidates), {core.label for core in study.CORE_SPECS})
        self.assertTrue(
            all(
                trial.probe_only and trial.track == "irnn"
                for trial in study.flatten_candidates(candidates)
            )
        )

    def test_command_routes_tracks_to_separate_groups(self):
        tanh_trial = next(
            iter(study.make_tanh_screen(fake_calibration()).values())
        )[0]
        irnn_trial = study.make_core_screen(fake_calibration())[
            "irnn-alpha1p0"
        ][0]
        with tempfile.TemporaryDirectory() as directory:
            tanh_command = study.build_two_track_command(
                tanh_trial,
                Path(directory),
                "project",
                "entity",
                "prefix",
            )
            irnn_command = study.build_two_track_command(
                irnn_trial,
                Path(directory),
                "project",
                "entity",
                "prefix",
            )
        self.assertIn("wandb.group=prefix-tanh", tanh_command)
        self.assertIn("wandb.group=prefix-irnn", irnn_command)
        self.assertIn("model.exclude_initial_memory_reconstruction=true", tanh_command)
        self.assertIn("model.use_terminal_loss=true", tanh_command)
        self.assertIn("model.activation=relu", irnn_command)
        self.assertIn("model.recurrent_init=identity", irnn_command)

    def test_ranking_is_accuracy_then_worst_seed_then_loss(self):
        calibration = fake_calibration()
        candidates = dict(
            list(study.make_tanh_screen(calibration).items())[:2]
        )
        values = ((0.80, 0.70), (0.80, 0.75))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for candidate_index, (label, trials) in enumerate(candidates.items()):
                for seed_index, trial in enumerate(trials):
                    result = root / "trials" / trial.trial_id / "result.json"
                    result.parent.mkdir(parents=True, exist_ok=True)
                    accuracy = values[candidate_index][seed_index]
                    result.write_text(
                        json.dumps(
                            {
                                "last": {
                                    "metrics": {
                                        "val/accuracy_ignore_index": accuracy,
                                        "val/loss": 1.0 - accuracy,
                                    }
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
            ranking = study.rank_candidates(root, candidates)
        self.assertEqual(ranking[0]["mean_accuracy"], 0.775)

    def test_controller_hard_deadline_prevents_new_trial(self):
        trial = next(
            iter(study.make_tanh_screen(fake_calibration()).values())
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = study.StudyController(
                output_root=root,
                gpu_ids=(0,),
                workers_per_gpu=1,
                project="unused",
                entity="unused",
                group="deadline-test",
                command_builder=study.build_two_track_command,
                dry_run=True,
                deadline_monotonic=time.monotonic() - 1.0,
            )
            result = controller.run_trials([trial])
            manifest = json.loads((root / "manifest.json").read_text())
        self.assertFalse(result[trial.trial_id])
        self.assertEqual(
            manifest["trials"][trial.trial_id]["status"], "budget_exhausted"
        )


if __name__ == "__main__":
    unittest.main()
