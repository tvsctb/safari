import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ar_rmt_scale_study.py"
SPEC = importlib.util.spec_from_file_location("run_ar_rmt_scale_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


class ArRMTScaleStudyTest(unittest.TestCase):
    def test_initial_grid_and_outward_extension(self):
        scales = study.initial_scales()
        self.assertEqual(len(scales), 9)
        center = study.Scale(1.0, 1.0)
        self.assertAlmostEqual(center.rho, study.RHO_BASE)
        self.assertAlmostEqual(center.tau, study.TAU_BASE)
        self.assertEqual(study.outward_scales(center), ())
        corner = study.outward_scales(study.Scale(0.5, 2.0))
        self.assertEqual(len(corner), 7)
        self.assertIn(study.Scale(0.25, 4.0), corner)

    def test_command_keeps_full_schedule_and_canonical_options(self):
        trial = study.make_screen_trials([study.Scale(1.0, 2.0)], False)[0]
        with tempfile.TemporaryDirectory() as directory:
            command = study.build_command(
                trial, Path(directory), "project", "entity", "group"
            )
        joined = "\n".join(command)
        self.assertIn("trainer.max_epochs=150", command)
        self.assertIn("scheduler.num_training_steps=62800", command)
        self.assertIn("scheduler.num_warmup_steps=12560", command)
        self.assertIn("model.num_memory_tokens=2", command)
        self.assertIn("model.stop_gradient_memory_target=false", command)
        self.assertIn("task.aux_weight=0.1", command)
        self.assertNotIn("write_input_mode", joined)
        self.assertNotIn("share_inverse", joined)
        self.assertNotIn("scale_mode", joined)

    def test_ranking_uses_paired_seed_mean_then_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = study.Scale(0.5, 1.0)
            second = study.Scale(1.0, 1.0)
            values = {
                first: [(0.80, 0.4), (0.82, 0.5)],
                second: [(0.79, 0.2), (0.80, 0.2)],
            }
            for scale, records in values.items():
                for seed, (accuracy, loss) in zip(study.TUNING_SEEDS, records):
                    trial = study.make_screen_trials([scale], False)
                    trial = next(item for item in trial if item.seed == seed)
                    result = root / "trials" / trial.trial_id / "result.json"
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
            ranking = study.rank_scales(root, False, [first, second])
        self.assertEqual(study.scale_from_row(ranking[0]), first)
        self.assertAlmostEqual(ranking[0]["mean_accuracy"], 0.81)

    def test_baseline_and_heldout_cardinality(self):
        self.assertEqual(len(study.make_baseline_trials()), 6)
        self.assertEqual(
            len(study.make_heldout_trials(True, study.Scale(1.0, 1.0))), 4
        )
        self.assertEqual(
            len(study.make_screen_trials(study.initial_scales(), False)), 18
        )

    def test_concurrency_selector_prefers_lower_near_tie(self):
        records = [
            {
                "workers_per_gpu": 2,
                "aggregate_updates_per_second": 10.0,
                "succeeded": True,
            },
            {
                "workers_per_gpu": 4,
                "aggregate_updates_per_second": 11.0,
                "succeeded": True,
            },
            {
                "workers_per_gpu": 6,
                "aggregate_updates_per_second": 11.1,
                "succeeded": True,
            },
            {
                "workers_per_gpu": 8,
                "aggregate_updates_per_second": 20.0,
                "succeeded": False,
            },
        ]
        self.assertEqual(study.choose_worker_count(records), 4)


if __name__ == "__main__":
    unittest.main()
