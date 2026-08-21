import tempfile
import unittest
from pathlib import Path

from scripts.run_ar_normalized_state_likelihood_study import (
    EXPECTED_RUNS,
    make_trial,
    trials,
)
from scripts.run_ar_rnn_controlled_lag_study import evaluation_command
from scripts.run_ar_rnn_study import build_rnn_command


class NormalizedStateLikelihoodStudyTest(unittest.TestCase):
    def test_exact_three_by_four_design(self):
        planned = trials()
        self.assertEqual(len(planned), EXPECTED_RUNS)
        self.assertEqual(len({trial.trial_id for trial in planned}), EXPECTED_RUNS)
        self.assertEqual({trial.seed for trial in planned}, {204, 205, 206, 207})
        self.assertEqual(
            {trial.trial_id.split("-")[2] for trial in planned},
            {"noaux", "gaussian", "vmf"},
        )

    def test_common_forward_and_likelihood_specific_options(self):
        noaux = make_trial("noaux", 204)
        gaussian = make_trial("gaussian", 204)
        vmf = make_trial("vmf", 204)
        for trial in (noaux, gaussian, vmf):
            self.assertTrue(trial.normalized_state)
            self.assertEqual(trial.activation, "tanh")
            self.assertEqual(trial.recurrent_init, "orthogonal")
            self.assertEqual(trial.state_likelihood_granularity, "layer")
            self.assertTrue(trial.exclude_initial_memory_reconstruction)
            self.assertTrue(trial.use_terminal_loss)
            self.assertFalse(trial.stop_gradient_memory_target)
            self.assertEqual(trial.num_active_associations, 5)
        self.assertTrue(noaux.probe_only)
        self.assertFalse(gaussian.probe_only)
        self.assertFalse(vmf.probe_only)
        self.assertEqual(gaussian.gaussian_scale_mode, "learned")
        self.assertEqual(vmf.state_aux_distribution, "vmf")
        self.assertEqual(vmf.vmf_kappa_mode, "learned")

    def test_training_and_evaluation_commands_preserve_new_options(self):
        trial = make_trial("vmf", 204)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = build_rnn_command(trial, root, "p", "e", "g")
            self.assertIn("model.normalized_state=true", train)
            self.assertIn("model.state_likelihood_granularity=layer", train)
            self.assertIn("model.vmf_kappa_mode=learned", train)
            args = type(
                "Args",
                (),
                dict(
                    examples_per_lag=10,
                    base_batch_size=2,
                    dataset_seed=1,
                    wandb_project="p",
                    wandb_entity="e",
                    eval_group="g",
                ),
            )()
            evaluate = evaluation_command(trial, root, 0, args)
            self.assertIn("--normalized-state", evaluate)
            self.assertEqual(
                evaluate[evaluate.index("--state-likelihood-granularity") + 1],
                "layer",
            )


if __name__ == "__main__":
    unittest.main()
