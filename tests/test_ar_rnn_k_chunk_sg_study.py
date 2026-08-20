import unittest

from scripts.run_ar_rnn_k_chunk_sg_study import (
    CHUNK_VARIANTS,
    difficulty_trials,
    make_trial,
    sg_trials,
)


class ArRNNKChunkSGStudyTest(unittest.TestCase):
    def test_difficulty_factorial_is_exact(self):
        trials = difficulty_trials()
        self.assertEqual(len(trials), 32)
        self.assertEqual({trial.seed for trial in trials}, {202, 203})
        self.assertEqual({trial.num_active_associations for trial in trials}, {3, 4, 5, 6})

    def test_multiscale_halves_each_cores_base_lambda(self):
        irnn = make_trial("irnn", 5, "aux", 202, chunks=(2, 4))
        tanh = make_trial("tanh", 5, "aux", 202, chunks=(1, 4))
        self.assertEqual(irnn.aux_chunk_sizes, (2, 4))
        self.assertAlmostEqual(irnn.aux_weight, 0.05)
        self.assertAlmostEqual(tanh.aux_weight, 0.025)
        self.assertEqual(set(CHUNK_VARIANTS), {(2,), (3,), (2, 4), (1, 4)})

    def test_sg_is_a_separate_fixed_current_best_arm(self):
        trials = sg_trials()
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(trial.stop_gradient_memory_target for trial in trials))
        self.assertTrue(all(trial.num_active_associations == 5 for trial in trials))
        self.assertTrue(all(trial.aux_chunk_sizes == (4,) for trial in trials))
        self.assertTrue(all(trial.rho == 8.0 for trial in trials))


if __name__ == "__main__":
    unittest.main()
