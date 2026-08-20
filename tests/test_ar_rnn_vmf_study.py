import unittest

from scripts.run_ar_rnn_vmf_study import make_trial


class ArRNNVMFStudyTest(unittest.TestCase):
    def setUp(self):
        self.kappas = {
            "irnn": {"memory": 3.0, "terminal": 5.0},
            "tanh": {"memory": 7.0, "terminal": 11.0},
        }

    def test_design_has_only_gg_or_vv_not_mixed_components(self):
        gg = make_trial("irnn", "gg", 202, self.kappas, "tuning", 150)
        fixed = make_trial("irnn", "vv-fixed", 202, self.kappas, "tuning", 150)
        learned = make_trial("irnn", "vv-learned", 202, self.kappas, "tuning", 150)
        self.assertEqual(gg.state_aux_distribution, "gaussian")
        self.assertEqual(fixed.state_aux_distribution, "vmf")
        self.assertEqual(fixed.vmf_kappa_mode, "fixed")
        self.assertEqual(learned.state_aux_distribution, "vmf")
        self.assertEqual(learned.vmf_kappa_mode, "learned")
        self.assertEqual(learned.memory_vmf_kappa, 3.0)
        self.assertEqual(learned.terminal_vmf_kappa, 5.0)

    def test_vmf_study_keeps_single_chunk_and_sg_off(self):
        trial = make_trial("tanh", "vv-learned", 203, self.kappas, "tuning", 150)
        self.assertEqual(trial.aux_chunk_sizes, (4,))
        self.assertFalse(trial.stop_gradient_memory_target)
        self.assertEqual(trial.num_active_associations, 5)


if __name__ == "__main__":
    unittest.main()
