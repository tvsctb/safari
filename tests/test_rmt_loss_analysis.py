import unittest

import torch

from src.models.sequence.rmt_aux import RMTAuxLM
from src.utils.rmt_loss_analysis import (
    apply_gradient_step,
    forward_parameter_items,
    restore_parameters,
    transition_gain_proxies,
)


class RMTLossAnalysisTest(unittest.TestCase):
    def _model(self):
        torch.manual_seed(7)
        return RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=1,
            vocab_size=10,
            chunk_size=2,
            num_memory_tokens=2,
            dropout=0.0,
            learnable_terminal_target=True,
        ).eval()

    def test_forward_parameters_exclude_inverse_only_parameters(self):
        names = [name for name, _ in forward_parameter_items(self._model())]
        self.assertTrue(any(name.startswith("blocks.") for name in names))
        self.assertTrue(any(name.startswith("embedding.") for name in names))
        self.assertFalse(any(name.startswith("inverse_blocks.") for name in names))
        self.assertNotIn("terminal_target", names)

    def test_normalized_step_can_be_restored_exactly(self):
        parameters = [parameter for _, parameter in forward_parameter_items(self._model())]
        gradients = [torch.ones_like(parameter) for parameter in parameters]
        before = [parameter.detach().clone() for parameter in parameters]
        originals, scale = apply_gradient_step(parameters, gradients, 1e-4)
        self.assertGreater(scale, 0.0)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, parameters)))
        restore_parameters(parameters, originals)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(before, parameters)))

    def test_transition_proxies_are_positive_and_repeatable(self):
        model = self._model()
        input_ids = torch.randint(0, 10, (3, 6))
        aux_tokens = torch.randint(0, 10, (3, 6))
        first = transition_gain_proxies(model, input_ids, aux_tokens, probes=2, seed=11)
        second = transition_gain_proxies(model, input_ids, aux_tokens, probes=2, seed=11)
        self.assertGreater(first["carry_gain"], 0.0)
        self.assertGreater(first["write_gain"], 0.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
