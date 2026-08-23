import unittest

import torch

from src.models.sequence.rnn_aux import (
    FactorizedStackedRNN,
    RNNAuxLM,
    StackedRNN,
)


class RNNInverseCapacityTest(unittest.TestCase):
    def model(self, multiplier):
        torch.manual_seed(37)
        return RNNAuxLM(
            d_model=64,
            n_layer=3,
            vocab_size=20,
            activation="relu",
            recurrent_init="identity",
            recurrent_identity_scale=1.0,
            rho=8.0,
            tau=243.242356581615,
            inverse_capacity_multiplier=multiplier,
        )

    def test_exact_inverse_parameter_counts(self):
        expected = {
            0.125: 5197,
            0.25: 10462,
            0.5: 20734,
            1.0: 41536,
            2.0: 83011,
        }
        for multiplier, count in expected.items():
            with self.subTest(multiplier=multiplier):
                self.assertEqual(
                    self.model(multiplier).inverse_parameter_count(), count
                )

    def test_forward_initialization_is_identical_across_capacities(self):
        control = self.model(1.0)
        for multiplier in (0.125, 0.25, 0.5, 2.0):
            candidate = self.model(multiplier)
            for name, value in control.rnn.state_dict().items():
                torch.testing.assert_close(value, candidate.rnn.state_dict()[name])
            torch.testing.assert_close(
                control.embedding.weight, candidate.embedding.weight
            )
            torch.testing.assert_close(
                control.initial_state, candidate.initial_state
            )

    def test_full_rank_factorization_preserves_initial_inverse_function(self):
        control = self.model(1.0).eval()
        doubled = self.model(2.0).eval()
        inputs = torch.randn(5, 7, 64)
        state = torch.randn(3, 5, 64)
        expected, expected_state, _ = control.inverse_rnn(inputs, state)
        actual, actual_state, _ = doubled.inverse_rnn(inputs, state)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=5e-5)
        torch.testing.assert_close(
            actual_state, expected_state, rtol=1e-3, atol=5e-5
        )

    def test_reduced_inverse_is_trainable_and_keeps_state_interface(self):
        model = self.model(0.125)
        self.assertIsInstance(model.rnn, StackedRNN)
        self.assertIsInstance(model.inverse_rnn, FactorizedStackedRNN)
        inputs = torch.randn(4, 3, 64)
        state = torch.randn(3, 4, 64)
        outputs, terminal, _ = model.inverse_rnn(inputs, state)
        self.assertEqual(outputs.shape, (4, 3, 64))
        self.assertEqual(terminal.shape, (3, 4, 64))
        outputs.square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.inverse_rnn.parameters())
        )

    def test_reduced_and_doubled_capacities_backpropagate_full_aux_objective(self):
        inputs = torch.randint(0, 20, (3, 11))
        targets = torch.randint(0, 20, (3, 11))
        for multiplier in (0.125, 2.0):
            with self.subTest(multiplier=multiplier):
                model = self.model(multiplier).train()
                output, _ = model(
                    inputs,
                    targets=targets,
                    aux_tokens=targets,
                    compute_aux=True,
                )
                output.aux_loss.backward()
                self.assertTrue(torch.isfinite(output.aux_loss))
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.rnn.parameters()
                    )
                )
                self.assertEqual(
                    model.metrics["aux/inverse_parameter_count"].item(),
                    float(model.inverse_parameter_count()),
                )

    def test_invalid_capacity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inverse_capacity_multiplier"):
            self.model(4.0)


if __name__ == "__main__":
    unittest.main()
