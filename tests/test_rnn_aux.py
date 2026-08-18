import unittest

import torch
import torch.nn as nn

from src.models.sequence.rnn_aux import RNNAuxLM


class RNNAuxLMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = torch.randint(0, 20, (8, 11))
        self.targets = torch.randint(0, 20, (8, 11))

    def make_model(self, **options):
        return RNNAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            chunk_size=4,
            **options,
        )

    def test_defaults_are_sg_off_random_offset_and_no_terminal(self):
        model = self.make_model()
        self.assertIs(type(model.rnn), nn.RNN)
        self.assertIs(type(model.inverse_rnn), nn.RNN)
        self.assertFalse(hasattr(model, "gru"))
        self.assertFalse(model.stop_gradient_memory_target)
        self.assertEqual(model.chunk_offset, "random")
        self.assertFalse(hasattr(model, "terminal_target"))
        self.assertFalse(hasattr(model, "tau"))

    def test_forward_inverse_are_unshared_with_identical_initialization(self):
        model = self.make_model()
        self.assertIsNot(model.rnn, model.inverse_rnn)
        for forward, inverse in zip(
            model.rnn.parameters(), model.inverse_rnn.parameters()
        ):
            torch.testing.assert_close(forward, inverse)
            self.assertIsNot(forward, inverse)

    def test_memory_predictor_is_initially_identity(self):
        model = self.make_model()
        hidden = torch.randn(1, 3, 8)
        torch.testing.assert_close(model._predict_memory(hidden), hidden)

    def test_forward_has_three_aux_components_and_finite_diagnostics(self):
        model = self.make_model(chunk_offset=0)
        output, state = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        self.assertEqual(output.logits.shape, (8, 11, 20))
        self.assertEqual(state.shape, (1, 8, 8))
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertEqual(
            set(model.loss_components),
            {"chunk_ce", "discrete_ce", "memory_nll", "total"},
        )
        self.assertNotIn("terminal_nll", model.loss_components)
        self.assertTrue(
            torch.isfinite(model.metrics["aux/memory_relative_mse"])
        )

    def test_random_offsets_are_per_sequence_and_seed_reproducible(self):
        model = self.make_model()
        model.train()
        torch.manual_seed(123)
        first = model._sample_offsets(128, self.inputs.device)
        torch.manual_seed(123)
        second = model._sample_offsets(128, self.inputs.device)
        torch.testing.assert_close(first, second)
        self.assertGreater(first.unique().numel(), 1)
        self.assertGreaterEqual(first.min().item(), 0)
        self.assertLess(first.max().item(), model.chunk_size)

    def test_random_offset_evaluation_is_fixed_zero(self):
        model = self.make_model()
        model.eval()
        offsets = model._sample_offsets(32, self.inputs.device)
        torch.testing.assert_close(offsets, torch.zeros_like(offsets))

    def test_fixed_offset_validation(self):
        with self.assertRaisesRegex(ValueError, "chunk_offset"):
            self.make_model(chunk_offset=True)
        with self.assertRaisesRegex(ValueError, "chunk_offset"):
            self.make_model(chunk_offset=4)
        with self.assertRaisesRegex(ValueError, "chunk_offset"):
            self.make_model(chunk_offset="sequence")

    def test_target_sg_blocks_only_the_direct_target_path(self):
        off = self.make_model(
            chunk_offset=0,
            stop_gradient_memory_target=False,
            stop_gradient_memory_observation=True,
            use_chunk_loss=False,
            use_discrete_loss=False,
        )
        on = self.make_model(
            chunk_offset=0,
            stop_gradient_memory_target=True,
            stop_gradient_memory_observation=True,
            use_chunk_loss=False,
            use_discrete_loss=False,
        )
        on.load_state_dict(off.state_dict())
        short_inputs = self.inputs[:, :4]
        short_targets = self.targets[:, :4]

        off_output, _ = off(
            short_inputs, targets=short_targets, aux_tokens=short_targets
        )
        off_output.aux_loss.backward()
        self.assertIsNotNone(off.initial_state.grad)
        self.assertGreater(off.initial_state.grad.norm().item(), 0.0)

        on_output, _ = on(
            short_inputs, targets=short_targets, aux_tokens=short_targets
        )
        on_output.aux_loss.backward()
        self.assertIsNone(on.initial_state.grad)

    def test_memory_loss_uses_ffn_prediction(self):
        model = self.make_model(
            chunk_offset=0,
            use_chunk_loss=False,
            use_discrete_loss=False,
        )
        baseline, _ = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
        )
        baseline_loss = baseline.aux_loss.detach().clone()
        final_projection = model.memory_predictor[2]
        with torch.no_grad():
            final_projection.bias.fill_(1.0)
        shifted, _ = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
        )
        self.assertFalse(torch.allclose(baseline_loss, shifted.aux_loss))

    def test_predictor_gradient_is_part_of_memory_loss(self):
        model = self.make_model(
            chunk_offset=0,
            use_chunk_loss=False,
            use_discrete_loss=False,
        )
        output, _ = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
        )
        output.aux_loss.backward()
        self.assertGreater(
            model.memory_predictor[2].weight.grad.norm().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
