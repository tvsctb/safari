import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.rnn_aux import RNNAuxLM, StackedTanhRNN


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
        self.assertIs(type(model.rnn), StackedTanhRNN)
        self.assertIs(type(model.rnn.layers[0]), nn.RNN)
        self.assertIs(type(model.inverse_rnn), StackedTanhRNN)
        self.assertFalse(hasattr(model, "gru"))
        self.assertFalse(model.stop_gradient_memory_target)
        self.assertEqual(model.chunk_offset, "random")
        self.assertFalse(hasattr(model, "terminal_target"))
        self.assertFalse(hasattr(model, "tau"))

    def test_core_transition_matches_vanilla_tanh_equation(self):
        model = self.make_model()
        layer = model.rnn.layers[0]
        inputs = torch.randn(3, 1, 8)
        state = torch.randn(1, 3, 8)
        expected = torch.tanh(
            F.linear(inputs[:, 0], layer.weight_ih_l0, layer.bias_ih_l0)
            + F.linear(state[0], layer.weight_hh_l0, layer.bias_hh_l0)
        )
        outputs, terminal, trajectory = model.rnn(
            inputs, state, return_trajectory=True
        )
        torch.testing.assert_close(outputs[:, 0], expected)
        torch.testing.assert_close(terminal[0], expected)
        torch.testing.assert_close(trajectory[0][:, 0], expected)

    def test_stacked_core_matches_native_multilayer_rnn_without_dropout(self):
        model = RNNAuxLM(
            d_model=8,
            n_layer=3,
            vocab_size=20,
            chunk_size=4,
            dropout=0.0,
        )
        native = nn.RNN(
            8,
            8,
            num_layers=3,
            nonlinearity="tanh",
            batch_first=True,
        )
        with torch.no_grad():
            for index, layer in enumerate(model.rnn.layers):
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(native, f"{name}_l{index}").copy_(
                        getattr(layer, f"{name}_l0")
                    )
        embedded = model.embedding(self.inputs)
        initial = model.default_state(self.inputs.size(0))
        actual_output, actual_state, _ = model.rnn(
            embedded, initial, return_trajectory=True
        )
        expected_output, expected_state = native(embedded, initial)
        torch.testing.assert_close(actual_output, expected_output)
        torch.testing.assert_close(actual_state, expected_state)

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

    def test_no_aux_never_samples_or_uses_offsets(self):
        model = self.make_model()
        model.train()
        with mock.patch.object(
            model, "_sample_offsets", side_effect=AssertionError("offset used")
        ), mock.patch.object(
            model.inverse_rnn,
            "forward",
            side_effect=AssertionError("inverse used"),
        ):
            output, state = model(self.inputs, compute_aux=False)
        self.assertEqual(output.logits.shape, (8, 11, 20))
        self.assertEqual(state.shape, (1, 8, 8))

    def test_offset_changes_aux_only_not_forward_trajectory(self):
        model = self.make_model(chunk_offset=0)
        model.eval()
        first, first_state = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        model.chunk_offset = 3
        second, second_state = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
        torch.testing.assert_close(first_state, second_state, rtol=0, atol=0)
        self.assertFalse(torch.equal(first.aux_loss, second.aux_loss))

    def test_aux_on_off_have_identical_forward_outputs(self):
        model = self.make_model(chunk_offset=2)
        model.eval()
        with_aux, with_aux_state = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        without_aux, without_aux_state = model(self.inputs, compute_aux=False)
        torch.testing.assert_close(
            with_aux.logits, without_aux.logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            with_aux_state, without_aux_state, rtol=0, atol=0
        )

    def test_offset_independence_with_training_dropout(self):
        model = RNNAuxLM(
            d_model=8,
            n_layer=2,
            vocab_size=20,
            chunk_size=4,
            chunk_offset=0,
            dropout=0.4,
        )
        model.train()
        rng_state = torch.random.get_rng_state()
        first, first_state = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        model.chunk_offset = 3
        torch.random.set_rng_state(rng_state)
        second, second_state = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
        torch.testing.assert_close(first_state, second_state, rtol=0, atol=0)

    def test_aux_toggle_does_not_shift_forward_dropout_rng(self):
        model = RNNAuxLM(
            d_model=8,
            n_layer=2,
            vocab_size=20,
            chunk_size=4,
            chunk_offset="random",
            dropout=0.4,
        )
        model.train()
        rng_state = torch.random.get_rng_state()
        with_aux, with_aux_state = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        torch.random.set_rng_state(rng_state)
        without_aux, without_aux_state = model(self.inputs, compute_aux=False)
        torch.testing.assert_close(
            with_aux.logits, without_aux.logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            with_aux_state, without_aux_state, rtol=0, atol=0
        )

    def test_forward_core_runs_once_and_inverse_batches_by_length(self):
        model = self.make_model(chunk_offset=0)
        model.eval()
        with mock.patch.object(
            model.rnn, "forward", wraps=model.rnn.forward
        ) as forward_call, mock.patch.object(
            model.inverse_rnn, "forward", wraps=model.inverse_rnn.forward
        ) as inverse_call:
            model(
                self.inputs,
                targets=self.targets,
                aux_tokens=self.targets,
                compute_aux=True,
            )
        self.assertEqual(forward_call.call_count, 1)
        # Length 11 with chunks 4/4/3 requires only two inverse batches.
        self.assertEqual(inverse_call.call_count, 2)

    def test_batched_random_offsets_equal_individual_fixed_offset_losses(self):
        model = self.make_model()
        model.eval()
        offsets = torch.arange(self.inputs.size(0)) % model.chunk_size
        with mock.patch.object(model, "_sample_offsets", return_value=offsets):
            batched, batched_state = model(
                self.inputs,
                targets=self.targets,
                aux_tokens=self.targets,
                compute_aux=True,
                compute_diagnostics=False,
            )

        individual_losses = []
        individual_logits = []
        individual_states = []
        for index, offset in enumerate(offsets.tolist()):
            model.chunk_offset = offset
            output, state = model(
                self.inputs[index : index + 1],
                targets=self.targets[index : index + 1],
                aux_tokens=self.targets[index : index + 1],
                compute_aux=True,
                compute_diagnostics=False,
            )
            individual_losses.append(output.aux_loss)
            individual_logits.append(output.logits)
            individual_states.append(state)
        torch.testing.assert_close(
            batched.aux_loss, torch.stack(individual_losses).mean()
        )
        torch.testing.assert_close(
            batched.logits, torch.cat(individual_logits, dim=0)
        )
        torch.testing.assert_close(
            batched_state, torch.cat(individual_states, dim=1)
        )

    def test_parameter_matched_default_shape_has_26432_forward_parameters(self):
        model = RNNAuxLM(
            d_model=64,
            n_layer=3,
            vocab_size=20,
            chunk_size=4,
        )
        forward_parameters = (
            model.embedding.weight.numel()
            + model.initial_state.numel()
            + sum(parameter.numel() for parameter in model.rnn.parameters())
        )
        self.assertEqual(forward_parameters, 26432)

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
