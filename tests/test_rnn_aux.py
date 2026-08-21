import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig

from src.models.sequence.rnn_aux import RNNAuxLM, StackedRNN


class RNNAuxLMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = torch.randint(0, 20, (8, 11))
        self.targets = torch.randint(0, 20, (8, 11))

    def make_model(self, **options):
        chunk_size = options.pop("chunk_size", 4)
        return RNNAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            chunk_size=chunk_size,
            **options,
        )

    def test_defaults_are_sg_off_random_offset_no_m0_and_learned_terminal(self):
        model = self.make_model()
        self.assertIs(type(model.rnn), StackedRNN)
        self.assertIs(type(model.rnn.layers[0]), nn.RNN)
        self.assertIs(type(model.inverse_rnn), StackedRNN)
        self.assertEqual(model.rnn.activation, "tanh")
        self.assertEqual(model.rnn.recurrent_init, "orthogonal")
        self.assertFalse(model.stop_gradient_memory_target)
        self.assertEqual(model.chunk_offset, "random")
        self.assertTrue(model.exclude_initial_memory_reconstruction)
        self.assertTrue(model.use_terminal_loss)
        self.assertFalse(model.condition_memory_reconstruction_on_boundary)
        self.assertFalse(model.auxiliary_probe_only)
        self.assertIsInstance(model.terminal_target, nn.Parameter)
        self.assertEqual(tuple(model.terminal_target.shape), (1, 1, 8))
        self.assertEqual(model.tau.item(), 1.0)

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

    def test_relu_orthogonal_changes_only_the_core_options(self):
        model = self.make_model(activation="relu", recurrent_init="orthogonal")
        layer = model.rnn.layers[0]
        inputs = torch.randn(3, 1, 8)
        state = torch.randn(1, 3, 8)
        expected = torch.relu(
            F.linear(inputs[:, 0], layer.weight_ih_l0, layer.bias_ih_l0)
            + F.linear(state[0], layer.weight_hh_l0, layer.bias_hh_l0)
        )
        outputs, terminal, _ = model.rnn(inputs, state)
        torch.testing.assert_close(outputs[:, 0], expected)
        torch.testing.assert_close(terminal[0], expected)

    def test_irnn_recurrent_weights_are_scaled_identities(self):
        model = RNNAuxLM(
            d_model=8,
            n_layer=3,
            vocab_size=20,
            chunk_size=4,
            activation="relu",
            recurrent_init="identity",
            recurrent_identity_scale=0.9,
        )
        expected = 0.9 * torch.eye(8)
        for layer in model.rnn.layers:
            torch.testing.assert_close(layer.weight_hh_l0, expected)
        for forward, inverse in zip(
            model.rnn.parameters(), model.inverse_rnn.parameters()
        ):
            torch.testing.assert_close(forward, inverse)

    def test_invalid_core_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "activation"):
            self.make_model(activation="gelu")
        with self.assertRaisesRegex(ValueError, "recurrent_init"):
            self.make_model(recurrent_init="xavier")
        with self.assertRaisesRegex(ValueError, "requires relu"):
            self.make_model(recurrent_init="identity")
        with self.assertRaisesRegex(ValueError, "recurrent_identity_scale"):
            self.make_model(
                activation="relu",
                recurrent_init="identity",
                recurrent_identity_scale=0.0,
            )

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

    def test_native_rnn_receives_contiguous_expanded_initial_state(self):
        model = self.make_model()
        initial = model.default_state(self.inputs.size(0))
        self.assertFalse(initial.is_contiguous())
        layer = model.rnn.layers[0]
        with mock.patch.object(layer, "forward", wraps=layer.forward) as call:
            model.rnn(model.embedding(self.inputs), initial)
        self.assertTrue(call.call_args.args[1].is_contiguous())

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

    def test_forward_has_aux_components_and_finite_diagnostics(self):
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
            {
                "chunk_ce",
                "discrete_ce",
                "memory_nll",
                "terminal_nll",
                "state_scale",
                "total",
            },
        )
        self.assertTrue(torch.isfinite(model.loss_components["terminal_nll"]))
        self.assertTrue(
            torch.isfinite(model.metrics["aux/memory_relative_mse"])
        )

    def test_learned_gaussian_scales_are_frozen_until_configured_step(self):
        model = self.make_model(
            chunk_offset=0,
            gaussian_scale_mode="learned",
            gaussian_scale_learning_start_step=1,
            gaussian_scale_learning_rate=1e-5,
            use_chunk_loss=False,
            use_discrete_loss=False,
            use_terminal_loss=False,
        )
        self.assertEqual(
            model.log_rho._optim, {"lr": 1e-5, "weight_decay": 0.0}
        )
        first, _ = model(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        first_gradient = torch.autograd.grad(
            first.aux_loss, model.log_rho, allow_unused=True
        )[0]
        self.assertIsNone(first_gradient)
        second, _ = model(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        second_gradient = torch.autograd.grad(second.aux_loss, model.log_rho)[0]
        self.assertGreater(second_gradient.abs().item(), 0.0)
        self.assertEqual(model.aux_training_step.item(), 2)

    def test_trajectory_scale_constraint_uses_all_post_input_states(self):
        model = self.make_model(
            chunk_offset=0,
            gaussian_scale_mode="learned",
            memory_scale_target=1.0,
            memory_scale_constraint_weight=2.0,
            memory_scale_constraint_start_step=0,
            memory_scale_constraint_ramp_steps=0,
            use_chunk_loss=False,
            use_discrete_loss=False,
            use_memory_loss=False,
            use_terminal_loss=False,
        )
        initial = model.default_state(self.inputs.size(0))
        _, _, trajectory = model.rnn(
            model.embedding(self.inputs), initial, return_trajectory=True
        )
        log_rms = model.trajectory_log_rms(trajectory)
        expected = 2.0 * log_rms.square().mean()
        output, _ = model(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        torch.testing.assert_close(model.loss_components["state_scale"], expected)
        torch.testing.assert_close(output.aux_loss, expected)
        self.assertEqual(model.metrics["aux/state_scale_active"].item(), 1.0)

    def test_learned_scale_target_initializes_from_trajectory_and_has_gradient(self):
        model = self.make_model(
            chunk_offset=0,
            gaussian_scale_mode="learned",
            memory_scale_target_mode="learned",
            memory_scale_target_learning_rate=3e-4,
            memory_scale_constraint_weight=0.2,
            memory_scale_constraint_start_step=0,
            use_chunk_loss=False,
            use_discrete_loss=False,
            use_memory_loss=False,
            use_terminal_loss=False,
        )
        self.assertEqual(
            model.log_memory_scale_target._optim,
            {"lr": 3e-4, "weight_decay": 0.0},
        )
        model(self.inputs, targets=self.targets, aux_tokens=self.targets)
        self.assertTrue(model.memory_scale_target_initialized.item())
        self.assertTrue(torch.isfinite(model.log_memory_scale_target))

        with torch.no_grad():
            model.log_memory_scale_target.add_(torch.log(torch.tensor(2.0)))
        output, _ = model(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        gradient = torch.autograd.grad(
            output.aux_loss, model.log_memory_scale_target
        )[0]
        self.assertGreater(gradient.abs().item(), 0.0)

    def test_learned_scale_target_checkpoint_preserves_lazy_initialization(self):
        source = self.make_model(
            memory_scale_target_mode="learned",
            memory_scale_constraint_weight=0.2,
        )
        restored = self.make_model(
            memory_scale_target_mode="learned",
            memory_scale_constraint_weight=0.2,
        )
        restored.load_state_dict(source.state_dict(), strict=True)
        self.assertFalse(restored.memory_scale_target_initialized.item())
        restored(self.inputs, targets=self.targets, aux_tokens=self.targets)
        self.assertTrue(restored.memory_scale_target_initialized.item())

    def test_validation_sanity_forward_does_not_initialize_scale_target(self):
        model = self.make_model(
            memory_scale_target_mode="learned",
            memory_scale_constraint_weight=0.2,
        ).eval()
        model(self.inputs, targets=self.targets, aux_tokens=self.targets)
        self.assertFalse(model.memory_scale_target_initialized.item())

    def test_fixed_and_learned_scale_modes_do_not_change_forward_initialization(self):
        torch.manual_seed(123)
        fixed = self.make_model(rho=8.0, tau=12.0)
        torch.manual_seed(123)
        learned = self.make_model(
            rho=8.0,
            tau=12.0,
            gaussian_scale_mode="learned",
            gaussian_scale_learning_start_step=100,
        )
        fixed_forward = (
            fixed.embedding.weight,
            fixed.initial_state,
            *tuple(fixed.rnn.parameters()),
        )
        learned_forward = (
            learned.embedding.weight,
            learned.initial_state,
            *tuple(learned.rnn.parameters()),
        )
        for actual, expected in zip(learned_forward, fixed_forward):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(learned._configured_gaussian_scale("rho").item(), 8.0)
        self.assertEqual(learned._configured_gaussian_scale("tau").item(), 12.0)

    def test_calibrated_branch_target_survives_nan_warmup_checkpoint(self):
        warmup = self.make_model(
            gaussian_scale_mode="learned",
            gaussian_scale_learning_start_step=100,
        )
        warmup.aux_training_step.fill_(12)
        branch = self.make_model(
            gaussian_scale_mode="learned",
            gaussian_scale_learning_start_step=12,
            memory_scale_target=3.25,
            memory_scale_constraint_weight=0.5,
        )
        branch.load_state_dict(warmup.state_dict(), strict=True)
        self.assertEqual(branch.aux_training_step.item(), 12)
        self.assertEqual(branch.memory_scale_target.item(), 3.25)

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

    def test_explicit_single_scale_is_exactly_the_legacy_scalar_path(self):
        legacy = self.make_model()
        explicit = self.make_model(aux_chunk_sizes=[4])
        explicit.load_state_dict(legacy.state_dict())
        legacy.train()
        explicit.train()
        rng = torch.random.get_rng_state()
        first, first_state = legacy(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        torch.random.set_rng_state(rng)
        second, second_state = explicit(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
        torch.testing.assert_close(first_state, second_state, rtol=0, atol=0)
        torch.testing.assert_close(first.aux_loss, second.aux_loss, rtol=0, atol=0)

    def test_hydra_listconfig_chunk_sizes_are_accepted(self):
        model = self.make_model(aux_chunk_sizes=ListConfig([1, 4]))
        self.assertEqual(model.aux_chunk_sizes, (1, 4))

    def test_multiscale_loss_is_sum_of_complete_individual_scale_losses(self):
        multi = self.make_model(aux_chunk_sizes=[2, 4], chunk_offset=0).eval()
        scale2 = self.make_model(
            chunk_size=2, aux_chunk_sizes=[2], chunk_offset=0
        ).eval()
        scale4 = self.make_model(aux_chunk_sizes=[4], chunk_offset=0).eval()
        scale2.load_state_dict(multi.state_dict())
        scale4.load_state_dict(multi.state_dict())
        combined, _ = multi(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        only2, _ = scale2(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        only4, _ = scale4(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        torch.testing.assert_close(combined.aux_loss, only2.aux_loss + only4.aux_loss)
        self.assertEqual(set(multi.scale_loss_components), {2, 4})

    def test_chunk_one_has_discrete_and_memory_but_no_chunk_ce(self):
        model = self.make_model(
            chunk_size=1,
            aux_chunk_sizes=[1],
            chunk_offset=0,
            use_terminal_loss=False,
        )
        output, _ = model(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        torch.testing.assert_close(
            model.loss_components["chunk_ce"], output.aux_loss.new_zeros(())
        )
        self.assertGreater(model.loss_components["discrete_ce"].item(), 0.0)
        self.assertGreater(model.loss_components["memory_nll"].abs().item(), 0.0)

    def test_vmf_fixed_and_learned_have_proper_finite_losses_and_kappa_gradients(self):
        fixed = self.make_model(
            chunk_offset=0,
            state_aux_distribution="vmf",
            vmf_kappa_mode="fixed",
            memory_vmf_kappa=3.0,
            terminal_vmf_kappa=5.0,
        )
        fixed_output, _ = fixed(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        self.assertTrue(torch.isfinite(fixed_output.aux_loss))
        learned = self.make_model(
            chunk_offset=0,
            state_aux_distribution="vmf",
            vmf_kappa_mode="learned",
            memory_vmf_kappa=3.0,
            terminal_vmf_kappa=5.0,
        )
        learned_output, _ = learned(
            self.inputs, targets=self.targets, aux_tokens=self.targets
        )
        learned_output.aux_loss.backward()
        self.assertTrue(torch.isfinite(learned_output.aux_loss))
        self.assertIsNotNone(learned.log_memory_vmf_kappa.grad)
        self.assertIsNotNone(learned.log_terminal_vmf_kappa.grad)
        self.assertTrue(learned.log_memory_vmf_kappa._no_weight_decay)
        self.assertTrue(learned.log_terminal_vmf_kappa._no_weight_decay)
        self.assertTrue(learned.terminal_target._no_weight_decay)

    def test_vmf_terminal_direction_does_not_shift_forward_initialization(self):
        torch.manual_seed(991)
        gaussian = self.make_model(state_aux_distribution="gaussian")
        torch.manual_seed(991)
        vmf = self.make_model(state_aux_distribution="vmf")
        gaussian_forward = (
            gaussian.embedding.weight,
            gaussian.initial_state,
            *tuple(gaussian.rnn.parameters()),
        )
        vmf_forward = (
            vmf.embedding.weight,
            vmf.initial_state,
            *tuple(vmf.rnn.parameters()),
        )
        for actual, expected in zip(vmf_forward, gaussian_forward):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(vmf.terminal_target.norm().item(), 0.0)

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

    def test_boundary_conditioning_preserves_token_reconstruction(self):
        default = self.make_model(chunk_offset=0).eval()
        conditioned = self.make_model(
            chunk_offset=0,
            condition_memory_reconstruction_on_boundary=True,
        ).eval()
        conditioned.load_state_dict(default.state_dict())
        chunks = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        boundaries = torch.tensor([1, 5])
        successor = torch.randn(1, 2, 8)

        default_logits, default_targets, _ = default._inverse_batch(
            chunks, boundaries, successor
        )
        conditioned_logits, conditioned_targets, conditioned_memory = (
            conditioned._inverse_batch(chunks, boundaries, successor)
        )

        torch.testing.assert_close(conditioned_logits[:, :4], default_logits)
        torch.testing.assert_close(conditioned_targets[:, :4], default_targets)
        self.assertTrue(torch.all(conditioned_targets[:, -1].eq(-100)))
        self.assertEqual(tuple(conditioned_memory.shape), (1, 2, 8))

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

    def test_all_core_options_have_26432_forward_parameters(self):
        conditions = (
            {},
            {"activation": "relu", "recurrent_init": "orthogonal"},
            {"activation": "relu", "recurrent_init": "identity"},
        )
        for options in conditions:
            model = RNNAuxLM(
                d_model=64,
                n_layer=3,
                vocab_size=20,
                chunk_size=4,
                **options,
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
            use_terminal_loss=False,
        )
        on = self.make_model(
            chunk_offset=0,
            stop_gradient_memory_target=True,
            stop_gradient_memory_observation=True,
            use_chunk_loss=False,
            use_discrete_loss=False,
            use_terminal_loss=False,
        )
        on.load_state_dict(off.state_dict())
        short_inputs = self.inputs[:, :8]
        short_targets = self.targets[:, :8]

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
            use_terminal_loss=False,
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
            use_terminal_loss=False,
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

    def test_initial_memory_reconstruction_is_optional_and_off_by_default(self):
        excluded = self.make_model(
            chunk_offset=0,
            use_chunk_loss=False,
            use_discrete_loss=False,
            use_terminal_loss=False,
        )
        included = self.make_model(
            chunk_offset=0,
            use_chunk_loss=False,
            use_discrete_loss=False,
            use_terminal_loss=False,
            exclude_initial_memory_reconstruction=False,
        )
        included.load_state_dict(excluded.state_dict())
        short_inputs = self.inputs[:, :4]
        short_targets = self.targets[:, :4]

        excluded_output, _ = excluded(
            short_inputs, targets=short_targets, aux_tokens=short_targets
        )
        included_output, _ = included(
            short_inputs, targets=short_targets, aux_tokens=short_targets
        )
        torch.testing.assert_close(
            excluded_output.aux_loss,
            excluded_output.aux_loss.new_zeros(()),
        )
        self.assertGreater(included_output.aux_loss.item(), 0.0)

        token_only = self.make_model(
            chunk_offset=0,
            use_memory_loss=False,
            use_terminal_loss=False,
        )
        token_output, _ = token_only(
            short_inputs, targets=short_targets, aux_tokens=short_targets
        )
        self.assertGreater(token_output.aux_loss.item(), 0.0)

    def test_terminal_target_is_learned(self):
        model = self.make_model(
            chunk_offset=0,
            use_chunk_loss=False,
            use_discrete_loss=False,
            use_memory_loss=False,
        )
        output, _ = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
        )
        output.aux_loss.backward()
        self.assertGreater(model.terminal_target.grad.norm().item(), 0.0)
        self.assertGreater(model.initial_state.grad.norm().item(), 0.0)

    def test_probe_only_aux_has_exactly_the_lm_forward_gradient(self):
        baseline = self.make_model(chunk_offset=0)
        probe = self.make_model(chunk_offset=0, auxiliary_probe_only=True)
        probe.load_state_dict(baseline.state_dict())

        baseline_output, _ = baseline(self.inputs, compute_aux=False)
        baseline_lm = F.cross_entropy(
            baseline_output.logits.flatten(0, 1), self.targets.flatten()
        )
        probe_output, _ = probe(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        probe_lm = F.cross_entropy(
            probe_output.logits.flatten(0, 1), self.targets.flatten()
        )

        def forward_parameters(model):
            return (
                model.embedding.weight,
                model.initial_state,
                *tuple(model.rnn.parameters()),
            )

        baseline_gradients = torch.autograd.grad(
            baseline_lm, forward_parameters(baseline)
        )
        probe_gradients = torch.autograd.grad(
            probe_lm + 0.1 * probe_output.aux_loss,
            forward_parameters(probe),
            retain_graph=True,
        )
        for actual, expected in zip(probe_gradients, baseline_gradients):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        inverse_gradients = torch.autograd.grad(
            probe_output.aux_loss,
            (*tuple(probe.inverse_rnn.parameters()), probe.terminal_target),
            allow_unused=True,
        )
        self.assertTrue(
            any(
                gradient is not None and gradient.norm().item() > 0.0
                for gradient in inverse_gradients
            )
        )

    def test_probe_only_adamw_step_leaves_forward_trajectory_unchanged(self):
        baseline = self.make_model(chunk_offset=0)
        probe = self.make_model(chunk_offset=0, auxiliary_probe_only=True)
        probe.load_state_dict(baseline.state_dict())
        baseline_optimizer = torch.optim.AdamW(
            baseline.parameters(), lr=1e-3, weight_decay=0.1
        )
        probe_optimizer = torch.optim.AdamW(
            probe.parameters(), lr=1e-3, weight_decay=0.1
        )

        baseline_output, _ = baseline(self.inputs, compute_aux=False)
        baseline_loss = F.cross_entropy(
            baseline_output.logits.flatten(0, 1), self.targets.flatten()
        )
        baseline_loss.backward()
        baseline_optimizer.step()

        probe_output, _ = probe(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        probe_loss = F.cross_entropy(
            probe_output.logits.flatten(0, 1), self.targets.flatten()
        ) + 0.1 * probe_output.aux_loss
        probe_loss.backward()
        probe_optimizer.step()

        baseline_forward = (
            baseline.embedding.weight,
            baseline.initial_state,
            *tuple(baseline.rnn.parameters()),
        )
        probe_forward = (
            probe.embedding.weight,
            probe.initial_state,
            *tuple(probe.rnn.parameters()),
        )
        for actual, expected in zip(probe_forward, baseline_forward):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        self.assertTrue(
            any(
                not torch.equal(actual, expected)
                for actual, expected in zip(
                    probe.inverse_rnn.parameters(),
                    baseline.inverse_rnn.parameters(),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
