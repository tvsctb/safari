import unittest

import torch
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    boundary_inverse_batch,
    chunk_ranges,
    cross_entropy_sum,
    gaussian_nll_sum,
    role_inverse_targets,
    terminal_gaussian_nll,
)
from src.models.sequence.gru_aux import GRUAuxLM
from src.models.sequence.rmt_aux import RMTAuxLM


class AuxiliaryUtilityTest(unittest.TestCase):
    def test_chunk_ranges_with_offset_and_remainder(self):
        self.assertEqual(chunk_ranges(10, 4, 0), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(chunk_ranges(10, 4, 2), [(0, 2), (2, 6), (6, 10)])

    def test_minibatch_offset_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed, sequence"):
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                random_chunk_offset="minibatch",
            )

    def test_boundary_reverse_alignment(self):
        chunk = torch.tensor([[11, 12, 13, 14]])
        boundary = torch.tensor([10])
        memory = torch.ones(1, 1, 3)
        data_inputs, memory_inputs, targets = boundary_inverse_batch(
            chunk, boundary, memory
        )
        torch.testing.assert_close(data_inputs, torch.tensor([[14, 13, 12, 11]]))
        torch.testing.assert_close(targets, torch.tensor([[13, 12, 11, 10]]))
        self.assertEqual(memory_inputs.shape, (1, 1, 3))

    def test_role_scheme_alignments(self):
        chunk = torch.tensor([[11, 12, 13, 14]])
        reverse_inputs, reverse_targets = role_inverse_targets(
            chunk, "role_reverse"
        )
        forward_inputs, forward_targets = role_inverse_targets(
            chunk, "role_forward"
        )
        torch.testing.assert_close(
            reverse_inputs, torch.tensor([[14, 13, 12, 11]])
        )
        torch.testing.assert_close(reverse_targets, reverse_inputs)
        torch.testing.assert_close(forward_inputs, chunk)
        torch.testing.assert_close(forward_targets, chunk)

    def test_report_losses_sum_within_sequence(self):
        batch_size = 2
        logits = torch.zeros(batch_size, 3, 4)
        targets = torch.zeros(batch_size, 3, dtype=torch.long)
        token_loss = cross_entropy_sum(
            [logits], [targets], logits, batch_size
        )
        self.assertAlmostEqual(token_loss.item(), 3.0 * torch.log(torch.tensor(4.0)).item())

        memory_target = torch.ones(2, batch_size, 3)
        memory_estimate = torch.zeros_like(memory_target)
        scale = torch.ones(())
        memory_loss = gaussian_nll_sum(
            [memory_target], [memory_estimate], scale, logits, batch_size
        )
        terminal_loss = terminal_gaussian_nll(
            memory_target, scale, batch_size
        )
        self.assertEqual(memory_loss.item(), 3.0)
        self.assertEqual(terminal_loss.item(), 3.0)

    def test_layerwise_and_slotwise_gaussian_scales(self):
        reference = torch.zeros(())
        gru_state = torch.ones(2, 1, 3)
        gru_scale = torch.tensor([1.0, 2.0])
        gru_loss = terminal_gaussian_nll(
            gru_state, gru_scale, batch_size=1, scale_axis=0
        )
        expected_gru = 3 * 0.5 + 3 * (0.125 + torch.log(torch.tensor(2.0)))
        torch.testing.assert_close(gru_loss, expected_gru)

        rmt_state = torch.ones(1, 2, 3)
        rmt_loss = gaussian_nll_sum(
            [rmt_state],
            [torch.zeros_like(rmt_state)],
            gru_scale,
            reference,
            batch_size=1,
            scale_axis=1,
        )
        torch.testing.assert_close(rmt_loss, expected_gru)


class AuxModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.inputs = torch.randint(0, 20, (2, 10))
        self.targets = torch.randint(0, 20, (2, 10))

    def _models(self):
        return [
            GRUAuxLM(
                d_model=16,
                n_layer=2,
                vocab_size=20,
                chunk_size=4,
                random_chunk_offset=False,
                scale_mode="learned",
                use_direction_embedding=True,
            ),
            RMTAuxLM(
                d_model=16,
                n_layer=2,
                d_inner=32,
                n_heads=4,
                vocab_size=20,
                chunk_size=4,
                num_memory_tokens=2,
                scale_mode="learned",
                use_direction_embedding=True,
            ),
        ]

    def test_aux_forward_and_backward(self):
        for model in self._models():
            with self.subTest(model=type(model).__name__):
                output, state = model(self.inputs, targets=self.targets, compute_aux=True)
                self.assertEqual(output.logits.shape, (2, 10, 20))
                self.assertTrue(torch.isfinite(output.aux_loss))
                self.assertEqual(state.size(1 if isinstance(model, GRUAuxLM) else 0), 2)

                loss = F.cross_entropy(
                    output.logits.reshape(-1, 20), self.targets.reshape(-1)
                ) + output.aux_loss
                loss.backward()
                self.assertIsNotNone(model.direction_embedding.grad)
                self.assertGreater(model.direction_embedding.grad.norm().item(), 0.0)
                self.assertIsNotNone(model.log_rho.grad)
                self.assertIsNotNone(model.log_tau.grad)
                component_sum = sum(
                    model.metrics[name]
                    for name in (
                        "aux/chunk_ce",
                        "aux/discrete_ce",
                        "aux/memory_nll",
                        "aux/terminal_nll",
                        "aux/terminal_chunk",
                    )
                )
                torch.testing.assert_close(output.aux_loss, component_sum)
                terminal_loss = terminal_gaussian_nll(
                    state, model.metrics["aux/tau"], self.inputs.size(0)
                ) / self.inputs.size(1)
                torch.testing.assert_close(
                    model.metrics["aux/terminal_nll"], terminal_loss
                )
                terminal_length = self.inputs.size(1) % model.chunk_size
                terminal_length = terminal_length or model.chunk_size
                terminal_ce = F.cross_entropy(
                    output.logits[:, -terminal_length:].reshape(-1, 20),
                    self.targets[:, -terminal_length:].reshape(-1),
                    reduction="sum",
                ) / (self.inputs.size(0) * self.inputs.size(1))
                torch.testing.assert_close(
                    model.metrics["aux/terminal_chunk"], terminal_ce
                )

    def test_eval_accepts_masked_targets_without_aux(self):
        masked_targets = torch.full_like(self.targets, -100)
        for model in self._models():
            with self.subTest(model=type(model).__name__):
                model.eval()
                output, _ = model(
                    self.inputs, targets=masked_targets, compute_aux=False
                )
                self.assertEqual(output.aux_loss.item(), 0.0)

    def test_models_are_causal(self):
        changed = self.inputs.clone()
        changed[:, 6:] = (changed[:, 6:] + 7) % 20
        for model in self._models():
            with self.subTest(model=type(model).__name__):
                model.eval()
                original, _ = model(self.inputs, compute_aux=False)
                modified, _ = model(changed, compute_aux=False)
                torch.testing.assert_close(
                    original.logits[:, :6], modified.logits[:, :6], atol=1e-6, rtol=1e-6
                )

    def test_gru_chunking_matches_single_call(self):
        model = GRUAuxLM(
            d_model=16,
            n_layer=2,
            vocab_size=20,
            chunk_size=4,
            dropout=0.0,
            random_chunk_offset=False,
        )
        model.eval()
        output, final_state = model(self.inputs, compute_aux=False)
        direct_output, direct_state = model.gru(model.embedding(self.inputs))
        direct_logits = model._lm_logits(direct_output)
        torch.testing.assert_close(output.logits, direct_logits)
        _, terminal_memory = model.gru(model.embedding(self.inputs[:, :-2]))
        torch.testing.assert_close(final_state, terminal_memory)
        self.assertFalse(torch.equal(final_state, direct_state))

    def test_rmt_query_parameters_are_distinct(self):
        model = self._models()[1]
        self.assertIsNot(model.forward_queries, model.inverse_queries)
        self.assertNotEqual(
            model.forward_queries.untyped_storage().data_ptr(),
            model.inverse_queries.untyped_storage().data_ptr(),
        )

    def test_rmt_terminal_chunk_has_no_memory_queries(self):
        model = RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=2,
            vocab_size=20,
            chunk_size=4,
            num_memory_tokens=2,
        )
        sequence_lengths = []

        def capture_length(module, args):
            sequence_lengths.append(args[0].size(1))

        handle = model.blocks[0].register_forward_pre_hook(capture_length)
        try:
            model.eval()
            model(self.inputs, compute_aux=False)
        finally:
            handle.remove()

        self.assertEqual(sequence_lengths, [8, 8, 4])

    def test_inverse_excludes_terminal_chunk(self):
        gru = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            chunk_size=4,
            random_chunk_offset=False,
            share_inverse=False,
        )
        gru_inverse_shapes = []
        gru_handle = gru.inverse_gru.register_forward_pre_hook(
            lambda module, args: gru_inverse_shapes.append(args[0].shape)
        )
        try:
            gru(self.inputs, targets=self.targets, compute_aux=True)
        finally:
            gru_handle.remove()
        self.assertEqual(gru_inverse_shapes, [torch.Size([4, 5, 8])])

        rmt = RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=2,
            vocab_size=20,
            chunk_size=4,
            num_memory_tokens=2,
            share_inverse=False,
        )
        rmt_inverse_shapes = []
        rmt_handle = rmt.inverse_blocks[0].register_forward_pre_hook(
            lambda module, args: rmt_inverse_shapes.append(args[0].shape)
        )
        try:
            rmt(self.inputs, targets=self.targets, compute_aux=True)
        finally:
            rmt_handle.remove()
        self.assertEqual(rmt_inverse_shapes, [torch.Size([4, 8, 8])])

    def test_single_terminal_chunk_has_no_inverse_terms(self):
        inputs = self.inputs[:, :3]
        targets = self.targets[:, :3]
        for model in self._models():
            with self.subTest(model=type(model).__name__):
                output, state = model(inputs, targets=targets, compute_aux=True)
                self.assertTrue(torch.isfinite(output.aux_loss))
                self.assertEqual(model.metrics["aux/chunk_ce"].item(), 0.0)
                self.assertEqual(model.metrics["aux/discrete_ce"].item(), 0.0)
                self.assertEqual(model.metrics["aux/memory_nll"].item(), 0.0)
                initial_state = model.default_state(2, device=inputs.device)
                torch.testing.assert_close(state, initial_state)

    def test_report_defaults(self):
        gru = GRUAuxLM(d_model=8, n_layer=1, vocab_size=20)
        self.assertEqual(gru.chunk_size, 4)
        self.assertTrue(gru.random_chunk_offset)
        self.assertEqual(gru.scale_mode, "fixed")
        self.assertFalse(hasattr(gru, "log_rho"))
        self.assertFalse(hasattr(gru, "log_tau"))
        self.assertIs(gru.gru, gru.inverse_gru)
        self.assertEqual(gru.memory_token_id, 20)

        rmt = RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=2,
            vocab_size=20,
        )
        self.assertEqual(rmt.chunk_size, 4)
        self.assertFalse(hasattr(rmt, "random_chunk_offset"))
        self.assertFalse(hasattr(rmt, "chunk_offset_mode"))
        self.assertEqual(rmt.scale_mode, "fixed")
        self.assertFalse(hasattr(rmt, "log_rho"))
        self.assertFalse(hasattr(rmt, "log_tau"))
        self.assertIs(rmt.blocks, rmt.inverse_blocks)

    def test_untied_mode_shares_embedding_and_vocabulary_head(self):
        gru = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            share_inverse=False,
        )
        self.assertIsNot(gru.gru, gru.inverse_gru)
        self.assertIsNone(gru.direction_embedding)
        gru_forward_ptrs = {
            parameter.untyped_storage().data_ptr()
            for parameter in gru.gru.parameters()
        }
        gru_inverse_ptrs = {
            parameter.untyped_storage().data_ptr()
            for parameter in gru.inverse_gru.parameters()
        }
        self.assertTrue(gru_forward_ptrs.isdisjoint(gru_inverse_ptrs))
        self.assertFalse(hasattr(gru, "inverse_head"))
        hidden = torch.randn(2, 3, 8)
        torch.testing.assert_close(
            gru._lm_logits(hidden),
            F.linear(hidden, gru.embedding.weight[:20]),
        )

        rmt = RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=2,
            vocab_size=20,
            share_inverse=False,
        )
        self.assertIsNot(rmt.blocks, rmt.inverse_blocks)
        self.assertIsNot(rmt.final_norm, rmt.inverse_final_norm)
        self.assertIsNone(rmt.direction_embedding)
        self.assertIsNotNone(rmt.inverse_position_embedding)
        torch.testing.assert_close(
            rmt.inverse_position_embedding, rmt.position_embedding
        )
        self.assertNotEqual(
            rmt.inverse_position_embedding.untyped_storage().data_ptr(),
            rmt.position_embedding.untyped_storage().data_ptr(),
        )
        self.assertFalse(hasattr(rmt, "inverse_head"))

        output, _ = rmt(self.inputs, targets=self.targets, compute_aux=True)
        output.aux_loss.backward()
        self.assertIsNotNone(rmt.embedding.weight.grad)
        self.assertIsNotNone(rmt.inverse_position_embedding.grad)

    def test_all_token_schemes(self):
        for token_scheme in ("boundary_reverse", "role_reverse", "role_forward"):
            models = [
                GRUAuxLM(
                    d_model=8,
                    n_layer=1,
                    vocab_size=20,
                    token_scheme=token_scheme,
                    random_chunk_offset=False,
                    use_direction_embedding=(token_scheme == "boundary_reverse"),
                ),
                RMTAuxLM(
                    d_model=8,
                    n_layer=1,
                    d_inner=16,
                    n_heads=2,
                    vocab_size=20,
                    num_memory_tokens=2,
                    token_scheme=token_scheme,
                    use_direction_embedding=(token_scheme == "boundary_reverse"),
                ),
            ]
            for model in models:
                with self.subTest(
                    token_scheme=token_scheme, model=type(model).__name__
                ):
                    output, _ = model(
                        self.inputs, targets=self.targets, compute_aux=True
                    )
                    self.assertEqual(output.logits.shape, (2, 10, 20))
                    self.assertTrue(torch.isfinite(output.aux_loss))
                    if token_scheme == "boundary_reverse":
                        self.assertIsNotNone(model.direction_embedding)
                        self.assertGreater(
                            model.metrics["aux/discrete_ce"].item(), 0.0
                        )
                    else:
                        self.assertIsNone(model.direction_embedding)
                        self.assertEqual(
                            model.metrics["aux/discrete_ce"].item(), 0.0
                        )

    def test_gru_memory_token_is_optional(self):
        observed_lengths = []
        model = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            use_memory_token=False,
            share_inverse=False,
            random_chunk_offset=False,
        )
        handle = model.inverse_gru.register_forward_pre_hook(
            lambda module, args: observed_lengths.append(args[0].size(1))
        )
        try:
            model(self.inputs, targets=self.targets, compute_aux=True)
        finally:
            handle.remove()
        self.assertEqual(observed_lengths, [4])

        role_model = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            token_scheme="role_reverse",
            use_memory_token=True,
            share_inverse=False,
            random_chunk_offset=False,
        )
        role_lengths = []
        handle = role_model.inverse_gru.register_forward_pre_hook(
            lambda module, args: role_lengths.append(args[0].size(1))
        )
        try:
            role_model(self.inputs, targets=self.targets, compute_aux=True)
        finally:
            handle.remove()
        self.assertEqual(role_lengths, [6])

    def test_fixed_scales_are_hyperparameters(self):
        gru = GRUAuxLM(
            d_model=8,
            n_layer=2,
            vocab_size=20,
            scale_mode="fixed",
            scale_granularity="layerwise",
            rho=[0.3, 1.0],
            tau=[1.0, 3.0],
            random_chunk_offset=False,
        )
        self.assertFalse(hasattr(gru, "log_rho"))
        self.assertFalse(hasattr(gru, "log_tau"))
        torch.testing.assert_close(gru.rho, torch.tensor([0.3, 1.0]))
        torch.testing.assert_close(gru.tau, torch.tensor([1.0, 3.0]))
        output, _ = gru(self.inputs, targets=self.targets, compute_aux=True)
        output.aux_loss.backward()
        self.assertIsNotNone(gru.gru.weight_ih_l0.grad)

        rmt = RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=2,
            vocab_size=20,
            num_memory_tokens=2,
            scale_mode="fixed",
            scale_granularity="slotwise",
            rho=[0.3, 1.0],
            tau=[1.0, 3.0],
        )
        output, _ = rmt(self.inputs, targets=self.targets, compute_aux=True)
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertEqual(rmt.metrics["aux/rho/0"].item(), rmt.rho[0].item())
        self.assertEqual(rmt.metrics["aux/tau/1"].item(), rmt.tau[1].item())

    def test_component_level_partial_sharing(self):
        for model in (
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                share_inverse=False,
                share_inverse_embedding=False,
                share_inverse_head=False,
            ),
            RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                num_memory_tokens=2,
                share_inverse=False,
                share_inverse_embedding=False,
                share_inverse_head=False,
            ),
        ):
            with self.subTest(model=type(model).__name__):
                self.assertIsNot(model.embedding, model.inverse_embedding)
                self.assertTrue(hasattr(model, "inverse_head"))
                output, _ = model(
                    self.inputs, targets=self.targets, compute_aux=True
                )
                output.aux_loss.backward()
                self.assertIsNotNone(model.inverse_embedding.weight.grad)
                self.assertIsNotNone(model.inverse_head.weight.grad)

    def test_direction_embedding_option(self):
        shared_boundary_models = [
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                use_direction_embedding=True,
            ),
            RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                use_direction_embedding=True,
            ),
        ]
        for model in shared_boundary_models:
            with self.subTest(model=type(model).__name__, mode="enabled"):
                self.assertTrue(model.use_direction_embedding)
                self.assertIsNotNone(model.direction_embedding)

        disabled_models = [
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                use_direction_embedding=False,
            ),
            RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                use_direction_embedding=False,
            ),
        ]
        for model in disabled_models:
            with self.subTest(model=type(model).__name__, mode="disabled"):
                self.assertFalse(model.use_direction_embedding)
                self.assertIsNone(model.direction_embedding)

        role_enabled = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            token_scheme="role_reverse",
            use_direction_embedding=True,
        )
        self.assertFalse(role_enabled.use_direction_embedding)
        self.assertIsNone(role_enabled.direction_embedding)

        untied_enabled_models = [
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                share_inverse=False,
                use_direction_embedding=True,
            ),
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                share_inverse_embedding=False,
                use_direction_embedding=True,
            ),
            RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                share_inverse_embedding=False,
                use_direction_embedding=True,
            ),
        ]
        for model in untied_enabled_models:
            with self.subTest(model=type(model).__name__, mode="untied"):
                self.assertTrue(model.use_direction_embedding)
                self.assertIsNotNone(model.direction_embedding)

        head_only_untied = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            share_inverse_head=False,
        )
        self.assertFalse(head_only_untied.use_direction_embedding)
        self.assertIsNone(head_only_untied.direction_embedding)

    def test_per_sequence_random_offsets(self):
        model = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            random_chunk_offset="sequence",
        )
        output, state = model(
            self.inputs, targets=self.targets, compute_aux=True
        )
        self.assertEqual(output.logits.shape, (2, 10, 20))
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertTrue(torch.isfinite(state).all())

    def test_chunk_offset_is_not_configurable(self):
        with self.assertRaisesRegex(ValueError, "always 0"):
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                random_chunk_offset="fixed",
                chunk_offset=2,
            )
        with self.assertRaisesRegex(ValueError, "always 0"):
            RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                chunk_offset=1,
            )

    def test_rmt_random_chunk_offset_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed at 0"):
            RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                random_chunk_offset="sequence",
            )

    def test_gru_state_type_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "global, layerwise"):
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                scale_granularity="state_type",
            )

    def test_rmt_role_terminal_is_query_free(self):
        model = RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=2,
            vocab_size=20,
            chunk_size=4,
            num_memory_tokens=2,
            token_scheme="role_reverse",
        )
        sequence_lengths = []
        handle = model.blocks[0].register_forward_pre_hook(
            lambda module, args: sequence_lengths.append(args[0].size(1))
        )
        try:
            model.eval()
            model(self.inputs, targets=self.targets, compute_aux=False)
        finally:
            handle.remove()
        # Two transitions: 2 memory + 5 token + 2 query. Terminal: no queries.
        self.assertEqual(sequence_lengths, [9, 9, 5])

    def test_role_models_share_the_same_unmasked_chunks(self):
        aux_tokens = torch.arange(10).unsqueeze(0).expand(2, -1)
        expected = torch.cat((aux_tokens[:, :4], aux_tokens[:, 4:8]), dim=0)
        models = [
            GRUAuxLM(
                d_model=8,
                n_layer=1,
                vocab_size=20,
                token_scheme="role_reverse",
                random_chunk_offset=False,
            ),
            RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                num_memory_tokens=2,
                token_scheme="role_reverse",
            ),
        ]
        for model in models:
            captured = []
            original = model._inverse_record

            def capture(chunks, *args):
                captured.append(chunks.detach().clone())
                return original(chunks, *args)

            model._inverse_record = capture
            with self.subTest(model=type(model).__name__):
                model(
                    self.inputs,
                    targets=self.targets,
                    aux_tokens=aux_tokens,
                    compute_aux=True,
                )
                self.assertEqual(len(captured), 1)
                torch.testing.assert_close(captured[0], expected)

    def test_rmt_role_accepts_masked_loss_targets(self):
        aux_tokens = self.targets.clone()
        masked_targets = torch.full_like(self.targets, -100)
        masked_targets[:, -1] = aux_tokens[:, -1]
        for token_scheme in ("role_reverse", "role_forward"):
            model = RMTAuxLM(
                d_model=8,
                n_layer=1,
                d_inner=16,
                n_heads=2,
                vocab_size=20,
                num_memory_tokens=2,
                token_scheme=token_scheme,
            )
            with self.subTest(token_scheme=token_scheme):
                model.eval()
                output, _ = model(
                    self.inputs,
                    targets=masked_targets,
                    aux_tokens=aux_tokens,
                    compute_aux=False,
                )
                self.assertEqual(output.logits.shape, (2, 10, 20))
                self.assertTrue(torch.isfinite(output.logits).all())

    def test_rmt_role_inference_uses_explicit_token_stream(self):
        model = RMTAuxLM(
            d_model=8,
            n_layer=1,
            d_inner=16,
            n_heads=2,
            vocab_size=20,
            num_memory_tokens=2,
            token_scheme="role_forward",
        )
        model.eval()
        output, _ = model(
            self.inputs,
            targets=None,
            aux_tokens=self.targets,
            compute_aux=False,
        )
        self.assertEqual(output.logits.shape, (2, 10, 20))
        with self.assertRaisesRegex(ValueError, "unmasked aux_tokens"):
            model(self.inputs, targets=None, compute_aux=False)

    def test_gru_role_state_boundaries_match_aux_chunks(self):
        model = GRUAuxLM(
            d_model=8,
            n_layer=1,
            vocab_size=20,
            token_scheme="role_reverse",
            dropout=0.0,
            random_chunk_offset=False,
        )
        model.eval()
        output, terminal_memory = model(
            self.inputs,
            targets=self.targets,
            aux_tokens=self.targets,
            compute_aux=True,
        )
        direct_output, _ = model.gru(model.embedding(self.inputs))
        _, expected_terminal_memory = model.gru(
            model.embedding(self.inputs[:, :9])
        )
        torch.testing.assert_close(output.logits, model._lm_logits(direct_output))
        torch.testing.assert_close(terminal_memory, expected_terminal_memory)


if __name__ == "__main__":
    unittest.main()
