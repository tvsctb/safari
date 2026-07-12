import unittest

import torch
import torch.nn.functional as F

from src.models.sequence.auxiliary import (
    boundary_inverse_batch,
    chunk_ranges,
    cross_entropy_sum,
    gaussian_nll_sum,
    terminal_gaussian_nll,
)
from src.models.sequence.gru_aux import GRUAuxLM
from src.models.sequence.rmt_aux import RMTAuxLM


class AuxiliaryUtilityTest(unittest.TestCase):
    def test_chunk_ranges_with_offset_and_remainder(self):
        self.assertEqual(chunk_ranges(10, 4, 0), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(chunk_ranges(10, 4, 2), [(0, 2), (2, 6), (6, 10)])

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
            ),
            RMTAuxLM(
                d_model=16,
                n_layer=2,
                d_inner=32,
                n_heads=4,
                vocab_size=20,
                chunk_size=4,
                num_memory_tokens=2,
                random_chunk_offset=False,
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
            random_chunk_offset=False,
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
            random_chunk_offset=False,
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
        self.assertFalse(rmt.random_chunk_offset)
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


if __name__ == "__main__":
    unittest.main()
