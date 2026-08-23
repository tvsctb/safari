import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from src.models.sequence.rnn_aux import RNNAuxLM


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_ar_rnn_past_info_probe.py"
SPEC = importlib.util.spec_from_file_location("past_probe_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PastInformationProbeTest(unittest.TestCase):
    def model(self):
        return RNNAuxLM(
            d_model=8,
            n_layer=2,
            vocab_size=20,
            activation="relu",
            recurrent_init="identity",
            recurrent_identity_scale=1.0,
            rho=8.0,
            tau=243.0,
        )

    def test_probe_matches_inverse_topology_and_does_not_alias_forward(self):
        model = self.model()
        model.requires_grad_(False)
        probe = MODULE.InverseMatchedPastProbe(model, initialization_seed=17)
        self.assertEqual(len(probe.inverse_rnn.layers), len(model.inverse_rnn.layers))
        self.assertEqual(probe.embedding_weight.shape, model.embedding.weight.shape)
        self.assertTrue(all(parameter.requires_grad for parameter in probe.parameters()))
        before = model.embedding.weight.detach().clone()
        states = torch.randn(5, 2, 8)
        keys = torch.arange(5)
        loss = torch.nn.functional.cross_entropy(probe(states, keys), torch.arange(5) + 9)
        torch.optim.AdamW(probe.parameters(), lr=1e-3).zero_grad()
        loss.backward()
        self.assertIsNone(model.embedding.weight.grad)
        self.assertTrue(torch.equal(before, model.embedding.weight))

    def test_common_seed_gives_identical_inverse_initialization(self):
        first = MODULE.InverseMatchedPastProbe(self.model(), initialization_seed=23)
        second = MODULE.InverseMatchedPastProbe(self.model(), initialization_seed=23)
        for left, right in zip(first.inverse_rnn.parameters(), second.inverse_rnn.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_global_unit_removes_only_global_norm(self):
        states = torch.randn(7, 3, 5)
        normalized = MODULE.transform_states(states, "global_unit")
        torch.testing.assert_close(normalized.flatten(1).norm(dim=1), torch.ones(7))
        ratios_before = states[:, 0].norm(dim=1) / states[:, 1].norm(dim=1)
        ratios_after = normalized[:, 0].norm(dim=1) / normalized[:, 1].norm(dim=1)
        torch.testing.assert_close(ratios_before, ratios_after)

    def test_extraction_uses_body_boundaries_and_past_targets(self):
        model = self.model().eval()
        dataset = MODULE.extract_examples(
            model,
            num_base_examples=2,
            dataset_seed=101,
            base_batch_size=16,
            device=torch.device("cpu"),
        )
        states, keys, targets, ages = dataset.tensors
        self.assertEqual(states.shape[1:], (2, 8))
        self.assertEqual(set(ages.tolist()), set(range(0, 40, 2)))
        self.assertTrue(torch.all((keys >= 0) & (keys < 20)))
        self.assertTrue(torch.all((targets >= 0) & (targets < 20)))
        # Twenty placements yield 5+5+... alternating eligible boundaries:
        # 10 + 10 + 9 + 9 + ... + 1 + 1 = 110 examples per base item.
        self.assertEqual(len(dataset), 220)


if __name__ == "__main__":
    unittest.main()
