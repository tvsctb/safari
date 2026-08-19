import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[1] / "src" / "dataloaders" / "controlled_assoc_recall.py"
SPEC = importlib.util.spec_from_file_location("controlled_assoc_recall", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
associative_recall_vocab = MODULE.associative_recall_vocab
generate_controlled_lag_batch = MODULE.generate_controlled_lag_batch


class ControlledAssociativeRecallTest(unittest.TestCase):
    def test_only_query_pair_position_changes(self):
        batch = generate_controlled_lag_batch(3, seed=17)
        self.assertEqual(tuple(batch.input_ids.shape), (3, 20, 42))
        self.assertEqual(tuple(batch.targets.shape), (3, 20))
        self.assertEqual(batch.lags.tolist(), list(range(40, 0, -2)))

        vocab = associative_recall_vocab()
        prefix = vocab.get_id(vocab.copy_prefix)
        for example in range(3):
            query_key = int(batch.input_ids[example, 0, -1])
            target = int(batch.targets[example, 0])
            reference_pairs = None
            for position in range(20):
                sequence = batch.input_ids[example, position]
                self.assertEqual(int(sequence[-2]), prefix)
                self.assertEqual(int(sequence[-1]), query_key)
                self.assertEqual(int(batch.targets[example, position]), target)
                pairs = [tuple(pair) for pair in sequence[:40].reshape(20, 2).tolist()]
                self.assertEqual(sum(key == query_key for key, _ in pairs), 1)
                self.assertEqual(pairs[position], (query_key, target))
                counter = Counter(pairs)
                if reference_pairs is None:
                    reference_pairs = counter
                self.assertEqual(counter, reference_pairs)

    def test_is_deterministic(self):
        first = generate_controlled_lag_batch(4, seed=99)
        second = generate_controlled_lag_batch(4, seed=99)
        self.assertTrue(torch.equal(first.input_ids, second.input_ids))
        self.assertTrue(torch.equal(first.targets, second.targets))

    def test_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            generate_controlled_lag_batch(0, seed=1)


if __name__ == "__main__":
    unittest.main()
