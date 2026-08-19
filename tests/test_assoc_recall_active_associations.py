import unittest

import numpy as np

from src.dataloaders.synthetics import Vocab, generate_assoc_recall


class ActiveAssociationGenerationTest(unittest.TestCase):
    def test_body_contains_exactly_requested_active_keys(self):
        vocab = Vocab(18, {"copy_prefix": "=>", "noop": "."})
        for active in (3, 5, 7, 9):
            text = generate_assoc_recall(
                vocab,
                input_seq_len=40,
                num_keys=1,
                rng=np.random.default_rng(100 + active),
                allow_dot=False,
                num_active_associations=active,
            )
            tokens = text.split()
            body = tokens[:40]
            query_key = tokens[-2]
            self.assertEqual(len(set(body[::2])), active)
            self.assertIn(query_key, body[::2])
            self.assertEqual(tokens[-3], "=>")

    def test_rejects_more_active_keys_than_available(self):
        vocab = Vocab(18, {"copy_prefix": "=>", "noop": "."})
        with self.assertRaises(ValueError):
            generate_assoc_recall(
                vocab,
                input_seq_len=40,
                num_keys=1,
                rng=np.random.default_rng(1),
                num_active_associations=10,
            )


if __name__ == "__main__":
    unittest.main()
