"""Counterbalanced associative-recall examples with an exactly controlled lag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

@dataclass(frozen=True)
class ControlledLagBatch:
    """Twenty counterfactual placements of the same queried key/value pair."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    lags: torch.Tensor


@dataclass(frozen=True)
class ControlledARVocab:
    non_special_vocab: tuple[str, ...]
    vocab: tuple[str, ...]
    v2id: dict[str, int]
    copy_prefix: str = "=>"
    noop: str = "."

    def get_id(self, token: str) -> int:
        return self.v2id[token]


def associative_recall_vocab(vocab_size: int = 20) -> ControlledARVocab:
    if vocab_size < 6 or vocab_size % 2:
        raise ValueError("vocab_size must be even and at least 6")
    # This exactly mirrors ``synthetics.Vocab`` without importing the entire
    # dataloader package (whose optional datasets have heavy dependencies).
    non_special = tuple(sorted(str(index) for index in range(vocab_size - 2)))
    tokens = tuple(sorted((*non_special, "=>", ".")))
    return ControlledARVocab(
        non_special_vocab=non_special,
        vocab=tokens,
        v2id={token: index for index, token in enumerate(tokens)},
    )


def generate_controlled_lag_batch(
    num_base_examples: int,
    *,
    seed: int,
    num_pairs: int = 20,
    vocab_size: int = 20,
) -> ControlledLagBatch:
    """Generate paired examples whose body stays fixed at ``2 * num_pairs``.

    For each base example, the queried key/value pair occurs exactly once. The
    other ``num_pairs - 1`` pairs are shared by all placements, in the same
    order, and the queried pair is inserted at every possible pair position.
    This makes position (and hence recurrent lag) the only within-example
    intervention.
    """
    if num_base_examples <= 0:
        raise ValueError("num_base_examples must be positive")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")

    vocab = associative_recall_vocab(vocab_size)
    non_special = vocab.non_special_vocab
    split = len(non_special) // 2
    keys, values = non_special[:split], non_special[split:]
    if len(keys) < 2:
        raise ValueError("controlled lag requires at least two keys")

    key_ids = np.asarray([vocab.get_id(token) for token in keys], dtype=np.int64)
    value_ids = np.asarray(
        [vocab.get_id(token) for token in values], dtype=np.int64
    )
    prefix_id = vocab.get_id(vocab.copy_prefix)
    rng = np.random.default_rng(seed)

    # Match the original generator: each example samples a key->value mapping,
    # with replacement on the value side.
    mappings = rng.choice(value_ids, size=(num_base_examples, len(keys)))
    query_indices = rng.integers(len(keys), size=num_base_examples)
    distractor_indices = np.empty(
        (num_base_examples, num_pairs - 1), dtype=np.int64
    )
    for row, query_index in enumerate(query_indices):
        choices = np.delete(np.arange(len(keys)), query_index)
        distractor_indices[row] = rng.choice(
            choices, size=num_pairs - 1, replace=True
        )

    sequence_length = 2 * num_pairs + 2
    inputs = np.empty(
        (num_base_examples, num_pairs, sequence_length), dtype=np.int64
    )
    targets = np.empty((num_base_examples, num_pairs), dtype=np.int64)
    for row in range(num_base_examples):
        query_index = int(query_indices[row])
        query_key = key_ids[query_index]
        query_value = mappings[row, query_index]
        distractors = [
            (key_ids[index], mappings[row, index])
            for index in distractor_indices[row]
        ]
        for position in range(num_pairs):
            pairs = list(distractors)
            pairs.insert(position, (query_key, query_value))
            body = np.asarray(pairs, dtype=np.int64).reshape(-1)
            inputs[row, position, : 2 * num_pairs] = body
            inputs[row, position, -2:] = (prefix_id, query_key)
            targets[row, position] = query_value

    # The queried value is at 2*p+1 and the query key is at 2*num_pairs+1.
    # Their recurrent separation is therefore 2*num_pairs - 2*p.
    lags = torch.arange(2 * num_pairs, 0, -2, dtype=torch.long)
    return ControlledLagBatch(
        input_ids=torch.from_numpy(inputs),
        targets=torch.from_numpy(targets),
        lags=lags,
    )
