from typing import NamedTuple

import torch
import torch.nn.functional as F


class AuxCausalLMOutput(NamedTuple):
    logits: torch.Tensor
    aux_loss: torch.Tensor


def chunk_ranges(length, chunk_size, offset=0):
    """Partition a sequence, optionally placing the first boundary before B."""
    if length <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= offset < chunk_size:
        raise ValueError("offset must be in [0, chunk_size)")

    first_length = chunk_size if offset == 0 else offset
    ranges = [(0, min(first_length, length))]
    start = ranges[0][1]
    while start < length:
        end = min(start + chunk_size, length)
        ranges.append((start, end))
        start = end
    return ranges


def boundary_inverse_targets(chunk, boundary):
    """Build reversed data inputs and their boundary-scheme vocabulary targets."""
    if chunk.ndim != 2 or boundary.ndim != 1:
        raise ValueError("chunk and boundary must have shapes (batch, length) and (batch,)")
    if chunk.size(0) != boundary.size(0) or chunk.size(1) == 0:
        raise ValueError("chunk must be non-empty and share boundary's batch size")

    data_inputs = chunk.flip(1)
    if chunk.size(1) == 1:
        data_targets = boundary[:, None]
    else:
        data_targets = torch.cat((chunk[:, :-1].flip(1), boundary[:, None]), dim=1)
    return data_inputs, data_targets


def boundary_inverse_batch(chunk, boundary, memory_embedding):
    """Append the GRU memory token to the default boundary reverse inputs."""
    data_inputs, data_targets = boundary_inverse_targets(chunk, boundary)
    memory_inputs = memory_embedding.expand(chunk.size(0), 1, -1)
    return data_inputs, memory_inputs, data_targets


def cross_entropy_sum(logits, targets, reference, batch_size):
    """Sum token losses within each sequence, then average the minibatch."""
    if not logits:
        return reference.new_zeros(())
    logits = torch.cat([x.reshape(-1, x.size(-1)) for x in logits], dim=0)
    targets = torch.cat([x.reshape(-1) for x in targets], dim=0)
    return F.cross_entropy(
        logits, targets, ignore_index=-100, reduction="sum"
    ) / batch_size


def gaussian_nll_sum(targets, estimates, scale, reference, batch_size):
    """Sum Gaussian coordinates and chunks, then average the minibatch."""
    if not targets:
        return reference.new_zeros(())
    squared_error = torch.stack([
        (target - estimate).float().pow(2).sum()
        for target, estimate in zip(targets, estimates)
    ]).sum()
    coordinates = sum(target.numel() for target in targets) / batch_size
    return (
        squared_error / (2.0 * scale.pow(2) * batch_size)
        + coordinates * torch.log(scale)
    )


def terminal_gaussian_nll(value, scale, batch_size):
    """Evaluate the report's terminal Gaussian term per sequence."""
    coordinates = value.numel() / batch_size
    return (
        value.float().pow(2).sum() / (2.0 * scale.pow(2) * batch_size)
        + coordinates * torch.log(scale)
    )


def bounded_exp(log_scale, minimum=1e-4, maximum=1e4):
    return log_scale.exp().clamp(min=minimum, max=maximum)
