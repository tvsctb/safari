import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


class AuxCausalLMOutput(NamedTuple):
    logits: torch.Tensor
    aux_loss: torch.Tensor


TOKEN_SCHEMES = {
    "boundary_reverse",
    "role_reverse",
    "role_forward",
}


def normalize_token_scheme(token_scheme):
    aliases = {
        "boundary": "boundary_reverse",
        "boundary-reverse": "boundary_reverse",
        "role-reverse": "role_reverse",
        "role-forward": "role_forward",
    }
    token_scheme = aliases.get(token_scheme, token_scheme)
    if token_scheme not in TOKEN_SCHEMES:
        choices = ", ".join(sorted(TOKEN_SCHEMES))
        raise ValueError(f"token_scheme must be one of: {choices}")
    return token_scheme


def normalize_offset_mode(random_chunk_offset):
    if isinstance(random_chunk_offset, bool):
        return "sequence" if random_chunk_offset else "fixed"
    aliases = {
        "random": "sequence",
        "random_sequence": "sequence",
        "per_sequence": "sequence",
    }
    mode = aliases.get(random_chunk_offset, random_chunk_offset)
    if mode not in {"fixed", "sequence"}:
        raise ValueError(
            "random_chunk_offset must be a bool or one of fixed, sequence"
        )
    return mode


def resolve_direction_embedding(
    option,
    token_scheme,
):
    """Apply the explicit switch on the boundary scheme where it is defined."""
    if not isinstance(option, bool):
        raise ValueError("use_direction_embedding must be true or false")
    return option and token_scheme == "boundary_reverse"


def validate_scale_configuration(mode, granularity, allowed_granularities):
    if mode not in {"fixed", "learned"}:
        raise ValueError("scale_mode must be 'fixed' or 'learned'")
    if granularity not in allowed_granularities:
        choices = ", ".join(sorted(allowed_granularities))
        raise ValueError(f"scale_granularity must be one of: {choices}")


def scale_tensor(value, size, granularity, name):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    expected = 1 if granularity == "global" else size
    if tensor.numel() not in {1, expected}:
        raise ValueError(f"{name} must contain 1 or {expected} positive values")
    if tensor.numel() == 1 and expected > 1:
        tensor = tensor.expand(expected).clone()
    tensor = tensor.reshape(()) if expected == 1 else tensor.reshape(expected)
    if not torch.isfinite(tensor).all() or not (tensor > 0).all():
        raise ValueError(f"{name} must contain finite positive values")
    return tensor


def initialize_scale(module, name, value, size, mode, granularity):
    value = scale_tensor(value, size, granularity, name)
    if mode == "learned":
        module.register_parameter(f"log_{name}", torch.nn.Parameter(value.log()))
    else:
        module.register_buffer(name, value)


def configured_scale(module, name, mode, minimum=1e-4, maximum=1e4):
    if mode == "learned":
        return bounded_exp(
            getattr(module, f"log_{name}"), minimum=minimum, maximum=maximum
        )
    return getattr(module, name)


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


def boundary_inverse_targets(
    chunk,
    boundary,
    condition_memory_reconstruction_on_boundary=False,
):
    """Build inverse inputs and boundary-scheme vocabulary targets.

    When requested, the observed boundary is appended after the reversed chunk.
    Its target is ignored, so token CE is unchanged while the final inverse
    state used for memory reconstruction can condition on the boundary.
    """
    if chunk.ndim != 2 or boundary.ndim != 1:
        raise ValueError("chunk and boundary must have shapes (batch, length) and (batch,)")
    if chunk.size(0) != boundary.size(0) or chunk.size(1) == 0:
        raise ValueError("chunk must be non-empty and share boundary's batch size")

    data_inputs = chunk.flip(1)
    if chunk.size(1) == 1:
        data_targets = boundary[:, None]
    else:
        data_targets = torch.cat((chunk[:, :-1].flip(1), boundary[:, None]), dim=1)
    if condition_memory_reconstruction_on_boundary:
        data_inputs = torch.cat((data_inputs, boundary[:, None]), dim=1)
        data_targets = F.pad(data_targets, (0, 1), value=-100)
    return data_inputs, data_targets


def role_inverse_targets(chunk, token_scheme):
    """Return role-scheme data order and the B vocabulary targets."""
    token_scheme = normalize_token_scheme(token_scheme)
    if token_scheme == "boundary_reverse":
        raise ValueError("role_inverse_targets requires a role token scheme")
    if chunk.ndim != 2 or chunk.size(1) == 0:
        raise ValueError("chunk must have shape (batch, nonzero length)")
    ordered = chunk.flip(1) if token_scheme == "role_reverse" else chunk
    return ordered, ordered


def cross_entropy_sum(logits, targets, reference, batch_size):
    """Sum token losses within each sequence, then average the minibatch."""
    if not logits:
        return reference.new_zeros(())
    logits = torch.cat([x.reshape(-1, x.size(-1)) for x in logits], dim=0)
    targets = torch.cat([x.reshape(-1) for x in targets], dim=0)
    return F.cross_entropy(
        logits, targets, ignore_index=-100, reduction="sum"
    ) / batch_size


def _broadcast_state_scale(scale, value, scale_axis):
    if scale.ndim == 0:
        return scale
    if scale_axis < 0:
        scale_axis += value.ndim
    if not 0 <= scale_axis < value.ndim:
        raise ValueError("scale_axis is outside the state tensor")
    if value.size(scale_axis) != scale.numel():
        raise ValueError("scale length does not match its state axis")
    shape = [1] * value.ndim
    shape[scale_axis] = scale.numel()
    return scale.reshape(shape)


def gaussian_nll_sum(
    targets, estimates, scale, reference, batch_size, scale_axis=0
):
    """Sum Gaussian coordinates and chunks, then average the minibatch."""
    if not targets:
        return reference.new_zeros(())
    if len(targets) != len(estimates):
        raise ValueError("targets and estimates must have the same length")
    target_values = torch.stack(tuple(targets))
    estimate_values = torch.stack(tuple(estimates))
    original_axis = scale_axis % target_values[0].ndim
    state_scale = _broadcast_state_scale(
        scale, target_values, original_axis + 1
    )
    losses = (
        (target_values - estimate_values).float().square()
        / (2.0 * state_scale.square())
        + torch.log(state_scale)
    )
    return losses.sum() / batch_size


def memory_reconstruction_target(value, stop_gradient=False):
    """Optionally remove the direct reconstruction gradient into its target."""
    return value.detach() if stop_gradient else value


def validate_memory_observation_gradient_scale(gradient_scale):
    """Return a finite successor-memory gradient multiplier in [0, 1]."""
    gradient_scale = float(gradient_scale)
    if not math.isfinite(gradient_scale) or not 0.0 <= gradient_scale <= 1.0:
        raise ValueError("memory observation gradient scale must be in [0, 1]")
    return gradient_scale


def memory_observation(value, stop_gradient=False, gradient_scale=1.0):
    """Control inverse-loss gradients entering the successor memory value."""
    gradient_scale = validate_memory_observation_gradient_scale(gradient_scale)
    if stop_gradient:
        gradient_scale = 0.0
    detached = value.detach()
    if gradient_scale == 0.0:
        return detached
    return detached + gradient_scale * (value - detached)


def mean_batch_variance(values, batch_axis):
    """Average coordinate-wise population variance across the minibatch."""
    if not values:
        raise ValueError("values must contain at least one state tensor")
    stacked = torch.stack(tuple(values)).detach().float()
    original_axis = batch_axis % values[0].ndim
    return stacked.var(dim=original_axis + 1, correction=0).mean()


def mean_reconstruction_mse(targets, estimates, reference):
    """Average raw coordinate MSE without Gaussian NLL scale constants."""
    if not targets:
        return reference.new_zeros(())
    if len(targets) != len(estimates):
        raise ValueError("targets and estimates must have the same length")
    target_values = torch.stack(tuple(targets)).detach().float()
    estimate_values = torch.stack(tuple(estimates)).detach().float()
    return (target_values - estimate_values).square().mean()


def memory_reconstruction_diagnostics(targets, estimates, reference):
    """Describe reconstruction quality relative to the target memory magnitude.

    These detached diagnostics do not alter the optimized loss.  Second moments
    measure absolute state size, while batch variance measures only dispersion
    across examples and is therefore reported separately.
    """
    if not targets:
        zero = reference.detach().new_zeros(())
        return {
            "target_second_moment": zero,
            "estimate_second_moment": zero,
            "residual_mse": zero,
            "relative_mse": zero,
            "relative_rmse": zero,
            "norm_ratio": zero,
            "target_estimate_cosine": zero,
            "target_batch_variance": zero,
            "estimate_batch_variance": zero,
            "batch_r2": zero,
        }
    if len(targets) != len(estimates):
        raise ValueError("targets and estimates must have the same length")

    target_values = torch.stack(tuple(targets)).detach().float()
    estimate_values = torch.stack(tuple(estimates)).detach().float()
    target_second_moment = target_values.square().mean()
    estimate_second_moment = estimate_values.square().mean()
    residual_mse = (target_values - estimate_values).square().mean()
    target_batch_variance = target_values.var(dim=1, correction=0).mean()
    estimate_batch_variance = estimate_values.var(dim=1, correction=0).mean()
    epsilon = torch.finfo(target_second_moment.dtype).eps
    relative_mse = residual_mse / target_second_moment.clamp_min(epsilon)

    flat_target = target_values.reshape(-1)
    flat_estimate = estimate_values.reshape(-1)
    cosine_denominator = flat_target.norm() * flat_estimate.norm()
    target_estimate_cosine = torch.where(
        cosine_denominator > 0,
        torch.dot(flat_target, flat_estimate)
        / cosine_denominator.clamp_min(epsilon),
        cosine_denominator.new_zeros(()),
    )
    return {
        "target_second_moment": target_second_moment,
        "estimate_second_moment": estimate_second_moment,
        "residual_mse": residual_mse,
        "relative_mse": relative_mse,
        "relative_rmse": relative_mse.sqrt(),
        "norm_ratio": (
            estimate_second_moment.clamp_min(0).sqrt()
            / target_second_moment.clamp_min(epsilon).sqrt()
        ),
        "target_estimate_cosine": target_estimate_cosine,
        "target_batch_variance": target_batch_variance,
        "estimate_batch_variance": estimate_batch_variance,
        "batch_r2": 1.0
        - residual_mse / target_batch_variance.clamp_min(epsilon),
    }


def terminal_reconstruction_mse(value, target=None):
    """Return raw terminal coordinate MSE, independent of terminal scale."""
    residual = value.detach() if target is None else value.detach() - target.detach()
    return residual.float().pow(2).mean()


def validate_observation_noise_std(value):
    """Return a finite, non-negative Gaussian observation-noise scale."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("observation_noise_std must be finite and non-negative")
    return value


def validate_generation_noise_std(value):
    """Return a finite, non-negative memory-generation noise scale."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("generation_noise_std must be finite and non-negative")
    return value


def noisy_observation(value, noise_std, training):
    """Apply training-only Gaussian noise to an inverse-path observation."""
    if not training or noise_std == 0.0:
        return value
    return value + noise_std * torch.randn_like(value)


def noisy_generation(value, noise_std, training):
    """Apply training-only Gaussian noise to a generated successor memory."""
    if not training or noise_std == 0.0:
        return value
    return value + noise_std * torch.randn_like(value)


def terminal_gaussian_nll(
    value, scale, batch_size, scale_axis=0, target=None
):
    """Evaluate the report's terminal Gaussian term per sequence."""
    state_scale = _broadcast_state_scale(scale, value, scale_axis)
    residual = value if target is None else value - target
    return (
        residual.float().pow(2) / (2.0 * state_scale.pow(2))
        + torch.log(state_scale)
    ).sum() / batch_size


def bounded_exp(log_scale, minimum=1e-4, maximum=1e4):
    return log_scale.exp().clamp(min=minimum, max=maximum)
