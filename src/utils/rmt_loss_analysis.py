import math
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.models.sequence.auxiliary import chunk_ranges


FORWARD_PARAMETER_PREFIXES = (
    "embedding.",
    "initial_memory",
    "forward_queries",
    "position_embedding",
    "blocks.",
    "final_norm.",
)


def forward_parameter_items(model):
    """Parameters that determine the inference-time recurrent transition."""
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(FORWARD_PARAMETER_PREFIXES)
    ]


def tensor_dict_norm(tensors):
    squares = [tensor.detach().float().square().sum() for tensor in tensors]
    if not squares:
        return 0.0
    return math.sqrt(torch.stack(squares).sum().item())


def tensor_dict_cosine(left, right):
    dot = sum(
        (x.detach().float() * y.detach().float()).sum()
        for x, y in zip(left, right)
    )
    left_norm = tensor_dict_norm(left)
    right_norm = tensor_dict_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot.item() / (left_norm * right_norm)


def normalized_step_scale(parameters, gradients, relative_step):
    parameter_norm = tensor_dict_norm(parameters)
    gradient_norm = tensor_dict_norm(gradients)
    if gradient_norm == 0.0:
        return 0.0
    return float(relative_step) * parameter_norm / gradient_norm


def apply_gradient_step(parameters, gradients, relative_step, sign=-1.0):
    """Apply a parameter-norm-relative step and return exact restoration copies."""
    originals = [parameter.detach().clone() for parameter in parameters]
    scale = normalized_step_scale(parameters, gradients, relative_step)
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            parameter.add_(gradient, alpha=float(sign) * scale)
    return originals, scale


def restore_parameters(parameters, originals):
    with torch.no_grad():
        for parameter, original in zip(parameters, originals):
            parameter.copy_(original)


def model_batch_metrics(model, input_ids, labels, aux_tokens):
    output, _ = model(
        input_ids,
        targets=labels,
        aux_tokens=aux_tokens,
        compute_aux=True,
    )
    lm_loss = F.cross_entropy(
        output.logits.reshape(-1, model.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    valid = labels.ne(-100)
    accuracy = (
        output.logits.argmax(dim=-1)[valid].eq(labels[valid]).float().mean()
        if valid.any()
        else lm_loss.new_zeros(())
    )
    components = model.loss_components
    differentiable = OrderedDict(
        lm=lm_loss,
        chunk=components["chunk_ce"],
        discrete=components["discrete_ce"],
        token=components["chunk_ce"] + components["discrete_ce"],
        memory=components["memory_nll"],
        terminal=components["terminal_nll"],
        full_aux=components["total"],
    )
    metrics = {
        "lm": lm_loss.detach().item(),
        "accuracy": accuracy.detach().item(),
        "chunk_ce": components["chunk_ce"].detach().item(),
        "discrete_ce": components["discrete_ce"].detach().item(),
        "token_ce": (
            components["chunk_ce"] + components["discrete_ce"]
        ).detach().item(),
        "memory_nll": components["memory_nll"].detach().item(),
        "terminal_nll": components["terminal_nll"].detach().item(),
        "full_aux": components["total"].detach().item(),
    }
    for name, value in model.metrics.items():
        if name.startswith("aux/memory_") or name in {
            "aux/terminal_reconstruction_mse",
            "aux/terminal_batch_variance",
            "aux/rho",
            "aux/tau",
        }:
            if torch.is_tensor(value) and value.numel() == 1:
                metrics[name.replace("aux/", "")] = value.detach().item()
    return differentiable, metrics


def component_gradients(losses, parameters):
    gradients = OrderedDict()
    names = list(losses)
    for index, name in enumerate(names):
        values = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=index + 1 < len(names),
            allow_unused=True,
        )
        gradients[name] = tuple(
            torch.zeros_like(parameter) if value is None else value.detach()
            for parameter, value in zip(parameters, values)
        )
    return gradients


def gradient_summary(parameter_items, gradients, aux_weight=0.1):
    names = [name for name, _ in parameter_items]
    groups = {
        "embedding": lambda name: name.startswith("embedding."),
        "memory_io": lambda name: name in {
            "initial_memory", "forward_queries", "position_embedding"
        },
        "blocks": lambda name: name.startswith("blocks."),
        "final_norm": lambda name: name.startswith("final_norm."),
    }
    summary = {"norms": {}, "weighted_norms": {}, "group_norms": {}, "cosines": {}}
    for loss_name, values in gradients.items():
        summary["norms"][loss_name] = tensor_dict_norm(values)
        weight = 1.0 if loss_name == "lm" else float(aux_weight)
        summary["weighted_norms"][loss_name] = weight * summary["norms"][loss_name]
        summary["group_norms"][loss_name] = {
            group_name: tensor_dict_norm([
                value for name, value in zip(names, values) if predicate(name)
            ])
            for group_name, predicate in groups.items()
        }
    loss_names = list(gradients)
    for left_index, left in enumerate(loss_names):
        for right in loss_names[left_index + 1:]:
            summary["cosines"][f"{left}:{right}"] = tensor_dict_cosine(
                gradients[left], gradients[right]
            )
    return summary


def recurrent_transition_records(model, input_ids, aux_tokens):
    memory = model.default_state(input_ids.size(0), device=input_ids.device)
    records = []
    with torch.no_grad():
        for start, end in chunk_ranges(input_ids.size(1), model.chunk_size):
            input_chunk = input_ids[:, start:end]
            token_chunk = aux_tokens[:, start:end]
            token_embeddings = model.embedding(input_chunk)
            records.append((memory.detach(), token_embeddings.detach()))
            _, memory = model._forward_chunk(
                memory, input_chunk, token_chunk, terminal=False
            )
    return records


def _rademacher_like(reference, generator):
    values = torch.empty_like(reference)
    values.bernoulli_(0.5, generator=generator)
    return values.mul_(2.0).sub_(1.0)


def transition_gain_proxies(
    model,
    input_ids,
    aux_tokens,
    probes=2,
    seed=1234,
    transition_indices=None,
):
    """Hutchinson JVP estimates of local carry and write mean-squared gain."""
    records = recurrent_transition_records(model, input_ids, aux_tokens)
    if transition_indices is None:
        transition_indices = sorted({0, len(records) // 2, len(records) - 1})
    generator = torch.Generator(device=input_ids.device).manual_seed(int(seed))
    carry_values = []
    write_values = []
    for index in transition_indices:
        memory, token_embeddings = records[index]

        def from_memory(value):
            return model._transform(
                value,
                token_embeddings,
                model.forward_queries,
                model.blocks,
                model.final_norm,
            )[1]

        def from_tokens(value):
            return model._transform(
                memory,
                value,
                model.forward_queries,
                model.blocks,
                model.final_norm,
            )[1]

        for _ in range(int(probes)):
            memory_probe = _rademacher_like(memory, generator)
            _, carry_jvp = torch.autograd.functional.jvp(
                from_memory, memory, memory_probe, create_graph=False
            )
            carry_values.append(
                carry_jvp.detach().float().square().sum()
                / memory_probe.detach().float().square().sum()
            )
            token_probe = _rademacher_like(token_embeddings, generator)
            _, write_jvp = torch.autograd.functional.jvp(
                from_tokens, token_embeddings, token_probe, create_graph=False
            )
            write_values.append(
                write_jvp.detach().float().square().sum()
                / token_probe.detach().float().square().sum()
            )
    carry = torch.stack(carry_values).mean().item()
    write = torch.stack(write_values).mean().item()
    return {
        "carry_gain": carry,
        "write_gain": write,
        "log_carry_gain": math.log(max(carry, 1e-30)),
        "log_write_gain": math.log(max(write, 1e-30)),
    }


def central_directional_effect(
    model,
    batch,
    parameters,
    gradient,
    relative_step,
    proxy_seed,
    probes=2,
):
    input_ids, labels, aux_tokens = batch
    base_losses, base_metrics = model_batch_metrics(
        model, input_ids, labels, aux_tokens
    )
    del base_losses
    base_proxies = transition_gain_proxies(
        model, input_ids, aux_tokens, probes=probes, seed=proxy_seed
    )
    results = {}
    scale = None
    for label, sign in (("minus", -1.0), ("plus", 1.0)):
        originals, current_scale = apply_gradient_step(
            parameters, gradient, relative_step, sign=sign
        )
        scale = current_scale
        try:
            with torch.no_grad():
                _, metrics = model_batch_metrics(
                    model, input_ids, labels, aux_tokens
                )
            proxies = transition_gain_proxies(
                model, input_ids, aux_tokens, probes=probes, seed=proxy_seed
            )
            results[label] = {**metrics, **proxies}
        finally:
            restore_parameters(parameters, originals)
    base = {**base_metrics, **base_proxies}
    delta = {
        name: results["minus"][name] - base[name]
        for name in base
        if name in results["minus"]
    }
    central = {
        name: (results["minus"][name] - results["plus"][name])
        / (2.0 * float(relative_step))
        for name in base
        if name in results["minus"] and name in results["plus"]
    }
    return {
        "relative_step": float(relative_step),
        "absolute_scale": float(scale),
        "base": base,
        "minus": results["minus"],
        "plus": results["plus"],
        "minus_delta": delta,
        "central_per_relative_step": central,
    }
