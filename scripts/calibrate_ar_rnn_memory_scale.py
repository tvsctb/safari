#!/usr/bin/env python3
"""Calibrate a fixed post-warmup RNN trajectory-scale target and stiffness."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataloaders.synthetics import ICLDataModule
from src.models.sequence.rnn_aux import RNNAuxLM


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def model_from_checkpoint(path: Path, args: argparse.Namespace) -> RNNAuxLM:
    model = RNNAuxLM(
        d_model=64,
        n_layer=3,
        vocab_size=20,
        chunk_size=4,
        aux_chunk_sizes=(4,),
        chunk_offset="random",
        dropout=0.0,
        activation="relu",
        recurrent_init="identity",
        recurrent_identity_scale=1.0,
        rho=args.rho,
        tau=args.tau,
        gaussian_scale_mode="learned",
        gaussian_scale_learning_start_step=args.training_steps + 1,
        gaussian_scale_learning_rate=args.scale_learning_rate,
        auxiliary_probe_only=False,
        stop_gradient_memory_target=False,
        stop_gradient_memory_observation=False,
        memory_observation_gradient_scale=1.0,
        use_chunk_loss=True,
        use_discrete_loss=True,
        use_memory_loss=True,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        condition_memory_reconstruction_on_boundary=False,
    )
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key[len("model.") :]: value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state
    model.load_state_dict(model_state, strict=True)
    return model


def calibration_dataset(args: argparse.Namespace):
    data = ICLDataModule(
        num_examples=5000,
        num_test_examples=500,
        vocab_size=20,
        input_seq_len=40,
        copy_method="assoc_recall",
        seed=args.seed,
        loader_seed=args.seed,
        batch_size=args.batch_size,
        num_active_associations=5,
        return_aux_tokens=True,
    )
    data.setup()
    count = min(args.examples, len(data.dataset["train"]))
    return Subset(data.dataset["train"], range(count))


def trajectory(model: RNNAuxLM, input_ids: torch.Tensor):
    initial = model.default_state(input_ids.size(0), device=input_ids.device)
    outputs, _, states = model.rnn(
        model.embedding(input_ids), initial, return_trajectory=True
    )
    return outputs, states


def gradient_norm(loss: torch.Tensor, parameters: list[torch.Tensor]) -> float:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    squared = loss.new_zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().double().square().sum()
    return float(squared.sqrt())


def calibrate(model: RNNAuxLM, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = calibration_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    log_rms_values = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch[0].to(device)
            _, states = trajectory(model, input_ids)
            log_rms_values.append(model.trajectory_log_rms(states).cpu())
    log_rms = torch.cat(log_rms_values).double()
    target_log_rms = float(log_rms.mean())
    target_rms = math.exp(target_log_rms)

    # Fetch the gradient-calibration batch outside inference_mode. Tensors
    # created inside inference_mode cannot be saved for backward, even when
    # they themselves do not require gradients (embedding backward saves the
    # integer indices).
    diagnostic_batch = tuple(
        value.to(device) for value in next(iter(loader))
    )
    model.train()
    input_ids, labels, _ = diagnostic_batch
    outputs, states = trajectory(model, input_ids)
    logits = model._lm_logits(outputs)
    lm_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    # Measure restoring stiffness at a meaningful 10% scale displacement;
    # exactly at the geometric-mean target its expected first derivative is 0.
    displaced_target_log = math.log(target_rms / 1.1)
    unit_scale_loss = (
        model.trajectory_log_rms(states) - displaced_target_log
    ).square().mean()
    forward_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name == "initial_state"
        or name == "embedding.weight"
        or name.startswith("rnn.")
    ]
    lm_gradient_norm = gradient_norm(lm_loss, forward_parameters)
    unit_scale_gradient_norm = gradient_norm(
        unit_scale_loss, forward_parameters
    )
    denominator = args.aux_weight * unit_scale_gradient_norm
    suggested_weight = (
        args.target_gradient_ratio * lm_gradient_norm / denominator
        if denominator > 0
        else args.maximum_constraint_weight
    )
    suggested_weight = min(
        max(suggested_weight, args.minimum_constraint_weight),
        args.maximum_constraint_weight,
    )
    return {
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "examples": len(dataset),
        "definition": "per-example RMS over every M1..MK, layer, and coordinate",
        "target_rms": target_rms,
        "target_log_rms": target_log_rms,
        "sample_log_rms_mean": float(log_rms.mean()),
        "sample_log_rms_sample_sd": float(log_rms.std(unbiased=True)),
        "sample_rms_arithmetic_mean": float(log_rms.exp().mean()),
        "sample_rms_min": float(log_rms.exp().min()),
        "sample_rms_max": float(log_rms.exp().max()),
        "gradient_calibration": {
            "displacement": 1.1,
            "target_weighted_scale_to_lm_ratio": args.target_gradient_ratio,
            "aux_weight": args.aux_weight,
            "lm_gradient_norm": lm_gradient_norm,
            "unit_scale_gradient_norm_at_displacement": (
                unit_scale_gradient_norm
            ),
            "suggested_constraint_weight": suggested_weight,
            "minimum_constraint_weight": args.minimum_constraint_weight,
            "maximum_constraint_weight": args.maximum_constraint_weight,
        },
        "rho": float(model._configured_gaussian_scale("rho").detach()),
        "tau": float(model._configured_gaussian_scale("tau").detach()),
        "aux_training_step": int(model.aux_training_step),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rho", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=243.242356581615)
    parser.add_argument("--scale-learning-rate", type=float, default=1e-5)
    parser.add_argument("--training-steps", type=int, default=62800)
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--target-gradient-ratio", type=float, default=0.002)
    parser.add_argument("--minimum-constraint-weight", type=float, default=1e-4)
    parser.add_argument("--maximum-constraint-weight", type=float, default=1e4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    report = calibrate(model_from_checkpoint(args.checkpoint, args), args)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
