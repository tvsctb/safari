#!/usr/bin/env python3
"""Evaluate one trained AR RNN on fixed-length, counterbalanced recall lags."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.sequence.rnn_aux import RNNAuxLM


_GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dataloaders"
    / "controlled_assoc_recall.py"
)
_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "controlled_assoc_recall", _GENERATOR_PATH
)
_GENERATOR_MODULE = importlib.util.module_from_spec(_GENERATOR_SPEC)
sys.modules[_GENERATOR_SPEC.name] = _GENERATOR_MODULE
_GENERATOR_SPEC.loader.exec_module(_GENERATOR_MODULE)
generate_controlled_lag_batch = _GENERATOR_MODULE.generate_controlled_lag_batch


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
        chunk_size=args.chunk_size,
        chunk_offset="random",
        dropout=0.0,
        activation=args.activation,
        recurrent_init=args.recurrent_init,
        recurrent_identity_scale=args.recurrent_identity_scale,
        rho=args.rho,
        tau=args.tau,
        auxiliary_probe_only=args.probe_only,
        stop_gradient_memory_target=False,
        stop_gradient_memory_observation=False,
        memory_observation_gradient_scale=1.0,
        use_chunk_loss=True,
        use_discrete_loss=True,
        use_memory_loss=True,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        condition_memory_reconstruction_on_boundary=(
            args.condition_memory_reconstruction_on_boundary
        ),
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


@torch.inference_mode()
def evaluate(model: RNNAuxLM, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model.to(device).eval()
    num_pairs = 20
    correct = torch.zeros(num_pairs, dtype=torch.long)
    nll_sum = torch.zeros(num_pairs, dtype=torch.float64)
    count = torch.zeros(num_pairs, dtype=torch.long)

    for start in range(0, args.examples_per_lag, args.base_batch_size):
        size = min(args.base_batch_size, args.examples_per_lag - start)
        batch = generate_controlled_lag_batch(
            size,
            seed=args.dataset_seed + start,
            num_pairs=num_pairs,
            vocab_size=20,
            num_active_associations=args.num_active_associations,
        )
        flat_inputs = batch.input_ids.flatten(0, 1).to(device)
        flat_targets = batch.targets.flatten().to(device)
        output, _ = model(flat_inputs, compute_aux=False)
        final_logits = output.logits[:, -1]
        losses = F.cross_entropy(final_logits, flat_targets, reduction="none")
        predictions = final_logits.argmax(dim=-1)
        losses = losses.reshape(size, num_pairs).double().cpu()
        matches = predictions.eq(flat_targets).reshape(size, num_pairs).cpu()
        nll_sum += losses.sum(dim=0)
        correct += matches.sum(dim=0)
        count += size

    rows = []
    for position, lag in enumerate(range(2 * num_pairs, 0, -2)):
        nll = float(nll_sum[position] / count[position])
        rows.append(
            {
                "pair_position": position,
                "lag": lag,
                "count": int(count[position]),
                "accuracy": float(correct[position] / count[position]),
                "nll": nll,
                "perplexity": math.exp(nll),
            }
        )
    rows.sort(key=lambda row: row["lag"])
    total_count = int(count.sum())
    total_correct = int(correct.sum())
    total_nll = float(nll_sum.sum() / total_count)
    return {
        "protocol": {
            "body_tokens": 40,
            "pairs": 20,
            "input_tokens": 42,
            "query_key_occurrences": 1,
            "same_distractor_multiset_across_lags": True,
            "examples_per_lag": args.examples_per_lag,
            "dataset_seed": args.dataset_seed,
            "num_active_associations": args.num_active_associations,
        },
        "overall": {
            "count": total_count,
            "accuracy": total_correct / total_count,
            "nll": total_nll,
            "perplexity": math.exp(total_nll),
        },
        "by_lag": rows,
    }


def log_wandb(report: dict, checkpoint: Path, args: argparse.Namespace) -> None:
    if args.wandb_mode == "disabled":
        return
    import wandb

    config = {
        "condition": args.condition,
        "seed": args.seed,
        "activation": args.activation,
        "recurrent_init": args.recurrent_init,
        "recurrent_identity_scale": args.recurrent_identity_scale,
        "probe_only": args.probe_only,
        "rho": args.rho,
        "tau": args.tau,
        "chunk_size": args.chunk_size,
        "condition_memory_reconstruction_on_boundary": (
            args.condition_memory_reconstruction_on_boundary
        ),
        **report["protocol"],
    }
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.run_name,
        id=f"{args.wandb_group}-{args.run_name}",
        resume="allow",
        config=config,
    ) as run:
        table = wandb.Table(
            columns=["lag", "pair_position", "count", "accuracy", "nll", "perplexity"]
        )
        for row in report["by_lag"]:
            table.add_data(*[row[column] for column in table.columns])
            run.log(
                {
                    "controlled_lag/lag": row["lag"],
                    "controlled_lag/accuracy": row["accuracy"],
                    "controlled_lag/nll": row["nll"],
                    "controlled_lag/perplexity": row["perplexity"],
                },
                step=row["lag"] // 2,
            )
        run.log({"controlled_lag/table": table})
        for key, value in report["overall"].items():
            run.summary[f"controlled_lag/overall_{key}"] = value
        artifact = wandb.Artifact(
            f"{args.wandb_group}-{args.run_name}-checkpoint", type="model"
        )
        artifact.add_file(str(checkpoint), name="last.ckpt")
        run.log_artifact(artifact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--activation", choices=("tanh", "relu"), required=True)
    parser.add_argument("--recurrent-init", choices=("orthogonal", "identity"), required=True)
    parser.add_argument("--recurrent-identity-scale", type=float, default=1.0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--rho", type=float, default=16.0)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument(
        "--condition-memory-reconstruction-on-boundary", action="store_true"
    )
    parser.add_argument("--examples-per-lag", type=int, default=10000)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=20260819)
    parser.add_argument("--num-active-associations", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--wandb-group", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    report = evaluate(model_from_checkpoint(args.checkpoint, args), args)
    report.update(
        {
            "condition": args.condition,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
        }
    )
    atomic_json(args.output, report)
    log_wandb(report, args.checkpoint, args)
    print(json.dumps(report["overall"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
