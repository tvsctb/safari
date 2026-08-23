#!/usr/bin/env python3
"""Train an inverse-matched frozen-state probe for past key/value information."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.models.sequence.rnn_aux import RNNAuxLM


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "src/dataloaders/controlled_assoc_recall.py"
SPEC = importlib.util.spec_from_file_location("past_probe_controlled_ar", GENERATOR_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
generate_controlled_lag_batch = MODULE.generate_controlled_lag_batch

BODY_TOKENS = 40
AGES = tuple(range(0, BODY_TOKENS, 2))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_forward(path: Path) -> RNNAuxLM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    config = dict(hyperparameters.get("model", {}))
    config.pop("_name_", None)
    accepted = set(inspect.signature(RNNAuxLM).parameters)
    config = {key: value for key, value in config.items() if key in accepted}
    model = RNNAuxLM(**config)
    raw_state = checkpoint.get("state_dict", checkpoint)
    state = {
        key.removeprefix("model."): value
        for key, value in raw_state.items()
        if key.startswith("model.")
    }
    if not state:
        state = raw_state
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    return model


class InverseMatchedPastProbe(nn.Module):
    """Use the trained model's inverse topology but no trained inverse weights."""

    def __init__(self, forward: RNNAuxLM, initialization_seed: int):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(initialization_seed)
            self.inverse_rnn = copy.deepcopy(forward.inverse_rnn)
            self.inverse_rnn.reset_parameters()
            self.inverse_rnn.requires_grad_(True)
        # Match the inverse's shared embedding/head without allowing probe
        # optimization to alter the frozen forward representation.
        self.register_buffer(
            "embedding_weight", forward.embedding.weight.detach().clone()
        )

    def forward(self, states: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("states must have shape (batch, layers, hidden)")
        embedded = F.embedding(keys, self.embedding_weight).unsqueeze(1)
        output, _, _ = self.inverse_rnn(
            embedded, states.transpose(0, 1).contiguous()
        )
        return F.linear(output[:, -1], self.embedding_weight)


def transform_states(states: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full":
        return states
    if mode != "global_unit":
        raise ValueError("state mode must be full or global_unit")
    norms = states.flatten(1).norm(dim=1).clamp_min(1e-12)
    return states / norms[:, None, None]


@torch.inference_mode()
def extract_examples(
    model: RNNAuxLM,
    *,
    num_base_examples: int,
    dataset_seed: int,
    base_batch_size: int,
    device: torch.device,
) -> TensorDataset:
    controlled = generate_controlled_lag_batch(
        num_base_examples,
        seed=dataset_seed,
        num_pairs=20,
        vocab_size=20,
        num_active_associations=5,
    )
    inputs = controlled.input_ids.flatten(0, 1)
    targets = controlled.targets.flatten()
    placements = torch.arange(20).repeat(num_base_examples)
    value_ends = 2 * (placements + 1)
    keys = inputs[:, -1]
    body = inputs[:, :BODY_TOKENS]

    state_parts, key_parts, target_parts, age_parts = [], [], [], []
    model.to(device).eval()
    for start in range(0, body.size(0), base_batch_size):
        end = min(start + base_batch_size, body.size(0))
        token_ids = body[start:end].to(device)
        initial = model.default_state(token_ids.size(0), device=device)
        _, _, trajectory = model.rnn(
            model.embedding(token_ids), initial, return_trajectory=True
        )
        # Hold the observation time fixed. Every queried association is decoded
        # from the state after the complete 40-token body, so ``age`` varies
        # only through the counterfactual insertion position and cannot be
        # confounded with how long the recurrent state has been allowed to form.
        states = torch.stack(
            [layer[:, BODY_TOKENS - 1] for layer in trajectory], dim=1
        )
        state_parts.append(states.float().cpu())
        key_parts.append(keys[start:end].clone())
        target_parts.append(targets[start:end].clone())
        age_parts.append((BODY_TOKENS - value_ends[start:end]).clone())
    dataset = TensorDataset(
        torch.cat(state_parts),
        torch.cat(key_parts),
        torch.cat(target_parts),
        torch.cat(age_parts),
    )
    observed_ages = set(dataset.tensors[3].tolist())
    if observed_ages != set(AGES):
        raise RuntimeError(f"unexpected ages: {sorted(observed_ages)}")
    return dataset


def balanced_loader(
    dataset: TensorDataset,
    batch_size: int,
    seed: int,
) -> DataLoader:
    ages = dataset.tensors[3]
    counts = Counter(ages.tolist())
    weights = torch.tensor([1.0 / counts[int(age)] for age in ages])
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True, generator=generator
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


@torch.inference_mode()
def evaluate(
    probe: InverseMatchedPastProbe,
    dataset: TensorDataset,
    *,
    state_mode: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    probe.eval()
    totals = {age: {"correct": 0, "nll": 0.0, "count": 0} for age in AGES}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for states, keys, targets, ages in loader:
        logits = probe(
            transform_states(states.to(device), state_mode), keys.to(device)
        )
        losses = F.cross_entropy(logits, targets.to(device), reduction="none").cpu()
        predictions = logits.argmax(dim=-1).cpu()
        for age in ages.unique().tolist():
            mask = ages == age
            totals[age]["correct"] += int(predictions[mask].eq(targets[mask]).sum())
            totals[age]["nll"] += float(losses[mask].sum())
            totals[age]["count"] += int(mask.sum())
    rows = []
    for age in AGES:
        item = totals[age]
        nll = item["nll"] / item["count"]
        rows.append(
            {
                "age": age,
                "count": item["count"],
                "accuracy": item["correct"] / item["count"],
                "nll": nll,
                "perplexity": math.exp(min(nll, 50.0)),
            }
        )

    def summarize(selected: tuple[int, ...]) -> dict:
        members = [row for row in rows if row["age"] in selected]
        return {
            "macro_accuracy": sum(row["accuracy"] for row in members) / len(members),
            "macro_nll": sum(row["nll"] for row in members) / len(members),
        }

    return {
        "by_age": rows,
        "overall": summarize(AGES),
        "age_20_38": summarize(tuple(age for age in AGES if age >= 20)),
        "age_30_38": summarize(tuple(age for age in AGES if age >= 30)),
    }


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    forward = load_forward(args.checkpoint)
    frozen_before = {
        name: value.detach().cpu().clone() for name, value in forward.named_parameters()
    }
    train_data = extract_examples(
        forward,
        num_base_examples=args.train_base_examples,
        dataset_seed=args.dataset_seed,
        base_batch_size=args.forward_batch_size,
        device=device,
    )
    val_data = extract_examples(
        forward,
        num_base_examples=args.val_base_examples,
        dataset_seed=args.dataset_seed + 1,
        base_batch_size=args.forward_batch_size,
        device=device,
    )
    test_data = extract_examples(
        forward,
        num_base_examples=args.test_base_examples,
        dataset_seed=args.dataset_seed + 2,
        base_batch_size=args.forward_batch_size,
        device=device,
    )
    forward.cpu()
    probe = InverseMatchedPastProbe(forward, args.probe_seed).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    loader = balanced_loader(train_data, args.batch_size, args.probe_seed)
    total_steps = args.max_epochs * len(loader)
    warmup_steps = max(1, int(args.warmup_fraction * total_steps))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / "best.pt"
    history = []
    best_nll, bad_epochs, step = float("inf"), 0, 0

    wandb_run = None
    if args.wandb_mode == "online":
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.run_name,
            id=f"{args.wandb_group}-{args.run_name}",
            resume="allow",
            config={
                "condition": args.condition,
                "forward_seed": args.forward_seed,
                "probe_seed": args.probe_seed,
                "state_mode": args.state_mode,
                "probe_architecture": "exact_inverse_rnn_M40_key_to_past_value",
                "forward_frozen": True,
                "num_active_associations": 5,
                "chance_accuracy": 1 / 9,
                "train_base_examples": args.train_base_examples,
                "val_base_examples": args.val_base_examples,
                "test_base_examples": args.test_base_examples,
            },
        )

    for epoch in range(args.max_epochs):
        probe.train()
        correct = count = 0
        loss_sum = 0.0
        for states, keys, targets, _ in loader:
            states = transform_states(states.to(device), args.state_mode)
            keys, targets = keys.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(states, keys)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            count += targets.numel()
            correct += int(logits.argmax(dim=-1).eq(targets).sum())
            loss_sum += float(loss.detach()) * targets.numel()
        val = evaluate(
            probe, val_data, state_mode=args.state_mode,
            batch_size=args.batch_size, device=device,
        )
        row = {
            "epoch": epoch,
            "train_accuracy": correct / count,
            "train_nll": loss_sum / count,
            "val_macro_accuracy": val["overall"]["macro_accuracy"],
            "val_macro_nll": val["overall"]["macro_nll"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log({f"probe/{key}": value for key, value in row.items()}, step=epoch)
        if row["val_macro_nll"] < best_nll - 1e-5:
            best_nll, bad_epochs = row["val_macro_nll"], 0
            torch.save({"probe": probe.state_dict(), "epoch": epoch}, best_path)
        else:
            bad_epochs += 1
        if epoch + 1 >= args.min_epochs and bad_epochs >= args.patience:
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    probe.load_state_dict(best["probe"], strict=True)
    test = evaluate(
        probe, test_data, state_mode=args.state_mode,
        batch_size=args.batch_size, device=device,
    )
    for name, value in forward.named_parameters():
        if not torch.equal(value.detach().cpu(), frozen_before[name]):
            raise RuntimeError(f"frozen forward parameter changed: {name}")
    report = {
        "condition": args.condition,
        "forward_seed": args.forward_seed,
        "probe_seed": args.probe_seed,
        "state_mode": args.state_mode,
        "checkpoint": str(args.checkpoint),
        "best_epoch": int(best["epoch"]),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
        "chance_accuracy": 1 / 9,
        "history": history,
        "test": test,
    }
    atomic_json(args.output / "result.json", report)
    if wandb_run is not None:
        import wandb
        table = wandb.Table(columns=["age", "count", "accuracy", "nll", "perplexity"])
        age_log_start = len(history)
        for row in test["by_age"]:
            table.add_data(*[row[column] for column in table.columns])
            wandb_run.log(
                {
                    "past_probe/age": row["age"],
                    "past_probe/accuracy": row["accuracy"],
                    "past_probe/nll": row["nll"],
                }, step=age_log_start + row["age"] // 2,
            )
        wandb_run.log({"past_probe/table": table})
        for section in ("overall", "age_20_38", "age_30_38"):
            for key, value in test[section].items():
                wandb_run.summary[f"past_probe/{section}/{key}"] = value
        artifact = wandb.Artifact(f"{args.wandb_group}-{args.run_name}-probe", type="model")
        artifact.add_file(str(best_path), name="best.pt")
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("aux", "noaux"), required=True)
    parser.add_argument("--state-mode", choices=("full", "global_unit"), required=True)
    parser.add_argument("--forward-seed", type=int, required=True)
    parser.add_argument("--probe-seed", type=int, required=True)
    parser.add_argument("--train-base-examples", type=int, default=10000)
    parser.add_argument("--val-base-examples", type=int, default=2000)
    parser.add_argument("--test-base-examples", type=int, default=5000)
    parser.add_argument("--dataset-seed", type=int, default=20260824)
    parser.add_argument("--forward-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.wandb_mode == "online" and (not args.wandb_group or not args.run_name):
        raise ValueError("online W&B requires group and run name")
    report = train(args)
    print(json.dumps(report["test"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
