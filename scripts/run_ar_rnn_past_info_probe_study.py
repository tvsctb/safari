#!/usr/bin/env python3
"""Download fixed K=5 checkpoints and run inverse-matched past probes."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


SOURCE_EVAL_GROUP = "ar-irnn-difficulty-heldout-eval-20260820-v1"
FORWARD_SEEDS = (204, 205, 206, 207)
STATE_MODES = ("full", "global_unit")


@dataclass(frozen=True)
class ProbeTrial:
    condition: str
    forward_seed: int
    state_mode: str
    checkpoint: Path

    @property
    def trial_id(self) -> str:
        return f"past-{self.condition}-{self.state_mode}-s{self.forward_seed}"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def artifact_run_name(condition: str, seed: int) -> str:
    label = "memory4x" if condition == "aux" else "noaux"
    return f"difficulty-k5-{label}-s{seed}"


def download_checkpoints(args: argparse.Namespace) -> dict[tuple[str, int], Path]:
    import wandb

    api = wandb.Api(timeout=90)
    output = {}
    for condition in ("aux", "noaux"):
        for seed in FORWARD_SEEDS:
            run_name = artifact_run_name(condition, seed)
            artifact_name = f"{SOURCE_EVAL_GROUP}-{run_name}-checkpoint"
            root = args.output_root / "forward_checkpoints" / run_name
            checkpoint = root / "last.ckpt"
            if not checkpoint.exists():
                artifact = api.artifact(
                    f"{args.wandb_entity}/{args.wandb_project}/{artifact_name}:latest"
                )
                artifact.download(root=str(root))
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            output[(condition, seed)] = checkpoint
    return output


def ensure_fresh_group(args: argparse.Namespace) -> None:
    import wandb

    api = wandb.Api(timeout=60)
    runs = list(
        api.runs(
            f"{args.wandb_entity}/{args.wandb_project}",
            filters={"group": args.wandb_group},
        )
    )
    if runs:
        raise RuntimeError(f"W&B group is not empty: {args.wandb_group}")


def command(
    trial: ProbeTrial,
    args: argparse.Namespace,
    *,
    preflight: bool = False,
) -> list[str]:
    output = args.output_root / ("preflight" if preflight else "trials") / trial.trial_id
    values = [
        sys.executable,
        "scripts/train_ar_rnn_past_info_probe.py",
        "--checkpoint", str(trial.checkpoint),
        "--output", str(output),
        "--condition", trial.condition,
        "--state-mode", trial.state_mode,
        "--forward-seed", str(trial.forward_seed),
        # Pair AUX/no-AUX decoder initialization within each forward seed.
        "--probe-seed", str(90260824 + trial.forward_seed),
        "--dataset-seed", str(args.dataset_seed),
        "--forward-batch-size", str(args.forward_batch_size),
        "--batch-size", str(args.batch_size),
    ]
    if preflight:
        values.extend(
            [
                "--train-base-examples", "8",
                "--val-base-examples", "4",
                "--test-base-examples", "4",
                "--max-epochs", "2",
                "--min-epochs", "1",
                "--patience", "2",
                "--wandb-mode", "disabled",
            ]
        )
    else:
        values.extend(
            [
                "--train-base-examples", str(args.train_base_examples),
                "--val-base-examples", str(args.val_base_examples),
                "--test-base-examples", str(args.test_base_examples),
                "--max-epochs", str(args.max_epochs),
                "--min-epochs", str(args.min_epochs),
                "--patience", str(args.patience),
                "--wandb-mode", "online",
                "--wandb-project", args.wandb_project,
                "--wandb-entity", args.wandb_entity,
                "--wandb-group", args.wandb_group,
                "--run-name", trial.trial_id,
            ]
        )
    return values


def run_preflight(trial: ProbeTrial, gpu_id: int, args: argparse.Namespace) -> None:
    marker = args.output_root / "preflight" / "PREFLIGHT_OK"
    if marker.exists():
        return
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu_id), "OMP_NUM_THREADS": "1"})
    log = args.output_root / "preflight" / "preflight.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command(trial, args, preflight=True),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = args.output_root / "preflight" / trial.trial_id / "result.json"
    if completed.returncode or not result.exists():
        raise RuntimeError(f"past-probe GPU preflight failed; inspect {log}")
    marker.write_text("ok\n")


def run_trials(trials: list[ProbeTrial], gpu_ids: list[int], args: argparse.Namespace) -> None:
    slots: queue.Queue[int] = queue.Queue()
    for _ in range(args.workers_per_gpu):
        for gpu_id in gpu_ids:
            slots.put(gpu_id)

    def run_one(trial: ProbeTrial) -> None:
        result = args.output_root / "trials" / trial.trial_id / "result.json"
        if result.exists():
            return
        gpu_id = slots.get()
        try:
            root = result.parent
            root.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            with (root / "probe.log").open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command(trial, args),
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode or not result.exists():
                raise RuntimeError(f"probe failed: {trial.trial_id}")
        finally:
            slots.put(gpu_id)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as executor:
        futures = {executor.submit(run_one, trial): trial for trial in trials}
        for future in as_completed(futures):
            future.result()


def aggregate(trials: list[ProbeTrial], args: argparse.Namespace) -> dict:
    records = {}
    for trial in trials:
        path = args.output_root / "trials" / trial.trial_id / "result.json"
        records[trial.trial_id] = json.loads(path.read_text())
    conditions = {}
    for state_mode in STATE_MODES:
        conditions[state_mode] = {}
        for condition in ("aux", "noaux"):
            members = [
                records[trial.trial_id]
                for trial in trials
                if trial.condition == condition and trial.state_mode == state_mode
            ]
            summary = {}
            for section in ("overall", "age_20_38", "age_30_38"):
                accuracies = [item["test"][section]["macro_accuracy"] for item in members]
                nlls = [item["test"][section]["macro_nll"] for item in members]
                summary[section] = {
                    "accuracy_mean": statistics.fmean(accuracies),
                    "accuracy_sample_sd": statistics.stdev(accuracies),
                    "nll_mean": statistics.fmean(nlls),
                    "seed_accuracies": dict(zip(FORWARD_SEEDS, accuracies)),
                }
            summary["by_age"] = []
            for age in range(0, 40, 2):
                rows = [
                    next(row for row in item["test"]["by_age"] if row["age"] == age)
                    for item in members
                ]
                values = [row["accuracy"] for row in rows]
                summary["by_age"].append(
                    {
                        "age": age,
                        "accuracy_mean": statistics.fmean(values),
                        "accuracy_sample_sd": statistics.stdev(values),
                    }
                )
            conditions[state_mode][condition] = summary
        for section in ("overall", "age_20_38", "age_30_38"):
            aux = conditions[state_mode]["aux"][section]["seed_accuracies"]
            noaux = conditions[state_mode]["noaux"][section]["seed_accuracies"]
            differences = [aux[seed] - noaux[seed] for seed in FORWARD_SEEDS]
            conditions[state_mode][f"paired_{section}"] = {
                "delta_mean": statistics.fmean(differences),
                "delta_sample_sd": statistics.stdev(differences),
                "wins": sum(value > 0 for value in differences),
                "seed_deltas": dict(zip(FORWARD_SEEDS, differences)),
            }
    return {
        "protocol": {
            "source_group": SOURCE_EVAL_GROUP,
            "probe": "exact inverse RNN initialized from M_i; key-only input; past-value target",
            "forward_frozen": True,
            "target_leakage": False,
            "chance_accuracy": 1 / 9,
            "forward_seeds": FORWARD_SEEDS,
        },
        "trials": [asdict(trial) | {"checkpoint": str(trial.checkpoint)} for trial in trials],
        "conditions": conditions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--dataset-seed", type=int, default=20260824)
    parser.add_argument("--train-base-examples", type=int, default=2000)
    parser.add_argument("--val-base-examples", type=int, default=500)
    parser.add_argument("--test-base-examples", type=int, default=1000)
    parser.add_argument("--forward-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value]
    if not gpu_ids or not 1 <= args.workers_per_gpu <= 8:
        raise ValueError("invalid GPU worker configuration")
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_fresh_group(args)
    checkpoints = download_checkpoints(args)
    trials = [
        ProbeTrial(condition, seed, state_mode, checkpoints[(condition, seed)])
        for seed in FORWARD_SEEDS
        for condition in ("aux", "noaux")
        for state_mode in STATE_MODES
    ]
    if len(trials) != 16 or len({trial.trial_id for trial in trials}) != 16:
        raise RuntimeError("expected exactly 16 unique probe trials")
    atomic_json(
        args.output_root / "plan.json",
        {
            "group": args.wandb_group,
            "source_group": SOURCE_EVAL_GROUP,
            "trials": [
                asdict(trial) | {"checkpoint": str(trial.checkpoint)} for trial in trials
            ],
        },
    )
    run_preflight(trials[0], gpu_ids[0], args)
    run_trials(trials, gpu_ids, args)
    report = aggregate(trials, args)
    atomic_json(args.output_root / "aggregate.json", report)
    (args.output_root / "PAST_INFO_PROBE_COMPLETE").write_text("ok\n")
    print(json.dumps(report["conditions"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
