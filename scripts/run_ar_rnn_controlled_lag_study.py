#!/usr/bin/env python3
"""Reproduce selected AR RNN endpoints and immediately evaluate controlled lag."""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from run_ar_rmt_scale_study import FULL_EPOCHS, HELDOUT_SEEDS, StudyController, atomic_json
from run_ar_rnn_study import RNNTrial, build_rnn_command


TANH_TAU = 47.799047874118955
IRNN_TAU = 243.242356581615
EXPECTED_TRAINING_RUNS = 16
EXPECTED_EVALUATION_RUNS = 16


def selected_trials() -> list[RNNTrial]:
    trials = []
    for seed in HELDOUT_SEEDS:
        trials.extend(
            [
                RNNTrial(
                    f"controlled-tanh-aux-s{seed}", seed, FULL_EPOCHS, 0.05,
                    16.0, "controlled_retrain", tau=TANH_TAU, track="tanh",
                ),
                RNNTrial(
                    f"controlled-tanh-noaux-probe-s{seed}", seed, FULL_EPOCHS,
                    0.05, 16.0, "controlled_retrain", tau=TANH_TAU,
                    probe_only=True, track="tanh",
                ),
                RNNTrial(
                    f"controlled-irnn-aux-s{seed}", seed, FULL_EPOCHS, 0.1,
                    16.0, "controlled_retrain", tau=IRNN_TAU,
                    activation="relu", recurrent_init="identity",
                    recurrent_identity_scale=1.0, track="irnn",
                ),
                RNNTrial(
                    f"controlled-irnn-noaux-probe-s{seed}", seed, FULL_EPOCHS,
                    0.1, 16.0, "controlled_retrain", tau=IRNN_TAU,
                    probe_only=True, activation="relu",
                    recurrent_init="identity", recurrent_identity_scale=1.0,
                    track="irnn",
                ),
            ]
        )
    return trials


def condition(trial: RNNTrial) -> str:
    return f"{trial.track}-{'noaux' if trial.probe_only else 'aux'}"


def evaluation_command(
    trial: RNNTrial,
    output_root: Path,
    gpu_id: int,
    args: argparse.Namespace,
) -> list[str]:
    checkpoint = output_root / "trials" / trial.trial_id / "checkpoints" / "last.ckpt"
    result = output_root / "controlled_lag" / f"{trial.trial_id}.json"
    command = [
        sys.executable,
        "scripts/evaluate_ar_rnn_controlled_lag.py",
        "--checkpoint", str(checkpoint),
        "--output", str(result),
        "--condition", condition(trial),
        "--run-name", trial.trial_id,
        "--seed", str(trial.seed),
        "--activation", trial.activation,
        "--recurrent-init", trial.recurrent_init,
        "--recurrent-identity-scale", str(trial.recurrent_identity_scale),
        "--rho", str(trial.rho),
        "--tau", str(trial.tau),
        "--chunk-size", str(trial.chunk_size),
        "--examples-per-lag", str(args.examples_per_lag),
        "--base-batch-size", str(args.base_batch_size),
        "--dataset-seed", str(args.dataset_seed),
        "--device", "cuda",
        "--wandb-project", args.wandb_project,
        "--wandb-entity", args.wandb_entity,
        "--wandb-group", args.eval_group,
    ]
    if trial.num_active_associations is not None:
        command.extend(
            ["--num-active-associations", str(trial.num_active_associations)]
        )
    if trial.probe_only:
        command.append("--probe-only")
    if trial.condition_memory_reconstruction_on_boundary:
        command.append("--condition-memory-reconstruction-on-boundary")
    return command


def run_evaluations(
    trials: list[RNNTrial], output_root: Path, gpu_ids: list[int], args: argparse.Namespace
) -> None:
    slots: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        slots.put(gpu_id)

    def run_one(trial: RNNTrial) -> None:
        gpu_id = slots.get()
        try:
            checkpoint = output_root / "trials" / trial.trial_id / "checkpoints" / "last.ckpt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            result = output_root / "controlled_lag" / f"{trial.trial_id}.json"
            if result.exists():
                return
            log = output_root / "controlled_lag" / f"{trial.trial_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    evaluation_command(trial, output_root, gpu_id, args),
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0 or not result.exists():
                raise RuntimeError(f"controlled-lag evaluation failed: {trial.trial_id}")
        finally:
            slots.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {executor.submit(run_one, trial): trial for trial in trials}
        for future in as_completed(futures):
            future.result()


def aggregate(trials: list[RNNTrial], output_root: Path) -> dict:
    reports = {}
    for trial in trials:
        path = output_root / "controlled_lag" / f"{trial.trial_id}.json"
        reports[trial.trial_id] = json.loads(path.read_text(encoding="utf-8"))

    conditions = {}
    for name in ("tanh-aux", "tanh-noaux", "irnn-aux", "irnn-noaux"):
        members = [trial for trial in trials if condition(trial) == name]
        by_lag = []
        for lag in range(2, 41, 2):
            values = [
                next(row for row in reports[t.trial_id]["by_lag"] if row["lag"] == lag)["accuracy"]
                for t in members
            ]
            nlls = [
                next(row for row in reports[t.trial_id]["by_lag"] if row["lag"] == lag)["nll"]
                for t in members
            ]
            by_lag.append(
                {
                    "lag": lag,
                    "accuracy_mean": statistics.fmean(values),
                    "accuracy_sample_sd": statistics.stdev(values),
                    "nll_mean": statistics.fmean(nlls),
                    "seed_accuracies": dict(zip([t.seed for t in members], values)),
                }
            )
        overall = [reports[t.trial_id]["overall"]["accuracy"] for t in members]
        conditions[name] = {
            "overall_accuracy_mean": statistics.fmean(overall),
            "overall_accuracy_sample_sd": statistics.stdev(overall),
            "by_lag": by_lag,
        }

    paired = {}
    for core in ("tanh", "irnn"):
        aux = {row["lag"]: row for row in conditions[f"{core}-aux"]["by_lag"]}
        noaux = {row["lag"]: row for row in conditions[f"{core}-noaux"]["by_lag"]}
        paired[core] = [
            {
                "lag": lag,
                "accuracy_delta": aux[lag]["accuracy_mean"] - noaux[lag]["accuracy_mean"],
                "wins": sum(
                    aux[lag]["seed_accuracies"][seed] > noaux[lag]["seed_accuracies"][seed]
                    for seed in HELDOUT_SEEDS
                ),
            }
            for lag in range(2, 41, 2)
        ]
    return {
        "protocol": next(iter(reports.values()))["protocol"],
        "trials": [asdict(trial) for trial in trials],
        "conditions": conditions,
        "paired_aux_minus_noaux": paired,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--training-group", required=True)
    parser.add_argument("--eval-group", required=True)
    parser.add_argument("--examples-per-lag", type=int, default=10000)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=20260819)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value]
    trials = selected_trials()
    if len(trials) != EXPECTED_TRAINING_RUNS:
        raise RuntimeError("unexpected controlled-lag training run count")
    atomic_json(
        args.output_root / "plan.json",
        {
            "training_runs": EXPECTED_TRAINING_RUNS,
            "evaluation_runs": EXPECTED_EVALUATION_RUNS,
            "training_group": args.training_group,
            "evaluation_group": args.eval_group,
            "trials": [asdict(trial) for trial in trials],
        },
    )
    controller = StudyController(
        args.output_root,
        gpu_ids,
        args.workers_per_gpu,
        args.wandb_project,
        args.wandb_entity,
        args.training_group,
        command_builder=build_rnn_command,
        dry_run=args.dry_run,
    )
    training_results = controller.run_trials(trials)
    if not all(training_results.get(trial.trial_id, False) for trial in trials):
        raise RuntimeError("one or more selected endpoint reproductions failed")
    if args.dry_run:
        return 0
    run_evaluations(trials, args.output_root, gpu_ids, args)
    atomic_json(args.output_root / "controlled_lag_report.json", aggregate(trials, args.output_root))
    (args.output_root / "CONTROLLED_LAG_COMPLETE").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
