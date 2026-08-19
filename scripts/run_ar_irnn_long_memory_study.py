#!/usr/bin/env python3
"""Test four isolated iRNN interventions against the saved AUX control."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

from run_ar_rmt_scale_study import FULL_EPOCHS, HELDOUT_SEEDS, StudyController, atomic_json
from run_ar_rnn_controlled_lag_study import run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command


CONTROL_RHO = 16.0
MEMORY_2X_RHO = CONTROL_RHO / math.sqrt(2.0)
MEMORY_4X_RHO = CONTROL_RHO / 2.0
IRNN_TAU = 243.242356581615
EXPECTED_TRAINING_RUNS = 16
EXPECTED_EVALUATION_RUNS = 16


def make_trial(label: str, seed: int, **changes) -> RNNTrial:
    options = {
        "trial_id": f"longmem-{label}-s{seed}",
        "seed": seed,
        "max_epochs": FULL_EPOCHS,
        "aux_weight": 0.1,
        "rho": CONTROL_RHO,
        "tau": IRNN_TAU,
        "phase": "irnn_long_memory",
        "activation": "relu",
        "recurrent_init": "identity",
        "recurrent_identity_scale": 1.0,
        "exclude_initial_memory_reconstruction": True,
        "use_terminal_loss": True,
        "chunk_size": 4,
        "condition_memory_reconstruction_on_boundary": False,
        "track": f"irnn-{label}",
    }
    options.update(changes)
    return RNNTrial(**options)


def study_trials() -> list[RNNTrial]:
    trials = []
    for seed in HELDOUT_SEEDS:
        trials.extend(
            [
                make_trial("memory2x", seed, rho=MEMORY_2X_RHO),
                make_trial("memory4x", seed, rho=MEMORY_4X_RHO),
                make_trial(
                    "boundary",
                    seed,
                    condition_memory_reconstruction_on_boundary=True,
                ),
                make_trial("chunk8", seed, chunk_size=8),
            ]
        )
    return trials


def run_preflight(output_root: Path, gpu_id: int) -> None:
    root = output_root / "preflight"
    marker = root / "PREFLIGHT_OK"
    if marker.exists():
        return
    trial = make_trial(
        "preflight-combined",
        20260819,
        max_epochs=1,
        rho=MEMORY_4X_RHO,
        chunk_size=8,
        condition_memory_reconstruction_on_boundary=True,
    )
    command = build_rnn_command(trial, root, "unused", "unused", "unused")
    replacements = {
        "+trainer.check_val_every_n_epoch=5": "+trainer.check_val_every_n_epoch=1",
        "trainer.limit_train_batches=1.0": "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=1.0": "trainer.limit_val_batches=2",
        "task.aux_gradient_norm_interval=1570": "task.aux_gradient_norm_interval=0",
        "task.aux_diagnostic_interval=157": "task.aux_diagnostic_interval=1",
        "wandb.mode=online": "wandb.mode=disabled",
        "+wandb.entity=unused": "+wandb.entity=null",
        "+wandb.resume=allow": "+wandb.resume=null",
    }
    command = [replacements.get(argument, argument) for argument in command]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    log = root / "preflight.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    trial_root = root / "trials" / trial.trial_id
    if (
        completed.returncode != 0
        or not (trial_root / "result.json").exists()
        or not (trial_root / "checkpoints" / "last.ckpt").exists()
    ):
        raise RuntimeError(f"iRNN long-memory GPU preflight failed; inspect {log}")
    marker.write_text("ok\n", encoding="utf-8")


def aggregate(trials: list[RNNTrial], output_root: Path) -> dict:
    reports = {
        trial.trial_id: json.loads(
            (output_root / "controlled_lag" / f"{trial.trial_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for trial in trials
    }
    conditions = {}
    for label in ("memory2x", "memory4x", "boundary", "chunk8"):
        members = [trial for trial in trials if trial.track == f"irnn-{label}"]
        by_lag = []
        for lag in range(2, 41, 2):
            accuracy = [
                next(
                    row
                    for row in reports[trial.trial_id]["by_lag"]
                    if row["lag"] == lag
                )["accuracy"]
                for trial in members
            ]
            nll = [
                next(
                    row
                    for row in reports[trial.trial_id]["by_lag"]
                    if row["lag"] == lag
                )["nll"]
                for trial in members
            ]
            by_lag.append(
                {
                    "lag": lag,
                    "accuracy_mean": statistics.fmean(accuracy),
                    "accuracy_sample_sd": statistics.stdev(accuracy),
                    "nll_mean": statistics.fmean(nll),
                    "seed_accuracies": dict(zip([trial.seed for trial in members], accuracy)),
                }
            )
        overall = [reports[trial.trial_id]["overall"] for trial in members]
        conditions[label] = {
            "overall_accuracy_mean": statistics.fmean(row["accuracy"] for row in overall),
            "overall_accuracy_sample_sd": statistics.stdev(row["accuracy"] for row in overall),
            "overall_nll_mean": statistics.fmean(row["nll"] for row in overall),
            "by_lag": by_lag,
        }
    return {
        "control": {
            "wandb_group": "ar-rnn-controlled-lag-eval-20260819-v1",
            "condition": "irnn-aux",
            "rho": CONTROL_RHO,
            "chunk_size": 4,
            "condition_memory_reconstruction_on_boundary": False,
        },
        "protocol": next(iter(reports.values()))["protocol"],
        "trials": [asdict(trial) for trial in trials],
        "conditions": conditions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=6)
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
    trials = study_trials()
    if len(trials) != EXPECTED_TRAINING_RUNS:
        raise RuntimeError("unexpected iRNN long-memory trial count")
    atomic_json(
        args.output_root / "plan.json",
        {
            "training_runs": EXPECTED_TRAINING_RUNS,
            "evaluation_runs": EXPECTED_EVALUATION_RUNS,
            "training_group": args.training_group,
            "evaluation_group": args.eval_group,
            "control_group": "ar-rnn-controlled-lag-eval-20260819-v1",
            "trials": [asdict(trial) for trial in trials],
        },
    )
    if not args.dry_run:
        run_preflight(args.output_root, gpu_ids[0])
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
    results = controller.run_trials(trials)
    if not all(results.get(trial.trial_id, False) for trial in trials):
        raise RuntimeError("one or more iRNN long-memory trials failed")
    if args.dry_run:
        return 0
    run_evaluations(trials, args.output_root, gpu_ids, args)
    atomic_json(
        args.output_root / "controlled_lag_report.json",
        aggregate(trials, args.output_root),
    )
    (args.output_root / "IRNN_LONG_MEMORY_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
