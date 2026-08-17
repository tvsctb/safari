#!/usr/bin/env python3
"""Run the full matched AR RMT terminal-off, memory-target-SG follow-up."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from run_ar_rmt_scale_study import (
    BASELINE_SEEDS,
    FULL_EPOCHS,
    Scale,
    StudyController,
    Trial,
    atomic_json,
    condition_metrics,
)


FOLLOWUP_SCALE = Scale(0.25, 2.0)


def make_trials():
    return [
        Trial(
            trial_id=f"terminal-off-targetsg-on-rp025-s{seed}",
            seed=seed,
            max_epochs=FULL_EPOCHS,
            aux_weight=0.1,
            target_sg=True,
            scale=FOLLOWUP_SCALE,
            phase="terminal_off_sgon",
            use_terminal_loss=False,
        )
        for seed in BASELINE_SEEDS
    ]


def build_report(output_root: Path, trials):
    metrics = condition_metrics(output_root, trials)
    accuracies = [
        float(row["accuracy"])
        for row in metrics["rows"]
        if row["accuracy"] is not None
    ]
    losses = [
        float(row["loss"])
        for row in metrics["rows"]
        if row["loss"] is not None
    ]
    return {
        "condition": {
            "aux_weight": 0.1,
            "target_sg": True,
            "use_terminal_loss": False,
            "memory_pressure": FOLLOWUP_SCALE.rho_pressure,
            "rho": FOLLOWUP_SCALE.rho,
            "tau": None,
            "epochs": FULL_EPOCHS,
            "seeds": list(BASELINE_SEEDS),
        },
        "trials": [asdict(trial) for trial in trials],
        "metrics": metrics,
        "mean_accuracy": statistics.fmean(accuracies) if accuracies else None,
        "sample_sd_accuracy": (
            statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        ),
        "mean_loss": statistics.fmean(losses) if losses else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument(
        "--wandb-group", default="ar-rmt-terminal-off-targetsg-on-20260817-v1"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    trials = make_trials()
    if args.dry_run:
        print(json.dumps([asdict(trial) for trial in trials], indent=2))
        return 0
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)
    controller = StudyController(
        output_root=args.output_root,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        dry_run=False,
    )
    results = controller.run_trials(trials)
    failed = sorted(trial_id for trial_id, ok in results.items() if not ok)
    if failed:
        raise RuntimeError(f"terminal-off SG-on trials failed: {failed}")
    atomic_json(args.output_root / "study-summary.json", build_report(args.output_root, trials))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
