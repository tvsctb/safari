#!/usr/bin/env python3
"""Tune a smooth target-to-Gaussian scale LR crossfade for K=5 iRNN AUX."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from run_ar_rmt_scale_study import (
    FULL_EPOCHS,
    FULL_WARMUP_STEPS,
    HELDOUT_SEEDS,
    TUNING_SEEDS,
    StudyController,
    atomic_json,
)
from run_ar_rnn_controlled_lag_study import run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command


RHO = 8.0
TAU = 243.242356581615
AUX_WEIGHT = 0.1
CONSTRAINT_WEIGHT = 0.2
ACTIVE_ASSOCIATIONS = 5
CROSSFADE_STEPS = FULL_WARMUP_STEPS
TARGET_LRS = (3e-4, 1e-3)
GAUSSIAN_LRS = (3e-5, 1e-4)


@dataclass(frozen=True)
class CrossfadeCondition:
    label: str
    target_lr_initial: float
    gaussian_lr_final: float


def conditions() -> tuple[CrossfadeCondition, ...]:
    return tuple(
        CrossfadeCondition(
            f"target{target_lr:g}-scale{gaussian_lr:g}",
            target_lr,
            gaussian_lr,
        )
        for target_lr in TARGET_LRS
        for gaussian_lr in GAUSSIAN_LRS
    )


def make_trial(
    condition: CrossfadeCondition,
    seed: int,
    phase: str,
    *,
    max_epochs: int,
) -> RNNTrial:
    return RNNTrial(
        trial_id=f"crossfade-{condition.label}-s{seed}",
        seed=seed,
        max_epochs=max_epochs,
        aux_weight=AUX_WEIGHT,
        rho=RHO,
        tau=TAU,
        phase=f"irnn_scale_crossfade_{phase}",
        activation="relu",
        recurrent_init="identity",
        recurrent_identity_scale=1.0,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        chunk_size=4,
        aux_chunk_sizes=(4,),
        stop_gradient_memory_target=False,
        state_aux_distribution="gaussian",
        gaussian_scale_mode="learned",
        gaussian_scale_learning_start_step=0,
        gaussian_scale_learning_rate=condition.gaussian_lr_final,
        memory_scale_target=None,
        memory_scale_target_mode="learned",
        memory_scale_target_learning_rate=condition.target_lr_initial,
        memory_scale_constraint_weight=CONSTRAINT_WEIGHT,
        memory_scale_constraint_start_step=0,
        memory_scale_constraint_ramp_steps=0,
        scale_crossfade_enabled=True,
        scale_crossfade_steps=CROSSFADE_STEPS,
        target_lr_final=0.0,
        gaussian_lr_initial=0.0,
        track=condition.label,
        num_active_associations=ACTIVE_ASSOCIATIONS,
    )


def tuning_trials(max_epochs: int = FULL_EPOCHS) -> list[RNNTrial]:
    return [
        make_trial(condition, seed, "tuning", max_epochs=max_epochs)
        for condition in conditions()
        for seed in TUNING_SEEDS
    ]


def heldout_trials(condition: CrossfadeCondition) -> list[RNNTrial]:
    return [
        make_trial(condition, seed, "heldout", max_epochs=FULL_EPOCHS)
        for seed in HELDOUT_SEEDS
    ]


def _last_metrics(root: Path, trial: RNNTrial) -> dict:
    path = root / "trials" / trial.trial_id / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))["last"]["metrics"]


def rank_validation(trials: list[RNNTrial], root: Path) -> list[dict]:
    rows = []
    for condition in conditions():
        members = [trial for trial in trials if trial.track == condition.label]
        metrics = [_last_metrics(root, trial) for trial in members]
        accuracies = [row["val/accuracy_ignore_index"] for row in metrics]
        losses = [row["val/loss"] for row in metrics]
        rows.append(
            {
                "condition": asdict(condition),
                "validation_accuracy_mean": statistics.fmean(accuracies),
                "validation_accuracy_sample_sd": statistics.stdev(accuracies),
                "validation_loss_mean": statistics.fmean(losses),
                "seed_validation_accuracy": dict(
                    zip([trial.seed for trial in members], accuracies)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["validation_accuracy_mean"],
            row["validation_loss_mean"],
            row["validation_accuracy_sample_sd"],
            row["condition"]["label"],
        )
    )
    return rows


def _report(root: Path, trial: RNNTrial) -> dict:
    path = root / "controlled_lag" / f"{trial.trial_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _lag_accuracy(report: dict, minimum: int) -> float:
    return statistics.fmean(
        row["accuracy"] for row in report["by_lag"] if row["lag"] >= minimum
    )


def rank_controlled_lag(trials: list[RNNTrial], root: Path) -> list[dict]:
    labels = sorted({trial.track for trial in trials})
    rows = []
    for label in labels:
        members = [trial for trial in trials if trial.track == label]
        reports = [_report(root, trial) for trial in members]
        rows.append(
            {
                "condition": asdict(
                    next(value for value in conditions() if value.label == label)
                ),
                "lag_22_40_accuracy_mean": statistics.fmean(
                    _lag_accuracy(report, 22) for report in reports
                ),
                "lag_32_40_accuracy_mean": statistics.fmean(
                    _lag_accuracy(report, 32) for report in reports
                ),
                "overall_accuracy_mean": statistics.fmean(
                    report["overall"]["accuracy"] for report in reports
                ),
                "overall_nll_mean": statistics.fmean(
                    report["overall"]["nll"] for report in reports
                ),
                "learned_scale_state": {
                    str(trial.seed): _report(root, trial)["learned_scale_state"]
                    for trial in members
                },
            }
        )
    rows.sort(
        key=lambda row: (
            -row["lag_22_40_accuracy_mean"],
            -row["lag_32_40_accuracy_mean"],
            row["overall_nll_mean"],
            row["condition"]["label"],
        )
    )
    return rows


def _paired_bootstrap(values: list[float], draws: int = 10000) -> list[float]:
    generator = random.Random(20260823)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def aggregate_heldout(trials: list[RNNTrial], root: Path) -> dict:
    reports = {trial.seed: _report(root, trial) for trial in trials}
    by_lag = []
    for lag in range(2, 41, 2):
        accuracy = [
            next(row for row in report["by_lag"] if row["lag"] == lag)[
                "accuracy"
            ]
            for report in reports.values()
        ]
        nll = [
            next(row for row in report["by_lag"] if row["lag"] == lag)["nll"]
            for report in reports.values()
        ]
        by_lag.append(
            {
                "lag": lag,
                "accuracy_mean": statistics.fmean(accuracy),
                "accuracy_sample_sd": statistics.stdev(accuracy),
                "nll_mean": statistics.fmean(nll),
                "seed_accuracy": dict(zip(reports, accuracy)),
            }
        )
    long_values = [_lag_accuracy(report, 22) for report in reports.values()]
    return {
        "overall_accuracy_mean": statistics.fmean(
            report["overall"]["accuracy"] for report in reports.values()
        ),
        "overall_nll_mean": statistics.fmean(
            report["overall"]["nll"] for report in reports.values()
        ),
        "lag_22_40_accuracy_mean": statistics.fmean(long_values),
        "lag_22_40_accuracy_sample_sd": statistics.stdev(long_values),
        "lag_22_40_bootstrap_95_ci": _paired_bootstrap(long_values),
        "lag_32_40_accuracy_mean": statistics.fmean(
            _lag_accuracy(report, 32) for report in reports.values()
        ),
        "learned_scale_state": {
            str(seed): report["learned_scale_state"]
            for seed, report in reports.items()
        },
        "by_lag": by_lag,
    }


def _eval_args(args: argparse.Namespace, group: str) -> argparse.Namespace:
    values = vars(args).copy()
    values["eval_group"] = group
    return argparse.Namespace(**values)


def run_preflight(root: Path, gpu_id: int) -> None:
    marker = root / "preflight" / "PREFLIGHT_OK"
    if marker.exists():
        return
    condition = conditions()[-1]
    trial = make_trial(condition, 20260823, "preflight", max_epochs=1)
    command = build_rnn_command(
        trial, root / "preflight", "unused", "unused", "unused"
    )
    replacements = {
        "+trainer.check_val_every_n_epoch=5": "+trainer.check_val_every_n_epoch=1",
        "trainer.limit_train_batches=1.0": "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=1.0": "trainer.limit_val_batches=2",
        "task.aux_gradient_norm_interval=1570": "task.aux_gradient_norm_interval=1",
        "task.aux_diagnostic_interval=157": "task.aux_diagnostic_interval=1",
        "callbacks.rnn_scale_crossfade.transition_steps=12560": (
            "callbacks.rnn_scale_crossfade.transition_steps=2"
        ),
        "wandb.mode=online": "wandb.mode=disabled",
        "+wandb.entity=unused": "+wandb.entity=null",
        "+wandb.resume=allow": "+wandb.resume=null",
    }
    command = [replacements.get(value, value) for value in command]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    log = marker.parent / "preflight.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    trial_root = marker.parent / "trials" / trial.trial_id
    if (
        completed.returncode != 0
        or not (trial_root / "result.json").exists()
        or not (trial_root / "checkpoints" / "last.ckpt").exists()
    ):
        raise RuntimeError(f"scale-crossfade GPU preflight failed; inspect {log}")
    marker.write_text("ok\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--tuning-training-group", required=True)
    parser.add_argument("--tuning-eval-group", required=True)
    parser.add_argument("--heldout-training-group", required=True)
    parser.add_argument("--heldout-eval-group", required=True)
    parser.add_argument("--examples-per-lag", type=int, default=10000)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=20260823)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value]
    tuning = tuning_trials()
    if len(tuning) != 8:
        raise RuntimeError("unexpected crossfade tuning cardinality")
    atomic_json(
        args.output_root / "plan.json",
        {
            "design": "2x2_target_lr_by_gaussian_lr_smooth_crossfade",
            "fixed_initial_values": {"rho": RHO, "tau": TAU},
            "no_fixed_scale_goal": True,
            "constraint_weight": CONSTRAINT_WEIGHT,
            "crossfade_steps": CROSSFADE_STEPS,
            "target_lr_final": 0.0,
            "gaussian_lr_initial": 0.0,
            "tuning_epochs": FULL_EPOCHS,
            "tuning_training_runs": 8,
            "tuning_evaluation_runs": 8,
            "heldout_training_runs": 4,
            "heldout_evaluation_runs": 4,
            "conditions": [asdict(value) for value in conditions()],
            "groups": {
                "tuning_training": args.tuning_training_group,
                "tuning_evaluation": args.tuning_eval_group,
                "heldout_training": args.heldout_training_group,
                "heldout_evaluation": args.heldout_eval_group,
            },
        },
    )
    if not args.dry_run:
        run_preflight(args.output_root, gpu_ids[0])

    tuning_root = args.output_root / "tuning"
    controller = StudyController(
        tuning_root,
        gpu_ids,
        args.workers_per_gpu,
        args.wandb_project,
        args.wandb_entity,
        args.tuning_training_group,
        command_builder=build_rnn_command,
        dry_run=args.dry_run,
    )
    status = controller.run_trials(tuning)
    if not all(status.get(trial.trial_id, False) for trial in tuning):
        raise RuntimeError("one or more full crossfade tuning runs failed")
    if args.dry_run:
        return 0

    validation_ranking = rank_validation(tuning, tuning_root)
    atomic_json(args.output_root / "validation-ranking.json", validation_ranking)
    run_evaluations(
        tuning,
        tuning_root,
        gpu_ids,
        _eval_args(args, args.tuning_eval_group),
    )
    full_ranking = rank_controlled_lag(tuning, tuning_root)
    atomic_json(args.output_root / "full-ranking.json", full_ranking)
    selected = CrossfadeCondition(**full_ranking[0]["condition"])

    heldout = heldout_trials(selected)
    heldout_root = args.output_root / "heldout"
    controller = StudyController(
        heldout_root,
        gpu_ids,
        args.workers_per_gpu,
        args.wandb_project,
        args.wandb_entity,
        args.heldout_training_group,
        command_builder=build_rnn_command,
        dry_run=False,
    )
    status = controller.run_trials(heldout)
    if not all(status.get(trial.trial_id, False) for trial in heldout):
        raise RuntimeError("one or more crossfade held-out trials failed")
    run_evaluations(
        heldout,
        heldout_root,
        gpu_ids,
        _eval_args(args, args.heldout_eval_group),
    )
    atomic_json(
        args.output_root / "crossfade-report.json",
        {
            "selected_condition": asdict(selected),
            "validation_ranking": validation_ranking,
            "full_ranking": full_ranking,
            "heldout": aggregate_heldout(heldout, heldout_root),
        },
    )
    (args.output_root / "IRNN_SCALE_LR_CROSSFADE_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
