#!/usr/bin/env python3
"""Run the common-warmup iRNN learned Gaussian-scale study end to end."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from run_ar_rmt_scale_study import (
    FULL_EPOCHS,
    FULL_TRAINING_STEPS,
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
SCALE_LR = 1e-5
CONSTRAINT_RAMP_STEPS = 500
CONTINUATION_LOADER_SEED_OFFSET = 10_000_000
WARMUP_EPOCHS = FULL_EPOCHS // 5


def make_trial(
    trial_id: str,
    seed: int,
    max_epochs: int,
    track: str,
    *,
    learning_start_step: int,
    target: float | None = None,
    constraint_weight: float = 0.0,
    initial_checkpoint: str | None = None,
    loader_seed: int | None = None,
) -> RNNTrial:
    return RNNTrial(
        trial_id=trial_id,
        seed=seed,
        max_epochs=max_epochs,
        aux_weight=AUX_WEIGHT,
        rho=RHO,
        tau=TAU,
        phase="warmup_scale",
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
        gaussian_scale_learning_start_step=learning_start_step,
        gaussian_scale_learning_rate=SCALE_LR,
        memory_scale_target=target,
        memory_scale_constraint_weight=constraint_weight,
        memory_scale_constraint_start_step=FULL_WARMUP_STEPS,
        memory_scale_constraint_ramp_steps=CONSTRAINT_RAMP_STEPS,
        initial_checkpoint=initial_checkpoint,
        loader_seed=loader_seed,
        track=track,
        num_active_associations=5,
    )


def warmup_trials(seeds: tuple[int, ...]) -> list[RNNTrial]:
    return [
        make_trial(
            f"warmup-s{seed}",
            seed,
            WARMUP_EPOCHS,
            "warmup",
            learning_start_step=FULL_TRAINING_STEPS + 1,
            loader_seed=seed,
        )
        for seed in seeds
    ]


def warmup_checkpoint(root: Path, seed: int) -> Path:
    return root / "trials" / f"warmup-s{seed}" / "checkpoints" / "last.ckpt"


def run_calibration(
    root: Path,
    warmup_root: Path,
    seed: int,
    gpu_id: int,
    *,
    examples: int,
) -> dict:
    output = root / f"seed-{seed}.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        "scripts/calibrate_ar_rnn_memory_scale.py",
        "--checkpoint", str(warmup_checkpoint(warmup_root, seed)),
        "--output", str(output),
        "--seed", str(seed),
        "--rho", str(RHO),
        "--tau", str(TAU),
        "--scale-learning-rate", str(SCALE_LR),
        "--training-steps", str(FULL_TRAINING_STEPS),
        "--examples", str(examples),
        "--batch-size", "128",
        "--aux-weight", str(AUX_WEIGHT),
        "--target-gradient-ratio", "0.002",
        "--device", "cuda",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    root.mkdir(parents=True, exist_ok=True)
    log = root / f"seed-{seed}.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not output.exists():
        details = log.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"memory-scale calibration failed for seed {seed}: {log}\n{details}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def run_calibrations(
    root: Path,
    warmup_root: Path,
    seeds: tuple[int, ...],
    gpu_ids: tuple[int, ...],
    *,
    examples: int,
) -> dict[int, dict]:
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            seed: executor.submit(
                run_calibration,
                root,
                warmup_root,
                seed,
                gpu_ids[index % len(gpu_ids)],
                examples=examples,
            )
            for index, seed in enumerate(seeds)
        }
        return {seed: future.result() for seed, future in futures.items()}


def branch_trials(
    seeds: tuple[int, ...],
    warmup_root: Path,
    calibrations: dict[int, dict],
    constraint_weight: float,
    conditions: tuple[str, ...],
) -> list[RNNTrial]:
    trials = []
    for seed in seeds:
        checkpoint = str(warmup_checkpoint(warmup_root, seed))
        common = dict(
            seed=seed,
            max_epochs=FULL_EPOCHS,
            initial_checkpoint=checkpoint,
            loader_seed=CONTINUATION_LOADER_SEED_OFFSET + seed,
        )
        for condition in conditions:
            if condition == "fixed":
                trials.append(
                    make_trial(
                        f"fixed-s{seed}",
                        track="fixed",
                        learning_start_step=FULL_TRAINING_STEPS + 1,
                        **common,
                    )
                )
            elif condition == "learnable-free":
                trials.append(
                    make_trial(
                        f"learnable-free-s{seed}",
                        track="learnable-free",
                        learning_start_step=FULL_WARMUP_STEPS,
                        **common,
                    )
                )
            elif condition == "warmup-anchored":
                trials.append(
                    make_trial(
                        f"warmup-anchored-s{seed}",
                        track="warmup-anchored",
                        learning_start_step=FULL_WARMUP_STEPS,
                        target=calibrations[seed]["target_rms"],
                        constraint_weight=constraint_weight,
                        **common,
                    )
                )
            else:
                raise ValueError(condition)
    return trials


def report_for(root: Path, trial: RNNTrial) -> dict:
    return json.loads(
        (root / "controlled_lag" / f"{trial.trial_id}.json").read_text(
            encoding="utf-8"
        )
    )


def long_accuracy(report: dict, minimum_lag: int) -> float:
    return statistics.fmean(
        row["accuracy"] for row in report["by_lag"] if row["lag"] >= minimum_lag
    )


def rank_tuning(trials: list[RNNTrial], root: Path) -> list[dict]:
    rows = []
    for condition in ("fixed", "learnable-free", "warmup-anchored"):
        members = [trial for trial in trials if trial.track == condition]
        reports = [report_for(root, trial) for trial in members]
        rows.append(
            {
                "condition": condition,
                "lag_22_40_accuracy": statistics.fmean(
                    long_accuracy(report, 22) for report in reports
                ),
                "lag_32_40_accuracy": statistics.fmean(
                    long_accuracy(report, 32) for report in reports
                ),
                "overall_accuracy": statistics.fmean(
                    report["overall"]["accuracy"] for report in reports
                ),
                "overall_nll": statistics.fmean(
                    report["overall"]["nll"] for report in reports
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["lag_22_40_accuracy"],
            -row["lag_32_40_accuracy"],
            row["overall_nll"],
            row["condition"],
        )
    )
    return rows


def select_learnable(ranking: list[dict]) -> str:
    return next(
        row["condition"]
        for row in ranking
        if row["condition"] in {"learnable-free", "warmup-anchored"}
    )


def paired_bootstrap(differences: list[float], samples: int = 10000):
    generator = random.Random(20260822)
    values = sorted(
        statistics.fmean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return [values[int(0.025 * samples)], values[int(0.975 * samples) - 1]]


def heldout_report(trials: list[RNNTrial], root: Path, selected: str) -> dict:
    reports = {trial.trial_id: report_for(root, trial) for trial in trials}
    conditions = {}
    for condition in ("fixed", selected):
        members = [trial for trial in trials if trial.track == condition]
        by_lag = []
        for lag in range(2, 41, 2):
            values = [
                next(
                    row
                    for row in reports[trial.trial_id]["by_lag"]
                    if row["lag"] == lag
                )["accuracy"]
                for trial in members
            ]
            by_lag.append(
                {
                    "lag": lag,
                    "accuracy_mean": statistics.fmean(values),
                    "accuracy_sample_sd": statistics.stdev(values),
                    "seed_accuracy": dict(zip([t.seed for t in members], values)),
                }
            )
        conditions[condition] = {
            "overall_accuracy_mean": statistics.fmean(
                reports[trial.trial_id]["overall"]["accuracy"] for trial in members
            ),
            "overall_nll_mean": statistics.fmean(
                reports[trial.trial_id]["overall"]["nll"] for trial in members
            ),
            "lag_22_40_accuracy_mean": statistics.fmean(
                long_accuracy(reports[trial.trial_id], 22) for trial in members
            ),
            "lag_32_40_accuracy_mean": statistics.fmean(
                long_accuracy(reports[trial.trial_id], 32) for trial in members
            ),
            "by_lag": by_lag,
        }
    fixed = {
        trial.seed: long_accuracy(reports[trial.trial_id], 22)
        for trial in trials
        if trial.track == "fixed"
    }
    learned = {
        trial.seed: long_accuracy(reports[trial.trial_id], 22)
        for trial in trials
        if trial.track == selected
    }
    differences = [learned[seed] - fixed[seed] for seed in HELDOUT_SEEDS]
    return {
        "selected_condition": selected,
        "conditions": conditions,
        "paired_lag_22_40_delta_mean": statistics.fmean(differences),
        "paired_lag_22_40_wins": sum(value > 0 for value in differences),
        "paired_lag_22_40_bootstrap_95_ci": paired_bootstrap(differences),
        "paired_seed_deltas": dict(zip(HELDOUT_SEEDS, differences)),
    }


def eval_args(args: argparse.Namespace, group: str) -> argparse.Namespace:
    values = vars(args).copy()
    values["eval_group"] = group
    return argparse.Namespace(**values)


def run_phase(
    trials: list[RNNTrial],
    root: Path,
    args: argparse.Namespace,
    train_group: str,
    eval_group: str | None,
    gpu_ids: tuple[int, ...],
) -> None:
    controller = StudyController(
        root,
        gpu_ids,
        args.workers_per_gpu,
        args.wandb_project,
        args.wandb_entity,
        train_group,
        command_builder=build_rnn_command,
        dry_run=args.dry_run,
    )
    status = controller.run_trials(trials)
    if not all(status.get(trial.trial_id, False) for trial in trials):
        raise RuntimeError(f"training phase failed: {train_group}")
    if eval_group is not None and not args.dry_run:
        run_evaluations(
            trials, root, list(gpu_ids), eval_args(args, eval_group)
        )


def run_preflight(root: Path, gpu_id: int) -> None:
    marker = root / "PREFLIGHT_OK"
    if marker.exists():
        return
    warmup_root = root / "warmup"
    warmup = replace(
        make_trial(
            "warmup-s20260822",
            20260822,
            1,
            "warmup",
            learning_start_step=100,
            loader_seed=20260822,
        ),
        max_epochs=1,
    )

    def smoke_builder(trial, output_root, project, entity, group):
        command = build_rnn_command(trial, output_root, project, entity, group)
        replacements = {
            "+trainer.check_val_every_n_epoch=5": "+trainer.check_val_every_n_epoch=1",
            "trainer.limit_train_batches=1.0": "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=1.0": "trainer.limit_val_batches=2",
            "task.aux_gradient_norm_interval=1570": "task.aux_gradient_norm_interval=1",
            "task.aux_diagnostic_interval=157": "task.aux_diagnostic_interval=1",
            "wandb.mode=online": "wandb.mode=disabled",
            "+wandb.entity=unused": "+wandb.entity=null",
            "+wandb.resume=allow": "+wandb.resume=null",
        }
        return [replacements.get(value, value) for value in command]

    controller = StudyController(
        warmup_root,
        (gpu_id,),
        1,
        "unused",
        "unused",
        "unused-preflight-warmup",
        command_builder=smoke_builder,
        dry_run=False,
    )
    if not controller.run_trials([warmup]).get(warmup.trial_id, False):
        raise RuntimeError("warmup-scale preflight warmup failed")
    calibration = run_calibration(
        root / "calibration",
        warmup_root,
        warmup.seed,
        gpu_id,
        examples=32,
    )
    branch = replace(
        make_trial(
            f"warmup-anchored-s{warmup.seed}",
            warmup.seed,
            2,
            "warmup-anchored",
            learning_start_step=2,
            target=calibration["target_rms"],
            constraint_weight=calibration["gradient_calibration"][
                "suggested_constraint_weight"
            ],
            initial_checkpoint=str(warmup_checkpoint(warmup_root, warmup.seed)),
            loader_seed=CONTINUATION_LOADER_SEED_OFFSET + warmup.seed,
        ),
        memory_scale_constraint_start_step=2,
        memory_scale_constraint_ramp_steps=1,
    )
    controller = StudyController(
        root / "branch",
        (gpu_id,),
        1,
        "unused",
        "unused",
        "unused-preflight-branch",
        command_builder=smoke_builder,
        dry_run=False,
    )
    if not controller.run_trials([branch]).get(branch.trial_id, False):
        raise RuntimeError("warmup-scale preflight branch failed")
    marker.write_text("ok\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--group-prefix", required=True)
    parser.add_argument("--examples-per-lag", type=int, default=10000)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=20260822)
    parser.add_argument("--calibration-examples", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)
    groups = {
        name: f"{args.group_prefix}-{name}"
        for name in (
            "warmup-train",
            "tuning-train",
            "tuning-eval",
            "heldout-train",
            "heldout-eval",
        )
    }
    atomic_json(
        args.output_root / "plan.json",
        {
            "protocol": "common_fixed_scale_warmup_then_branch",
            "rho": RHO,
            "tau": TAU,
            "aux_weight": AUX_WEIGHT,
            "scale_learning_rate": SCALE_LR,
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup_steps": FULL_WARMUP_STEPS,
            "full_epochs": FULL_EPOCHS,
            "full_steps": FULL_TRAINING_STEPS,
            "continuation_loader_seed_offset": CONTINUATION_LOADER_SEED_OFFSET,
            "tuning_conditions": [
                "fixed", "learnable-free", "warmup-anchored"
            ],
            "groups": groups,
        },
    )
    if not args.dry_run:
        run_preflight(args.output_root / "preflight", gpu_ids[0])
        if args.preflight_only:
            return 0

    tuning_warmup_root = args.output_root / "tuning-warmup"
    run_phase(
        warmup_trials(TUNING_SEEDS),
        tuning_warmup_root,
        args,
        groups["warmup-train"],
        None,
        gpu_ids,
    )
    if args.dry_run:
        return 0
    tuning_calibrations = run_calibrations(
        args.output_root / "tuning-calibration",
        tuning_warmup_root,
        TUNING_SEEDS,
        gpu_ids,
        examples=args.calibration_examples,
    )
    constraint_weight = statistics.median(
        report["gradient_calibration"]["suggested_constraint_weight"]
        for report in tuning_calibrations.values()
    )
    atomic_json(
        args.output_root / "constraint-calibration.json",
        {
            "constraint_weight": constraint_weight,
            "aggregation": "median_across_tuning_seeds",
            "reports": tuning_calibrations,
        },
    )
    tuning = branch_trials(
        TUNING_SEEDS,
        tuning_warmup_root,
        tuning_calibrations,
        constraint_weight,
        ("fixed", "learnable-free", "warmup-anchored"),
    )
    tuning_root = args.output_root / "tuning"
    run_phase(
        tuning,
        tuning_root,
        args,
        groups["tuning-train"],
        groups["tuning-eval"],
        gpu_ids,
    )
    ranking = rank_tuning(tuning, tuning_root)
    selected = select_learnable(ranking)
    atomic_json(
        args.output_root / "selection.json",
        {
            "selected_learnable_condition": selected,
            "criterion": "lag22-40_then_lag32-40_then_overall_nll",
            "ranking": ranking,
        },
    )

    heldout_warmup_root = args.output_root / "heldout-warmup"
    run_phase(
        warmup_trials(HELDOUT_SEEDS),
        heldout_warmup_root,
        args,
        groups["warmup-train"],
        None,
        gpu_ids,
    )
    heldout_calibrations = run_calibrations(
        args.output_root / "heldout-calibration",
        heldout_warmup_root,
        HELDOUT_SEEDS,
        gpu_ids,
        examples=args.calibration_examples,
    )
    heldout = branch_trials(
        HELDOUT_SEEDS,
        heldout_warmup_root,
        heldout_calibrations,
        constraint_weight,
        ("fixed", selected),
    )
    heldout_root = args.output_root / "heldout"
    run_phase(
        heldout,
        heldout_root,
        args,
        groups["heldout-train"],
        groups["heldout-eval"],
        gpu_ids,
    )
    final = {
        "selection": json.loads(
            (args.output_root / "selection.json").read_text(encoding="utf-8")
        ),
        "constraint_weight": constraint_weight,
        "heldout_calibrations": heldout_calibrations,
        "heldout": heldout_report(heldout, heldout_root, selected),
        "reference_noaux_group": "ar-irnn-difficulty-heldout-eval-20260820-v1",
        "trials": [asdict(trial) for trial in tuning + heldout],
    }
    atomic_json(args.output_root / "study-summary.json", final)
    (args.output_root / "STUDY_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    print(json.dumps(final["heldout"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
