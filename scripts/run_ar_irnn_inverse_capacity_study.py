#!/usr/bin/env python3
"""Automate the K=5 iRNN inverse-decoder capacity study."""

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

from run_ar_rmt_scale_study import (
    FULL_EPOCHS,
    HELDOUT_SEEDS,
    TUNING_SEEDS,
    StudyController,
    atomic_json,
)
from run_ar_rnn_controlled_lag_study import run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command


CAPACITIES = (
    ("one-eighth", 0.125, 5197),
    ("quarter", 0.25, 10462),
    ("half", 0.5, 20734),
    ("full", 1.0, 41536),
    ("double", 2.0, 83011),
)
ACTIVE_ASSOCIATIONS = 5
RHO = 8.0
TAU = 243.242356581615
AUX_WEIGHT = 0.1
REFERENCE_EVAL_GROUP = "ar-irnn-difficulty-heldout-eval-20260820-v1"


def make_trial(label: str, multiplier: float, seed: int, phase: str) -> RNNTrial:
    return RNNTrial(
        trial_id=f"invcap-{phase}-{label}-s{seed}",
        seed=seed,
        max_epochs=FULL_EPOCHS,
        aux_weight=AUX_WEIGHT,
        rho=RHO,
        tau=TAU,
        phase=f"irnn_inverse_capacity_{phase}",
        activation="relu",
        recurrent_init="identity",
        recurrent_identity_scale=1.0,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        chunk_size=4,
        condition_memory_reconstruction_on_boundary=False,
        track=f"invcap-{label}",
        num_active_associations=ACTIVE_ASSOCIATIONS,
        inverse_capacity_multiplier=multiplier,
    )


def tuning_trials() -> list[RNNTrial]:
    return [
        make_trial(label, multiplier, seed, "tuning")
        for label, multiplier, _ in CAPACITIES
        for seed in TUNING_SEEDS
    ]


def heldout_trials(selected: dict) -> list[RNNTrial]:
    return [
        make_trial(
            selected["label"], selected["multiplier"], seed, "heldout"
        )
        for seed in HELDOUT_SEEDS
    ]


def capacity_for_trial(trial: RNNTrial) -> dict:
    label = trial.track.removeprefix("invcap-")
    for candidate, multiplier, parameters in CAPACITIES:
        if label == candidate:
            return {
                "label": candidate,
                "multiplier": multiplier,
                "inverse_parameters": parameters,
            }
    raise KeyError(label)


def evaluation_args(args: argparse.Namespace, group: str) -> argparse.Namespace:
    values = vars(args).copy()
    values["eval_group"] = group
    return argparse.Namespace(**values)


def controlled_report(root: Path, trial: RNNTrial) -> dict:
    return json.loads(
        (root / "controlled_lag" / f"{trial.trial_id}.json").read_text()
    )


def mean_lag(report: dict, minimum: int) -> float:
    return statistics.fmean(
        row["accuracy"] for row in report["by_lag"] if row["lag"] >= minimum
    )


def rank_capacities(trials: list[RNNTrial], root: Path) -> list[dict]:
    ranking = []
    for label, multiplier, parameters in CAPACITIES:
        members = [
            trial
            for trial in trials
            if capacity_for_trial(trial)["label"] == label
        ]
        reports = [controlled_report(root, trial) for trial in members]
        long = [mean_lag(report, 22) for report in reports]
        very_long = [mean_lag(report, 32) for report in reports]
        overall = [report["overall"]["accuracy"] for report in reports]
        nll = [report["overall"]["nll"] for report in reports]
        ranking.append(
            {
                "label": label,
                "multiplier": multiplier,
                "inverse_parameters": parameters,
                "lag_22_40_accuracy_mean": statistics.fmean(long),
                "lag_22_40_accuracy_sample_sd": statistics.stdev(long),
                "lag_32_40_accuracy_mean": statistics.fmean(very_long),
                "overall_accuracy_mean": statistics.fmean(overall),
                "overall_nll_mean": statistics.fmean(nll),
                "seed_lag_22_40_accuracy": dict(
                    zip([trial.seed for trial in members], long)
                ),
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["lag_22_40_accuracy_mean"],
            -row["lag_32_40_accuracy_mean"],
            row["overall_nll_mean"],
            row["inverse_parameters"],
        )
    )
    return ranking


def probe_command(
    trial: RNNTrial,
    root: Path,
    state_mode: str,
    args: argparse.Namespace,
    group: str,
    *,
    preflight: bool = False,
) -> list[str]:
    checkpoint = root / "trials" / trial.trial_id / "checkpoints" / "last.ckpt"
    run_name = f"{trial.trial_id}-{state_mode}"
    output = root / "past_probe" / run_name
    command = [
        sys.executable,
        "scripts/train_ar_rnn_past_info_probe.py",
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--condition", "aux",
        "--state-mode", state_mode,
        "--forward-seed", str(trial.seed),
        "--probe-seed", str(90260824 + trial.seed),
        "--dataset-seed", str(args.probe_dataset_seed),
        "--forward-batch-size", str(args.probe_forward_batch_size),
        "--batch-size", str(args.probe_batch_size),
    ]
    if preflight:
        command.extend(
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
        command.extend(
            [
                "--train-base-examples", str(args.probe_train_base_examples),
                "--val-base-examples", str(args.probe_val_base_examples),
                "--test-base-examples", str(args.probe_test_base_examples),
                "--max-epochs", str(args.probe_max_epochs),
                "--min-epochs", str(args.probe_min_epochs),
                "--patience", str(args.probe_patience),
                "--wandb-mode", "online",
                "--wandb-project", args.wandb_project,
                "--wandb-entity", args.wandb_entity,
                "--wandb-group", group,
                "--run-name", run_name,
            ]
        )
    return command


def run_probes(
    trials: list[RNNTrial],
    root: Path,
    gpu_ids: list[int],
    args: argparse.Namespace,
    group: str,
) -> None:
    jobs = [(trial, mode) for trial in trials for mode in ("full", "global_unit")]
    slots: queue.Queue[int] = queue.Queue()
    for _ in range(args.probe_workers_per_gpu):
        for gpu_id in gpu_ids:
            slots.put(gpu_id)

    def run_one(job) -> None:
        trial, mode = job
        result = root / "past_probe" / f"{trial.trial_id}-{mode}" / "result.json"
        if result.exists():
            return
        gpu_id = slots.get()
        try:
            result.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            with (result.parent / "probe.log").open("w") as stream:
                completed = subprocess.run(
                    probe_command(trial, root, mode, args, group),
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode or not result.exists():
                raise RuntimeError(f"capacity probe failed: {trial.trial_id}-{mode}")
        finally:
            slots.put(gpu_id)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


def ensure_fresh_groups(args: argparse.Namespace) -> None:
    import wandb

    api = wandb.Api(timeout=90)
    project = f"{args.wandb_entity}/{args.wandb_project}"
    for group in (
        args.tuning_training_group,
        args.tuning_eval_group,
        args.tuning_probe_group,
        args.heldout_training_group,
        args.heldout_eval_group,
        args.heldout_probe_group,
    ):
        if list(api.runs(project, filters={"group": group})):
            raise RuntimeError(f"W&B group is not empty: {group}")


def run_preflight(root: Path, gpu_id: int, args: argparse.Namespace) -> None:
    marker = root / "preflight" / "PREFLIGHT_OK"
    if marker.exists():
        return
    trial = make_trial("one-eighth", 0.125, 20260824, "preflight")
    trial = RNNTrial(**{**asdict(trial), "max_epochs": 1})
    command = build_rnn_command(trial, root / "preflight", "unused", "unused", "unused")
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
    command = [replacements.get(value, value) for value in command]
    environment = os.environ.copy()
    environment.update(
        {"CUDA_VISIBLE_DEVICES": str(gpu_id), "OMP_NUM_THREADS": "1"}
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    log = marker.parent / "preflight.log"
    with log.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    checkpoint = (
        root
        / "preflight"
        / "trials"
        / trial.trial_id
        / "checkpoints"
        / "last.ckpt"
    )
    if completed.returncode or not checkpoint.exists():
        raise RuntimeError(f"inverse-capacity GPU preflight failed; inspect {log}")
    probe = subprocess.run(
        probe_command(
            trial, root / "preflight", "full", args, "unused", preflight=True
        ),
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=False,
    )
    probe_result = (
        root
        / "preflight"
        / "past_probe"
        / f"{trial.trial_id}-full"
        / "result.json"
    )
    if probe.returncode or not probe_result.exists():
        raise RuntimeError("inverse-capacity standard-probe preflight failed")
    marker.write_text("ok\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--probe-workers-per-gpu", type=int, default=4)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--tuning-training-group", required=True)
    parser.add_argument("--tuning-eval-group", required=True)
    parser.add_argument("--tuning-probe-group", required=True)
    parser.add_argument("--heldout-training-group", required=True)
    parser.add_argument("--heldout-eval-group", required=True)
    parser.add_argument("--heldout-probe-group", required=True)
    parser.add_argument("--examples-per-lag", type=int, default=10000)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=20260820)
    parser.add_argument("--probe-dataset-seed", type=int, default=20260824)
    parser.add_argument("--probe-train-base-examples", type=int, default=10000)
    parser.add_argument("--probe-val-base-examples", type=int, default=2000)
    parser.add_argument("--probe-test-base-examples", type=int, default=5000)
    parser.add_argument("--probe-forward-batch-size", type=int, default=512)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--probe-max-epochs", type=int, default=120)
    parser.add_argument("--probe-min-epochs", type=int, default=30)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value]
    tuning = tuning_trials()
    if len(tuning) != 10:
        raise RuntimeError("expected exactly ten capacity tuning trials")
    if not args.dry_run:
        ensure_fresh_groups(args)
        run_preflight(args.output_root, gpu_ids[0], args)
    atomic_json(
        args.output_root / "plan.json",
        {
            "design": "inverse_capacity_1of8_to_2x",
            "capacities": [
                {"label": label, "multiplier": multiplier, "parameters": count}
                for label, multiplier, count in CAPACITIES
            ],
            "tuning_training_runs": 10,
            "tuning_evaluation_runs": 10,
            "tuning_standard_probe_runs": 20,
            "heldout_training_runs": 4,
            "heldout_evaluation_runs": 4,
            "heldout_standard_probe_runs": 8,
            "reference_noaux_eval_group": REFERENCE_EVAL_GROUP,
            "trials": [asdict(trial) for trial in tuning],
        },
    )
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
    result = controller.run_trials(tuning)
    if not all(result.get(trial.trial_id, False) for trial in tuning):
        raise RuntimeError("one or more capacity tuning trials failed")
    if args.dry_run:
        return 0
    run_evaluations(
        tuning,
        tuning_root,
        gpu_ids,
        evaluation_args(args, args.tuning_eval_group),
    )
    run_probes(tuning, tuning_root, gpu_ids, args, args.tuning_probe_group)
    ranking = rank_capacities(tuning, tuning_root)
    selected = ranking[0]
    atomic_json(
        args.output_root / "selection.json",
        {
            "metric": "controlled_lag_mean_accuracy_22_40",
            "tie_breakers": ["accuracy_32_40", "overall_nll", "fewer_parameters"],
            "selected": selected,
            "ranking": ranking,
        },
    )
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
    result = controller.run_trials(heldout)
    if not all(result.get(trial.trial_id, False) for trial in heldout):
        raise RuntimeError("one or more capacity held-out trials failed")
    run_evaluations(
        heldout,
        heldout_root,
        gpu_ids,
        evaluation_args(args, args.heldout_eval_group),
    )
    run_probes(
        heldout, heldout_root, gpu_ids, args, args.heldout_probe_group
    )
    atomic_json(
        args.output_root / "inverse_capacity_report.json",
        {
            "selection": selected,
            "ranking": ranking,
            "heldout_trials": [asdict(trial) for trial in heldout],
            "reference_noaux_eval_group": REFERENCE_EVAL_GROUP,
        },
    )
    (args.output_root / "IRNN_INVERSE_CAPACITY_COMPLETE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
