#!/usr/bin/env python3
"""Select an AR association load using no-AUX, then compare AUX held out."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

from run_ar_rmt_scale_study import FULL_EPOCHS, HELDOUT_SEEDS, StudyController, atomic_json
from run_ar_rnn_controlled_lag_study import evaluation_command, run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command


ACTIVE_ASSOCIATIONS = (3, 5, 7, 9)
TUNING_SEEDS = (202, 203)
IRNN_TAU = 243.242356581615
BASE_RHO = 16.0
MEMORY_4X_RHO = 8.0
TARGET_LONG_LAG_INTERVAL = (0.25, 0.50)


def make_trial(
    label: str,
    active_associations: int,
    seed: int,
    *,
    probe_only: bool,
    rho: float,
) -> RNNTrial:
    return RNNTrial(
        trial_id=f"difficulty-k{active_associations}-{label}-s{seed}",
        seed=seed,
        max_epochs=FULL_EPOCHS,
        aux_weight=0.1,
        rho=rho,
        tau=IRNN_TAU,
        phase="difficulty_ladder",
        probe_only=probe_only,
        activation="relu",
        recurrent_init="identity",
        recurrent_identity_scale=1.0,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        chunk_size=4,
        condition_memory_reconstruction_on_boundary=False,
        track=f"k{active_associations}-{label}",
        num_active_associations=active_associations,
    )


def screen_trials() -> list[RNNTrial]:
    return [
        make_trial("screen", k, seed, probe_only=True, rho=BASE_RHO)
        for k in ACTIVE_ASSOCIATIONS
        for seed in TUNING_SEEDS
    ]


def heldout_trials(active_associations: int) -> list[RNNTrial]:
    trials = []
    for seed in HELDOUT_SEEDS:
        trials.extend(
            [
                make_trial(
                    "noaux", active_associations, seed,
                    probe_only=True, rho=BASE_RHO,
                ),
                make_trial(
                    "aux", active_associations, seed,
                    probe_only=False, rho=BASE_RHO,
                ),
                make_trial(
                    "memory4x", active_associations, seed,
                    probe_only=False, rho=MEMORY_4X_RHO,
                ),
            ]
        )
    return trials


def _report(output_root: Path, trial: RNNTrial) -> dict:
    return json.loads(
        (output_root / "controlled_lag" / f"{trial.trial_id}.json").read_text(
            encoding="utf-8"
        )
    )


def select_difficulty(trials: list[RNNTrial], output_root: Path) -> dict:
    rows = []
    for active_associations in ACTIVE_ASSOCIATIONS:
        members = [
            trial
            for trial in trials
            if trial.num_active_associations == active_associations
        ]
        seed_means = {}
        for trial in members:
            report = _report(output_root, trial)
            seed_means[trial.seed] = statistics.fmean(
                row["accuracy"] for row in report["by_lag"] if row["lag"] >= 22
            )
        rows.append(
            {
                "num_active_associations": active_associations,
                "lag_22_40_accuracy_mean": statistics.fmean(seed_means.values()),
                "lag_22_40_accuracy_sample_sd": statistics.stdev(seed_means.values()),
                "seed_means": seed_means,
            }
        )

    lower, upper = TARGET_LONG_LAG_INTERVAL
    in_interval = [
        row for row in rows
        if lower <= row["lag_22_40_accuracy_mean"] <= upper
    ]
    if in_interval:
        selected = max(in_interval, key=lambda row: row["num_active_associations"])
        reason = "largest_k_inside_predeclared_interval"
    else:
        def distance(row: dict) -> tuple[float, int]:
            value = row["lag_22_40_accuracy_mean"]
            interval_distance = max(lower - value, 0.0, value - upper)
            return interval_distance, -row["num_active_associations"]

        selected = min(rows, key=distance)
        reason = "closest_to_predeclared_interval_tie_larger_k"
    return {
        "selection_metric": "noaux_mean_accuracy_lags_22_40",
        "target_interval": [lower, upper],
        "selection_rule": reason,
        "selected_num_active_associations": selected["num_active_associations"],
        "screen": rows,
    }


def _evaluation_args(args: argparse.Namespace, group: str) -> argparse.Namespace:
    values = vars(args).copy()
    values["eval_group"] = group
    return argparse.Namespace(**values)


def run_preflight(output_root: Path, gpu_id: int) -> None:
    root = output_root / "preflight"
    marker = root / "PREFLIGHT_OK"
    if marker.exists():
        return
    trial = make_trial("preflight", 3, 20260820, probe_only=False, rho=MEMORY_4X_RHO)
    trial = RNNTrial(**{**asdict(trial), "max_epochs": 1})
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
        {"CUDA_VISIBLE_DEVICES": str(gpu_id), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
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
        raise RuntimeError(f"difficulty-ladder GPU preflight failed; inspect {log}")
    marker.write_text("ok\n", encoding="utf-8")


def aggregate(
    screen: list[RNNTrial],
    selected: list[RNNTrial],
    heldout_root: Path,
    selection: dict,
) -> dict:
    heldout = {}
    for label in ("noaux", "aux", "memory4x"):
        members = [trial for trial in selected if trial.track.endswith(f"-{label}")]
        reports = {trial.seed: _report(heldout_root, trial) for trial in members}
        by_lag = []
        for lag in range(2, 41, 2):
            accuracy = [
                next(row for row in report["by_lag"] if row["lag"] == lag)["accuracy"]
                for report in reports.values()
            ]
            by_lag.append(
                {
                    "lag": lag,
                    "accuracy_mean": statistics.fmean(accuracy),
                    "accuracy_sample_sd": statistics.stdev(accuracy),
                    "seed_accuracies": dict(zip(reports, accuracy)),
                }
            )
        overall = [report["overall"]["accuracy"] for report in reports.values()]
        long_lag = [
            statistics.fmean(
                row["accuracy"] for row in report["by_lag"] if row["lag"] >= 22
            )
            for report in reports.values()
        ]
        heldout[label] = {
            "overall_accuracy_mean": statistics.fmean(overall),
            "overall_accuracy_sample_sd": statistics.stdev(overall),
            "lag_22_40_accuracy_mean": statistics.fmean(long_lag),
            "lag_22_40_accuracy_sample_sd": statistics.stdev(long_lag),
            "by_lag": by_lag,
        }
    return {
        "protocol": {
            "body_tokens": 40,
            "pair_slots": 20,
            "lags": list(range(2, 41, 2)),
            "selection_uses_aux_results": False,
        },
        "selection": selection,
        "screen_trials": [asdict(trial) for trial in screen],
        "heldout_trials": [asdict(trial) for trial in selected],
        "heldout": heldout,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--screen-training-group", required=True)
    parser.add_argument("--screen-eval-group", required=True)
    parser.add_argument("--heldout-training-group", required=True)
    parser.add_argument("--heldout-eval-group", required=True)
    parser.add_argument("--examples-per-lag", type=int, default=10000)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=20260820)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value]
    screen = screen_trials()
    if len(screen) != 8:
        raise RuntimeError("unexpected screen cardinality")
    atomic_json(
        args.output_root / "plan.json",
        {
            "screen_training_runs": 8,
            "screen_evaluation_runs": 8,
            "heldout_training_runs": 12,
            "heldout_evaluation_runs": 12,
            "groups": {
                "screen_training": args.screen_training_group,
                "screen_evaluation": args.screen_eval_group,
                "heldout_training": args.heldout_training_group,
                "heldout_evaluation": args.heldout_eval_group,
            },
            "screen_trials": [asdict(trial) for trial in screen],
        },
    )
    if not args.dry_run:
        run_preflight(args.output_root, gpu_ids[0])
    screen_root = args.output_root / "screen"
    controller = StudyController(
        screen_root, gpu_ids, args.workers_per_gpu,
        args.wandb_project, args.wandb_entity, args.screen_training_group,
        command_builder=build_rnn_command, dry_run=args.dry_run,
    )
    results = controller.run_trials(screen)
    if not all(results.get(trial.trial_id, False) for trial in screen):
        raise RuntimeError("one or more no-AUX difficulty screens failed")
    if args.dry_run:
        return 0
    run_evaluations(
        screen, screen_root, gpu_ids,
        _evaluation_args(args, args.screen_eval_group),
    )
    selection = select_difficulty(screen, screen_root)
    atomic_json(args.output_root / "selection.json", selection)

    selected_k = selection["selected_num_active_associations"]
    selected = heldout_trials(selected_k)
    heldout_root = args.output_root / "heldout"
    controller = StudyController(
        heldout_root, gpu_ids, args.workers_per_gpu,
        args.wandb_project, args.wandb_entity, args.heldout_training_group,
        command_builder=build_rnn_command, dry_run=False,
    )
    results = controller.run_trials(selected)
    if not all(results.get(trial.trial_id, False) for trial in selected):
        raise RuntimeError("one or more held-out difficulty trials failed")
    run_evaluations(
        selected, heldout_root, gpu_ids,
        _evaluation_args(args, args.heldout_eval_group),
    )
    atomic_json(
        args.output_root / "difficulty_ladder_report.json",
        aggregate(screen, selected, heldout_root, selection),
    )
    (args.output_root / "DIFFICULTY_LADDER_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
