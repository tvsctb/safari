#!/usr/bin/env python3
"""Run a fixed 2x2 iRNN AUX-weight x memory-pressure factorial study."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
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


ACTIVE_ASSOCIATIONS = 5
BASE_AUX_WEIGHT = 0.1
HIGH_AUX_WEIGHT = 0.2
BASE_RHO = 8.0
HIGH_MEMORY_RHO = BASE_RHO / math.sqrt(2.0)
BASE_TAU = 243.242356581615
HIGH_LAMBDA_TAU = BASE_TAU * math.sqrt(HIGH_AUX_WEIGHT / BASE_AUX_WEIGHT)
REFERENCE_EVAL_GROUP = "ar-irnn-difficulty-heldout-eval-20260820-v1"
REFERENCE_CONDITION = "memory4x"


@dataclass(frozen=True)
class FactorialCondition:
    label: str
    aux_weight: float
    rho: float
    tau: float
    lambda_high: bool
    memory_high: bool

    @property
    def memory_pressure(self) -> float:
        return (
            self.aux_weight / BASE_AUX_WEIGHT
            * (BASE_RHO / self.rho) ** 2
        )

    @property
    def token_pressure(self) -> float:
        return self.aux_weight / BASE_AUX_WEIGHT

    @property
    def terminal_pressure(self) -> float:
        return (
            self.aux_weight / BASE_AUX_WEIGHT
            * (BASE_TAU / self.tau) ** 2
        )


def conditions() -> tuple[FactorialCondition, ...]:
    return (
        FactorialCondition(
            "control", BASE_AUX_WEIGHT, BASE_RHO, BASE_TAU, False, False
        ),
        FactorialCondition(
            "memory2x",
            BASE_AUX_WEIGHT,
            HIGH_MEMORY_RHO,
            BASE_TAU,
            False,
            True,
        ),
        FactorialCondition(
            "lambda2x",
            HIGH_AUX_WEIGHT,
            BASE_RHO,
            HIGH_LAMBDA_TAU,
            True,
            False,
        ),
        FactorialCondition(
            "combined",
            HIGH_AUX_WEIGHT,
            HIGH_MEMORY_RHO,
            HIGH_LAMBDA_TAU,
            True,
            True,
        ),
    )


def make_trial(
    condition: FactorialCondition,
    seed: int,
    phase: str,
    *,
    max_epochs: int = FULL_EPOCHS,
) -> RNNTrial:
    return RNNTrial(
        trial_id=f"factorial-{phase}-{condition.label}-s{seed}",
        seed=seed,
        max_epochs=max_epochs,
        aux_weight=condition.aux_weight,
        rho=condition.rho,
        tau=condition.tau,
        phase=f"irnn_aux_pressure_{phase}",
        probe_only=False,
        activation="relu",
        recurrent_init="identity",
        recurrent_identity_scale=1.0,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        chunk_size=4,
        condition_memory_reconstruction_on_boundary=False,
        track=f"factorial-{condition.label}",
        num_active_associations=ACTIVE_ASSOCIATIONS,
    )


def tuning_trials() -> list[RNNTrial]:
    return [
        make_trial(condition, seed, "tuning")
        for condition in conditions()
        for seed in TUNING_SEEDS
    ]


def heldout_trials(condition: FactorialCondition) -> list[RNNTrial]:
    return [make_trial(condition, seed, "heldout") for seed in HELDOUT_SEEDS]


def condition_for_trial(trial: RNNTrial) -> FactorialCondition:
    label = trial.track.removeprefix("factorial-")
    return next(condition for condition in conditions() if condition.label == label)


def _report(output_root: Path, trial: RNNTrial) -> dict:
    return json.loads(
        (output_root / "controlled_lag" / f"{trial.trial_id}.json").read_text(
            encoding="utf-8"
        )
    )


def _mean_accuracy(report: dict, minimum_lag: int = 2) -> float:
    return statistics.fmean(
        row["accuracy"] for row in report["by_lag"] if row["lag"] >= minimum_lag
    )


def rank_conditions(trials: list[RNNTrial], output_root: Path) -> list[dict]:
    rows = []
    for condition in conditions():
        members = [trial for trial in trials if condition_for_trial(trial) == condition]
        reports = [_report(output_root, trial) for trial in members]
        long_accuracy = [_mean_accuracy(report, 22) for report in reports]
        very_long_accuracy = [_mean_accuracy(report, 32) for report in reports]
        overall_accuracy = [report["overall"]["accuracy"] for report in reports]
        overall_nll = [report["overall"]["nll"] for report in reports]
        rows.append(
            {
                "condition": asdict(condition),
                "memory_pressure": condition.memory_pressure,
                "token_pressure": condition.token_pressure,
                "terminal_pressure": condition.terminal_pressure,
                "lag_22_40_accuracy_mean": statistics.fmean(long_accuracy),
                "lag_22_40_accuracy_sample_sd": statistics.stdev(long_accuracy),
                "lag_32_40_accuracy_mean": statistics.fmean(very_long_accuracy),
                "overall_accuracy_mean": statistics.fmean(overall_accuracy),
                "overall_nll_mean": statistics.fmean(overall_nll),
                "seed_lag_22_40_accuracy": dict(
                    zip([trial.seed for trial in members], long_accuracy)
                ),
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


def select_treatment(ranking: list[dict]) -> dict:
    treatments = [row for row in ranking if row["condition"]["label"] != "control"]
    selected = treatments[0]
    control = next(row for row in ranking if row["condition"]["label"] == "control")
    return {
        "selection_metric": "mean_accuracy_lags_22_40",
        "tie_breakers": ["mean_accuracy_lags_32_40", "overall_nll"],
        "selected_condition": selected["condition"],
        "selected_minus_control_lag_22_40": (
            selected["lag_22_40_accuracy_mean"]
            - control["lag_22_40_accuracy_mean"]
        ),
        "ranking": ranking,
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
    combined = next(condition for condition in conditions() if condition.label == "combined")
    trial = make_trial(combined, 20260820, "preflight", max_epochs=1)
    command = build_rnn_command(trial, root, "unused", "unused", "unused")
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
        raise RuntimeError(f"iRNN factorial GPU preflight failed; inspect {log}")
    marker.write_text("ok\n", encoding="utf-8")


def aggregate_condition(trials: list[RNNTrial], output_root: Path) -> dict:
    reports = {trial.seed: _report(output_root, trial) for trial in trials}
    by_lag = []
    for lag in range(2, 41, 2):
        accuracy = [
            next(row for row in report["by_lag"] if row["lag"] == lag)["accuracy"]
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
                "seed_accuracies": dict(zip(reports, accuracy)),
            }
        )
    overall = [report["overall"] for report in reports.values()]
    long_accuracy = [_mean_accuracy(report, 22) for report in reports.values()]
    very_long_accuracy = [_mean_accuracy(report, 32) for report in reports.values()]
    return {
        "overall_accuracy_mean": statistics.fmean(row["accuracy"] for row in overall),
        "overall_accuracy_sample_sd": statistics.stdev(
            row["accuracy"] for row in overall
        ),
        "overall_nll_mean": statistics.fmean(row["nll"] for row in overall),
        "lag_22_40_accuracy_mean": statistics.fmean(long_accuracy),
        "lag_22_40_accuracy_sample_sd": statistics.stdev(long_accuracy),
        "lag_32_40_accuracy_mean": statistics.fmean(very_long_accuracy),
        "lag_32_40_accuracy_sample_sd": statistics.stdev(very_long_accuracy),
        "by_lag": by_lag,
    }


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
    # Match the saved K=5 control exactly for paired held-out evaluation.
    parser.add_argument("--dataset-seed", type=int, default=20260820)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value]
    tuning = tuning_trials()
    if len(tuning) != 8:
        raise RuntimeError("unexpected factorial tuning cardinality")
    atomic_json(
        args.output_root / "plan.json",
        {
            "design": "2x2_aux_weight_by_memory_pressure",
            "active_associations": ACTIVE_ASSOCIATIONS,
            "tuning_training_runs": 8,
            "tuning_evaluation_runs": 8,
            "heldout_training_runs": 4,
            "heldout_evaluation_runs": 4,
            "reference": {
                "eval_group": REFERENCE_EVAL_GROUP,
                "condition": REFERENCE_CONDITION,
            },
            "groups": {
                "tuning_training": args.tuning_training_group,
                "tuning_evaluation": args.tuning_eval_group,
                "heldout_training": args.heldout_training_group,
                "heldout_evaluation": args.heldout_eval_group,
            },
            "conditions": [
                {
                    **asdict(condition),
                    "memory_pressure": condition.memory_pressure,
                    "token_pressure": condition.token_pressure,
                    "terminal_pressure": condition.terminal_pressure,
                }
                for condition in conditions()
            ],
            "tuning_trials": [asdict(trial) for trial in tuning],
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
    results = controller.run_trials(tuning)
    if not all(results.get(trial.trial_id, False) for trial in tuning):
        raise RuntimeError("one or more factorial tuning trials failed")
    if args.dry_run:
        return 0

    run_evaluations(
        tuning,
        tuning_root,
        gpu_ids,
        _evaluation_args(args, args.tuning_eval_group),
    )
    selection = select_treatment(rank_conditions(tuning, tuning_root))
    atomic_json(args.output_root / "selection.json", selection)
    selected = FactorialCondition(**selection["selected_condition"])

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
    results = controller.run_trials(heldout)
    if not all(results.get(trial.trial_id, False) for trial in heldout):
        raise RuntimeError("one or more factorial held-out trials failed")
    run_evaluations(
        heldout,
        heldout_root,
        gpu_ids,
        _evaluation_args(args, args.heldout_eval_group),
    )
    atomic_json(
        args.output_root / "factorial_report.json",
        {
            "selection": selection,
            "selected_heldout": aggregate_condition(heldout, heldout_root),
            "selected_trials": [asdict(trial) for trial in heldout],
            "reference": {
                "eval_group": REFERENCE_EVAL_GROUP,
                "condition": REFERENCE_CONDITION,
            },
        },
    )
    (args.output_root / "IRNN_AUX_PRESSURE_FACTORIAL_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
