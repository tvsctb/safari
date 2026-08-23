#!/usr/bin/env python3
"""Automate the K=5 iRNN gradient-matched inverse-capacity study."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from run_ar_irnn_inverse_capacity_study import (
    ACTIVE_ASSOCIATIONS,
    RHO,
    TAU,
    controlled_report,
    ensure_fresh_groups,
    evaluation_args,
    mean_lag,
    run_preflight,
    run_probes,
)
from run_ar_rmt_scale_study import (
    FULL_EPOCHS,
    HELDOUT_SEEDS,
    TUNING_SEEDS,
    StudyController,
    atomic_json,
)
from run_ar_rnn_controlled_lag_study import run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command


# Fixed before training from the final-five-diagnostic means of the original
# lambda=0.1 capacity screen.  Each value targets the dense-1x forward
# total-AUX/LM gradient ratio (~2.25%) without any adaptive rescaling.
CONDITIONS = (
    ("one-eighth", 0.125, 0.030, 5197, 0.02475),
    ("quarter", 0.25, 0.035, 10462, 0.02334),
    ("half", 0.5, 0.040, 20734, 0.02296),
)
REFERENCE_TUNING_EVAL_GROUP = (
    "ar-irnn-inverse-capacity-tuning-eval-20260824-v1"
)
REFERENCE_HELDOUT_EVAL_GROUP = "ar-irnn-difficulty-heldout-eval-20260820-v1"
REFERENCE_DENSE_CONDITION = "memory4x"
REFERENCE_NOAUX_CONDITION = "noaux"


def make_trial(
    label: str,
    multiplier: float,
    aux_weight: float,
    seed: int,
    phase: str,
) -> RNNTrial:
    return RNNTrial(
        trial_id=f"invcap-gradmatch-{phase}-{label}-s{seed}",
        seed=seed,
        max_epochs=FULL_EPOCHS,
        aux_weight=aux_weight,
        rho=RHO,
        tau=TAU,
        phase=f"irnn_inverse_capacity_gradmatch_{phase}",
        activation="relu",
        recurrent_init="identity",
        recurrent_identity_scale=1.0,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        chunk_size=4,
        condition_memory_reconstruction_on_boundary=False,
        track=f"invcap-gradmatch-{label}",
        num_active_associations=ACTIVE_ASSOCIATIONS,
        inverse_capacity_multiplier=multiplier,
    )


def tuning_trials() -> list[RNNTrial]:
    return [
        make_trial(label, multiplier, aux_weight, seed, "tuning")
        for label, multiplier, aux_weight, _, _ in CONDITIONS
        for seed in TUNING_SEEDS
    ]


def heldout_trials(selected: dict) -> list[RNNTrial]:
    return [
        make_trial(
            selected["label"],
            selected["multiplier"],
            selected["aux_weight"],
            seed,
            "heldout",
        )
        for seed in HELDOUT_SEEDS
    ]


def condition_for_trial(trial: RNNTrial) -> dict:
    label = trial.track.removeprefix("invcap-gradmatch-")
    for (
        candidate,
        multiplier,
        aux_weight,
        parameters,
        predicted_ratio,
    ) in CONDITIONS:
        if label == candidate:
            return {
                "label": candidate,
                "multiplier": multiplier,
                "aux_weight": aux_weight,
                "inverse_parameters": parameters,
                "predicted_forward_aux_lm_ratio": predicted_ratio,
            }
    raise KeyError(label)


def rank_conditions(trials: list[RNNTrial], root: Path) -> list[dict]:
    ranking = []
    for (
        label,
        multiplier,
        aux_weight,
        parameters,
        predicted_ratio,
    ) in CONDITIONS:
        members = [
            trial
            for trial in trials
            if condition_for_trial(trial)["label"] == label
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
                "aux_weight": aux_weight,
                "inverse_parameters": parameters,
                "predicted_forward_aux_lm_ratio": predicted_ratio,
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
    if len(tuning) != 6:
        raise RuntimeError("expected exactly six gradient-matched tuning trials")
    if not args.dry_run:
        ensure_fresh_groups(args)
        preflight = make_trial(
            CONDITIONS[0][0],
            CONDITIONS[0][1],
            CONDITIONS[0][2],
            20260824,
            "preflight",
        )
        run_preflight(args.output_root, gpu_ids[0], args, preflight)
    atomic_json(
        args.output_root / "plan.json",
        {
            "design": "inverse_capacity_gradient_matched_fixed_lambda",
            "gradient_target": "dense_1x_forward_total_aux_over_lm_0.0225",
            "conditions": [
                {
                    "label": label,
                    "multiplier": multiplier,
                    "aux_weight": aux_weight,
                    "inverse_parameters": parameters,
                    "predicted_forward_aux_lm_ratio": predicted_ratio,
                }
                for (
                    label,
                    multiplier,
                    aux_weight,
                    parameters,
                    predicted_ratio,
                ) in CONDITIONS
            ],
            "tuning_training_runs": 6,
            "tuning_evaluation_runs": 6,
            "tuning_standard_probe_runs": 12,
            "heldout_training_runs": 4,
            "heldout_evaluation_runs": 4,
            "heldout_standard_probe_runs": 8,
            "reference_tuning_eval_group": REFERENCE_TUNING_EVAL_GROUP,
            "reference_heldout_eval_group": REFERENCE_HELDOUT_EVAL_GROUP,
            "reference_dense_condition": REFERENCE_DENSE_CONDITION,
            "reference_noaux_condition": REFERENCE_NOAUX_CONDITION,
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
        raise RuntimeError("one or more gradient-matched tuning trials failed")
    if args.dry_run:
        return 0
    run_evaluations(
        tuning,
        tuning_root,
        gpu_ids,
        evaluation_args(args, args.tuning_eval_group),
    )
    run_probes(tuning, tuning_root, gpu_ids, args, args.tuning_probe_group)
    ranking = rank_conditions(tuning, tuning_root)
    selected = ranking[0]
    atomic_json(
        args.output_root / "selection.json",
        {
            "metric": "controlled_lag_mean_accuracy_22_40",
            "tie_breakers": [
                "accuracy_32_40",
                "overall_nll",
                "fewer_parameters",
            ],
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
        raise RuntimeError("one or more gradient-matched held-out trials failed")
    run_evaluations(
        heldout,
        heldout_root,
        gpu_ids,
        evaluation_args(args, args.heldout_eval_group),
    )
    run_probes(
        heldout,
        heldout_root,
        gpu_ids,
        args,
        args.heldout_probe_group,
    )
    atomic_json(
        args.output_root / "inverse_capacity_gradmatch_report.json",
        {
            "selection": selected,
            "ranking": ranking,
            "heldout_trials": [asdict(trial) for trial in heldout],
            "reference_tuning_eval_group": REFERENCE_TUNING_EVAL_GROUP,
            "reference_heldout_eval_group": REFERENCE_HELDOUT_EVAL_GROUP,
            "reference_dense_condition": REFERENCE_DENSE_CONDITION,
            "reference_noaux_condition": REFERENCE_NOAUX_CONDITION,
        },
    )
    marker = args.output_root / "IRNN_INVERSE_CAPACITY_GRADMATCH_COMPLETE"
    marker.write_text("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
