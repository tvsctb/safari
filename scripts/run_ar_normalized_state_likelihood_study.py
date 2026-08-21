#!/usr/bin/env python3
"""Compare no-AUX, learned Gaussian, and learned vMF normalized-state RNNs."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import torch

from run_ar_rmt_scale_study import FULL_EPOCHS, HELDOUT_SEEDS, StudyController, atomic_json
from run_ar_rnn_controlled_lag_study import run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command
from src.models.sequence.rnn_aux import RNNAuxLM


ACTIVE_ASSOCIATIONS = 5
CONDITIONS = ("noaux", "gaussian", "vmf")
EXPECTED_RUNS = len(CONDITIONS) * len(HELDOUT_SEEDS)


def run_vmf_fused_optimizer_probe(gpu_id: int) -> None:
    """Exercise and diagnose the exact learned-kappa fused optimizer path."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.manual_seed(20260822)
    model = RNNAuxLM(
        d_model=64,
        n_layer=3,
        vocab_size=20,
        chunk_size=4,
        chunk_offset="sequence",
        activation="tanh",
        recurrent_init="orthogonal",
        normalized_state=True,
        state_aux_distribution="vmf",
        vmf_kappa_mode="learned",
        state_likelihood_granularity="layer",
        memory_vmf_kappa=1.0,
        terminal_vmf_kappa=1.0,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
    ).to(device)
    named_parameters = list(model.named_parameters())
    regular = [parameter for _, parameter in named_parameters if not hasattr(parameter, "_optim")]
    special = [parameter for _, parameter in named_parameters if hasattr(parameter, "_optim")]
    optimizer = torch.optim.AdamW(
        regular, lr=1e-3, weight_decay=0.1, fused=True
    )
    optimizer.add_param_group(
        {"params": special, "lr": 1e-4, "weight_decay": 0.0}
    )
    tokens = torch.randint(0, 20, (32, 42), device=device)
    output, _ = model(tokens, targets=tokens, aux_tokens=tokens)
    output.aux_loss.backward()
    try:
        optimizer.step()
    except RuntimeError:
        names = {id(parameter): name for name, parameter in named_parameters}
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = optimizer.state.get(parameter, {})
                fields = {
                    "group": group_index,
                    "name": names[id(parameter)],
                    "parameter": (
                        str(parameter.dtype),
                        str(parameter.device),
                        str(parameter.layout),
                        tuple(parameter.stride()),
                    ),
                    "gradient": (
                        str(parameter.grad.dtype),
                        str(parameter.grad.device),
                        str(parameter.grad.layout),
                        tuple(parameter.grad.stride()),
                    ),
                    "state": {
                        key: (
                            str(value.dtype),
                            str(value.device),
                            str(value.layout),
                            tuple(value.stride()),
                        )
                        for key, value in state.items()
                        if isinstance(value, torch.Tensor)
                    },
                }
                print(f"VMF_FUSED_PARAMETER={fields}", flush=True)
        raise
    print("VMF_FUSED_OPTIMIZER_PROBE_OK", flush=True)


def make_trial(condition: str, seed: int, max_epochs: int = FULL_EPOCHS) -> RNNTrial:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    is_vmf = condition == "vmf"
    return RNNTrial(
        trial_id=f"normalized-state-{condition}-s{seed}",
        seed=seed,
        max_epochs=max_epochs,
        aux_weight=0.1,
        rho=1.0,
        tau=1.0,
        phase="normalized_state_likelihood",
        probe_only=condition == "noaux",
        activation="tanh",
        recurrent_init="orthogonal",
        recurrent_identity_scale=1.0,
        normalized_state=True,
        normalization_epsilon=1e-5,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        chunk_size=4,
        aux_chunk_sizes=(4,),
        condition_memory_reconstruction_on_boundary=False,
        track=f"normalized-{condition}",
        num_active_associations=ACTIVE_ASSOCIATIONS,
        stop_gradient_memory_target=False,
        state_aux_distribution="vmf" if is_vmf else "gaussian",
        vmf_kappa_mode="learned" if is_vmf else "fixed",
        vmf_kappa_learning_rate=1e-4,
        memory_vmf_kappa=1.0,
        terminal_vmf_kappa=1.0,
        gaussian_scale_mode="fixed" if is_vmf else "learned",
        state_likelihood_granularity="layer",
        gaussian_scale_learning_start_step=0,
        gaussian_scale_learning_rate=1e-4,
        memory_scale_constraint_weight=0.0,
    )


def trials() -> list[RNNTrial]:
    return [
        make_trial(condition, seed)
        for condition in CONDITIONS
        for seed in HELDOUT_SEEDS
    ]


def preflight_command(trial, output_root, project, entity, group):
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
    return [replacements.get(argument, argument) for argument in command]


def run_preflight(output_root: Path, gpu_id: int) -> None:
    marker = output_root / "PREFLIGHT_OK"
    if marker.exists():
        return
    checks = [
        replace(make_trial(condition, 20260822 + index), max_epochs=1)
        for index, condition in enumerate(CONDITIONS)
    ]
    controller = StudyController(
        output_root,
        (gpu_id,),
        1,
        "unused",
        "unused",
        "unused",
        command_builder=preflight_command,
        dry_run=False,
    )
    status = controller.run_trials(checks)
    if not all(status.get(trial.trial_id, False) for trial in checks):
        print(f"PREFLIGHT_STATUS={status}", flush=True)
        for trial in checks:
            log_path = output_root / "trials" / trial.trial_id / "train.log"
            if log_path.exists():
                print(f"\n===== PREFLIGHT LOG: {trial.trial_id} =====", flush=True)
                text = log_path.read_text(encoding="utf-8", errors="replace")
                print(text[-20_000:], flush=True)
        raise RuntimeError("normalized-state GPU preflight failed")
    for trial in checks:
        root = output_root / "trials" / trial.trial_id
        if not (root / "result.json").exists() or not (
            root / "checkpoints" / "last.ckpt"
        ).exists():
            raise RuntimeError(f"incomplete preflight output: {trial.trial_id}")
    marker.write_text("ok\n", encoding="utf-8")


def read_eval(output_root: Path, trial: RNNTrial) -> dict:
    return json.loads(
        (output_root / "controlled_lag" / f"{trial.trial_id}.json").read_text(
            encoding="utf-8"
        )
    )


def mean_for_lags(report: dict, minimum: int, maximum: int = 40) -> float:
    return statistics.fmean(
        row["accuracy"]
        for row in report["by_lag"]
        if minimum <= row["lag"] <= maximum
    )


def aggregate(all_trials: list[RNNTrial], output_root: Path) -> dict:
    reports = {trial.trial_id: read_eval(output_root, trial) for trial in all_trials}
    conditions = {}
    for condition in CONDITIONS:
        members = [trial for trial in all_trials if f"-{condition}-" in trial.trial_id]
        seed_reports = {trial.seed: reports[trial.trial_id] for trial in members}
        by_lag = []
        for lag in range(2, 41, 2):
            accuracies = {
                seed: next(row for row in report["by_lag"] if row["lag"] == lag)[
                    "accuracy"
                ]
                for seed, report in seed_reports.items()
            }
            nlls = {
                seed: next(row for row in report["by_lag"] if row["lag"] == lag)[
                    "nll"
                ]
                for seed, report in seed_reports.items()
            }
            by_lag.append(
                {
                    "lag": lag,
                    "accuracy_mean": statistics.fmean(accuracies.values()),
                    "accuracy_sample_sd": statistics.stdev(accuracies.values()),
                    "nll_mean": statistics.fmean(nlls.values()),
                    "seed_accuracies": accuracies,
                }
            )
        overall = [report["overall"]["accuracy"] for report in seed_reports.values()]
        long_22 = [mean_for_lags(report, 22) for report in seed_reports.values()]
        long_32 = [mean_for_lags(report, 32) for report in seed_reports.values()]
        conditions[condition] = {
            "overall_accuracy_mean": statistics.fmean(overall),
            "overall_accuracy_sample_sd": statistics.stdev(overall),
            "lag_22_40_accuracy_mean": statistics.fmean(long_22),
            "lag_22_40_accuracy_sample_sd": statistics.stdev(long_22),
            "lag_32_40_accuracy_mean": statistics.fmean(long_32),
            "lag_32_40_accuracy_sample_sd": statistics.stdev(long_32),
            "by_lag": by_lag,
        }

    paired = {}
    for condition in ("gaussian", "vmf"):
        condition_trials = {
            trial.seed: reports[trial.trial_id]
            for trial in all_trials
            if f"-{condition}-" in trial.trial_id
        }
        controls = {
            trial.seed: reports[trial.trial_id]
            for trial in all_trials
            if "-noaux-" in trial.trial_id
        }
        deltas = {
            seed: mean_for_lags(condition_trials[seed], 22)
            - mean_for_lags(controls[seed], 22)
            for seed in HELDOUT_SEEDS
        }
        paired[condition] = {
            "lag_22_40_accuracy_delta_mean": statistics.fmean(deltas.values()),
            "lag_22_40_accuracy_delta_sample_sd": statistics.stdev(deltas.values()),
            "wins": sum(value > 0 for value in deltas.values()),
            "seed_deltas": deltas,
        }
    return {
        "protocol": next(iter(reports.values()))["protocol"],
        "trials": [asdict(trial) for trial in all_trials],
        "conditions": conditions,
        "paired_aux_minus_noaux": paired,
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
    parser.add_argument("--dataset-seed", type=int, default=20260822)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)
    all_trials = trials()
    if len(all_trials) != EXPECTED_RUNS or len({t.trial_id for t in all_trials}) != EXPECTED_RUNS:
        raise RuntimeError("unexpected normalized-state study cardinality")
    atomic_json(
        args.output_root / "plan.json",
        {
            "training_runs": EXPECTED_RUNS,
            "evaluation_runs": EXPECTED_RUNS,
            "training_group": args.training_group,
            "evaluation_group": args.eval_group,
            "trials": [asdict(trial) for trial in all_trials],
        },
    )
    if not args.dry_run:
        run_vmf_fused_optimizer_probe(gpu_ids[0])
        run_preflight(args.output_root / "preflight", gpu_ids[0])
        if args.preflight_only:
            return 0
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
    status = controller.run_trials(all_trials)
    if not all(status.get(trial.trial_id, False) for trial in all_trials):
        raise RuntimeError("one or more normalized-state training runs failed")
    if args.dry_run:
        return 0
    run_evaluations(all_trials, args.output_root, list(gpu_ids), args)
    atomic_json(
        args.output_root / "normalized_state_report.json",
        aggregate(all_trials, args.output_root),
    )
    (args.output_root / "NORMALIZED_STATE_STUDY_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
