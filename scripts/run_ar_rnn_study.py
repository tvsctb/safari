#!/usr/bin/env python3
"""Run the parameter-matched vanilla-RNN AR no-AUX/AUX study in one Vessl Run."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, List, Sequence

from run_ar_rmt_scale_study import (
    BASELINE_SEEDS,
    FULL_EPOCHS,
    FULL_TRAINING_STEPS,
    FULL_WARMUP_STEPS,
    HELDOUT_SEEDS,
    SCREEN_EPOCHS,
    TUNING_SEEDS,
    StudyController,
    atomic_json,
    condition_metrics,
)


RHO_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def rho_label(rho: float) -> str:
    return str(rho).replace(".", "p")


@dataclass(frozen=True)
class RNNTrial:
    trial_id: str
    seed: int
    max_epochs: int
    aux_weight: float
    rho: float
    phase: str
    tau: float = 1.0
    probe_only: bool = False
    activation: str = "tanh"
    recurrent_init: str = "orthogonal"
    recurrent_identity_scale: float = 1.0
    exclude_initial_memory_reconstruction: bool = True
    use_terminal_loss: bool = True
    chunk_size: int = 4
    condition_memory_reconstruction_on_boundary: bool = False
    track: str = "legacy"
    num_active_associations: int | None = None
    aux_chunk_sizes: tuple[int, ...] | None = None
    stop_gradient_memory_target: bool = False
    state_aux_distribution: str = "gaussian"
    vmf_kappa_mode: str = "fixed"
    memory_vmf_kappa: float = 1.0
    terminal_vmf_kappa: float = 1.0


def make_baseline_trials() -> List[RNNTrial]:
    return [
        RNNTrial(
            trial_id=f"rnn-noaux-parammatched-s{seed}",
            seed=seed,
            max_epochs=FULL_EPOCHS,
            # The inverse probe trains online, but every path from it into the
            # forward RNN is detached.  Thus this remains an exact no-forward-
            # AUX baseline while yielding matched reconstruction telemetry.
            aux_weight=0.1,
            rho=16.0,
            phase="baseline",
            probe_only=True,
        )
        for seed in BASELINE_SEEDS
    ]


def make_screen_trials(
    rhos: Iterable[float], max_epochs: int = SCREEN_EPOCHS
) -> List[RNNTrial]:
    return [
        RNNTrial(
            trial_id=f"rnn-aux-rho{rho_label(rho)}-s{seed}",
            seed=seed,
            max_epochs=max_epochs,
            aux_weight=0.1,
            rho=float(rho),
            phase="screen",
        )
        for rho in rhos
        for seed in TUNING_SEEDS
    ]


def make_heldout_trials(rho: float) -> List[RNNTrial]:
    return [
        RNNTrial(
            trial_id=f"rnn-aux-selected-rho{rho_label(rho)}-s{seed}",
            seed=seed,
            max_epochs=FULL_EPOCHS,
            aux_weight=0.1,
            rho=float(rho),
            phase="heldout",
        )
        for seed in HELDOUT_SEEDS
    ]


def build_rnn_command(
    trial: RNNTrial,
    output_root: Path,
    project: str,
    entity: str,
    group: str,
):
    trial_root = output_root / "trials" / trial.trial_id
    checkpoint_dir = trial_root / "checkpoints"
    checkpoint = checkpoint_dir / "last.ckpt"
    has_aux = trial.aux_weight != 0.0
    chunk_sizes = (
        "null"
        if trial.aux_chunk_sizes is None
        else "[" + ",".join(str(value) for value in trial.aux_chunk_sizes) + "]"
    )
    command = [
        sys.executable,
        "-m",
        "train",
        "experiment=synthetics/associative_recall/rnn_aux",
        f"trainer.max_epochs={trial.max_epochs}",
        "callbacks=study",
        "+trainer.check_val_every_n_epoch=5",
        "+trainer.num_sanity_val_steps=0",
        "trainer.log_every_n_steps=50",
        "trainer.limit_train_batches=1.0",
        "trainer.limit_val_batches=1.0",
        "+trainer.precision=32",
        "trainer.gradient_clip_val=0.0",
        "train.test=false",
        "train.fused_adamw=true",
        "train.log_model_metrics_on_step=false",
        f"train.seed={trial.seed}",
        f"train.model_seed={trial.seed}",
        f"train.runtime_seed={trial.seed}",
        f"+dataset.seed={trial.seed}",
        "dataset.num_active_associations="
        f"{trial.num_active_associations if trial.num_active_associations is not None else 'null'}",
        f"dataset.loader_seed={trial.seed}",
        "loader.num_workers=0",
        "scheduler=linear_warmup",
        f"scheduler.num_warmup_steps={FULL_WARMUP_STEPS}",
        f"scheduler.num_training_steps={FULL_TRAINING_STEPS}",
        "optimizer.lr=0.001",
        "optimizer.weight_decay=0.1",
        "model.d_model=64",
        "model.n_layer=3",
        f"model.chunk_size={trial.chunk_size}",
        f"model.aux_chunk_sizes={chunk_sizes}",
        "model.chunk_offset=random",
        "model.dropout=0.0",
        f"model.activation={trial.activation}",
        f"model.recurrent_init={trial.recurrent_init}",
        f"model.recurrent_identity_scale={trial.recurrent_identity_scale}",
        f"model.rho={trial.rho}",
        f"model.tau={trial.tau}",
        f"model.state_aux_distribution={trial.state_aux_distribution}",
        f"model.vmf_kappa_mode={trial.vmf_kappa_mode}",
        f"model.memory_vmf_kappa={trial.memory_vmf_kappa}",
        f"model.terminal_vmf_kappa={trial.terminal_vmf_kappa}",
        f"model.auxiliary_probe_only={str(trial.probe_only).lower()}",
        "model.stop_gradient_memory_target="
        f"{str(trial.stop_gradient_memory_target).lower()}",
        "model.stop_gradient_memory_observation=false",
        "model.memory_observation_gradient_scale=1.0",
        "model.use_chunk_loss=true",
        "model.use_discrete_loss=true",
        "model.use_memory_loss=true",
        "model.exclude_initial_memory_reconstruction="
        f"{str(trial.exclude_initial_memory_reconstruction).lower()}",
        f"model.use_terminal_loss={str(trial.use_terminal_loss).lower()}",
        "model.condition_memory_reconstruction_on_boundary="
        f"{str(trial.condition_memory_reconstruction_on_boundary).lower()}",
        f"task.aux_weight={trial.aux_weight}",
        f"task.aux_weight_final={trial.aux_weight}",
        "task.aux_weight_schedule=fixed",
        f"task.aux_gradient_norm_interval={1570 if has_aux else 0}",
        f"task.aux_diagnostic_interval={157 if has_aux else 0}",
        "wandb.mode=online",
        f"wandb.project={project}",
        f"+wandb.entity={entity}",
        f"wandb.group={group}",
        f"wandb.name={trial.trial_id}",
        f"wandb.id={group}-{trial.trial_id}",
        "+wandb.resume=allow",
        f"callbacks.model_checkpoint.dirpath={checkpoint_dir}",
        f"callbacks.study_summary.result_path={trial_root / 'result.json'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
    ]
    if checkpoint.exists():
        command.append(f"train.ckpt={checkpoint}")
    return command


def run_preflight(output_root: Path, gpu_id: int):
    preflight_root = output_root / "preflight"
    marker = preflight_root / "PREFLIGHT_OK"
    if marker.exists():
        return
    trial = RNNTrial(
        "rnn-aux-preflight",
        20260819,
        1,
        aux_weight=0.1,
        rho=1.0,
        phase="preflight",
    )
    command = build_rnn_command(trial, preflight_root, "unused", "unused", "unused")
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
    preflight_root.mkdir(parents=True, exist_ok=True)
    log_path = preflight_root / "preflight.log"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    trial_root = preflight_root / "trials" / trial.trial_id
    if (
        completed.returncode != 0
        or not (trial_root / "result.json").exists()
        or not (trial_root / "checkpoints" / "last.ckpt").exists()
    ):
        log = log_path.read_text(encoding="utf-8", errors="replace")
        print(
            f"RNN GPU preflight log ({log_path}):\n{log}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            "RNN GPU preflight failed: "
            f"returncode={completed.returncode}, "
            f"result={bool((trial_root / 'result.json').exists())}, "
            f"checkpoint={bool((trial_root / 'checkpoints' / 'last.ckpt').exists())}"
        )
    marker.write_text("ok\n", encoding="utf-8")


def _last_metrics(output_root: Path, trial_id: str):
    try:
        with (output_root / "trials" / trial_id / "result.json").open(
            encoding="utf-8"
        ) as stream:
            return json.load(stream)["last"]["metrics"]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def rank_rhos(
    output_root: Path, rhos: Iterable[float], seeds: Sequence[int] = TUNING_SEEDS
):
    ranked = []
    for rho in rhos:
        records = []
        for seed in seeds:
            trial_id = f"rnn-aux-rho{rho_label(rho)}-s{seed}"
            metrics = _last_metrics(output_root, trial_id)
            if metrics is None:
                break
            accuracy = metrics.get("val/accuracy_ignore_index")
            loss = metrics.get("val/loss")
            if accuracy is None or loss is None:
                break
            records.append((seed, float(accuracy), float(loss)))
        if len(records) != len(seeds):
            continue
        accuracies = [record[1] for record in records]
        losses = [record[2] for record in records]
        ranked.append(
            {
                "rho": float(rho),
                "records": records,
                "mean_accuracy": statistics.fmean(accuracies),
                "sample_sd_accuracy": statistics.stdev(accuracies),
                "mean_loss": statistics.fmean(losses),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["mean_accuracy"],
            row["mean_loss"],
            row["sample_sd_accuracy"],
            row["rho"],
        )
    )
    return ranked


def paired_bootstrap_ci(differences: Sequence[float], samples: int = 10000):
    generator = random.Random(20260819)
    means = []
    for _ in range(samples):
        draw = [generator.choice(differences) for _ in differences]
        means.append(statistics.fmean(draw))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def build_report(output_root: Path, baseline_trials, selected_trials, rho: float):
    baseline = condition_metrics(output_root, baseline_trials)
    selected = condition_metrics(output_root, selected_trials)
    baseline_by_seed = {row["seed"]: row for row in baseline["rows"]}
    differences = []
    wins = 0
    for row in selected["rows"]:
        baseline_row = baseline_by_seed.get(row["seed"])
        if baseline_row is None or row["accuracy"] is None:
            continue
        difference = float(row["accuracy"]) - float(baseline_row["accuracy"])
        differences.append(difference)
        wins += difference > 0
    return {
        "condition": {
            "model": "vanilla_tanh_rnn",
            "d_model": 64,
            "n_layer": 3,
            "forward_parameters": 26432,
            "epochs": FULL_EPOCHS,
            "training_steps": FULL_TRAINING_STEPS,
            "warmup_steps": FULL_WARMUP_STEPS,
            "aux_weight": 0.1,
            "selected_rho": rho,
            "tau": 1.0,
            "target_sg": False,
            "terminal_loss": True,
            "terminal_target": "learned",
            "exclude_initial_memory_reconstruction": True,
            "baseline_inverse_probe": "online_detached",
            "seeds": list(BASELINE_SEEDS),
        },
        "baseline": baseline,
        "selected_aux": selected,
        "paired_accuracy_delta": (
            statistics.fmean(differences) if differences else None
        ),
        "paired_wins": wins,
        "paired_count": len(differences),
        "paired_bootstrap_95_ci": (
            paired_bootstrap_ci(differences) if differences else None
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument(
        "--wandb-group", default="ar-rnn-noaux-aux-scale-20260819-v1"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_trials = make_baseline_trials()
    screen_trials = make_screen_trials(RHO_GRID)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "baseline_trials": [asdict(trial) for trial in baseline_trials],
                    "screen_trials": [asdict(trial) for trial in screen_trials],
                    "sample_command": build_rnn_command(
                        screen_trials[0],
                        args.output_root,
                        args.wandb_project,
                        args.wandb_entity,
                        args.wandb_group,
                    ),
                },
                indent=2,
            )
        )
        return 0
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY secret was not injected")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is unavailable")
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")

    run_preflight(args.output_root, gpu_ids[0])
    if args.preflight_only:
        print(args.output_root / "preflight" / "PREFLIGHT_OK")
        return 0
    controller = StudyController(
        output_root=args.output_root,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        command_builder=build_rnn_command,
        dry_run=False,
    )

    baseline_result = {}

    def run_baselines():
        baseline_result.update(controller.run_trials(baseline_trials))

    baseline_thread = threading.Thread(target=run_baselines, daemon=False)
    baseline_thread.start()
    controller.run_trials(screen_trials)
    screen_ranking = rank_rhos(args.output_root, RHO_GRID)
    if len(screen_ranking) < 2:
        raise RuntimeError("fewer than two viable RNN AUX scales")
    atomic_json(args.output_root / "screen-ranking.json", screen_ranking)

    shortlist_rhos = [row["rho"] for row in screen_ranking[:2]]
    shortlist_trials = [
        replace(trial, max_epochs=FULL_EPOCHS, phase="shortlist")
        for trial in screen_trials
        if trial.rho in shortlist_rhos
    ]
    controller.run_trials(shortlist_trials)
    full_ranking = rank_rhos(args.output_root, shortlist_rhos)
    if not full_ranking:
        raise RuntimeError("no viable full RNN AUX scale")
    atomic_json(args.output_root / "full-ranking.json", full_ranking)
    selected_rho = full_ranking[0]["rho"]

    heldout_trials = make_heldout_trials(selected_rho)
    heldout_results = controller.run_trials(heldout_trials)
    if not all(heldout_results.get(trial.trial_id, False) for trial in heldout_trials):
        raise RuntimeError("one or more held-out RNN AUX trials failed")
    baseline_thread.join()
    if not all(baseline_result.get(trial.trial_id, False) for trial in baseline_trials):
        raise RuntimeError("one or more RNN no-AUX baselines failed")

    tuning_selected = [
        trial for trial in shortlist_trials if trial.rho == selected_rho
    ]
    selected_trials = tuning_selected + heldout_trials
    report_path = args.output_root / "study-summary.json"
    atomic_json(
        report_path,
        build_report(args.output_root, baseline_trials, selected_trials, selected_rho),
    )
    (args.output_root / "STUDY_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
