#!/usr/bin/env python3
"""Run independent tanh-objective and iRNN studies in one bounded Vessl run."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from run_ar_rmt_scale_study import (
    FULL_EPOCHS,
    FULL_TRAINING_STEPS,
    FULL_WARMUP_STEPS,
    HELDOUT_SEEDS,
    TUNING_SEEDS,
    StudyController,
    atomic_json,
    condition_metrics,
    paired_bootstrap_ci,
)
from run_ar_rnn_study import RNNTrial, _last_metrics, build_rnn_command
from src.models.sequence.rnn_aux import RNNAuxLM


SCREEN_EPOCHS = 150
SHORTLIST_EPOCHS = 250
RHO = 16.0
LAMBDA_GRID = (0.025, 0.05, 0.1, 0.2, 0.4)
# Terminal pressure is deliberately kept negligible.  These are weighted
# terminal/LM forward-gradient ratios (0.01%, 0.03%, 0.08%), all strictly
# below 0.1%.  Tau is recomputed for every lambda so the terminal-pressure
# axis and the total-AUX-weight axis remain independent.
TERMINAL_RATIO_TARGETS = (0.0001, 0.0003, 0.0008)
DEFAULT_TERMINAL_RATIO_TARGET = 0.0003
REFERENCE_AUX_WEIGHT = 0.1
EXPECTED_UNIQUE_RUNS = 64


@dataclass(frozen=True)
class CoreSpec:
    label: str
    activation: str
    recurrent_init: str
    recurrent_identity_scale: float = 1.0


CORE_SPECS = (
    CoreSpec("tanh-orthogonal", "tanh", "orthogonal"),
    CoreSpec("relu-orthogonal", "relu", "orthogonal"),
    CoreSpec("irnn-alpha0p9", "relu", "identity", 0.9),
    CoreSpec("irnn-alpha1p0", "relu", "identity", 1.0),
)


def value_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def ratio_label(value: float) -> str:
    return f"{value * 100:g}pct".replace(".", "p")


def core_kwargs(core: CoreSpec) -> dict:
    return {
        "activation": core.activation,
        "recurrent_init": core.recurrent_init,
        "recurrent_identity_scale": core.recurrent_identity_scale,
    }


def gradient_norm(gradients: Iterable[torch.Tensor | None]) -> float:
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def terminal_ratio_at_unit_tau(core: CoreSpec, seed: int) -> float:
    """Measure the initial weighted terminal/LM forward-gradient ratio."""
    torch.manual_seed(seed)
    model = RNNAuxLM(
        d_model=64,
        n_layer=3,
        vocab_size=20,
        chunk_size=4,
        chunk_offset=0,
        dropout=0.0,
        rho=RHO,
        tau=1.0,
        exclude_initial_memory_reconstruction=True,
        use_terminal_loss=True,
        **core_kwargs(core),
    )
    model.train()
    generator = torch.Generator().manual_seed(seed + 10000)
    inputs = torch.randint(0, 20, (32, 41), generator=generator)
    targets = torch.randint(0, 20, (32, 41), generator=generator)
    output, _ = model(
        inputs,
        targets=targets,
        aux_tokens=targets,
        compute_aux=True,
        compute_diagnostics=False,
    )
    lm_loss = F.cross_entropy(
        output.logits.flatten(0, 1), targets.flatten()
    )
    forward_parameters = (
        model.embedding.weight,
        model.initial_state,
        *tuple(model.rnn.parameters()),
    )
    lm_gradients = torch.autograd.grad(
        lm_loss, forward_parameters, retain_graph=True, allow_unused=True
    )
    terminal_gradients = torch.autograd.grad(
        0.1 * model.loss_components["terminal_nll"],
        forward_parameters,
        allow_unused=True,
    )
    lm_norm = gradient_norm(lm_gradients)
    if lm_norm == 0.0:
        raise RuntimeError(f"zero initial LM gradient for {core.label}")
    return gradient_norm(terminal_gradients) / lm_norm


def calibrate_terminal_scales(
    cores: Sequence[CoreSpec] = CORE_SPECS,
    seeds: Sequence[int] = TUNING_SEEDS,
) -> dict:
    calibration = {}
    for core in cores:
        ratios = [terminal_ratio_at_unit_tau(core, seed) for seed in seeds]
        reference_ratio = statistics.median(ratios)
        if not math.isfinite(reference_ratio) or reference_ratio <= 0.0:
            raise RuntimeError(f"invalid terminal calibration for {core.label}")
        calibration[core.label] = {
            "unit_tau_weighted_terminal_lm_ratios": ratios,
            "median_unit_tau_ratio": reference_ratio,
            "reference_aux_weight": REFERENCE_AUX_WEIGHT,
        }
    return calibration


def calibrated_tau(
    calibration: dict,
    core: CoreSpec,
    target_ratio: float,
    aux_weight: float,
) -> float:
    """Approximate tau from the existing initial-gradient probe.

    ``median_unit_tau_ratio`` was measured with REFERENCE_AUX_WEIGHT.  The
    terminal gradient scales linearly with lambda and approximately with
    inverse tau squared, so this keeps the requested terminal pressure fixed
    while lambda is varied.
    """
    reference_ratio = calibration[core.label]["median_unit_tau_ratio"]
    scaled_ratio = reference_ratio * aux_weight / REFERENCE_AUX_WEIGHT
    return min(max(math.sqrt(scaled_ratio / target_ratio), 0.25), 1024.0)


def make_trial(
    trial_id: str,
    seed: int,
    max_epochs: int,
    aux_weight: float,
    tau: float,
    phase: str,
    track: str,
    core: CoreSpec,
    probe_only: bool = False,
    exclude_initial_memory_reconstruction: bool = True,
    use_terminal_loss: bool = True,
) -> RNNTrial:
    return RNNTrial(
        trial_id=trial_id,
        seed=seed,
        max_epochs=max_epochs,
        aux_weight=aux_weight,
        rho=RHO,
        tau=tau,
        phase=phase,
        probe_only=probe_only,
        exclude_initial_memory_reconstruction=(
            exclude_initial_memory_reconstruction
        ),
        use_terminal_loss=use_terminal_loss,
        track=track,
        **core_kwargs(core),
    )


def make_tanh_screen(calibration: dict) -> Dict[str, List[RNNTrial]]:
    core = CORE_SPECS[0]
    candidates = {}
    for target_ratio in TERMINAL_RATIO_TARGETS:
        target_name = ratio_label(target_ratio)
        for aux_weight in LAMBDA_GRID:
            tau = calibrated_tau(
                calibration, core, target_ratio, aux_weight
            )
            candidate = f"tr{target_name}-lam{value_label(aux_weight)}"
            candidates[candidate] = [
                make_trial(
                    f"tanh-screen-{candidate}-s{seed}",
                    seed,
                    SCREEN_EPOCHS,
                    aux_weight,
                    tau,
                    "tanh_screen",
                    "tanh",
                    core,
                )
                for seed in TUNING_SEEDS
            ]
    return candidates


def make_core_screen(calibration: dict) -> Dict[str, List[RNNTrial]]:
    candidates = {}
    for core in CORE_SPECS:
        tau = calibrated_tau(
            calibration, core, DEFAULT_TERMINAL_RATIO_TARGET, 0.1
        )
        candidates[core.label] = [
            make_trial(
                f"core-screen-{core.label}-s{seed}",
                seed,
                SCREEN_EPOCHS,
                0.1,
                tau,
                "core_screen",
                "irnn",
                core,
                probe_only=True,
            )
            for seed in TUNING_SEEDS
        ]
    return candidates


def make_irnn_tau_screen(
    calibration: dict, core: CoreSpec
) -> Dict[str, List[RNNTrial]]:
    candidates = {}
    for target_ratio in TERMINAL_RATIO_TARGETS:
        target_name = ratio_label(target_ratio)
        tau = calibrated_tau(calibration, core, target_ratio, 0.1)
        candidate = f"tr{target_name}"
        candidates[candidate] = [
            make_trial(
                f"irnn-aux-screen-{core.label}-{candidate}-s{seed}",
                seed,
                SCREEN_EPOCHS,
                0.1,
                tau,
                "irnn_aux_screen",
                "irnn",
                core,
            )
            for seed in TUNING_SEEDS
        ]
    return candidates


def flatten_candidates(candidates: Dict[str, Sequence[RNNTrial]]) -> List[RNNTrial]:
    return [trial for trials in candidates.values() for trial in trials]


def rank_candidates(output_root: Path, candidates: dict) -> List[dict]:
    ranked = []
    for label, trials in candidates.items():
        rows = []
        for trial in trials:
            metrics = _last_metrics(output_root, trial.trial_id)
            if metrics is None:
                break
            accuracy = metrics.get("val/accuracy_ignore_index")
            loss = metrics.get("val/loss")
            if accuracy is None or loss is None:
                break
            rows.append(
                {
                    "seed": trial.seed,
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                }
            )
        if len(rows) != len(trials):
            continue
        accuracies = [row["accuracy"] for row in rows]
        losses = [row["loss"] for row in rows]
        ranked.append(
            {
                "label": label,
                "rows": rows,
                "mean_accuracy": statistics.fmean(accuracies),
                "sample_sd_accuracy": statistics.stdev(accuracies),
                "min_accuracy": min(accuracies),
                "mean_loss": statistics.fmean(losses),
                "trial": asdict(trials[0]),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["mean_accuracy"],
            -row["min_accuracy"],
            row["mean_loss"],
            row["sample_sd_accuracy"],
            row["label"],
        )
    )
    return ranked


def require_trials(controller: StudyController, trials: Sequence[RNNTrial]) -> None:
    results = controller.run_trials(trials)
    failed = [trial.trial_id for trial in trials if not results.get(trial.trial_id)]
    if failed:
        raise RuntimeError("required trials failed: " + ", ".join(failed))


def paired_report(
    output_root: Path,
    aux_trials: Sequence[RNNTrial],
    noaux_trials: Sequence[RNNTrial],
) -> dict:
    aux = condition_metrics(output_root, aux_trials)
    noaux = condition_metrics(output_root, noaux_trials)
    noaux_by_seed = {row["seed"]: row for row in noaux["rows"]}
    differences = []
    for row in aux["rows"]:
        baseline = noaux_by_seed.get(row["seed"])
        if baseline is not None and row["accuracy"] is not None:
            differences.append(
                float(row["accuracy"]) - float(baseline["accuracy"])
            )
    return {
        "aux": aux,
        "detached_noaux": noaux,
        "paired_accuracy_delta": (
            statistics.fmean(differences) if differences else None
        ),
        "paired_wins": sum(difference > 0 for difference in differences),
        "paired_count": len(differences),
        "paired_bootstrap_95_ci": (
            paired_bootstrap_ci(differences) if differences else None
        ),
    }


def build_two_track_command(
    trial: RNNTrial,
    output_root: Path,
    project: str,
    entity: str,
    group_prefix: str,
):
    return build_rnn_command(
        trial,
        output_root,
        project,
        entity,
        f"{group_prefix}-{trial.track}",
    )


def run_preflight(output_root: Path, gpu_id: int) -> None:
    root = output_root / "preflight"
    marker = root / "PREFLIGHT_OK"
    if marker.exists():
        return
    tanh = CORE_SPECS[0]
    relu = CORE_SPECS[1]
    irnn = CORE_SPECS[3]
    trials = (
        make_trial("preflight-tanh-aux", 9101, 1, 0.1, 16.0, "preflight", "tanh", tanh),
        make_trial(
            "preflight-tanh-probe", 9102, 1, 0.1, 16.0, "preflight", "tanh", tanh, probe_only=True
        ),
        make_trial(
            "preflight-relu-orth", 9103, 1, 0.1, 16.0, "preflight", "irnn", relu, probe_only=True
        ),
        make_trial("preflight-irnn", 9104, 1, 0.1, 16.0, "preflight", "irnn", irnn),
    )
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
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    for trial in trials:
        command = build_two_track_command(
            trial, root, "unused", "unused", "unused"
        )
        command = [replacements.get(argument, argument) for argument in command]
        trial_root = root / "trials" / trial.trial_id
        trial_root.mkdir(parents=True, exist_ok=True)
        with (trial_root / "train.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if (
            completed.returncode != 0
            or _last_metrics(root, trial.trial_id) is None
            or not (trial_root / "checkpoints" / "last.ckpt").exists()
        ):
            raise RuntimeError(f"preflight failed: {trial.trial_id}")
    marker.write_text("ok\n", encoding="utf-8")


def run_tanh_track(
    controller: StudyController,
    output_root: Path,
    screen: Dict[str, List[RNNTrial]],
) -> dict:
    ranking = rank_candidates(output_root, screen)
    if len(ranking) < 2:
        raise RuntimeError("fewer than two viable tanh configurations")
    shortlist_labels = [row["label"] for row in ranking[:2]]
    shortlist = {
        label: [
            replace(trial, max_epochs=SHORTLIST_EPOCHS, phase="tanh_shortlist")
            for trial in screen[label]
        ]
        for label in shortlist_labels
    }
    require_trials(controller, flatten_candidates(shortlist))
    shortlist_ranking = rank_candidates(output_root, shortlist)
    if not shortlist_ranking:
        raise RuntimeError("no viable extended tanh configuration")
    selected_label = shortlist_ranking[0]["label"]
    selected = shortlist[selected_label][0]
    core = CORE_SPECS[0]
    ablations = [
        make_trial(
            f"tanh-ablation-m0-on-s{seed}",
            seed,
            SCREEN_EPOCHS,
            selected.aux_weight,
            selected.tau,
            "tanh_ablation_m0_on",
            "tanh",
            core,
            exclude_initial_memory_reconstruction=False,
        )
        for seed in TUNING_SEEDS
    ] + [
        make_trial(
            f"tanh-ablation-terminal-off-s{seed}",
            seed,
            SCREEN_EPOCHS,
            selected.aux_weight,
            selected.tau,
            "tanh_ablation_terminal_off",
            "tanh",
            core,
            use_terminal_loss=False,
        )
        for seed in TUNING_SEEDS
    ]
    aux_final = [
        make_trial(
            f"tanh-final-aux-s{seed}",
            seed,
            FULL_EPOCHS,
            selected.aux_weight,
            selected.tau,
            "tanh_final_aux",
            "tanh",
            core,
        )
        for seed in HELDOUT_SEEDS
    ]
    noaux_final = [
        make_trial(
            f"tanh-final-noaux-probe-s{seed}",
            seed,
            FULL_EPOCHS,
            selected.aux_weight,
            selected.tau,
            "tanh_final_noaux",
            "tanh",
            core,
            probe_only=True,
        )
        for seed in HELDOUT_SEEDS
    ]
    require_trials(controller, ablations + aux_final + noaux_final)
    report = {
        "default": "no-M0 reconstruction + learned terminal ON",
        "selected": asdict(selected),
        "screen_ranking_150e": ranking,
        "shortlist_ranking_250e": shortlist_ranking,
        "ablations": {
            "m0_on": condition_metrics(output_root, ablations[:2]),
            "terminal_off": condition_metrics(output_root, ablations[2:]),
        },
        "final": paired_report(output_root, aux_final, noaux_final),
    }
    track_root = output_root / "tanh"
    atomic_json(track_root / "report.json", report)
    (track_root / "STUDY_COMPLETE").write_text("complete\n", encoding="utf-8")
    return report


def run_irnn_track(
    controller: StudyController,
    output_root: Path,
    core_screen: Dict[str, List[RNNTrial]],
    calibration: dict,
) -> dict:
    core_ranking = rank_candidates(output_root, core_screen)
    identity_rows = [
        row for row in core_ranking if row["label"].startswith("irnn-")
    ]
    if not identity_rows:
        raise RuntimeError("no viable identity-initialized ReLU RNN")
    selected_core = next(
        core for core in CORE_SPECS if core.label == identity_rows[0]["label"]
    )
    tau_screen = make_irnn_tau_screen(calibration, selected_core)
    controller.run_trials(flatten_candidates(tau_screen))
    tau_ranking = rank_candidates(output_root, tau_screen)
    if not tau_ranking:
        raise RuntimeError("no viable iRNN AUX terminal scale")
    selected_label = tau_ranking[0]["label"]
    extended = {
        selected_label: [
            replace(trial, max_epochs=SHORTLIST_EPOCHS, phase="irnn_aux_selected")
            for trial in tau_screen[selected_label]
        ]
    }
    require_trials(controller, extended[selected_label])
    extended_ranking = rank_candidates(output_root, extended)
    selected = extended[selected_label][0]
    aux_final = [
        make_trial(
            f"irnn-final-aux-s{seed}",
            seed,
            FULL_EPOCHS,
            0.1,
            selected.tau,
            "irnn_final_aux",
            "irnn",
            selected_core,
        )
        for seed in HELDOUT_SEEDS
    ]
    noaux_final = [
        make_trial(
            f"irnn-final-noaux-probe-s{seed}",
            seed,
            FULL_EPOCHS,
            0.1,
            selected.tau,
            "irnn_final_noaux",
            "irnn",
            selected_core,
            probe_only=True,
        )
        for seed in HELDOUT_SEEDS
    ]
    require_trials(controller, aux_final + noaux_final)
    report = {
        "core_ranking_150e": core_ranking,
        "selected_core": asdict(selected_core),
        "tau_ranking_150e": tau_ranking,
        "selected_tau_250e": extended_ranking,
        "default": "no-M0 reconstruction + learned terminal ON",
        "final": paired_report(output_root, aux_final, noaux_final),
    }
    track_root = output_root / "irnn"
    atomic_json(track_root / "report.json", report)
    (track_root / "STUDY_COMPLETE").write_text("complete\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument(
        "--wandb-group-prefix", default="ar-rnn-tanh-irnn-auto5h-20260819-v1"
    )
    parser.add_argument("--max-runtime-minutes", type=float, default=285.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    calibration = calibrate_terminal_scales()
    tanh_screen = make_tanh_screen(calibration)
    core_screen = make_core_screen(calibration)
    plan = {
        "expected_unique_wandb_runs": EXPECTED_UNIQUE_RUNS,
        "groups": [
            f"{args.wandb_group_prefix}-tanh",
            f"{args.wandb_group_prefix}-irnn",
        ],
        "calibration": calibration,
        "tanh_screen": {
            label: [asdict(trial) for trial in trials]
            for label, trials in tanh_screen.items()
        },
        "core_screen": {
            label: [asdict(trial) for trial in trials]
            for label, trials in core_screen.items()
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY secret was not injected")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is unavailable")
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)
    if len(gpu_ids) != 2:
        raise ValueError("the automated study requires exactly two GPUs")
    if args.workers_per_gpu != 8:
        raise ValueError("the validated study uses exactly eight workers per GPU")

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "study-plan.json", plan)
    run_preflight(args.output_root, gpu_ids[0])
    if args.preflight_only:
        print(args.output_root / "preflight" / "PREFLIGHT_OK")
        return 0

    deadline = started + args.max_runtime_minutes * 60.0
    controller = StudyController(
        output_root=args.output_root,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group_prefix,
        command_builder=build_two_track_command,
        deadline_monotonic=deadline,
    )
    # Only this initial scheduling barrier is shared. Subsequent selections and
    # continuations are independent and execute concurrently.
    controller.run_trials(
        flatten_candidates(tanh_screen) + flatten_candidates(core_screen)
    )

    reports = {}
    errors = {}

    def tanh_worker():
        try:
            reports["tanh"] = run_tanh_track(
                controller, args.output_root, tanh_screen
            )
        except Exception as error:
            errors["tanh"] = repr(error)

    def irnn_worker():
        try:
            reports["irnn"] = run_irnn_track(
                controller, args.output_root, core_screen, calibration
            )
        except Exception as error:
            errors["irnn"] = repr(error)

    threads = (
        threading.Thread(target=tanh_worker, daemon=False),
        threading.Thread(target=irnn_worker, daemon=False),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    combined = {
        "reports": reports,
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "deadline_minutes": args.max_runtime_minutes,
    }
    atomic_json(args.output_root / "combined-report.json", combined)
    if errors:
        raise RuntimeError("track failures: " + json.dumps(errors, sort_keys=True))
    (args.output_root / "STUDY_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
