#!/usr/bin/env python3
"""Automate separate GG versus VV-fixed/VV-learned recurrent-state losses."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from run_ar_rmt_scale_study import HELDOUT_SEEDS, SCREEN_EPOCHS, TUNING_SEEDS, StudyController, atomic_json
from run_ar_rnn_controlled_lag_study import run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command
from src.models.sequence.rnn_aux import RNNAuxLM


FULL_EPOCHS = 400
ACTIVE_ASSOCIATIONS = 5


def core_options(core: str) -> dict:
    if core == "irnn":
        return dict(activation="relu", recurrent_init="identity", aux_weight=0.1, rho=8.0, tau=243.242356581615)
    if core == "tanh":
        return dict(activation="tanh", recurrent_init="orthogonal", aux_weight=0.05, rho=16.0, tau=47.799047874118955)
    raise ValueError(core)


def make_trial(core, condition, seed, kappas, phase, max_epochs):
    options = core_options(core)
    distribution = "gaussian" if condition == "gg" else "vmf"
    mode = "learned" if condition == "vv-learned" else "fixed"
    return RNNTrial(
        trial_id=f"{core}-{condition}-{phase}-s{seed}",
        seed=seed,
        max_epochs=max_epochs,
        aux_weight=options["aux_weight"],
        rho=options["rho"],
        tau=options["tau"],
        phase=f"state_distribution_{phase}",
        activation=options["activation"],
        recurrent_init=options["recurrent_init"],
        recurrent_identity_scale=1.0,
        chunk_size=4,
        aux_chunk_sizes=(4,),
        state_aux_distribution=distribution,
        vmf_kappa_mode=mode,
        memory_vmf_kappa=kappas[core]["memory"],
        terminal_vmf_kappa=kappas[core]["terminal"],
        track=f"{core}-{condition}",
        num_active_associations=ACTIVE_ASSOCIATIONS,
    )


def forward_parameters(model):
    return (model.embedding.weight, model.initial_state, *tuple(model.rnn.parameters()))


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    squares = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    return torch.stack(squares).sum().sqrt().item() if squares else 0.0


def calibration_model(core, distribution, component, seed, device):
    options = core_options(core)
    torch.manual_seed(seed)
    model = RNNAuxLM(
        d_model=64, n_layer=3, vocab_size=20,
        chunk_size=4, aux_chunk_sizes=(4,), chunk_offset=0,
        activation=options["activation"], recurrent_init=options["recurrent_init"],
        recurrent_identity_scale=1.0, rho=options["rho"], tau=options["tau"],
        state_aux_distribution=distribution, vmf_kappa_mode="fixed",
        memory_vmf_kappa=1.0, terminal_vmf_kappa=1.0,
        use_chunk_loss=False, use_discrete_loss=False,
        use_memory_loss=component == "memory",
        use_terminal_loss=component == "terminal",
    ).to(device)
    return model


def calibrate_core(core, device):
    ratios = {"memory": [], "terminal": []}
    for component in ratios:
        for seed in TUNING_SEEDS:
            gg = calibration_model(core, "gaussian", component, seed, device)
            vv = calibration_model(core, "vmf", component, seed, device)
            shared = {
                name: value for name, value in gg.state_dict().items()
                if name != "terminal_target"
            }
            vv.load_state_dict(shared, strict=False)
            generator = torch.Generator(device="cpu").manual_seed(seed + 991)
            inputs = torch.randint(0, 20, (32, 42), generator=generator).to(device)
            targets = torch.randint(0, 20, (32, 42), generator=generator).to(device)
            gg_output, _ = gg(inputs, targets=targets, aux_tokens=targets)
            vv_output, _ = vv(inputs, targets=targets, aux_tokens=targets)
            gg_norm = gradient_norm(gg.loss_components[f"{component}_nll"], forward_parameters(gg))
            vv_norm = gradient_norm(vv.loss_components[f"{component}_nll"], forward_parameters(vv))
            if not math.isfinite(gg_norm) or not math.isfinite(vv_norm) or vv_norm <= 0:
                raise RuntimeError(f"invalid {core} {component} vMF calibration")
            ratios[component].append(gg_norm / vv_norm)
    return {
        component: min(max(statistics.median(values), 1e-4), 1e4)
        for component, values in ratios.items()
    } | {"raw_ratios": ratios}


def calibrate(device):
    return {core: calibrate_core(core, device) for core in ("irnn", "tanh")}


def report_for(root: Path, trial: RNNTrial):
    return json.loads((root / "controlled_lag" / f"{trial.trial_id}.json").read_text())


def long_accuracy(report, minimum=22):
    return statistics.fmean(row["accuracy"] for row in report["by_lag"] if row["lag"] >= minimum)


def select_vv(trials, root):
    selection = {}
    for core in ("irnn", "tanh"):
        rows = []
        for condition in ("gg", "vv-fixed", "vv-learned"):
            members = [trial for trial in trials if trial.track == f"{core}-{condition}"]
            reports = [report_for(root, trial) for trial in members]
            rows.append({
                "condition": condition,
                "lag_22_40": statistics.fmean(long_accuracy(row, 22) for row in reports),
                "lag_32_40": statistics.fmean(long_accuracy(row, 32) for row in reports),
                "overall_nll": statistics.fmean(row["overall"]["nll"] for row in reports),
            })
        vv = [row for row in rows if row["condition"].startswith("vv-")]
        chosen = max(vv, key=lambda row: (row["lag_22_40"], row["lag_32_40"], -row["overall_nll"]))
        selection[core] = {"selected_vv": chosen["condition"], "tuning": rows}
    return selection


def eval_args(args, group):
    values = vars(args).copy()
    values["eval_group"] = group
    return argparse.Namespace(**values)


def run_phase(trials, root, args, train_group, eval_group, gpu_ids):
    controller = StudyController(
        root, gpu_ids, args.workers_per_gpu,
        args.wandb_project, args.wandb_entity, train_group,
        command_builder=build_rnn_command, dry_run=args.dry_run,
    )
    status = controller.run_trials(trials)
    if not all(status.get(trial.trial_id, False) for trial in trials):
        raise RuntimeError(f"state-distribution training failed: {train_group}")
    if not args.dry_run:
        run_evaluations(trials, root, gpu_ids, eval_args(args, eval_group))


def run_preflight(root: Path, gpu_id: int):
    marker = root / "PREFLIGHT_OK"
    if marker.exists():
        return
    kappas = {
        "irnn": {"memory": 3.0, "terminal": 5.0},
        "tanh": {"memory": 0.04, "terminal": 0.01},
    }
    trials = [
        make_trial("irnn", "vv-learned", 20260821, kappas, "preflight", 1),
        make_trial("tanh", "gg", 20260822, kappas, "preflight", 1),
    ]

    def command_builder(trial, output_root, project, entity, group):
        command = build_rnn_command(trial, output_root, project, entity, group)
        replacements = {
            "+trainer.check_val_every_n_epoch=5": "+trainer.check_val_every_n_epoch=1",
            "trainer.limit_train_batches=1.0": "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=1.0": "trainer.limit_val_batches=2",
            "wandb.mode=online": "wandb.mode=disabled",
            "+wandb.entity=unused": "+wandb.entity=null",
            "+wandb.resume=allow": "+wandb.resume=null",
        }
        return [replacements.get(argument, argument) for argument in command]

    controller = StudyController(
        root, (gpu_id,), 1, "unused", "unused", "unused",
        command_builder=command_builder, dry_run=False,
    )
    status = controller.run_trials(trials)
    if not all(status.get(trial.trial_id, False) for trial in trials):
        for trial in trials:
            log = root / "trials" / trial.trial_id / "train.log"
            if log.exists():
                print(
                    f"GG/VV preflight log ({log}):\n"
                    + log.read_text(encoding="utf-8", errors="replace")[-20000:],
                    file=sys.stderr,
                    flush=True,
                )
        raise RuntimeError("GG/VV GPU preflight failed")
    marker.write_text("ok\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument("--group-prefix", required=True)
    parser.add_argument("--examples-per-lag", type=int, default=10000)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=20260821)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)
    if not args.dry_run:
        run_preflight(args.output_root / "preflight", gpu_ids[0])
        if args.preflight_only:
            return 0
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    calibration = calibrate(device)
    kappas = {
        core: {"memory": calibration[core]["memory"], "terminal": calibration[core]["terminal"]}
        for core in calibration
    }
    atomic_json(args.output_root / "calibration.json", calibration)
    tuning = [
        make_trial(core, condition, seed, kappas, "tuning", SCREEN_EPOCHS)
        for core in ("irnn", "tanh")
        for condition in ("gg", "vv-fixed", "vv-learned")
        for seed in TUNING_SEEDS
    ]
    groups = {phase: f"{args.group_prefix}-{phase}" for phase in ("tuning-train", "tuning-eval", "heldout-train", "heldout-eval")}
    atomic_json(args.output_root / "plan.json", {"groups": groups, "kappas": kappas, "tuning": [asdict(t) for t in tuning]})
    run_phase(tuning, args.output_root / "tuning", args, groups["tuning-train"], groups["tuning-eval"], gpu_ids)
    if args.dry_run:
        return 0
    selection = select_vv(tuning, args.output_root / "tuning")
    atomic_json(args.output_root / "selection.json", selection)
    heldout = [
        make_trial(core, condition, seed, kappas, "heldout", FULL_EPOCHS)
        for core in ("irnn", "tanh")
        for condition in ("gg", selection[core]["selected_vv"])
        for seed in HELDOUT_SEEDS
    ]
    run_phase(heldout, args.output_root / "heldout", args, groups["heldout-train"], groups["heldout-eval"], gpu_ids)
    atomic_json(args.output_root / "study.json", {"calibration": calibration, "selection": selection, "heldout": [asdict(t) for t in heldout]})
    (args.output_root / "VMF_STUDY_COMPLETE").write_text("complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
