#!/usr/bin/env python3
"""Automate AR core difficulty, multi-chunk AUX, and iRNN target-SG tests."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from run_ar_rmt_scale_study import HELDOUT_SEEDS, SCREEN_EPOCHS, TUNING_SEEDS, StudyController, atomic_json
from run_ar_rnn_controlled_lag_study import run_evaluations
from run_ar_rnn_study import RNNTrial, build_rnn_command


K_GRID = (3, 4, 5, 6)
CHUNK_VARIANTS = ((2,), (3,), (2, 4), (1, 4))
FULL_EPOCHS = 400


def core_options(core: str) -> dict:
    if core == "irnn":
        return dict(
            activation="relu", recurrent_init="identity",
            recurrent_identity_scale=1.0, aux_weight=0.1,
            rho=8.0, tau=243.242356581615,
        )
    if core == "tanh":
        return dict(
            activation="tanh", recurrent_init="orthogonal",
            recurrent_identity_scale=1.0, aux_weight=0.05,
            rho=16.0, tau=47.799047874118955,
        )
    raise ValueError(core)


def chunk_label(values: tuple[int, ...]) -> str:
    return "p".join(str(value) for value in values)


def make_trial(
    core: str,
    k: int,
    condition: str,
    seed: int,
    *,
    chunks: tuple[int, ...] = (4,),
    max_epochs: int = SCREEN_EPOCHS,
    target_sg: bool = False,
) -> RNNTrial:
    options = core_options(core)
    probe = condition == "noaux"
    scale_count = len(chunks)
    aux_weight = options["aux_weight"] / scale_count
    return RNNTrial(
        trial_id=(
            f"{core}-k{k}-{condition}-c{chunk_label(chunks)}"
            f"-{'sgon' if target_sg else 'sgoff'}-s{seed}"
        ),
        seed=seed,
        max_epochs=max_epochs,
        aux_weight=aux_weight,
        rho=options["rho"],
        tau=options["tau"],
        phase="k_chunk_sg",
        probe_only=probe,
        activation=options["activation"],
        recurrent_init=options["recurrent_init"],
        recurrent_identity_scale=options["recurrent_identity_scale"],
        chunk_size=chunks[0],
        aux_chunk_sizes=chunks,
        stop_gradient_memory_target=target_sg,
        track=f"{core}-k{k}-{condition}-c{chunk_label(chunks)}",
        num_active_associations=k,
    )


def difficulty_trials() -> list[RNNTrial]:
    return [
        make_trial(core, k, condition, seed)
        for core in ("irnn", "tanh")
        for k in K_GRID
        for condition in ("noaux", "aux")
        for seed in TUNING_SEEDS
    ]


def chunk_trials(selected_k: dict[str, int]) -> list[RNNTrial]:
    return [
        make_trial(core, selected_k[core], "aux", seed, chunks=chunks)
        for core in ("irnn", "tanh")
        for chunks in CHUNK_VARIANTS
        for seed in TUNING_SEEDS
    ]


def heldout_trials(selection: dict) -> list[RNNTrial]:
    trials = []
    for core in ("irnn", "tanh"):
        k = selection[core]["selected_k"]
        chunks = tuple(selection[core]["selected_chunks"])
        for seed in HELDOUT_SEEDS:
            trials.append(
                make_trial(
                    core, k, "noaux", seed, chunks=chunks,
                    max_epochs=FULL_EPOCHS,
                )
            )
            trials.append(
                make_trial(
                    core, k, "aux", seed, chunks=chunks,
                    max_epochs=FULL_EPOCHS,
                )
            )
    return trials


def sg_trials() -> list[RNNTrial]:
    return [
        make_trial(
            "irnn", 5, "aux", seed, chunks=(4,),
            max_epochs=FULL_EPOCHS, target_sg=True,
        )
        for seed in HELDOUT_SEEDS
    ]


def report_for(root: Path, trial: RNNTrial) -> dict:
    return json.loads(
        (root / "controlled_lag" / f"{trial.trial_id}.json").read_text()
    )


def long_accuracy(report: dict, minimum=22) -> float:
    return statistics.fmean(
        row["accuracy"] for row in report["by_lag"] if row["lag"] >= minimum
    )


def select_k(trials: list[RNNTrial], root: Path) -> tuple[dict[str, int], dict]:
    selected = {}
    detail = {}
    for core in ("irnn", "tanh"):
        rows = []
        for k in K_GRID:
            values = {}
            for condition in ("noaux", "aux"):
                members = [
                    trial for trial in trials
                    if trial.track == f"{core}-k{k}-{condition}-c4"
                ]
                values[condition] = [long_accuracy(report_for(root, t)) for t in members]
            row = {
                "k": k,
                "noaux": statistics.fmean(values["noaux"]),
                "aux": statistics.fmean(values["aux"]),
                "gap": statistics.fmean(values["aux"]) - statistics.fmean(values["noaux"]),
                "aux_min": min(values["aux"]),
            }
            row["near_solve"] = (
                row["aux"] >= 0.80 and row["aux_min"] >= 0.75
                and row["noaux"] <= 0.70 and row["gap"] >= 0.10
            )
            rows.append(row)
        eligible = [row for row in rows if row["near_solve"]]
        pool = eligible or rows
        choice = max(pool, key=lambda row: (row["gap"], row["k"]))
        selected[core] = choice["k"]
        detail[core] = {
            "selected_k": choice["k"],
            "criterion_satisfied": bool(eligible),
            "rows": rows,
        }
    return selected, detail


def select_chunks(
    difficulty: list[RNNTrial], difficulty_root: Path,
    chunks: list[RNNTrial], chunk_root: Path,
    k_detail: dict,
) -> dict:
    selection = {}
    for core in ("irnn", "tanh"):
        k = k_detail[core]["selected_k"]
        rows = []
        candidates = ((4,),) + CHUNK_VARIANTS
        for values in candidates:
            source_trials = difficulty if values == (4,) else chunks
            source_root = difficulty_root if values == (4,) else chunk_root
            members = [
                trial for trial in source_trials
                if trial.track == f"{core}-k{k}-aux-c{chunk_label(values)}"
            ]
            reports = [report_for(source_root, trial) for trial in members]
            rows.append(
                {
                    "chunks": list(values),
                    "lag_22_40": statistics.fmean(long_accuracy(row, 22) for row in reports),
                    "lag_32_40": statistics.fmean(long_accuracy(row, 32) for row in reports),
                    "overall_nll": statistics.fmean(row["overall"]["nll"] for row in reports),
                }
            )
        choice = max(
            rows, key=lambda row: (
                row["lag_22_40"], row["lag_32_40"], -row["overall_nll"]
            )
        )
        selection[core] = {
            **k_detail[core],
            "selected_chunks": choice["chunks"],
            "chunk_ranking": sorted(
                rows,
                key=lambda row: (-row["lag_22_40"], -row["lag_32_40"], row["overall_nll"]),
            ),
        }
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
        raise RuntimeError(f"training phase failed: {train_group}")
    if not args.dry_run:
        run_evaluations(trials, root, gpu_ids, eval_args(args, eval_group))


def run_preflight(root: Path, gpu_id: int):
    marker = root / "PREFLIGHT_OK"
    if marker.exists():
        return
    trials = [
        replace(make_trial("irnn", 5, "aux", 20260821, chunks=(1, 4)), max_epochs=1),
        replace(make_trial("irnn", 5, "aux", 20260822, target_sg=True), max_epochs=1),
    ]
    def command_builder(trial, output_root, project, entity, group):
        command = build_rnn_command(
            trial, output_root, project, entity, group
        )
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
        raise RuntimeError("multi-chunk/SG GPU preflight failed")
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
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)
    groups = {
        phase: f"{args.group_prefix}-{phase}"
        for phase in ("difficulty-train", "difficulty-eval", "chunk-train", "chunk-eval", "heldout-train", "heldout-eval", "sg-train", "sg-eval")
    }
    atomic_json(args.output_root / "plan.json", {"groups": groups, "k_grid": K_GRID, "chunk_variants": CHUNK_VARIANTS})
    if not args.dry_run:
        run_preflight(args.output_root / "preflight", gpu_ids[0])
    difficulty = difficulty_trials()
    difficulty_root = args.output_root / "difficulty"
    run_phase(difficulty, difficulty_root, args, groups["difficulty-train"], groups["difficulty-eval"], gpu_ids)
    if args.dry_run:
        return 0
    selected_k, k_detail = select_k(difficulty, difficulty_root)
    chunk = chunk_trials(selected_k)
    chunk_root = args.output_root / "chunk"
    run_phase(chunk, chunk_root, args, groups["chunk-train"], groups["chunk-eval"], gpu_ids)
    selection = select_chunks(difficulty, difficulty_root, chunk, chunk_root, k_detail)
    atomic_json(args.output_root / "selection.json", selection)
    heldout = heldout_trials(selection)
    run_phase(heldout, args.output_root / "heldout", args, groups["heldout-train"], groups["heldout-eval"], gpu_ids)
    sg = sg_trials()
    run_phase(sg, args.output_root / "sg", args, groups["sg-train"], groups["sg-eval"], gpu_ids)
    atomic_json(args.output_root / "study.json", {"selection": selection, "heldout": [asdict(t) for t in heldout], "sg": [asdict(t) for t in sg]})
    (args.output_root / "K_CHUNK_SG_COMPLETE").write_text("complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
