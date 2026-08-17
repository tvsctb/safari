#!/usr/bin/env python3
"""Run the complete associative-recall RMT scale study inside one Vessl Run."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


STEPS_PER_EPOCH = 157
FULL_EPOCHS = 400
SCREEN_EPOCHS = 150
FULL_TRAINING_STEPS = STEPS_PER_EPOCH * FULL_EPOCHS
FULL_WARMUP_STEPS = FULL_TRAINING_STEPS // 5
RHO_BASE = 31.622777
TAU_BASE = 11.925695
INITIAL_PRESSURES = (0.5, 1.0, 2.0)
TUNING_SEEDS = (202, 203)
HELDOUT_SEEDS = (204, 205, 206, 207)
BASELINE_SEEDS = TUNING_SEEDS + HELDOUT_SEEDS
INFRASTRUCTURE_MARKERS = (
    "cuda error",
    "driver shutting down",
    "connection reset",
    "connection timed out",
    "preempt",
    "socket timeout",
    "transport endpoint",
    "worker exited unexpectedly",
)
MODEL_FAILURE_MARKERS = ("nan", "not finite", "out of memory")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def pressure_label(value: float) -> str:
    return f"{round(value * 100):03d}"


def scale_from_pressure(base: float, pressure: float) -> float:
    return base / math.sqrt(pressure)


@dataclass(frozen=True, order=True)
class Scale:
    rho_pressure: float
    tau_pressure: float

    @property
    def rho(self) -> float:
        return scale_from_pressure(RHO_BASE, self.rho_pressure)

    @property
    def tau(self) -> float:
        return scale_from_pressure(TAU_BASE, self.tau_pressure)

    @property
    def label(self) -> str:
        return (
            f"rp{pressure_label(self.rho_pressure)}-"
            f"tp{pressure_label(self.tau_pressure)}"
        )


@dataclass(frozen=True)
class Trial:
    trial_id: str
    seed: int
    max_epochs: int
    aux_weight: float
    target_sg: bool
    scale: Scale
    phase: str
    use_terminal_loss: bool = True


def initial_scales() -> Tuple[Scale, ...]:
    return tuple(
        Scale(rho_pressure, tau_pressure)
        for rho_pressure in INITIAL_PRESSURES
        for tau_pressure in INITIAL_PRESSURES
    )


def outward_scales(best: Scale) -> Tuple[Scale, ...]:
    additions = set()
    rho_edge = None
    tau_edge = None
    if best.rho_pressure == min(INITIAL_PRESSURES):
        rho_edge = 0.25
    elif best.rho_pressure == max(INITIAL_PRESSURES):
        rho_edge = 4.0
    if best.tau_pressure == min(INITIAL_PRESSURES):
        tau_edge = 0.25
    elif best.tau_pressure == max(INITIAL_PRESSURES):
        tau_edge = 4.0
    if rho_edge is not None:
        additions.update(Scale(rho_edge, tau) for tau in INITIAL_PRESSURES)
    if tau_edge is not None:
        additions.update(Scale(rho, tau_edge) for rho in INITIAL_PRESSURES)
    if rho_edge is not None and tau_edge is not None:
        additions.add(Scale(rho_edge, tau_edge))
    return tuple(sorted(additions))


def make_screen_trials(
    scales: Iterable[Scale], target_sg: bool, max_epochs: int = SCREEN_EPOCHS
) -> List[Trial]:
    sg_label = "on" if target_sg else "off"
    return [
        Trial(
            trial_id=f"screen-targetsg-{sg_label}-{scale.label}-s{seed}",
            seed=seed,
            max_epochs=max_epochs,
            aux_weight=0.1,
            target_sg=target_sg,
            scale=scale,
            phase="screen",
        )
        for scale in scales
        for seed in TUNING_SEEDS
    ]


def make_baseline_trials() -> List[Trial]:
    center = Scale(1.0, 1.0)
    return [
        Trial(
            trial_id=f"noaux-s{seed}",
            seed=seed,
            max_epochs=FULL_EPOCHS,
            aux_weight=0.0,
            target_sg=False,
            scale=center,
            phase="baseline",
        )
        for seed in BASELINE_SEEDS
    ]


def make_heldout_trials(target_sg: bool, scale: Scale) -> List[Trial]:
    sg_label = "on" if target_sg else "off"
    return [
        Trial(
            trial_id=f"heldout-targetsg-{sg_label}-{scale.label}-s{seed}",
            seed=seed,
            max_epochs=FULL_EPOCHS,
            aux_weight=0.1,
            target_sg=target_sg,
            scale=scale,
            phase="heldout",
        )
        for seed in HELDOUT_SEEDS
    ]


def _last_metrics(result_path: Path) -> Optional[dict]:
    try:
        with result_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload["last"]["metrics"]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def rank_scales(
    output_root: Path,
    target_sg: bool,
    scales: Iterable[Scale],
    seeds: Sequence[int] = TUNING_SEEDS,
) -> List[dict]:
    sg_label = "on" if target_sg else "off"
    ranked = []
    for scale in sorted(set(scales)):
        records = []
        for seed in seeds:
            trial_id = f"screen-targetsg-{sg_label}-{scale.label}-s{seed}"
            metrics = _last_metrics(output_root / "trials" / trial_id / "result.json")
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
                "scale": asdict(scale),
                "rho": scale.rho,
                "tau": scale.tau,
                "records": records,
                "mean_accuracy": statistics.fmean(accuracies),
                "sample_sd_accuracy": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
                "mean_loss": statistics.fmean(losses),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["mean_accuracy"],
            row["mean_loss"],
            row["sample_sd_accuracy"],
            row["rho"],
            row["tau"],
        )
    )
    return ranked


def build_command(
    trial: Trial,
    output_root: Path,
    project: str,
    entity: str,
    group: str,
) -> List[str]:
    trial_root = output_root / "trials" / trial.trial_id
    checkpoint_dir = trial_root / "checkpoints"
    checkpoint = checkpoint_dir / "last.ckpt"
    command = [
        sys.executable,
        "-m",
        "train",
        "experiment=synthetics/associative_recall/rmt_aux",
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
        f"dataset.loader_seed={trial.seed}",
        "loader.num_workers=0",
        "scheduler=linear_warmup",
        f"scheduler.num_warmup_steps={FULL_WARMUP_STEPS}",
        f"scheduler.num_training_steps={FULL_TRAINING_STEPS}",
        "optimizer.lr=0.001",
        "optimizer.weight_decay=0.1",
        "model.d_model=32",
        "model.d_inner=128",
        "model.n_layer=2",
        "model.n_heads=1",
        "model.chunk_size=4",
        "model.num_memory_tokens=2",
        "model.use_chunk_loss=true",
        "model.use_discrete_loss=true",
        "model.use_memory_loss=true",
        f"model.use_terminal_loss={str(trial.use_terminal_loss).lower()}",
        f"model.learnable_terminal_target={str(trial.use_terminal_loss).lower()}",
        f"model.stop_gradient_memory_target={str(trial.target_sg).lower()}",
        "model.stop_gradient_memory_observation=false",
        "model.memory_observation_gradient_scale=1.0",
        f"model.rho={trial.scale.rho:.9f}",
        f"model.tau={trial.scale.tau:.9f}",
        f"task.aux_weight={trial.aux_weight}",
        f"task.aux_weight_final={trial.aux_weight}",
        "task.aux_weight_schedule=fixed",
        f"task.aux_gradient_norm_interval={1570 if trial.aux_weight else 0}",
        "task.aux_diagnostic_interval=157",
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


class StudyController:
    def __init__(
        self,
        output_root: Path,
        gpu_ids: Sequence[int],
        workers_per_gpu: int,
        project: str,
        entity: str,
        group: str,
        dry_run: bool = False,
    ):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.project = project
        self.entity = entity
        self.group = group
        self.dry_run = dry_run
        self.manifest_path = output_root / "manifest.json"
        self.lock = threading.Lock()
        self.slots = queue.Queue()
        for gpu_id in gpu_ids:
            for _ in range(workers_per_gpu):
                self.slots.put(gpu_id)
        if self.slots.qsize() == 0:
            raise ValueError("at least one GPU worker slot is required")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        try:
            with self.manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("group") != self.group:
                raise RuntimeError("existing manifest belongs to a different W&B group")
            for state in manifest.get("trials", {}).values():
                if state.get("status") == "running":
                    state["status"] = "interrupted"
            return manifest
        except FileNotFoundError:
            return {
                "version": 1,
                "group": self.group,
                "created_at": time.time(),
                "trials": {},
            }

    def _update(self, trial: Trial, **updates) -> None:
        with self.lock:
            state = self.manifest["trials"].setdefault(trial.trial_id, {})
            state.update(asdict(trial))
            state.update(updates)
            state["updated_at"] = time.time()
            atomic_json(self.manifest_path, self.manifest)

    def _already_complete(self, trial: Trial) -> bool:
        state = self.manifest["trials"].get(trial.trial_id, {})
        if (
            state.get("status") == "completed"
            and int(state.get("completed_epochs", 0)) >= trial.max_epochs
        ):
            return True
        result = self.output_root / "trials" / trial.trial_id / "result.json"
        try:
            with result.open(encoding="utf-8") as stream:
                last = json.load(stream)["last"]
            if int(last["epoch"]) + 1 >= trial.max_epochs:
                self._update(
                    trial, status="completed", completed_epochs=trial.max_epochs
                )
                return True
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return False

    def _retryable(self, log_path: Path) -> bool:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")[-20000:].lower()
        except OSError:
            return False
        if any(marker in text for marker in MODEL_FAILURE_MARKERS):
            return False
        return any(marker in text for marker in INFRASTRUCTURE_MARKERS)

    def _run_one(self, trial: Trial) -> bool:
        if self._already_complete(trial):
            return True
        gpu_id = self.slots.get()
        try:
            trial_root = self.output_root / "trials" / trial.trial_id
            trial_root.mkdir(parents=True, exist_ok=True)
            log_path = trial_root / "train.log"
            total_attempts = int(
                self.manifest["trials"].get(trial.trial_id, {}).get("attempts", 0)
            )
            rung_attempts = 0
            while True:
                total_attempts += 1
                rung_attempts += 1
                command = build_command(
                    trial, self.output_root, self.project, self.entity, self.group
                )
                self._update(
                    trial,
                    status="running",
                    attempts=total_attempts,
                    rung_attempts=rung_attempts,
                    gpu_id=gpu_id,
                    command=command,
                )
                if self.dry_run:
                    self._update(
                        trial,
                        status="completed",
                        completed_epochs=trial.max_epochs,
                        dry_run=True,
                    )
                    return True
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu_id),
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                    }
                )
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        f"\n=== attempt {total_attempts}, target {trial.max_epochs} epochs, "
                        f"GPU {gpu_id} ===\n"
                    )
                    stream.flush()
                    completed = subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[1],
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode == 0:
                    self._update(
                        trial,
                        status="completed",
                        completed_epochs=trial.max_epochs,
                        returncode=0,
                    )
                    return True
                retryable = rung_attempts < 2 and self._retryable(log_path)
                self._update(
                    trial,
                    status="retrying" if retryable else "failed",
                    returncode=completed.returncode,
                )
                if not retryable:
                    return False
        finally:
            self.slots.put(gpu_id)

    def run_trials(self, trials: Sequence[Trial]) -> Dict[str, bool]:
        pending = [trial for trial in trials if not self._already_complete(trial)]
        results = {
            trial.trial_id: True for trial in trials if trial not in pending
        }
        if not pending:
            return results
        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            future_trials = {
                executor.submit(self._run_one, trial): trial for trial in pending
            }
            for future in as_completed(future_trials):
                trial = future_trials[future]
                try:
                    results[trial.trial_id] = bool(future.result())
                except Exception as error:
                    self._update(trial, status="failed", error=repr(error))
                    results[trial.trial_id] = False
        return results


def local_smoke_command(
    trial: Trial, output_root: Path, train_batches: int, val_batches: int
) -> List[str]:
    command = build_command(trial, output_root, "unused", "unused", "unused")
    replacements = {
        "+trainer.check_val_every_n_epoch=5": "+trainer.check_val_every_n_epoch=1",
        "trainer.limit_train_batches=1.0": (
            f"trainer.limit_train_batches={train_batches}"
        ),
        "trainer.limit_val_batches=1.0": f"trainer.limit_val_batches={val_batches}",
        "task.aux_gradient_norm_interval=1570": "task.aux_gradient_norm_interval=0",
        "wandb.mode=online": "wandb.mode=disabled",
        "+wandb.entity=unused": "+wandb.entity=null",
        "+wandb.resume=allow": "+wandb.resume=null",
    }
    return [replacements.get(argument, argument) for argument in command]


def run_preflight(output_root: Path, gpu_id: int) -> None:
    """Exercise the real one-GPU train/checkpoint/summary path before fan-out."""
    preflight_root = output_root / "preflight"
    if (preflight_root / "PREFLIGHT_OK").exists():
        return
    trial = Trial(
        trial_id="preflight",
        seed=20260817,
        max_epochs=1,
        aux_weight=0.1,
        target_sg=False,
        scale=Scale(1.0, 1.0),
        phase="preflight",
    )
    command = local_smoke_command(trial, preflight_root, 2, 2)
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
    result = preflight_root / "trials" / "preflight" / "result.json"
    checkpoint = (
        preflight_root / "trials" / "preflight" / "checkpoints" / "last.ckpt"
    )
    if completed.returncode != 0 or _last_metrics(result) is None or not checkpoint.exists():
        raise RuntimeError(f"GPU preflight failed; inspect {log_path}")
    (preflight_root / "PREFLIGHT_OK").write_text("ok\n", encoding="utf-8")


def choose_worker_count(records: Sequence[dict], tolerance: float = 0.02) -> int:
    viable = [record for record in records if record.get("succeeded")]
    if not viable:
        raise RuntimeError("all concurrency benchmark candidates failed")
    best_throughput = max(record["aggregate_updates_per_second"] for record in viable)
    near_best = [
        record
        for record in viable
        if record["aggregate_updates_per_second"]
        >= best_throughput * (1.0 - tolerance)
    ]
    return min(int(record["workers_per_gpu"]) for record in near_best)


def benchmark_worker_counts(
    output_root: Path,
    gpu_ids: Sequence[int],
    candidates: Sequence[int] = (2, 4, 6, 8),
) -> int:
    """Measure whole-host trial throughput and select a safe near-optimal fan-out."""
    benchmark_root = output_root / "concurrency_benchmark"
    report_path = benchmark_root / "benchmark.json"
    if report_path.exists():
        with report_path.open(encoding="utf-8") as stream:
            return int(json.load(stream)["selected_workers_per_gpu"])
    partial_path = benchmark_root / "partial.json"
    records = []
    if partial_path.exists():
        try:
            with partial_path.open(encoding="utf-8") as stream:
                records = list(json.load(stream).get("records", []))
        except (OSError, ValueError, TypeError):
            records = []
    completed_candidates = {
        int(record["workers_per_gpu"]) for record in records
    }
    for workers_per_gpu in candidates:
        if workers_per_gpu in completed_candidates:
            continue
        candidate_root = benchmark_root / f"workers-{workers_per_gpu}"
        if candidate_root.exists():
            shutil.rmtree(candidate_root)
        jobs = []
        for gpu_id in gpu_ids:
            for worker in range(workers_per_gpu):
                trial = Trial(
                    trial_id=f"bench-c{workers_per_gpu}-g{gpu_id}-w{worker}",
                    seed=20260817 + worker,
                    max_epochs=2,
                    aux_weight=0.1,
                    target_sg=False,
                    scale=Scale(1.0, 1.0),
                    phase="benchmark",
                )
                command = local_smoke_command(trial, candidate_root, 157, 2)
                jobs.append((gpu_id, trial, command))

        started = time.monotonic()

        def run_job(job) -> bool:
            gpu_id, trial, command = job
            trial_root = candidate_root / "trials" / trial.trial_id
            trial_root.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            with (trial_root / "train.log").open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result = trial_root / "result.json"
            checkpoint = trial_root / "checkpoints" / "last.ckpt"
            return (
                completed.returncode == 0
                and _last_metrics(result) is not None
                and checkpoint.exists()
            )

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            job_results = list(executor.map(run_job, jobs))
            succeeded = all(job_results)
        elapsed = time.monotonic() - started
        total_updates = len(jobs) * 2 * STEPS_PER_EPOCH
        records.append(
            {
                "workers_per_gpu": workers_per_gpu,
                "processes": len(jobs),
                "elapsed_seconds": elapsed,
                "total_updates": total_updates,
                "aggregate_updates_per_second": total_updates / elapsed,
                "succeeded": succeeded,
            }
        )
        atomic_json(partial_path, {"records": records})
    selected = choose_worker_count(records)
    atomic_json(
        report_path,
        {"records": records, "selected_workers_per_gpu": selected},
    )
    return selected


def scale_from_row(row: dict) -> Scale:
    return Scale(**row["scale"])


def paired_bootstrap_ci(differences: Sequence[float], samples: int = 10000):
    generator = random.Random(20260817)
    means = []
    for _ in range(samples):
        draw = [generator.choice(differences) for _ in differences]
        means.append(statistics.fmean(draw))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def condition_metrics(output_root: Path, trials: Sequence[Trial]) -> dict:
    rows = []
    for trial in trials:
        metrics = _last_metrics(
            output_root / "trials" / trial.trial_id / "result.json"
        )
        if metrics is None:
            continue
        rows.append(
            {
                "seed": trial.seed,
                "accuracy": metrics.get("val/accuracy_ignore_index"),
                "loss": metrics.get("val/loss"),
                "perplexity": metrics.get("val/perplexity"),
            }
        )
    accuracies = [float(row["accuracy"]) for row in rows if row["accuracy"] is not None]
    losses = [float(row["loss"]) for row in rows if row["loss"] is not None]
    return {
        "rows": rows,
        "mean_accuracy": statistics.fmean(accuracies) if accuracies else None,
        "sample_sd_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min_accuracy": min(accuracies) if accuracies else None,
        "mean_loss": statistics.fmean(losses) if losses else None,
    }


def final_report(
    output_root: Path,
    baseline_trials: Sequence[Trial],
    selected: Dict[bool, Scale],
    heldout: Dict[bool, Sequence[Trial]],
) -> dict:
    report = {
        "baseline": condition_metrics(output_root, baseline_trials),
        "selected": {},
    }
    baseline_by_seed = {
        row["seed"]: row
        for row in report["baseline"]["rows"]
        if row["seed"] in HELDOUT_SEEDS
    }
    for target_sg in (False, True):
        label = "target_sg_on" if target_sg else "target_sg_off"
        metrics = condition_metrics(output_root, heldout[target_sg])
        differences = []
        wins = 0
        for row in metrics["rows"]:
            baseline = baseline_by_seed.get(row["seed"])
            if baseline is None or row["accuracy"] is None:
                continue
            difference = float(row["accuracy"]) - float(baseline["accuracy"])
            differences.append(difference)
            wins += difference > 0
        metrics.update(
            {
                "scale": asdict(selected[target_sg]),
                "rho": selected[target_sg].rho,
                "tau": selected[target_sg].tau,
                "paired_accuracy_delta": (
                    statistics.fmean(differences) if differences else None
                ),
                "paired_wins": wins,
                "paired_count": len(differences),
                "paired_bootstrap_95_ci": (
                    paired_bootstrap_ci(differences) if differences else None
                ),
            }
        )
        report["selected"][label] = metrics
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", default="auto")
    parser.add_argument("--wandb-project", default="aux-assc-recall")
    parser.add_argument("--wandb-entity", default="bjjin07-x")
    parser.add_argument(
        "--wandb-group", default="ar-rmt-noaux-targetsg-scale-20260817-v1"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        key = os.environ.get("WANDB_API_KEY", "")
        if not key or key == "WANDB_API_KEY":
            raise RuntimeError("WANDB_API_KEY secret was not injected")
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi is unavailable")
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value)

    if args.dry_run:
        sample = make_screen_trials([Scale(1.0, 1.0)], False)[0]
        plan = {
            "gpu_ids": gpu_ids,
            "workers_per_gpu": args.workers_per_gpu,
            "initial_screen_trials": 2 * len(initial_scales()) * len(TUNING_SEEDS),
            "full_noaux_trials": len(BASELINE_SEEDS),
            "shortlist_full_trials": 2 * 2 * len(TUNING_SEEDS),
            "heldout_full_trials": 2 * len(HELDOUT_SEEDS),
            "maximum_boundary_extension_trials_per_sg": 7 * len(TUNING_SEEDS),
            "sample_command": build_command(
                sample,
                args.output_root,
                args.wandb_project,
                args.wandb_entity,
                args.wandb_group,
            ),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    run_preflight(args.output_root, gpu_ids[0])
    if args.workers_per_gpu == "auto":
        workers_per_gpu = benchmark_worker_counts(args.output_root, gpu_ids)
    else:
        workers_per_gpu = int(args.workers_per_gpu)
        if workers_per_gpu < 1 or workers_per_gpu > 8:
            raise ValueError("workers-per-gpu must be auto or an integer in [1, 8]")
    print(f"Selected {workers_per_gpu} concurrent trials per GPU", flush=True)
    controller = StudyController(
        output_root=args.output_root,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        dry_run=False,
    )

    baseline_trials = make_baseline_trials()
    baseline_result = {}

    def run_baselines():
        baseline_result.update(controller.run_trials(baseline_trials))

    baseline_thread = threading.Thread(target=run_baselines, daemon=False)
    baseline_thread.start()

    all_scales: Dict[bool, set] = {False: set(initial_scales()), True: set(initial_scales())}
    initial_trials = []
    for target_sg in (False, True):
        initial_trials.extend(make_screen_trials(initial_scales(), target_sg))
    controller.run_trials(initial_trials)

    for target_sg in (False, True):
        ranking = rank_scales(args.output_root, target_sg, all_scales[target_sg])
        if not ranking:
            raise RuntimeError(f"no complete initial scale for target_sg={target_sg}")
        additions = outward_scales(scale_from_row(ranking[0]))
        if additions:
            controller.run_trials(make_screen_trials(additions, target_sg))
            all_scales[target_sg].update(additions)

    screen_rankings = {}
    shortlist_trials = []
    shortlist_scales = {}
    for target_sg in (False, True):
        ranking = rank_scales(args.output_root, target_sg, all_scales[target_sg])
        if len(ranking) < 2:
            raise RuntimeError(f"fewer than two viable scales for target_sg={target_sg}")
        screen_rankings[str(target_sg).lower()] = ranking
        shortlist_scales[target_sg] = [scale_from_row(row) for row in ranking[:2]]
        for trial in make_screen_trials(shortlist_scales[target_sg], target_sg):
            shortlist_trials.append(replace(trial, max_epochs=FULL_EPOCHS))
    atomic_json(args.output_root / "screen_rankings.json", screen_rankings)
    controller.run_trials(shortlist_trials)

    selected = {}
    full_rankings = {}
    heldout = {}
    for target_sg in (False, True):
        ranking = rank_scales(
            args.output_root, target_sg, shortlist_scales[target_sg]
        )
        if not ranking:
            raise RuntimeError(f"no complete full scale for target_sg={target_sg}")
        full_rankings[str(target_sg).lower()] = ranking
        selected[target_sg] = scale_from_row(ranking[0])
        heldout[target_sg] = make_heldout_trials(target_sg, selected[target_sg])
    atomic_json(args.output_root / "full_rankings.json", full_rankings)

    heldout_trials = heldout[False] + heldout[True]
    heldout_result = controller.run_trials(heldout_trials)
    if not all(heldout_result.get(trial.trial_id, False) for trial in heldout_trials):
        raise RuntimeError("one or more held-out AUX trials failed")
    baseline_thread.join()
    if not all(baseline_result.get(trial.trial_id, False) for trial in baseline_trials):
        raise RuntimeError("one or more no-AUX baselines failed")

    report = final_report(args.output_root, baseline_trials, selected, heldout)
    atomic_json(args.output_root / "final_report.json", report)
    (args.output_root / "STUDY_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
