import json
import math
import os
import tempfile
from pathlib import Path

import pytorch_lightning as pl
import torch


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


class StudySummary(pl.Callback):
    """Persist validation metrics for an external, resumable study controller."""

    def __init__(self, result_path: str):
        self.result_path = Path(result_path)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            scalar = _scalar(value)
            if scalar is not None and (
                str(name).startswith("val/") or str(name).startswith("test/")
            ):
                metrics[str(name)] = scalar
        if not metrics:
            return

        payload = {"history": []}
        if self.result_path.exists():
            try:
                with self.result_path.open(encoding="utf-8") as stream:
                    existing = json.load(stream)
                if isinstance(existing.get("history"), list):
                    payload = existing
            except (OSError, ValueError, TypeError):
                pass

        record = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "metrics": metrics,
        }
        history = [
            item
            for item in payload["history"]
            if int(item.get("global_step", -1)) != record["global_step"]
        ]
        history.append(record)
        history.sort(key=lambda item: (item["global_step"], item["epoch"]))
        payload.update(
            {
                "history": history,
                "last": history[-1],
                "best_val_accuracy_ignore_index": max(
                    (
                        item["metrics"]["val/accuracy_ignore_index"]
                        for item in history
                        if "val/accuracy_ignore_index" in item["metrics"]
                    ),
                    default=None,
                ),
            }
        )
        _atomic_json(self.result_path, payload)
