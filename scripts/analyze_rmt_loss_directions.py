#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

import torch

from src.dataloaders.synthetics import ICLDataModule
from src.models.sequence.rmt_aux import RMTAuxLM
from src.utils.rmt_loss_analysis import (
    central_directional_effect,
    component_gradients,
    forward_parameter_items,
    gradient_summary,
    model_batch_metrics,
)


PRESSURE_PATTERN = re.compile(r"-p(sqrt2|2sqrt2|2)-s191-")


def checkpoint_metadata(path):
    match = PRESSURE_PATTERN.search(str(path))
    if match is None:
        raise ValueError(f"cannot infer pressure from {path}")
    return {
        "pressure": match.group(1),
        "endpoint": "best" if path.name == "best.ckpt" else "last",
        "path": str(path),
    }


def build_model():
    return RMTAuxLM(
        d_model=32,
        n_layer=2,
        d_inner=128,
        n_heads=1,
        vocab_size=20,
        chunk_size=4,
        num_memory_tokens=4,
        dropout=0.0,
        token_scheme="boundary_reverse",
        share_inverse=False,
        share_inverse_embedding=True,
        share_inverse_head=True,
        share_inverse_position_embedding=False,
        inverse_position_initialization="copy",
        use_direction_embedding=False,
        memory_scale_mode="fixed",
        memory_scale_granularity="global",
        terminal_scale_mode="fixed",
        terminal_scale_granularity="global",
        rho=1.0,
        tau=1.0,
        stop_gradient_memory_target=False,
        stop_gradient_memory_observation=False,
        memory_observation_gradient_scale=1.0,
        learnable_terminal_target=True,
        observation_noise_std=0.0,
        generation_noise_std=0.0,
        use_terminal_chunk=False,
        use_chunk_loss=True,
        use_discrete_loss=True,
        use_memory_loss=True,
        use_terminal_loss=True,
        use_terminal_chunk_loss=False,
    )


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        key[len("model."):]: value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not state:
        raise ValueError(f"checkpoint has no model.* state: {path}")
    model = build_model()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def deterministic_batches(batch_size, count, device):
    dataset = ICLDataModule(
        num_examples=5000,
        num_test_examples=500,
        vocab_size=20,
        input_seq_len=40,
        copy_method="assoc_recall",
        seed=0,
        loader_seed=None,
        batch_size=batch_size,
        split_train_test=False,
        induction_len=1,
        induction_num_triggers=1,
        allow_dot=False,
        max_copy_len=10,
        test_seq_len=None,
        num_keys=1,
        return_aux_tokens=True,
        data_dir=None,
    )
    dataset.setup()
    records = dataset.dataset["train"]
    batches = []
    for index in range(count):
        samples = [records[offset] for offset in range(index * batch_size, (index + 1) * batch_size)]
        batches.append(tuple(torch.stack(values).to(device) for values in zip(*samples)))
    return batches


def discover_checkpoints(root):
    paths = sorted(Path(root).rglob("*.ckpt"))
    paths = [path for path in paths if path.name in {"best.ckpt", "last.ckpt"}]
    if len(paths) != 6:
        raise ValueError(f"expected six seed191 endpoint checkpoints, found {len(paths)}")
    metadata = [checkpoint_metadata(path) for path in paths]
    cells = {(item["pressure"], item["endpoint"]) for item in metadata}
    expected = {
        (pressure, endpoint)
        for pressure in ("sqrt2", "2", "2sqrt2")
        for endpoint in ("best", "last")
    }
    if cells != expected:
        raise ValueError(f"checkpoint cells differ: {cells} != {expected}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--steps", type=float, nargs="+", default=(1e-4, 3e-4))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(20260815)
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint mechanism analysis requires CUDA")
    device = torch.device("cuda")
    checkpoints = discover_checkpoints(args.checkpoint_root)
    batches = deterministic_batches(args.batch_size, args.batches, device)
    report = {
        "schema_version": 1,
        "design": {
            "seed": 191,
            "dataset_seed": 0,
            "data_split": "train",
            "batch_size": args.batch_size,
            "batches": args.batches,
            "probes": args.probes,
            "relative_steps": args.steps,
            "aux_weight": 0.1,
            "proxy": "Hutchinson JVP mean-squared local transition gain",
            "theory_signs": {
                "lm": {"write_gain": 1, "carry_gain": None},
                "token": {"write_gain": 1, "carry_gain": -1},
                "memory": {"write_gain": -1, "carry_gain": 1},
                "terminal": {"write_gain": -1, "carry_gain": -1},
            },
        },
        "checkpoints": [],
    }
    for checkpoint_index, path in enumerate(checkpoints):
        metadata = checkpoint_metadata(path)
        print(f"Analyzing {metadata['pressure']} {metadata['endpoint']}: {path}", flush=True)
        model = load_model(path, device)
        parameter_items = forward_parameter_items(model)
        parameters = [parameter for _, parameter in parameter_items]
        checkpoint_record = {**metadata, "batches": []}
        for batch_index, batch in enumerate(batches):
            losses, metrics = model_batch_metrics(model, *batch)
            gradients = component_gradients(losses, parameters)
            batch_record = {
                "batch_index": batch_index,
                "base_metrics": metrics,
                "gradients": gradient_summary(parameter_items, gradients, aux_weight=0.1),
                "effects": {},
            }
            for loss_name in ("lm", "chunk", "discrete", "token", "memory", "terminal", "full_aux"):
                batch_record["effects"][loss_name] = []
                for relative_step in args.steps:
                    effect = central_directional_effect(
                        model,
                        batch,
                        parameters,
                        gradients[loss_name],
                        relative_step,
                        proxy_seed=100000 * checkpoint_index + 1000 * batch_index + 17,
                        probes=args.probes,
                    )
                    batch_record["effects"][loss_name].append(effect)
            checkpoint_record["batches"].append(batch_record)
            print(f"  completed batch {batch_index + 1}/{len(batches)}", flush=True)
        report["checkpoints"].append(checkpoint_record)
        del model
        torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
