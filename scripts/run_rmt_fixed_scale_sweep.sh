#!/usr/bin/env bash

set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

group="${WANDB_GROUP:-rmt-fixed-scale-3x3-20260713-v1}"
suffix="${RUN_SUFFIX:-v1}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-fixed-scale-3x3}"
mkdir -p "$output_root"

pids=()
names=()

launch() {
  local name="$1"
  local aux_weight="$2"
  local rho="$3"
  local tau="$4"
  echo "Launching $name (aux_weight=$aux_weight, rho=$rho, tau=$tau)"
  python -m train \
    experiment=synthetics/associative_recall/rmt_aux \
    trainer.max_epochs=400 \
    train.seed=0 \
    train.test=false \
    loader.num_workers=0 \
    model.d_model=32 \
    model.d_inner=128 \
    model.n_heads=1 \
    model.n_layer=2 \
    task.aux_weight="$aux_weight" \
    model.scale_mode=fixed \
    model.scale_granularity=global \
    model.rho="$rho" \
    model.tau="$tau" \
    wandb.mode=online \
    wandb.project=aux-assoc-recall \
    wandb.group="$group" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

launch "rmt-baseline-s0-$suffix" 0.0 1.0 1.0
launch "rmt-rho0p1-tau0p1-s0-$suffix" 0.1 0.1 0.1
launch "rmt-rho0p1-tau0p3-s0-$suffix" 0.1 0.1 0.3
launch "rmt-rho0p1-tau1p0-s0-$suffix" 0.1 0.1 1.0
launch "rmt-rho0p3-tau0p1-s0-$suffix" 0.1 0.3 0.1
launch "rmt-rho0p3-tau0p3-s0-$suffix" 0.1 0.3 0.3
launch "rmt-rho0p3-tau1p0-s0-$suffix" 0.1 0.3 1.0
launch "rmt-rho1p0-tau0p1-s0-$suffix" 0.1 1.0 0.1
launch "rmt-rho1p0-tau0p3-s0-$suffix" 0.1 1.0 0.3
launch "rmt-rho1p0-tau1p0-s0-$suffix" 0.1 1.0 1.0

status=0
for index in "${!pids[@]}"; do
  name="${names[$index]}"
  if wait "${pids[$index]}"; then
    echo "Completed $name"
  else
    echo "Failed $name"
    status=1
  fi
  tail -n 40 "$output_root/$name.log"
done
exit "$status"
