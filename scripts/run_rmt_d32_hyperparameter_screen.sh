#!/usr/bin/env bash

# First d32 (Transformer-capacity-matched) screen. All five conditions share
# the current RMT objective; only lambda and the fixed memory scale vary.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-hparam-a-20260718-v1}"
suffix="${RUN_SUFFIX:-v1}"
max_epochs="${MAX_EPOCHS:-120}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-hparam}/wave-a"
mkdir -p "$output_root"

common_args=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs="$max_epochs"
  +trainer.check_val_every_n_epoch=5
  +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50
  trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0
  +trainer.precision=32
  train.seed=0
  train.test=false
  loader.num_workers=0
  task.aux_gradient_norm_interval=157
  model.use_chunk_loss=true
  model.use_discrete_loss=true
  model.use_memory_loss=true
  model.use_terminal_loss=true
  model.use_terminal_chunk=false
  model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true
  model.stop_gradient_memory_target=true
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.tau=2.0
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
)

# Validate the exact d32 configuration before allocating the GPU.
python -m train --cfg job \
  "${common_args[@]}" \
  task.aux_weight=0.03 \
  model.rho=10.0 \
  wandb.mode=disabled \
  >"$output_root/preflight-config.yaml"

pids=()
names=()

launch() {
  local label="$1"
  local aux_weight="$2"
  local rho="$3"
  local name="rmt-d32-${label}-s0-${suffix}"

  echo "Launching $name (lambda=$aux_weight, rho=$rho, tau=2.0)"
  python -m train \
    "${common_args[@]}" \
    task.aux_weight="$aux_weight" \
    model.rho="$rho" \
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

# The d64 screen favored lambda=.1, rho=20.  Re-establish that result at d32
# while checking the smaller lambda/rho region that was previously competitive.
launch "baseline" 0.0 20.0
launch "rho10-lambda0p01" 0.01 10.0
launch "rho10-lambda0p03" 0.03 10.0
launch "rho10-lambda0p1" 0.1 10.0
launch "rho20-lambda0p1" 0.1 20.0

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
