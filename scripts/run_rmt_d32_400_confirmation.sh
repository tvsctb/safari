#!/usr/bin/env bash

# Full repo-horizon comparison for the d32 capacity-matched model. Five
# conditions share the 400-epoch base scheduler on one GPU.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-400-confirmation-20260718-v1}"
suffix="${RUN_SUFFIX:-v1}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-400-confirmation}"
mkdir -p "$output_root"

common_args=(
  experiment=synthetics/associative_recall/rmt_aux
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
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
)

python -m train --cfg job \
  "${common_args[@]}" \
  model.n_heads=4 \
  task.aux_weight=0.1 \
  model.rho=10.0 \
  model.tau=1.5 \
  wandb.mode=disabled \
  >"$output_root/preflight-config.yaml"

pids=()
names=()

launch() {
  local label="$1"
  local heads="$2"
  local aux_weight="$3"
  local rho="$4"
  local tau="$5"
  local name="rmt-d32-400-${label}-s0-${suffix}"

  echo "Launching $name (heads=$heads, lambda=$aux_weight, rho=$rho, tau=$tau)"
  python -m train \
    "${common_args[@]}" \
    model.n_heads="$heads" \
    task.aux_weight="$aux_weight" \
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

launch "h1-baseline" 1 0.0 10.0 2.0
launch "h1-lambda0p1-rho10-tau2" 1 0.1 10.0 2.0
launch "h4-baseline" 4 0.0 10.0 2.0
launch "h4-lambda0p1-rho10-tau1p5" 4 0.1 10.0 1.5
launch "h4-lambda0p1-rho20-tau2" 4 0.1 20.0 2.0

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
