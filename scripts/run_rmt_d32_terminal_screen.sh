#!/usr/bin/env bash

# Second d32 screen: terminal scale and observation noise after the lambda/rho
# screen. The two surviving memory-scale candidates both use lambda=.1.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-hparam-b-20260718-v1}"
suffix="${RUN_SUFFIX:-v1}"
max_epochs="${MAX_EPOCHS:-120}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-hparam}/wave-b"
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
  task.aux_weight=0.1
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
  model.generation_noise_std=0.0
)

python -m train --cfg job \
  "${common_args[@]}" \
  model.rho=20.0 \
  model.tau=2.0 \
  model.observation_noise_std=0.1 \
  wandb.mode=disabled \
  >"$output_root/preflight-config.yaml"

pids=()
names=()

launch() {
  local label="$1"
  local rho="$2"
  local tau="$3"
  local observation_noise="$4"
  local name="rmt-d32-${label}-s0-${suffix}"

  echo "Launching $name (lambda=.1, rho=$rho, tau=$tau, observation_noise=$observation_noise)"
  python -m train \
    "${common_args[@]}" \
    model.rho="$rho" \
    model.tau="$tau" \
    model.observation_noise_std="$observation_noise" \
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

launch "rho10-tau1p5" 10.0 1.5 0.0
launch "rho10-tau2" 10.0 2.0 0.0
launch "rho10-tau3" 10.0 3.0 0.0
launch "rho20-tau2" 20.0 2.0 0.0
launch "rho20-tau2-observation0p1" 20.0 2.0 0.1

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
