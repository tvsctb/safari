#!/usr/bin/env bash

# Multi-seed confirmation of the selected d32 full-auxiliary setting.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-selected-confirmation-20260719-v1}"
suffix="${RUN_SUFFIX:-selected-v1}"
max_epochs="${MAX_EPOCHS:-180}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-selected-confirmation}"
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
  train.test=false
  loader.num_workers=0
  task.aux_gradient_norm_interval=157
  task.aux_weight=3.0
  optimizer.lr=4e-3
  optimizer.weight_decay=0.1
  model.d_model=32
  model.d_inner=128
  model.n_layer=2
  model.n_heads=1
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
  model.rho=54.772256
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.tau=10.954451
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
)

pids=()
names=()

for seed in 0 1 2 3 4; do
  name="rmt-d32-selected-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    "${common_args[@]}" \
    train.seed="$seed" \
    wandb.mode=online \
    wandb.project=aux-assoc-recall \
    wandb.group="$group" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

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
