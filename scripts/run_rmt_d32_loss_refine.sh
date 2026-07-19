#!/usr/bin/env bash

# Refine the loss balance around the memory-x4 result. The optimizer, LR,
# schedule, model size, and data stay fixed; only lambda, rho, and tau vary.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0, 1, or 2}"
if (( shard < 0 || shard > 2 )); then
  echo "SHARD must be 0, 1, or 2" >&2
  exit 2
fi

group="${WANDB_GROUP:-rmt-d32-loss-refine-20260719-v1}"
suffix="${RUN_SUFFIX:-loss-refine-v1}"
max_epochs="${MAX_EPOCHS:-240}"
steps_per_epoch="157"
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 10))"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-loss-refine}/shard-${shard}"
mkdir -p "$output_root"

common_args=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs="$max_epochs"
  callbacks=full_run
  +trainer.check_val_every_n_epoch=5
  +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50
  trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0
  +trainer.precision=32
  trainer.gradient_clip_val=0.0
  train.test=false
  loader.num_workers=0
  scheduler=linear_warmup
  scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
  task.aux_gradient_norm_interval="$steps_per_epoch"
  task.aux_weight_schedule=fixed
  optimizer.lr=0.0005
  optimizer.weight_decay=0.1
  model.d_model=32
  model.d_inner=128
  model.n_layer=2
  model.n_heads=1
  model.chunk_size=4
  model.num_memory_tokens=4
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
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
)

pids=()
names=()

launch() {
  local label="$1"
  local aux_weight="$2"
  local rho="$3"
  local tau="$4"
  local seed="$5"
  local name="rmt-d32-${label}-s${seed}-${suffix}"

  echo "Launching $name (lambda=$aux_weight, rho=$rho, tau=$tau)"
  python -m train \
    "${common_args[@]}" \
    train.seed="$seed" \
    task.aux_weight="$aux_weight" \
    model.rho="$rho" \
    model.tau="$tau" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

# Effective memory and terminal pressures are lambda/rho^2 and lambda/tau^2.
# "token-*" conditions lower token reconstruction weight while compensating
# rho/tau so the named continuous-loss pressures stay unchanged.
conditions=(
  "memory-x3|1.0|36.514837|12.649111"
  "memory-x5|1.0|28.284271|12.649111"
  "memory-x4-terminal-075|1.0|31.622777|14.605935"
  "memory-x4-terminal-125|1.0|31.622777|11.313708"
  "memory-x4-token-075|0.75|27.386128|10.954451"
  "memory-x4-token-half|0.5|22.360680|8.944272"
  "memory-x3-token-075|0.75|31.622777|10.954451"
  "memory-x5-token-075|0.75|24.494897|10.954451"
)

jobs=()
for condition in "${conditions[@]}"; do
  for seed in 0 1 2; do
    jobs+=("$condition|$seed")
  done
done

for index in "${!jobs[@]}"; do
  if (( index % 3 != shard )); then
    continue
  fi
  IFS='|' read -r label aux_weight rho tau seed <<<"${jobs[$index]}"
  launch "$label" "$aux_weight" "$rho" "$tau" "$seed"
done

if (( ${#pids[@]} > 8 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "Shard $shard launched ${#pids[@]} processes"

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
