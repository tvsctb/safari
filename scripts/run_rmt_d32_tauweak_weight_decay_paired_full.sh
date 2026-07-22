#!/usr/bin/env bash

# Paired full-run check of AdamW weight decay at the selected natural setup.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0 or 1}"
if (( shard < 0 || shard > 1 )); then
  echo "SHARD must be set to 0 or 1" >&2
  exit 2
fi

group="${WANDB_GROUP:-rmt-d32-tauweak-wd-paired-full-20260722-v1}"
suffix="${RUN_SUFFIX:-wdfull400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 5))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-tauweak-wd-paired-full}/shard-${shard}"
mkdir -p "$output_root"

common=(
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
  scheduler=linear_warmup
  scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
  optimizer.lr=0.001
  task.aux_weight=1.0
  task.aux_weight_final=1.0
  task.aux_weight_schedule=fixed
  task.aux_weight_decay_start_step=0
  task.aux_weight_decay_end_step=1
  task.aux_gradient_norm_interval="$steps_per_epoch"
  task.aux_activation_lm_loss_threshold=null
  task.aux_solved_ce_threshold=null
  model.d_model=32
  model.d_inner=128
  model.n_layer=2
  model.n_heads=1
  model.chunk_size=4
  model.num_memory_tokens=4
  model.share_inverse=true
  model.share_inverse_embedding=true
  model.share_inverse_head=true
  model.share_inverse_position_embedding=true
  model.inverse_position_initialization=copy
  model.use_direction_embedding=true
  model.use_chunk_loss=true
  model.use_discrete_loss=true
  model.use_memory_loss=true
  model.use_terminal_loss=true
  model.use_terminal_chunk=false
  model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true
  model.stop_gradient_memory_target=true
  model.stop_gradient_memory_observation=false
  model.memory_observation_gradient_scale=1.0
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.rho=31.622777
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.tau=11.925695
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
)

pids=()
names=()
jobs=()
for seed in 120 121 122 123 124; do
  jobs+=("wd01|0.1|$seed")
  jobs+=("wd0|0.0|$seed")
done

for index in "${!jobs[@]}"; do
  if (( index % 2 != shard )); then
    continue
  fi
  IFS='|' read -r label weight_decay seed <<<"${jobs[$index]}"
  name="rmt-d32-tauweak-static-aux-lr1e-3-warm20-${label}-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    optimizer.weight_decay="$weight_decay" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
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
