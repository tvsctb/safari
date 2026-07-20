#!/usr/bin/env bash

# Five-seed, 400-epoch confirmation of the exact high-ceiling control against
# the most stable screen candidate (split PE, no direction, AUX warm-up).
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0 or 1}"
if (( shard < 0 || shard > 1 )); then
  echo "SHARD must be 0 or 1" >&2
  exit 2
fi

group="${WANDB_GROUP:-rmt-d32-memoryx4-terminal125-full-confirm-20260721-v1}"
suffix="${RUN_SUFFIX:-full-confirm-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
optimizer_warmup_steps="$((training_steps / 10))"
aux_warmup_steps="$optimizer_warmup_steps"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-memoryx4-terminal125-full-confirm}/shard-${shard}"
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
  loader.num_workers=0
  scheduler=linear_warmup
  scheduler.num_warmup_steps="$optimizer_warmup_steps"
  scheduler.num_training_steps="$training_steps"
  task.aux_gradient_norm_interval="$steps_per_epoch"
  optimizer.lr=0.0005
  optimizer.weight_decay=0.1
  model.d_model=32
  model.d_inner=128
  model.n_layer=2
  model.n_heads=1
  model.chunk_size=4
  model.num_memory_tokens=4
  model.share_inverse=true
  model.share_inverse_embedding=true
  model.share_inverse_head=true
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
  model.rho=31.622777
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.tau=11.313708
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
  local seed="$2"
  local share_position="$3"
  local direction="$4"
  local schedule="$5"
  local initial_weight="$6"
  local final_weight="$7"
  local schedule_end="$8"
  local name="rmt-d32-mx4-t125-${label}-s${seed}-${suffix}"

  echo "Launching $name"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    model.share_inverse_position_embedding="$share_position" \
    model.inverse_position_initialization=copy \
    model.use_direction_embedding="$direction" \
    task.aux_weight="$initial_weight" \
    task.aux_weight_final="$final_weight" \
    task.aux_weight_schedule="$schedule" \
    task.aux_weight_decay_start_step=0 \
    task.aux_weight_decay_end_step="$schedule_end" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

jobs=()
for seed in 0 1 2 3 4; do
  jobs+=("control|$seed|true|true|fixed|1.0|1.0|1")
  jobs+=("splitpe-warmup|$seed|false|false|linear|0.1|1.0|$aux_warmup_steps")
done

for index in "${!jobs[@]}"; do
  if (( index % 2 != shard )); then
    continue
  fi
  IFS='|' read -r label seed share_position direction schedule initial_weight final_weight schedule_end <<<"${jobs[$index]}"
  launch "$label" "$seed" "$share_position" "$direction" "$schedule" "$initial_weight" "$final_weight" "$schedule_end"
done

if (( ${#pids[@]} != 5 )); then
  echo "Expected 5 processes, launched ${#pids[@]}" >&2
  exit 2
fi

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
