#!/usr/bin/env bash

# Locked final evaluation of the natural, static tau-weak RMT configuration.
# Ten unseen seeds are paired between full AUX and no AUX across three GPU
# shards.  No schedule, adaptive gate, or answer-token reweighting is used.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0, 1, or 2}"
case "$shard" in
  0) seeds=(100 101 102 103) ;;
  1) seeds=(104 105 106 107) ;;
  2) seeds=(108 109) ;;
  *) echo "SHARD must be 0, 1, or 2" >&2; exit 2 ;;
esac

group="${WANDB_GROUP:-rmt-d32-tauweak-static-full-eval-20260722-v1}"
suffix="${RUN_SUFFIX:-unseen400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 10))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-tauweak-static-full-eval}/shard-${shard}"
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
  scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
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
  model.share_inverse_position_embedding=true
  model.inverse_position_initialization=copy
  model.use_direction_embedding=true
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
  task.aux_weight_schedule=fixed
  task.aux_weight_final=1.0
  task.aux_weight_decay_start_step=0
  task.aux_weight_decay_end_step=1
  task.aux_activation_lm_loss_threshold=null
  task.aux_solved_ce_threshold=null
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
)

pids=()
names=()

launch() {
  local condition="$1"
  local seed="$2"
  local use_aux aux_weight grad_interval

  if [[ "$condition" == "fullaux" ]]; then
    use_aux=true
    aux_weight=1.0
    grad_interval="$steps_per_epoch"
  else
    use_aux=false
    aux_weight=0.0
    grad_interval=0
  fi

  local name="rmt-d32-tauweak-static-${condition}-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    task.aux_weight="$aux_weight" \
    task.aux_gradient_norm_interval="$grad_interval" \
    model.use_chunk_loss="$use_aux" \
    model.use_discrete_loss="$use_aux" \
    model.use_memory_loss="$use_aux" \
    model.use_terminal_loss="$use_aux" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

for seed in "${seeds[@]}"; do
  launch fullaux "$seed"
  launch noaux "$seed"
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
