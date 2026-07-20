#!/usr/bin/env bash

# Interpolate between the seed-dependent failures at fixed AUX=1.0 and
# split-PE AUX warm-up from 0.1 by screening intermediate initial weights.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY"'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be 0, 1, or 2}"
if (( shard < 0 || shard > 2 )); then
  echo "SHARD must be 0, 1, or 2" >&2
  exit 2
fi

group="${WANDB_GROUP:-rmt-d32-mx4-t125-auxfloor-screen-20260721-v1}"
suffix="${RUN_SUFFIX:-auxfloor-screen-v1}"
max_epochs="${MAX_EPOCHS:-240}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 10))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-mx4-t125-auxfloor}/seed-${shard}"
mkdir -p "$output_root"

common=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs="$max_epochs" callbacks=full_run
  +trainer.check_val_every_n_epoch=5 +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50 trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0 +trainer.precision=32
  trainer.gradient_clip_val=0.0 train.test=false train.seed="$shard"
  loader.num_workers=0 scheduler=linear_warmup
  scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
  task.aux_gradient_norm_interval="$steps_per_epoch"
  task.aux_weight_final=1.0 task.aux_weight_schedule=linear
  task.aux_weight_decay_start_step=0
  task.aux_weight_decay_end_step="$warmup_steps"
  optimizer.lr=0.0005 optimizer.weight_decay=0.1
  model.d_model=32 model.d_inner=128 model.n_layer=2 model.n_heads=1
  model.chunk_size=4 model.num_memory_tokens=4
  model.share_inverse=true model.share_inverse_embedding=true
  model.share_inverse_head=true model.share_inverse_position_embedding=false
  model.inverse_position_initialization=copy model.use_direction_embedding=false
  model.use_chunk_loss=true model.use_discrete_loss=true
  model.use_memory_loss=true model.use_terminal_loss=true
  model.use_terminal_chunk=false model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true model.stop_gradient_memory_target=true
  model.memory_scale_mode=fixed model.memory_scale_granularity=global
  model.rho=31.622777 model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global model.tau=11.313708
  model.observation_noise_std=0.0 model.generation_noise_std=0.0
  wandb.mode=online wandb.project=aux-assoc-recall wandb.group="$group"
)

pids=()
names=()
launch() {
  local floor="$1"
  local label="floor${floor/./p}"
  local name="rmt-d32-mx4-t125-${label}-s${shard}-${suffix}"
  echo "Launching $name (AUX $floor -> 1.0 over $warmup_steps steps)"
  python -m train "${common[@]}" task.aux_weight="$floor" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" wandb.id="$name" hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

launch 0.3
launch 0.5
launch 0.7

status=0
for index in "${!pids[@]}"; do
  name="${names[$index]}"
  wait "${pids[$index]}" || status=1
  tail -n 40 "$output_root/$name.log"
done
exit "$status"
