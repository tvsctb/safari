#!/usr/bin/env bash

# Local static rho/tau refinement around the memory-x4 + terminal-1.25 anchor.
# The exact anchor already exists for seeds 0-2, so this screen only launches
# the four +/-10% effective-pressure coordinate points.
set -euo pipefail

python -c 'import os; assert os.environ.get("WANDB_API_KEY")'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

seed="${SHARD:?SHARD must be seed 0, 1, or 2}"
if (( seed < 0 || seed > 2 )); then
  echo "SHARD must be seed 0, 1, or 2" >&2
  exit 2
fi

group="${WANDB_GROUP:-rmt-d32-static-scale-local-20260721-v1}"
max_epochs="${MAX_EPOCHS:-240}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 10))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-static-scale-local}/seed-${seed}"
mkdir -p "$output_root"

common=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs="$max_epochs" callbacks=full_run
  +trainer.check_val_every_n_epoch=5 +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50 trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0 +trainer.precision=32
  trainer.gradient_clip_val=0.0 train.test=false loader.num_workers=0
  scheduler=linear_warmup scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
  task.aux_gradient_norm_interval="$steps_per_epoch"
  task.aux_weight=1.0 task.aux_weight_final=1.0
  task.aux_weight_schedule=fixed
  task.aux_activation_lm_loss_threshold=null
  task.aux_solved_ce_threshold=null
  optimizer.lr=0.0005 optimizer.weight_decay=0.1
  model.d_model=32 model.d_inner=128 model.n_layer=2 model.n_heads=1
  model.chunk_size=4 model.num_memory_tokens=4
  model.share_inverse=true model.share_inverse_embedding=true
  model.share_inverse_head=true model.share_inverse_position_embedding=true
  model.inverse_position_initialization=copy model.use_direction_embedding=true
  model.use_chunk_loss=true model.use_discrete_loss=true
  model.use_memory_loss=true model.use_terminal_loss=true
  model.use_terminal_chunk=false model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true model.stop_gradient_memory_target=true
  model.stop_gradient_memory_observation=false
  model.memory_observation_gradient_scale=1.0
  model.memory_scale_mode=fixed model.memory_scale_granularity=global
  model.terminal_scale_mode=fixed model.terminal_scale_granularity=global
  model.observation_noise_std=0.0 model.generation_noise_std=0.0
  wandb.mode=online wandb.project=aux-assoc-recall wandb.group="$group"
)

# pressure is proportional to 1/scale^2. These values are exact +/-10%
# pressure moves around rho=31.622777 and tau=11.313708.
conditions=(
  "rho-weak10|33.333334|11.313708"
  "rho-strong10|30.151135|11.313708"
  "tau-weak10|31.622777|11.925695"
  "tau-strong10|31.622777|10.787197"
)

pids=()
names=()
for condition in "${conditions[@]}"; do
  IFS='|' read -r label rho tau <<<"$condition"
  name="rmt-d32-static-${label}-s${seed}-v1"
  python -m train "${common[@]}" train.seed="$seed" \
    model.rho="$rho" model.tau="$tau" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" wandb.id="$name" hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

status=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}" || status=1
  tail -n 40 "$output_root/${names[$index]}.log"
done
exit "$status"
