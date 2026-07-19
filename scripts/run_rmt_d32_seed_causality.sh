#!/usr/bin/env bash

# Separate model-initialization randomness from shuffled training-batch order.
# Dataset content and post-initialization runtime RNG remain fixed.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-seed-causality-20260719-v1}"
suffix="${RUN_SUFFIX:-seed-causality-v1}"
max_epochs="${MAX_EPOCHS:-180}"
steps_per_epoch="157"
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 10))"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-seed-causality}"
mkdir -p "$output_root"

common_args=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs="$max_epochs"
  callbacks=full_run
  +trainer.check_val_every_n_epoch=5
  +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=25
  trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0
  +trainer.precision=32
  trainer.gradient_clip_val=0.0
  train.test=false
  train.seed=0
  train.runtime_seed=0
  +dataset.seed=0
  loader.num_workers=0
  scheduler=linear_warmup
  scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
  task.aux_gradient_norm_interval=25
  task.aux_weight=1.0
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
  model.rho=31.622777
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.tau=12.649111
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
  +wandb.log_model=true
)

pids=()
names=()

launch() {
  local label="$1"
  local model_seed="$2"
  local loader_seed="$3"
  local name="rmt-d32-${label}-${suffix}"

  echo "Launching $name (model_seed=$model_seed, loader_seed=$loader_seed)"
  python -m train \
    "${common_args[@]}" \
    train.model_seed="$model_seed" \
    dataset.loader_seed="$loader_seed" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

# The init0-order0 run is shared by both controlled comparisons.
conditions=(
  "init0-order0|0|0"
  "init1-order0|1|0"
  "init2-order0|2|0"
  "init3-order0|3|0"
  "init0-order1|0|1"
  "init0-order2|0|2"
  "init0-order3|0|3"
)

for condition in "${conditions[@]}"; do
  IFS='|' read -r label model_seed loader_seed <<<"$condition"
  launch "$label" "$model_seed" "$loader_seed"
done

if (( ${#pids[@]} > 8 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "Launched ${#pids[@]} controlled processes"

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
