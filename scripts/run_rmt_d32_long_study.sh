#!/usr/bin/env bash

# Long selected-setting confirmation, ablation, and warmup studies.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

wave="${EXPERIMENT_WAVE:-CONF}"
if [[ "$wave" != "CONF" && "$wave" != "ABL" && "$wave" != "WARM" ]]; then
  echo "EXPERIMENT_WAVE must be CONF, ABL, or WARM" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-long-${wave}-20260719-v1}"
suffix="${RUN_SUFFIX:-long-${wave}-v1}"
max_epochs="${MAX_EPOCHS:-240}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-long}/wave-${wave}"
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
  task.aux_weight=1.0
  optimizer.lr=3e-3
  optimizer.weight_decay=0.1
  model.d_model=32
  model.d_inner=128
  model.n_layer=2
  model.n_heads=1
  model.use_discrete_loss=true
  model.use_terminal_chunk=false
  model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true
  model.stop_gradient_memory_target=true
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.rho=63.245553
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.tau=12.649111
  model.generation_noise_std=0.0
)

pids=()
names=()

launch() {
  local label="$1"
  local seed="$2"
  local observation_noise="$3"
  local use_chunk="$4"
  local use_memory="$5"
  local use_terminal="$6"
  local warmup_steps="$7"
  local name="rmt-d32-${label}-s${seed}-${suffix}"

  echo "Launching $name (obs=$observation_noise, chunk=$use_chunk, memory=$use_memory, terminal=$use_terminal, warmup=$warmup_steps)"
  python -m train \
    "${common_args[@]}" \
    train.seed="$seed" \
    model.observation_noise_std="$observation_noise" \
    model.use_chunk_loss="$use_chunk" \
    model.use_memory_loss="$use_memory" \
    model.use_terminal_loss="$use_terminal" \
    scheduler.num_warmup_steps="$warmup_steps" \
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

if [[ "$wave" == "CONF" ]]; then
  # 157 steps/epoch * 240 epochs = 37680 steps; 3768 is the default 10% warmup.
  for seed in 0 1 2 3 4; do
    launch "selected" "$seed" 0.0 true true true 3768
  done
elif [[ "$wave" == "ABL" ]]; then
  # The no-chunk ablation retains the boundary discrete CE.
  launch "observation0p03" 0 0.03 true true true 3768
  launch "observation0p1" 0 0.1 true true true 3768
  launch "no-chunk" 0 0.0 false true true 3768
  launch "no-memory" 0 0.0 true false true 3768
  launch "no-terminal" 0 0.0 true true false 3768
else
  launch "warmup0pct" 0 0.0 true true true 0
  launch "warmup5pct" 0 0.0 true true true 1884
  launch "warmup10pct" 0 0.0 true true true 3768
  launch "warmup15pct" 0 0.0 true true true 5652
  launch "warmup20pct" 0 0.0 true true true 7536
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
