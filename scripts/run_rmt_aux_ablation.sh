#!/usr/bin/env bash

# One-factor ablations of the successful fixed-scale RMT setting.  Each wave
# launches exactly five independent processes on a single GPU.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

wave="${EXPERIMENT_WAVE:-A}"
if [[ "$wave" != "A" && "$wave" != "B" && "$wave" != "C" ]]; then
  echo "EXPERIMENT_WAVE must be A, B, or C" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-aux-ablation-tau-20260718-v1}"
suffix="${RUN_SUFFIX:-v1}"
max_epochs="${MAX_EPOCHS:-120}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-aux-ablation}/wave-${wave}"
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
  task.aux_weight=0.1
  task.aux_gradient_norm_interval=157
  model.rho=20.0
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.use_terminal_chunk_loss=false
)

python -m train --cfg job \
  "${common_args[@]}" \
  model.use_chunk_loss=true \
  model.use_discrete_loss=true \
  model.use_memory_loss=true \
  model.use_terminal_loss=true \
  model.stop_gradient_memory_target=true \
  model.learnable_terminal_target=true \
  model.use_terminal_chunk=false \
  model.observation_noise_std=0.0 \
  model.generation_noise_std=0.0 \
  model.tau=3.0 \
  wandb.mode=disabled \
  >"$output_root/preflight-config.yaml"

pids=()
names=()

launch() {
  local label="$1"
  local use_terminal_loss="$2"
  local use_discrete_loss="$3"
  local use_memory_loss="$4"
  local use_chunk_loss="$5"
  local stop_gradient="$6"
  local learnable_target="$7"
  local use_terminal_chunk="$8"
  local observation_noise="$9"
  local generation_noise="${10}"
  local tau="${11}"
  local name="rmt-ablate-${label}-s0-${suffix}"

  echo "Launching $name"
  python -m train \
    "${common_args[@]}" \
    model.use_terminal_loss="$use_terminal_loss" \
    model.use_discrete_loss="$use_discrete_loss" \
    model.use_memory_loss="$use_memory_loss" \
    model.use_chunk_loss="$use_chunk_loss" \
    model.stop_gradient_memory_target="$stop_gradient" \
    model.learnable_terminal_target="$learnable_target" \
    model.use_terminal_chunk="$use_terminal_chunk" \
    model.observation_noise_std="$observation_noise" \
    model.generation_noise_std="$generation_noise" \
    model.tau="$tau" \
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

# label, terminal, discrete, memory, chunk, stopgrad, learned target, CK,
# observation noise, generation noise, tau
if [[ "$wave" == "A" ]]; then
  launch "control" true true true true true true false 0.0 0.0 3.0
  launch "no-terminal" false true true true true true false 0.0 0.0 3.0
  launch "no-memory-nll" true true false true true true false 0.0 0.0 3.0
  launch "no-discrete-ce" true false true true true true false 0.0 0.0 3.0
  launch "no-memory-block" true false false true true true false 0.0 0.0 3.0
elif [[ "$wave" == "B" ]]; then
  launch "no-chunk-ce" true true true false true true false 0.0 0.0 3.0
  launch "no-stopgrad" true true true true false true false 0.0 0.0 3.0
  launch "zero-terminal-target" true true true true true false false 0.0 0.0 3.0
  launch "ck" true true true true true true true 0.0 0.0 3.0
  launch "observation-noise" true true true true true true false 0.1 0.0 3.0
else
  launch "generation-noise" true true true true true true false 0.0 0.1 3.0
  launch "tau1" true true true true true true false 0.0 0.0 1.0
  launch "tau2" true true true true true true false 0.0 0.0 2.0
  launch "tau5" true true true true true true false 0.0 0.0 5.0
  launch "tau10" true true true true true true false 0.0 0.0 10.0
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
