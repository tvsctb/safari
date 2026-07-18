#!/usr/bin/env bash

# Short, fixed-scale screen for the RMT auxiliary objective.  Five independent
# training processes share one GPU; wave A/B intentionally stay disjoint so
# each VESSL run uses exactly five processes.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

wave="${EXPERIMENT_WAVE:-A}"
if [[ "$wave" != "A" && "$wave" != "B" && "$wave" != "C" ]]; then
  echo "EXPERIMENT_WAVE must be A, B, or C" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-fixed-scale-lambda-20260718-v1}"
suffix="${RUN_SUFFIX:-v1}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-fixed-scale-screen}/wave-${wave}"
mkdir -p "$output_root"

common_args=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs=100
  +trainer.check_val_every_n_epoch=5
  +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50
  trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0
  +trainer.precision=32
  train.seed=0
  train.test=false
  loader.num_workers=0
  task.aux_gradient_norm_interval=157
  model.use_chunk_loss=true
  model.use_memory_loss=true
  model.use_terminal_loss=true
  model.use_terminal_chunk=false
  model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true
  model.memory_scale_mode=fixed
  model.terminal_scale_mode=fixed
  model.memory_scale_granularity=global
  model.terminal_scale_granularity=global
  model.tau=3.0
  model.stop_gradient_memory_target=true
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
)

# Validate the exact Hydra option set without creating a run or allocating the
# five workers.
python -m train --cfg job \
  "${common_args[@]}" \
  task.aux_weight=0.03 \
  model.rho=15.0 \
  wandb.mode=disabled \
  >"$output_root/preflight-config.yaml"

pids=()
names=()

launch() {
  local label="$1"
  local aux_weight="$2"
  local rho="$3"
  local seed="${4:-0}"
  local name="rmt-fixed-${label}-s${seed}-${suffix}"

  echo "Launching $name (lambda=$aux_weight, rho=$rho, tau=3.0)"
  python -m train \
    "${common_args[@]}" \
    train.seed="$seed" \
    task.aux_weight="$aux_weight" \
    model.rho="$rho" \
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

if [[ "$wave" == "A" ]]; then
  launch "baseline" 0.0 15.0
  launch "rho10-lambda0p01" 0.01 10.0
  launch "rho15-lambda0p01" 0.01 15.0
  launch "rho20-lambda0p01" 0.01 20.0
  launch "rho10-lambda0p03" 0.03 10.0
elif [[ "$wave" == "B" ]]; then
  launch "rho15-lambda0p03" 0.03 15.0
  launch "rho20-lambda0p03" 0.03 20.0
  launch "rho10-lambda0p1" 0.1 10.0
  launch "rho15-lambda0p1" 0.1 15.0
  launch "rho20-lambda0p1" 0.1 20.0
else
  # Follow-up around the first screen's effective-memory-gradient window.
  # The first condition changes only the seed of its leading candidate.
  launch "rho10-lambda0p03-rep" 0.03 10.0 1
  launch "rho12-lambda0p03" 0.03 12.0
  launch "rho12-lambda0p05" 0.05 12.0
  launch "rho15-lambda0p05" 0.05 15.0
  launch "rho15-lambda0p07" 0.07 15.0
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
