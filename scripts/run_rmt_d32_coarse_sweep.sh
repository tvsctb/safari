#!/usr/bin/env bash

# Wide d32 full-auxiliary sweep parameterized by the effective gradient
# coefficients lambda, lambda / rho^2, and lambda / tau^2.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

wave="${EXPERIMENT_WAVE:-A}"
if [[ "$wave" != "A" && "$wave" != "B" && "$wave" != "C" ]]; then
  echo "EXPERIMENT_WAVE must be A, B, or C" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-coarse-${wave}-20260718-v1}"
suffix="${RUN_SUFFIX:-coarse-${wave}-v1}"
max_epochs="${MAX_EPOCHS:-180}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-coarse}/wave-${wave}"
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
  task.aux_gradient_norm_interval=157
  model.d_model=32
  model.d_inner=128
  model.n_layer=2
  model.n_heads=1
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
)

pids=()
names=()

launch() {
  local label="$1"
  local aux_weight="$2"
  local rho="$3"
  local tau="$4"
  local learning_rate="$5"
  local name="rmt-d32-${label}-s0-${suffix}"

  echo "Launching $name (lambda=$aux_weight, rho=$rho, tau=$tau, lr=$learning_rate)"
  python -m train \
    "${common_args[@]}" \
    task.aux_weight="$aux_weight" \
    model.rho="$rho" \
    model.tau="$tau" \
    optimizer.lr="$learning_rate" \
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
  # Preserve lambda/rho^2=0.001 and lambda/tau^2=0.025 while
  # increasing the token-level auxiliary pressure by two orders of magnitude.
  launch "ce-lambda0p1" 0.1 10.0 2.0 5e-4
  launch "ce-lambda0p3" 0.3 17.320508 3.464102 5e-4
  launch "ce-lambda1" 1.0 31.622777 6.324555 5e-4
  launch "ce-lambda3" 3.0 54.772256 10.954451 5e-4
  launch "ce-lambda10" 10.0 100.0 20.0 5e-4
elif [[ "$wave" == "B" ]]; then
  # At lambda=3, vary one continuous-state pressure by four-fold around
  # the center while leaving the other pressure unchanged.
  launch "balance-center" 3.0 54.772256 10.954451 5e-4
  launch "memory-pressure-x4" 3.0 27.386128 10.954451 5e-4
  launch "memory-pressure-div4" 3.0 109.544512 10.954451 5e-4
  launch "terminal-pressure-x4" 3.0 54.772256 5.477226 5e-4
  launch "terminal-pressure-div4" 3.0 54.772256 21.908902 5e-4
else
  # Wide optimizer screen at the balanced lambda=3 point. The 5e-4
  # reference is already present in waves A and B.
  launch "lr1e-4" 3.0 54.772256 10.954451 1e-4
  launch "lr2e-4" 3.0 54.772256 10.954451 2e-4
  launch "lr1e-3" 3.0 54.772256 10.954451 1e-3
  launch "lr2e-3" 3.0 54.772256 10.954451 2e-3
  launch "lr4e-3" 3.0 54.772256 10.954451 4e-3
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
