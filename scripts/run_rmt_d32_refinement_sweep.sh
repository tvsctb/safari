#!/usr/bin/env bash

# d32 full-auxiliary refinement after the coarse pressure/LR screen.
# The anchor uses lambda/rho^2=0.00025 and lambda/tau^2=0.00625,
# one quarter of each continuous-state pressure in the original setting.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

wave="${EXPERIMENT_WAVE:-D}"
if [[ "$wave" != "D" && "$wave" != "E" && "$wave" != "F" ]]; then
  echo "EXPERIMENT_WAVE must be D, E, or F" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-refine-${wave}-20260719-v1}"
suffix="${RUN_SUFFIX:-refine-${wave}-v1}"
max_epochs="${MAX_EPOCHS:-160}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-d32-refine}/wave-${wave}"
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
  local weight_decay="$6"
  local name="rmt-d32-${label}-s0-${suffix}"

  echo "Launching $name (lambda=$aux_weight, rho=$rho, tau=$tau, lr=$learning_rate, wd=$weight_decay)"
  python -m train \
    "${common_args[@]}" \
    task.aux_weight="$aux_weight" \
    model.rho="$rho" \
    model.tau="$tau" \
    optimizer.lr="$learning_rate" \
    optimizer.weight_decay="$weight_decay" \
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

if [[ "$wave" == "D" ]]; then
  # Preserve both weakened continuous-state pressures while sweeping CE pressure.
  launch "lambda0p25" 0.25 31.622777 6.324555 4e-3 0.1
  launch "lambda0p5" 0.5 44.721360 8.944272 4e-3 0.1
  launch "lambda1" 1.0 63.245553 12.649111 4e-3 0.1
  launch "lambda2" 2.0 89.442719 17.888544 4e-3 0.1
  launch "lambda4" 4.0 126.491106 25.298221 4e-3 0.1
elif [[ "$wave" == "E" ]]; then
  # LR refinement around the 4e-3 transition found by the coarse screen.
  # The 4e-3 anchor is supplied by wave D.
  launch "lr1e-3" 1.0 63.245553 12.649111 1e-3 0.1
  launch "lr2e-3" 1.0 63.245553 12.649111 2e-3 0.1
  launch "lr3e-3" 1.0 63.245553 12.649111 3e-3 0.1
  launch "lr5e-3" 1.0 63.245553 12.649111 5e-3 0.1
  launch "lr7e-3" 1.0 63.245553 12.649111 7e-3 0.1
else
  # Weight-decay refinement. The 0.1 anchor is supplied by wave D.
  launch "wd0" 1.0 63.245553 12.649111 4e-3 0.0
  launch "wd1e-4" 1.0 63.245553 12.649111 4e-3 1e-4
  launch "wd1e-3" 1.0 63.245553 12.649111 4e-3 1e-3
  launch "wd1e-2" 1.0 63.245553 12.649111 4e-3 1e-2
  launch "wd3e-2" 1.0 63.245553 12.649111 4e-3 3e-2
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
