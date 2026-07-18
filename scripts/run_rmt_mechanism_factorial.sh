#!/usr/bin/env bash

set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

wave="${EXPERIMENT_WAVE:-A}"
if [[ "$wave" != "A" && "$wave" != "B" ]]; then
  echo "EXPERIMENT_WAVE must be A or B" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-mechanism-3x2x2-20260718-v1}"
suffix="${RUN_SUFFIX:-v1}"
output_root="${OUTPUT_ROOT:-/tmp/safari-rmt-mechanism}/wave-${wave}"
mkdir -p "$output_root"

pids=()
names=()

launch() {
  local label="$1"
  local aux_weight="$2"
  local noise="$3"
  local scale_mode="$4"
  local stop_gradient="$5"
  local observation_noise=0.0
  local generation_noise=0.0
  local name="rmt-mech-${label}-s0-${suffix}"

  case "$noise" in
    none) ;;
    obs) observation_noise=0.1 ;;
    gen) generation_noise=0.1 ;;
    *) echo "unknown noise condition: $noise" >&2; return 2 ;;
  esac

  echo "Launching $name (aux_weight=$aux_weight, noise=$noise, scale=$scale_mode, stop_gradient=$stop_gradient)"
  python -m train \
    experiment=synthetics/associative_recall/rmt_aux \
    trainer.max_epochs=160 \
    trainer.check_val_every_n_epoch=5 \
    trainer.num_sanity_val_steps=0 \
    trainer.log_every_n_steps=50 \
    trainer.limit_train_batches=1.0 \
    trainer.limit_val_batches=1.0 \
    trainer.precision=32 \
    train.seed=0 \
    train.test=false \
    loader.num_workers=0 \
    task.aux_weight="$aux_weight" \
    model.use_chunk_loss=true \
    model.use_memory_loss=true \
    model.use_terminal_loss=true \
    model.use_terminal_chunk=false \
    model.use_terminal_chunk_loss=false \
    model.learnable_terminal_target=true \
    model.memory_scale_mode="$scale_mode" \
    model.memory_scale_granularity=global \
    model.terminal_scale_mode="$scale_mode" \
    model.terminal_scale_granularity=global \
    model.rho=1.0 \
    model.tau=10.0 \
    model.stop_gradient_memory_target="$stop_gradient" \
    model.observation_noise_std="$observation_noise" \
    model.generation_noise_std="$generation_noise" \
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
  launch "baseline" 0.0 none fixed false
  launch "none-fixed-nosg" 0.1 none fixed false
  launch "none-fixed-sg" 0.1 none fixed true
  launch "none-learned-nosg" 0.1 none learned false
  launch "none-learned-sg" 0.1 none learned true
  launch "obs-fixed-nosg" 0.1 obs fixed false
  launch "obs-fixed-sg" 0.1 obs fixed true
else
  launch "obs-learned-nosg" 0.1 obs learned false
  launch "obs-learned-sg" 0.1 obs learned true
  launch "gen-fixed-nosg" 0.1 gen fixed false
  launch "gen-fixed-sg" 0.1 gen fixed true
  launch "gen-learned-nosg" 0.1 gen learned false
  launch "gen-learned-sg" 0.1 gen learned true
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
