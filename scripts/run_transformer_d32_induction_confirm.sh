#!/usr/bin/env bash

# Verify the repository's stock d32 Transformer on induction-head copying.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-transformer-d32-induction-confirm-20260812-v1}"
suffix="${RUN_SUFFIX:-transformer-induction-default400-v1}"
output_root="${OUTPUT_ROOT:-/tmp/transformer-d32-induction-confirm}"
mkdir -p "$output_root"

seeds=(167 168 169 170 171)
pids=()
names=()
for seed in "${seeds[@]}"; do
  name="transformer-d32-induction-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    experiment=synthetics/induction_head/transformer \
    trainer.max_epochs=400 \
    callbacks=full_run \
    +trainer.check_val_every_n_epoch=5 \
    +trainer.num_sanity_val_steps=0 \
    trainer.log_every_n_steps=50 \
    trainer.limit_train_batches=1.0 \
    trainer.limit_val_batches=1.0 \
    +trainer.precision=32 \
    train.test=false \
    loader.num_workers=0 \
    train.seed="$seed" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.mode=online \
    wandb.project=aux-assoc-recall \
    wandb.group="$group" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

if (( ${#pids[@]} > 8 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "Transformer induction confirmation launched ${#pids[@]} processes"

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
