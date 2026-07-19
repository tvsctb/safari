#!/usr/bin/env bash

# Final 400-epoch comparison: a 2x2 RMT AUX/LR factorial plus the repository
# Transformer, each at five paired training seeds. SHARD selects one of four
# balanced groups (7/6/6/6 processes) so a single GPU never hosts > 8 jobs.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0, 1, 2, or 3}"
if (( shard < 0 || shard > 3 )); then
  echo "SHARD must be 0, 1, 2, or 3" >&2
  exit 2
fi

group="${WANDB_GROUP:-d32-final-25-20260719-v1}"
suffix="${RUN_SUFFIX:-final25-v1}"
max_epochs="${MAX_EPOCHS:-400}"
output_root="${OUTPUT_ROOT:-/tmp/safari-d32-final-25}/shard-${shard}"
mkdir -p "$output_root"

common_args=(
  trainer.max_epochs="$max_epochs"
  callbacks=full_run
  +trainer.check_val_every_n_epoch=5
  +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50
  trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0
  +trainer.precision=32
  train.test=false
  loader.num_workers=0
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
  +wandb.log_model=all
)

rmt_args=(
  experiment=synthetics/associative_recall/rmt_aux
  task.aux_weight=1.0
  task.aux_gradient_norm_interval=157
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
  model.rho=63.245553
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.tau=12.649111
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
)

pids=()
names=()

launch_rmt() {
  local aux="$1"
  local lr="$2"
  local seed="$3"
  local lr_label
  local aux_label
  local name
  local aux_weight
  local grad_interval
  local use_aux

  if [[ "$lr" == "0.0005" ]]; then
    lr_label="lr5e-4"
  else
    lr_label="lr3e-3"
  fi
  if [[ "$aux" == "true" ]]; then
    aux_label="aux"
    aux_weight="1.0"
    grad_interval="157"
    use_aux="true"
  else
    aux_label="noaux"
    aux_weight="0.0"
    grad_interval="0"
    use_aux="false"
  fi

  name="rmt-d32-${aux_label}-${lr_label}-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    "${common_args[@]}" \
    "${rmt_args[@]}" \
    train.seed="$seed" \
    optimizer.lr="$lr" \
    task.aux_weight="$aux_weight" \
    task.aux_gradient_norm_interval="$grad_interval" \
    model.use_chunk_loss="$use_aux" \
    model.use_discrete_loss="$use_aux" \
    model.use_memory_loss="$use_aux" \
    model.use_terminal_loss="$use_aux" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

launch_transformer() {
  local seed="$1"
  local name="transformer-d32-repo-lr5e-4-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    "${common_args[@]}" \
    experiment=synthetics/associative_recall/transformer \
    train.seed="$seed" \
    optimizer.lr=0.0005 \
    optimizer.weight_decay=0.1 \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

# Keep the five conditions adjacent for readability, then distribute them by
# global index to obtain balanced 7/6/6/6 shards.
jobs=()
for seed in 0 1 2 3 4; do
  jobs+=("rmt|false|0.0005|$seed")
  jobs+=("rmt|false|0.003|$seed")
  jobs+=("rmt|true|0.0005|$seed")
  jobs+=("rmt|true|0.003|$seed")
  jobs+=("transformer|||$seed")
done

for index in "${!jobs[@]}"; do
  if (( index % 4 != shard )); then
    continue
  fi
  IFS='|' read -r kind aux lr seed <<<"${jobs[$index]}"
  if [[ "$kind" == "rmt" ]]; then
    launch_rmt "$aux" "$lr" "$seed"
  else
    launch_transformer "$seed"
  fi
done

if (( ${#pids[@]} > 8 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "Shard $shard launched ${#pids[@]} processes"

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
