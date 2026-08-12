#!/usr/bin/env bash

# Full-length static rho/tau screen for the GRU AUX model with an unshared
# inverse recurrent stack, shared token embedding/head, and no direction
# embedding. Each shard runs one seed and six scale conditions on one GPU.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0, 1, or 2}"
case "$shard" in
  0) seed=177 ;;
  1) seed=178 ;;
  2) seed=179 ;;
  *) echo "SHARD must be 0, 1, or 2" >&2; exit 2 ;;
esac

group="${WANDB_GROUP:-gru-d32-shared-token-unshared-inverse-scale-20260812-v1}"
suffix="${RUN_SUFFIX:-stuinv-nodir-scale400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 10))"
output_root="${OUTPUT_ROOT:-/tmp/gru-d32-shared-token-unshared-inverse-scale}/shard-${shard}"
mkdir -p "$output_root"

common=(
  experiment=synthetics/associative_recall/gru_aux
  trainer.max_epochs="$max_epochs"
  callbacks=full_run
  +trainer.check_val_every_n_epoch=5
  +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50
  trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0
  +trainer.precision=32
  trainer.gradient_clip_val=0.0
  train.test=false
  train.seed="$seed"
  loader.num_workers=0
  scheduler=linear_warmup
  scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
  optimizer.lr=0.0005
  optimizer.weight_decay=0.1
  model.d_model=32
  model.n_layer=2
  model.chunk_size=4
  model.share_inverse=false
  model.share_inverse_embedding=true
  model.share_inverse_head=true
  model.use_direction_embedding=false
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
  task.aux_weight=0.1
  task.aux_weight_final=0.1
  task.aux_weight_schedule=fixed
  task.aux_weight_decay_start_step=0
  task.aux_weight_decay_end_step=1
  task.aux_activation_lm_loss_threshold=null
  task.aux_solved_ce_threshold=null
  task.aux_gradient_norm_interval="$steps_per_epoch"
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
)

# Refuse to run if the requested sharing topology is not exactly the intended
# shared-token/head, unshared-inverse architecture.
python - <<'PY'
from src.models.sequence.gru_aux import GRUAuxLM

model = GRUAuxLM(
    d_model=32,
    n_layer=2,
    vocab_size=20,
    share_inverse=False,
    share_inverse_embedding=True,
    share_inverse_head=True,
    use_direction_embedding=False,
)
assert model.gru is not model.inverse_gru
forward_ptrs = {parameter.data_ptr() for parameter in model.gru.parameters()}
inverse_ptrs = {parameter.data_ptr() for parameter in model.inverse_gru.parameters()}
assert forward_ptrs.isdisjoint(inverse_ptrs)
assert model.embedding is model.inverse_embedding
assert not hasattr(model, "inverse_head")
assert model.direction_embedding is None
print("Verified unshared inverse GRU, shared token embedding/head, no direction embedding")
PY

pids=()
names=()
launch() {
  local condition="$1"
  local rho="$2"
  local tau="$3"
  local name="gru-d32-stuinv-${condition}-s${seed}-${suffix}"
  echo "Launching $name (rho=$rho, tau=$tau)"
  python -m train \
    "${common[@]}" \
    model.rho="$rho" \
    model.tau="$tau" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

# A log-coordinate screen around the natural RMT scale. The old GRU screen
# covered only rho/tau <= 1, which imposed much stronger Gaussian pressure.
launch anchor       31.622777 11.925695
launch rho-strong   10.000000 11.925695
launch rho-weak    100.000000 11.925695
launch tau-strong   31.622777  3.771236
launch tau-weak     31.622777 37.712359
launch both-weak   100.000000 37.712359

if (( ${#pids[@]} != 6 )); then
  echo "Expected exactly 6 processes, got ${#pids[@]}" >&2
  exit 2
fi
echo "GRU scale shard $shard launched ${#pids[@]} processes for seed $seed"

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
