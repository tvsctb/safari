#!/usr/bin/env bash

# Fresh paired full-AUX versus no-AUX evaluation for the selected natural
# shared-token/head, unshared-inverse RMT recipe with AdamW weight decay 0.1.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cohort="${COHORT:-confirmation}"
shard="${SHARD:?SHARD must be set to 0, 1, 2, or 3}"
case "$cohort:$shard" in
  confirmation:0) seeds=(157 158 159) ;;
  confirmation:1) seeds=(160 161 162) ;;
  confirmation:2) seeds=(163 164 165) ;;
  confirmation:3) seeds=(166) ;;
  replication:0) seeds=(167 168 169) ;;
  replication:1) seeds=(170 171 172) ;;
  replication:2) seeds=(173 174 175) ;;
  replication:3) seeds=(176) ;;
  *) echo "COHORT must be confirmation or replication and SHARD must be 0-3" >&2; exit 2 ;;
esac

group="${WANDB_GROUP:-rmt-d32-unshared-wd01-aux-paired-20260811-v1}"
suffix="${RUN_SUFFIX:-stuinvwd01paired400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 5))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-unshared-wd01-aux-paired}/shard-${shard}"
mkdir -p "$output_root"

common=(
  experiment=synthetics/associative_recall/rmt_aux
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
  loader.num_workers=0
  scheduler=linear_warmup
  scheduler.num_warmup_steps="$warmup_steps"
  scheduler.num_training_steps="$training_steps"
  optimizer.lr=0.001
  optimizer.weight_decay=0.1
  model.d_model=32
  model.d_inner=128
  model.n_layer=2
  model.n_heads=1
  model.chunk_size=4
  model.num_memory_tokens=4
  model.share_inverse=false
  model.share_inverse_embedding=true
  model.share_inverse_head=true
  model.share_inverse_position_embedding=false
  model.inverse_position_initialization=copy
  model.use_direction_embedding=false
  model.use_terminal_chunk=false
  model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true
  model.stop_gradient_memory_target=true
  model.stop_gradient_memory_observation=false
  model.memory_observation_gradient_scale=1.0
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.rho=31.622777
  model.tau=11.925695
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
  task.aux_weight_schedule=fixed
  task.aux_weight_decay_start_step=0
  task.aux_weight_decay_end_step=1
  task.aux_activation_lm_loss_threshold=null
  task.aux_solved_ce_threshold=null
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
)

# The component flags affect loss computation only. Reconstruct both variants
# under the same RNG seed and verify every inference-time tensor is identical.
python - <<'PY'
import torch
from src.models.sequence.rmt_aux import RMTAuxLM

kwargs = dict(
    d_model=32, n_layer=2, d_inner=128, n_heads=1, vocab_size=20,
    chunk_size=4, num_memory_tokens=4, share_inverse=False,
    share_inverse_embedding=True, share_inverse_head=True,
    share_inverse_position_embedding=False, inverse_position_initialization="copy",
    use_direction_embedding=False, use_terminal_chunk=False,
    use_terminal_chunk_loss=False, learnable_terminal_target=True,
    stop_gradient_memory_target=True, stop_gradient_memory_observation=False,
    memory_observation_gradient_scale=1.0, memory_scale_mode="fixed",
    memory_scale_granularity="global", terminal_scale_mode="fixed",
    terminal_scale_granularity="global", rho=31.622777, tau=11.925695,
)
torch.manual_seed(157)
full = RMTAuxLM(**kwargs, use_chunk_loss=True, use_discrete_loss=True,
                use_memory_loss=True, use_terminal_loss=True)
torch.manual_seed(157)
none = RMTAuxLM(**kwargs, use_chunk_loss=False, use_discrete_loss=False,
                use_memory_loss=False, use_terminal_loss=False)
prefixes = ("embedding.", "initial_memory", "forward_queries", "position_embedding",
            "blocks.", "final_norm.")
full_state = full.state_dict()
none_state = none.state_dict()
keys = [key for key in full_state if key.startswith(prefixes)]
assert keys
assert all(torch.equal(full_state[key], none_state[key]) for key in keys)
print(f"Verified {len(keys)} inference tensors are bitwise identical")
PY

pids=()
names=()
launch() {
  local condition="$1"
  local seed="$2"
  local use_aux aux_weight grad_interval
  if [[ "$condition" == "fullaux" ]]; then
    use_aux=true
    aux_weight=0.10
    grad_interval="$steps_per_epoch"
  else
    use_aux=false
    aux_weight=0.0
    grad_interval=0
  fi

  local name="rmt-d32-stuinv-wd01-${condition}-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    task.aux_weight="$aux_weight" \
    task.aux_weight_final="$aux_weight" \
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

for seed in "${seeds[@]}"; do
  launch fullaux "$seed"
  launch noaux "$seed"
done

if (( ${#pids[@]} > 8 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "Paired shard $shard launched ${#pids[@]} processes"

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
