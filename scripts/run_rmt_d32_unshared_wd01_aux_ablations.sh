#!/usr/bin/env bash

# No-retuning component ablations for the locked unshared-inverse RMT recipe.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0, 1, 2, 3, or 4}"
case "$shard" in
  0) seeds=(167 168) ;;
  1) seeds=(169 170) ;;
  2) seeds=(171 172) ;;
  3) seeds=(173 174) ;;
  4) seeds=(175 176) ;;
  *) echo "SHARD must be 0, 1, 2, 3, or 4" >&2; exit 2 ;;
esac

group="${WANDB_GROUP:-rmt-d32-unshared-wd01-aux-ablation-20260812-v1}"
suffix="${RUN_SUFFIX:-stuinvwd01abl400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 5))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-unshared-wd01-aux-ablation}/shard-${shard}"
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
  task.aux_weight=0.10
  task.aux_weight_final=0.10
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

# Loss flags must not perturb the forward-path initialization.
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
flags = [
    dict(use_chunk_loss=True, use_discrete_loss=True, use_memory_loss=True, use_terminal_loss=True),
    dict(use_chunk_loss=False, use_discrete_loss=False, use_memory_loss=True, use_terminal_loss=True),
    dict(use_chunk_loss=True, use_discrete_loss=True, use_memory_loss=False, use_terminal_loss=True),
    dict(use_chunk_loss=True, use_discrete_loss=True, use_memory_loss=True, use_terminal_loss=False),
]
models = []
for flag in flags:
    torch.manual_seed(167)
    models.append(RMTAuxLM(**kwargs, **flag))
prefixes = ("embedding.", "initial_memory", "forward_queries", "position_embedding",
            "blocks.", "final_norm.")
states = [model.state_dict() for model in models]
keys = [key for key in states[0] if key.startswith(prefixes)]
assert keys
assert all(torch.equal(states[0][key], state[key]) for state in states[1:] for key in keys)
print(f"Verified {len(keys)} inference tensors across all four loss configurations")
PY

pids=()
names=()
launch() {
  local condition="$1"
  local seed="$2"
  local chunk discrete memory terminal
  case "$condition" in
    minus-token) chunk=false; discrete=false; memory=true; terminal=true ;;
    minus-memory) chunk=true; discrete=true; memory=false; terminal=true ;;
    minus-terminal) chunk=true; discrete=true; memory=true; terminal=false ;;
    *) echo "Unknown condition $condition" >&2; exit 2 ;;
  esac

  local name="rmt-d32-stuinv-wd01-${condition}-s${seed}-${suffix}"
  echo "Launching $name"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    model.use_chunk_loss="$chunk" \
    model.use_discrete_loss="$discrete" \
    model.use_memory_loss="$memory" \
    model.use_terminal_loss="$terminal" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

for seed in "${seeds[@]}"; do
  launch minus-token "$seed"
  launch minus-memory "$seed"
  launch minus-terminal "$seed"
done

if (( ${#pids[@]} > 8 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "Ablation shard $shard launched ${#pids[@]} processes"

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
