#!/usr/bin/env bash

# No-retuning component ablations for a locked SG-off pressure recipe.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to an integer from 0 through 9}"
if (( shard < 0 || shard > 9 )); then
  echo "SHARD must be between 0 and 9" >&2
  exit 2
fi
first_seed="$((157 + 2 * shard))"
seeds=("$first_seed" "$((first_seed + 1))")

pressure_label="${PRESSURE_LABEL:-p1}"
rho="${RHO:-31.622777}"
group="${WANDB_GROUP:-rmt-d32-unshared-targetsgoff-${pressure_label}-ablation-20260815-v1}"
suffix="${RUN_SUFFIX:-targetsgoff-${pressure_label}-abl400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 5))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-unshared-targetsgoff-${pressure_label}-ablation}/shard-${shard}"
mkdir -p "$output_root"
gradient_steps="[0,3140,12560,31400,62643]"

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
  train.log_model_metrics_on_step=false
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
  model.stop_gradient_memory_target=false
  model.stop_gradient_memory_observation=false
  model.memory_observation_gradient_scale=1.0
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
  model.rho="$rho"
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
  task.aux_gradient_norm_interval=0
  task.aux_gradient_norm_steps="$gradient_steps"
  task.aux_metric_profile=compact
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
)

# Loss flags must not perturb the selected full model's inference path.
RHO="$rho" PRESSURE_LABEL="$pressure_label" python - <<'PY'
import os
import torch
from src.models.sequence.rmt_aux import RMTAuxLM

kwargs = dict(
    d_model=32, n_layer=2, d_inner=128, n_heads=1, vocab_size=20,
    chunk_size=4, num_memory_tokens=4, share_inverse=False,
    share_inverse_embedding=True, share_inverse_head=True,
    share_inverse_position_embedding=False, inverse_position_initialization="copy",
    use_direction_embedding=False, use_terminal_chunk=False,
    use_terminal_chunk_loss=False, learnable_terminal_target=True,
    stop_gradient_memory_target=False, stop_gradient_memory_observation=False,
    memory_observation_gradient_scale=1.0, memory_scale_mode="fixed",
    memory_scale_granularity="global", terminal_scale_mode="fixed",
    terminal_scale_granularity="global", rho=float(os.environ["RHO"]), tau=11.925695,
)
flags = [
    dict(use_chunk_loss=True, use_discrete_loss=True, use_memory_loss=True, use_terminal_loss=True),
    dict(use_chunk_loss=False, use_discrete_loss=False, use_memory_loss=True, use_terminal_loss=True),
    dict(use_chunk_loss=True, use_discrete_loss=True, use_memory_loss=False, use_terminal_loss=True),
    dict(use_chunk_loss=True, use_discrete_loss=True, use_memory_loss=True, use_terminal_loss=False),
]
models = []
for flag in flags:
    torch.manual_seed(157)
    models.append(RMTAuxLM(**kwargs, **flag))
prefixes = (
    "embedding.", "initial_memory", "forward_queries", "position_embedding",
    "blocks.", "final_norm.",
)
states = [model.state_dict() for model in models]
keys = [key for key in states[0] if key.startswith(prefixes)]
assert len(keys) == 30, len(keys)
assert all(torch.equal(states[0][key], state[key]) for state in states[1:] for key in keys)
print(
    "Verified 30 inference tensors across full and three SG-off "
    f"{os.environ['PRESSURE_LABEL']} ablations"
)
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

  local name="rmt-d32-stuinv-wd01-targetsgoff-${pressure_label}-${condition}-s${seed}-${suffix}"
  local wandb_dir="/tmp/wandb/$name"
  mkdir -p "$wandb_dir"
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
    wandb.save_dir="$wandb_dir" \
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

if (( ${#pids[@]} != 6 )); then
  echo "Expected exactly 6 processes, got ${#pids[@]}" >&2
  exit 2
fi
echo "SG-off ${pressure_label} ablation shard $shard launched exactly 6 processes"

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
