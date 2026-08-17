#!/usr/bin/env bash
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY"'
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to 0, 1, 2, or 3}"
case "$shard" in
  0) seeds=(157 158 159 160 161 162) ;;
  1) seeds=(163 164 165 166 167 168) ;;
  2) seeds=(169 170 171 172 173 174) ;;
  3) seeds=(175 176) ;;
  *) echo "SHARD must be between 0 and 3" >&2; exit 2 ;;
esac

group="${WANDB_GROUP:-rmt-d32-unshared-targetsgoff-p1-secondary-20260815-v1}"
suffix="${RUN_SUFFIX:-targetsgoff-p1-secondary400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 5))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-unshared-targetsgoff-p1-secondary}/shard-${shard}"
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
  model.use_chunk_loss=true
  model.use_discrete_loss=true
  model.use_memory_loss=true
  model.use_terminal_loss=true
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
  task.aux_gradient_norm_interval=0
  task.aux_gradient_norm_steps="$gradient_steps"
  wandb.mode=online
  wandb.project=aux-assoc-recall
  wandb.group="$group"
)

# The selected objective must preserve the paired baseline's forward-path
# initialization exactly.
python - <<'PY'
import torch
from src.models.sequence.rmt_aux import RMTAuxLM

base = dict(
    d_model=32, n_layer=2, d_inner=128, n_heads=1, vocab_size=20,
    chunk_size=4, num_memory_tokens=4, share_inverse=False,
    share_inverse_embedding=True, share_inverse_head=True,
    share_inverse_position_embedding=False, inverse_position_initialization="copy",
    use_direction_embedding=False, use_terminal_chunk=False,
    use_terminal_chunk_loss=False, learnable_terminal_target=True,
    stop_gradient_memory_observation=False, memory_observation_gradient_scale=1.0,
    memory_scale_mode="fixed", memory_scale_granularity="global",
    terminal_scale_mode="fixed", terminal_scale_granularity="global",
    tau=11.925695,
)
torch.manual_seed(157)
selected = RMTAuxLM(
    **base, rho=31.622777, stop_gradient_memory_target=False,
    use_chunk_loss=True, use_discrete_loss=True,
    use_memory_loss=True, use_terminal_loss=True,
)
torch.manual_seed(157)
baseline = RMTAuxLM(
    **base, rho=31.622777, stop_gradient_memory_target=True,
    use_chunk_loss=False, use_discrete_loss=False,
    use_memory_loss=False, use_terminal_loss=False,
)
prefixes = (
    "embedding.", "initial_memory", "forward_queries", "position_embedding",
    "blocks.", "final_norm.",
)
states = selected.state_dict(), baseline.state_dict()
keys = [key for key in states[0] if key.startswith(prefixes)]
assert len(keys) == 30, len(keys)
assert all(torch.equal(states[0][key], states[1][key]) for key in keys)
print("Verified 30 inference tensors against the paired no-AUX baseline")
PY

pids=()
names=()
for seed in "${seeds[@]}"; do
  name="rmt-d32-stuinv-wd01-fullaux-targetsgoff-p1-s${seed}-${suffix}"
  wandb_dir="/tmp/wandb/$name"
  mkdir -p "$wandb_dir"
  echo "Launching $name"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    wandb.save_dir="$wandb_dir" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

if (( ${#pids[@]} > 6 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "SG-off p1 secondary shard $shard launched ${#pids[@]} processes"

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${names[$index]}"
  else
    echo "Failed ${names[$index]}"
    status=1
  fi
  tail -n 40 "$output_root/${names[$index]}.log"
done
exit "$status"
