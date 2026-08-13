#!/usr/bin/env bash

# Fixed, pre-registered target-stop-gradient x memory-pressure screen for the
# locked shared-token/head, unshared-inverse RMT recipe.  SG pairs are adjacent
# in the job list so each pair runs on the same GPU shard.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be set to an integer from 0 through 8}"
if (( shard < 0 || shard > 8 )); then
  echo "SHARD must be between 0 and 8" >&2
  exit 2
fi

group="${WANDB_GROUP:-rmt-d32-unshared-targetsg-pressure-tune-20260813-v1}"
suffix="${RUN_SUFFIX:-targetsg-pressure400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 5))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-unshared-targetsg-pressure-tune}/shard-${shard}"
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
  model.use_chunk_loss=true
  model.use_discrete_loss=true
  model.use_memory_loss=true
  model.use_terminal_loss=true
  model.use_terminal_chunk=false
  model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true
  model.stop_gradient_memory_observation=false
  model.memory_observation_gradient_scale=1.0
  model.memory_scale_mode=fixed
  model.memory_scale_granularity=global
  model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global
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

# Target SG and fixed rho affect only training losses/gradients, never the
# inference graph or its initialization.
python - <<'PY'
import torch
from src.models.sequence.rmt_aux import RMTAuxLM

base = dict(
    d_model=32, n_layer=2, d_inner=128, n_heads=1, vocab_size=20,
    chunk_size=4, num_memory_tokens=4, share_inverse=False,
    share_inverse_embedding=True, share_inverse_head=True,
    share_inverse_position_embedding=False, inverse_position_initialization="copy",
    use_direction_embedding=False, use_chunk_loss=True, use_discrete_loss=True,
    use_memory_loss=True, use_terminal_loss=True, use_terminal_chunk=False,
    use_terminal_chunk_loss=False, learnable_terminal_target=True,
    stop_gradient_memory_observation=False, memory_observation_gradient_scale=1.0,
    memory_scale_mode="fixed", memory_scale_granularity="global",
    terminal_scale_mode="fixed", terminal_scale_granularity="global",
    tau=11.925695,
)
models = []
for stop_gradient in (False, True):
    for rho in (63.245553, 44.721360, 31.622777, 22.360680, 15.811388):
        torch.manual_seed(182)
        models.append(RMTAuxLM(
            **base, rho=rho, stop_gradient_memory_target=stop_gradient
        ))
prefixes = ("embedding.", "initial_memory", "forward_queries", "position_embedding",
            "blocks.", "final_norm.")
states = [model.state_dict() for model in models]
keys = [key for key in states[0] if key.startswith(prefixes)]
assert keys
assert all(torch.equal(states[0][key], state[key]) for state in states[1:] for key in keys)
print(f"Verified {len(keys)} inference tensors across all ten SG/pressure configurations")
PY

seeds=(182 183 184 185 186)
pressure_labels=(025 050 100 200 400)
rhos=(63.245553 44.721360 31.622777 22.360680 15.811388)
jobs=()
for seed in "${seeds[@]}"; do
  for index in "${!rhos[@]}"; do
    jobs+=("$seed|${pressure_labels[$index]}|${rhos[$index]}|on")
    jobs+=("$seed|${pressure_labels[$index]}|${rhos[$index]}|off")
  done
done

start="$((shard * 6))"
end="$((start + 6))"
if (( end > ${#jobs[@]} )); then
  end="${#jobs[@]}"
fi

pids=()
names=()
for ((job_index = start; job_index < end; job_index++)); do
  IFS='|' read -r seed pressure rho sg <<<"${jobs[$job_index]}"
  if [[ "$sg" == "on" ]]; then
    stop_gradient=true
  else
    stop_gradient=false
  fi
  name="rmt-d32-stuinv-wd01-targetsg-${sg}-p${pressure}-s${seed}-${suffix}"
  echo "Launching $name (rho=$rho, target_sg=$stop_gradient)"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    model.rho="$rho" \
    model.stop_gradient_memory_target="$stop_gradient" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" \
    wandb.id="$name" \
    hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

if (( ${#pids[@]} > 6 )); then
  echo "Refusing to run ${#pids[@]} processes on one GPU" >&2
  exit 2
fi
echo "Target-SG pressure shard $shard launched ${#pids[@]} processes"

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
