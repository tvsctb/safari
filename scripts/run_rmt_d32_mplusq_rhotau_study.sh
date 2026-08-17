#!/usr/bin/env bash

# Fresh memory-plus-query associative-recall study.  The no-AUX arm establishes
# a 20-seed architecture-specific baseline.  The screen is a fixed 5x5
# memory/terminal pressure grid crossed with target stop-gradient on/off on five
# matched tuning seeds.  No adaptive schedule or mid-run retuning is permitted.
set -euo pipefail

python -c 'import os; key=os.environ.get("WANDB_API_KEY", ""); assert key and key != "WANDB_API_KEY", "WANDB_API_KEY secret was not injected"'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

study="${STUDY:?STUDY must be noaux or screen}"
shard="${SHARD:?SHARD must be set}"
case "$study" in
  noaux) max_shard=3 ;;
  screen) max_shard=41 ;;
  *) echo "STUDY must be noaux or screen" >&2; exit 2 ;;
esac
if (( shard < 0 || shard > max_shard )); then
  echo "SHARD must be between 0 and $max_shard for $study" >&2
  exit 2
fi

project="${WANDB_PROJECT:-assoc-recall-rmt-write-memory}"
noaux_group="${NOAUX_GROUP:-rmt-d32-mplusq-noaux-20260817-v1}"
screen_group="${SCREEN_GROUP:-rmt-d32-mplusq-targetsg-rhotau-20260817-v1}"
suffix="${RUN_SUFFIX:-mplusq-rhotau400-v1}"
max_epochs="${MAX_EPOCHS:-400}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
warmup_steps="$((training_steps / 5))"
output_root="${OUTPUT_ROOT:-/output/rmt-d32-mplusq-rhotau-study}/${study}/shard-${shard}"
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
  train.fused_adamw=true
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
  model.write_input_mode=memory_plus_query
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
  model.observation_noise_std=0.0
  model.generation_noise_std=0.0
  task.aux_weight_schedule=fixed
  task.aux_weight_final=0.10
  task.aux_weight_decay_start_step=0
  task.aux_weight_decay_end_step=1
  task.aux_activation_lm_loss_threshold=null
  task.aux_solved_ce_threshold=null
  task.aux_diagnostic_interval=50
  wandb.mode=online
  wandb.project="$project"
)

# Scale and SG switches affect only the objective.  Verify that all 50 screen
# cells and no-AUX have identical fresh inference parameters, with nonzero
# learned write offsets under the new default architecture.
python - <<'PY'
import torch
from src.models.sequence.rmt_aux import RMTAuxLM

base = dict(
    d_model=32, n_layer=2, d_inner=128, n_heads=1, vocab_size=20,
    chunk_size=4, num_memory_tokens=4, write_input_mode="memory_plus_query",
    share_inverse=False, share_inverse_embedding=True, share_inverse_head=True,
    share_inverse_position_embedding=False, inverse_position_initialization="copy",
    use_direction_embedding=False, use_chunk_loss=True, use_discrete_loss=True,
    use_memory_loss=True, use_terminal_loss=True, use_terminal_chunk=False,
    use_terminal_chunk_loss=False, learnable_terminal_target=True,
    stop_gradient_memory_observation=False, memory_observation_gradient_scale=1.0,
    memory_scale_mode="fixed", memory_scale_granularity="global",
    terminal_scale_mode="fixed", terminal_scale_granularity="global",
)
rhos = (63.245553, 44.721360, 31.622777, 22.360680, 15.811388)
taus = (23.851390, 16.865480, 11.925695, 8.432740, 5.962847)
models = []
for stop_gradient in (False, True):
    for rho in rhos:
        for tau in taus:
            torch.manual_seed(202)
            models.append(RMTAuxLM(
                **base, rho=rho, tau=tau,
                stop_gradient_memory_target=stop_gradient,
            ))
torch.manual_seed(202)
baseline = RMTAuxLM(
    **base, rho=rhos[2], tau=taus[2],
    stop_gradient_memory_target=False,
)
prefixes = (
    "embedding.", "initial_memory", "forward_queries", "position_embedding",
    "blocks.", "final_norm.",
)
reference = baseline.state_dict()
keys = [key for key in reference if key.startswith(prefixes)]
assert len(keys) == 30, keys
assert all(
    torch.equal(reference[key], model.state_dict()[key])
    for model in models
    for key in keys
)
assert torch.count_nonzero(baseline.forward_queries) > 0
assert torch.count_nonzero(baseline.inverse_queries) > 0
print("Verified 30 inference tensors, 50 SG/rho/tau cells, no-AUX baseline, and nonzero write offsets")
PY

if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
  exit 0
fi

jobs=()
if [[ "$study" == "noaux" ]]; then
  for seed in $(seq 202 221); do
    jobs+=("$seed|noaux|100|31.622777|100|11.925695|off|false")
  done
else
  seeds=(202 203 204 205 206)
  pressure_labels=(025 050 100 200 400)
  rhos=(63.245553 44.721360 31.622777 22.360680 15.811388)
  taus=(23.851390 16.865480 11.925695 8.432740 5.962847)
  for seed in "${seeds[@]}"; do
    for rho_index in "${!rhos[@]}"; do
      for tau_index in "${!taus[@]}"; do
        jobs+=("$seed|full|${pressure_labels[$rho_index]}|${rhos[$rho_index]}|${pressure_labels[$tau_index]}|${taus[$tau_index]}|on|true")
        jobs+=("$seed|full|${pressure_labels[$rho_index]}|${rhos[$rho_index]}|${pressure_labels[$tau_index]}|${taus[$tau_index]}|off|false")
      done
    done
  done
fi

expected=20
[[ "$study" == "screen" ]] && expected=250
if (( ${#jobs[@]} != expected )); then
  echo "Expected $expected jobs, found ${#jobs[@]}" >&2
  exit 2
fi

start="$((shard * 6))"
end="$((start + 6))"
if (( end > ${#jobs[@]} )); then
  end="${#jobs[@]}"
fi

pids=()
names=()
for ((job_index = start; job_index < end; job_index++)); do
  IFS='|' read -r seed condition rho_label rho tau_label tau sg stop_gradient <<<"${jobs[$job_index]}"
  if [[ "$condition" == "noaux" ]]; then
    group="$noaux_group"
    aux_weight=0.0
    gradient_interval=0
    name="rmt-d32-mplusq-noaux-s${seed}-${suffix}"
  else
    group="$screen_group"
    aux_weight=0.10
    gradient_interval="$steps_per_epoch"
    name="rmt-d32-mplusq-targetsg-${sg}-rp${rho_label}-tp${tau_label}-s${seed}-${suffix}"
  fi
  echo "Launching $name (rho=$rho, tau=$tau, target_sg=$stop_gradient, aux=$aux_weight)"
  python -m train \
    "${common[@]}" \
    train.seed="$seed" \
    model.rho="$rho" \
    model.tau="$tau" \
    model.stop_gradient_memory_target="$stop_gradient" \
    task.aux_weight="$aux_weight" \
    task.aux_gradient_norm_interval="$gradient_interval" \
    wandb.group="$group" \
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
echo "$study shard $shard launched exactly ${#pids[@]} processes"

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
