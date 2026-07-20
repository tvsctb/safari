#!/usr/bin/env bash
set -euo pipefail

python -c 'import os; assert os.environ.get("WANDB_API_KEY")'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

group="${WANDB_GROUP:-rmt-d32-block-position-localization-20260720-v1}"
max_epochs="${MAX_EPOCHS:-180}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-block-position-localization}"
mkdir -p "$output_root"

common=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs="$max_epochs" callbacks=full_run
  +trainer.check_val_every_n_epoch=5 +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50 trainer.gradient_clip_val=0.0
  +trainer.precision=32 train.test=false train.seed=0 train.model_seed=0
  train.runtime_seed=0 +dataset.seed=0 dataset.loader_seed=0 loader.num_workers=0
  scheduler=linear_warmup scheduler.num_warmup_steps="$((training_steps / 10))"
  scheduler.num_training_steps="$training_steps"
  task.aux_gradient_norm_interval=50 task.aux_weight=1.0
  optimizer.lr=0.0005 optimizer.weight_decay=0.1
  model.d_model=32 model.d_inner=128 model.n_layer=2 model.n_heads=1
  model.chunk_size=4 model.num_memory_tokens=4
  model.use_chunk_loss=true model.use_discrete_loss=true
  model.use_memory_loss=true model.use_terminal_loss=true
  model.use_terminal_chunk=false model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true model.stop_gradient_memory_target=true
  model.memory_scale_mode=fixed model.rho=31.622777
  model.terminal_scale_mode=fixed model.tau=12.649111
  model.observation_noise_std=0.0 model.generation_noise_std=0.0
  wandb.mode=online wandb.project=aux-assoc-recall wandb.group="$group"
)

position='position_embedding'
block0='blocks.0.*'
block1='blocks.1.*'
attention0='blocks.0.attention.*'
attention1='blocks.1.attention.*'
mlp0='blocks.0.mlp.*'
mlp1='blocks.1.mlp.*'

pids=(); names=()
launch() {
  local label="$1" patterns="$2"
  local name="rmt-d32-${label}-block-position-v1"
  python -m train "${common[@]}" train.initialization_donor_seed=2 \
    "train.initialization_transplant_patterns=[$patterns]" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" wandb.id="$name" hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!"); names+=("$name")
}

launch position-block0 "$position,$block0"
launch position-block1 "$position,$block1"
launch position-both-attention "$position,$attention0,$attention1"
launch position-both-mlp "$position,$mlp0,$mlp1"
launch position-attention0 "$position,$attention0"
launch position-attention1 "$position,$attention1"
launch position-block0-attention1 "$position,$block0,$attention1"
launch position-attention0-block1 "$position,$attention0,$block1"

status=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || status=1
  tail -n 30 "$output_root/${names[$i]}.log"
done
exit "$status"
