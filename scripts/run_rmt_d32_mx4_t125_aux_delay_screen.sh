#!/usr/bin/env bash
set -euo pipefail

python -c 'import os; assert os.environ.get("WANDB_API_KEY")'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

shard="${SHARD:?SHARD must be 0 or 1}"
case "$shard" in
  0) seeds=(0 1) ;;
  1) seeds=(2) ;;
  *) echo "SHARD must be 0 or 1" >&2; exit 2 ;;
esac

group="${WANDB_GROUP:-rmt-d32-mx4-t125-aux-delay-20260721-v1}"
max_epochs="${MAX_EPOCHS:-240}"
steps_per_epoch=157
training_steps="$((steps_per_epoch * max_epochs))"
lr_warmup_steps="$((training_steps / 10))"
output_root="${OUTPUT_ROOT:-/tmp/rmt-d32-mx4-t125-aux-delay}/shard-${shard}"
mkdir -p "$output_root"

common=(
  experiment=synthetics/associative_recall/rmt_aux
  trainer.max_epochs="$max_epochs" callbacks=full_run
  +trainer.check_val_every_n_epoch=5 +trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=50 trainer.limit_train_batches=1.0
  trainer.limit_val_batches=1.0 +trainer.precision=32
  trainer.gradient_clip_val=0.0 train.test=false loader.num_workers=0
  scheduler=linear_warmup scheduler.num_warmup_steps="$lr_warmup_steps"
  scheduler.num_training_steps="$training_steps"
  task.aux_gradient_norm_interval="$steps_per_epoch"
  task.aux_weight=0.0 task.aux_weight_final=1.0 task.aux_weight_schedule=linear
  optimizer.lr=0.0005 optimizer.weight_decay=0.1
  model.d_model=32 model.d_inner=128 model.n_layer=2 model.n_heads=1
  model.chunk_size=4 model.num_memory_tokens=4
  model.share_inverse=true model.share_inverse_embedding=true
  model.share_inverse_head=true model.share_inverse_position_embedding=false
  model.inverse_position_initialization=copy model.use_direction_embedding=false
  model.use_chunk_loss=true model.use_discrete_loss=true
  model.use_memory_loss=true model.use_terminal_loss=true
  model.use_terminal_chunk=false model.use_terminal_chunk_loss=false
  model.learnable_terminal_target=true model.stop_gradient_memory_target=true
  model.memory_scale_mode=fixed model.memory_scale_granularity=global
  model.rho=31.622777 model.terminal_scale_mode=fixed
  model.terminal_scale_granularity=global model.tau=11.313708
  model.observation_noise_std=0.0 model.generation_noise_std=0.0
  wandb.mode=online wandb.project=aux-assoc-recall wandb.group="$group"
)

pids=()
names=()
launch() {
  local seed="$1"
  local delay_percent="$2"
  local start_step="$((training_steps * delay_percent / 100))"
  local end_step="$((start_step + training_steps / 10))"
  local name="rmt-d32-mx4-t125-delay${delay_percent}p-s${seed}-aux-delay-v1"
  python -m train "${common[@]}" train.seed="$seed" \
    task.aux_weight_decay_start_step="$start_step" \
    task.aux_weight_decay_end_step="$end_step" \
    callbacks.model_checkpoint.dirpath="$output_root/$name/checkpoints" \
    wandb.name="$name" wandb.id="$name" hydra.run.dir="$output_root/$name" \
    >"$output_root/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

for seed in "${seeds[@]}"; do
  launch "$seed" 5
  launch "$seed" 10
  launch "$seed" 20
done

status=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}" || status=1
  tail -n 40 "$output_root/${names[$index]}.log"
done
exit "$status"
