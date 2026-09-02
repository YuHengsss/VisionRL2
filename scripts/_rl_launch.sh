#!/usr/bin/env bash
# Internal launcher for the region-level RL stage. Called by the per-model scripts
# (scripts/train_rl_*.sh), which set the model-specific variables. Every recipe
# constant not listed there is fixed inside the trainer (qwenvl/train/region_level_grpo).
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"

: "${PHASE_A_CKPT:?SD-RPN checkpoint dir}"
: "${FILTERED_JSONL:?RL pool jsonl}"
: "${TWIG_K:?}" "${MIN_PIXELS:?}" "${MAX_PIXELS:?}" "${PLACEBO_KAPPA:?}"
: "${BATCH_SIZE:?}" "${GRAD_ACCUM_STEPS:?}" "${GPU_IDS:?}" "${ATTN_IMPL:?}" "${RUN_NAME:?}"

export DATASET_ROOT=${DATASET_ROOT:-datasets}     # image folders (see qwenvl/train/region_level_grpo/dataset.py)
export EV_MAPS_ROOT=${EV_MAPS_ROOT:-}             # optional root for relative ev_maps_path entries
export ATTN_IMPL
export REWARD_MASK_PIL_WORKERS=${REWARD_MASK_PIL_WORKERS:-4}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=3600 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUTPUT_DIR=${OUTPUT_DIR:-output/rl/${RUN_NAME}}
NPROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
mkdir -p "${OUTPUT_DIR}"
echo "[rl] ${RUN_NAME}: init=${PHASE_A_CKPT} pool=${FILTERED_JSONL} K=${TWIG_K} kappa=${PLACEBO_KAPPA}"
echo "[rl] gpus=${GPU_IDS} bs=${BATCH_SIZE} x ga=${GRAD_ACCUM_STEPS} x ${NPROC} = $((BATCH_SIZE*GRAD_ACCUM_STEPS*NPROC))  pixels=[${MIN_PIXELS},${MAX_PIXELS}]  out=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node="${NPROC}" \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}" \
    qwen-vl-finetune/qwenvl/train/region_level_grpo/train_phase_b1.py \
    --deepspeed qwen-vl-finetune/scripts/zero2_safe.json \
    --model_name_or_path "${PHASE_A_CKPT}" \
    --filtered_jsonl "${FILTERED_JSONL}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES:-0}" \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}} \
    --bf16 --output_dir "${OUTPUT_DIR}" --run_name "${RUN_NAME}" --report_to tensorboard \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size "${BATCH_SIZE}" --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate 1.5e-5 --lr_scheduler_type cosine --warmup_ratio 0.2 --weight_decay 0 --max_grad_norm 1 \
    --gradient_checkpointing True --model_max_length 2048 --dataloader_num_workers 2 \
    --eval_strategy no --save_strategy no --save_total_limit 1 --logging_steps 1 \
    --remove_unused_columns False --seed 42 \
    --enable_twig True --twig_K "${TWIG_K}" --twig_T 3 --roi_loss bce --roi_multi_head True \
    --min_pixels "${MIN_PIXELS}" --max_pixels "${MAX_PIXELS}" \
    --lambda_kl 0.5 \
    --placebo_kappa "${PLACEBO_KAPPA}"
