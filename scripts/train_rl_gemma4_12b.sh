#!/usr/bin/env bash
# Vision-RL2 on Gemma-4-12B-it (encoder-free, discrete visual tiers).
# Same recipe as the Qwen families (scripts/_rl_launch.sh) except:
#   * no DeepSpeed (plain torchrun DDP; the policy trains only the twig),
#   * visual budget = tier MAX_SOFT_TOKENS (560) instead of min/max pixels,
#   * ATTN_IMPL=sdpa (flash-attn is not built for the gemma env).
# Env: PHASE_A_CKPT (assembled full SD-RPN ckpt), FILTERED_JSONL (pool with
# ev_maps_path), DATASET_ROOT, GPU_IDS, RUN_NAME, PLACEBO_KAPPA (1.0).
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"

export PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/gemma4-12b-sdrpn-K27T3}
export FILTERED_JSONL=${FILTERED_JSONL:-data/VisionRL2-data/rl_pools/rl_pool_gemma4_12b.jsonl}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
export EV_MAPS_ROOT=${EV_MAPS_ROOT:-data/ev_maps}
export GPU_IDS=${GPU_IDS:-0,1}
export RUN_NAME=${RUN_NAME:-gemma4-12b-rl}
TWIG_K=${TWIG_K:-27}
MAX_SOFT_TOKENS=${MAX_SOFT_TOKENS:-560}
PLACEBO_KAPPA=${PLACEBO_KAPPA:-1.0}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}      # effective batch 32 on 2 GPUs
export ATTN_IMPL=${ATTN_IMPL:-sdpa}
export REWARD_MASK_PIL_WORKERS=${REWARD_MASK_PIL_WORKERS:-4}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=3600 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
OUTPUT_DIR=${OUTPUT_DIR:-output/rl/${RUN_NAME}}
NPROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
mkdir -p "${OUTPUT_DIR}"
echo "[rl] ${RUN_NAME}: init=${PHASE_A_CKPT} pool=${FILTERED_JSONL} K=${TWIG_K} kappa=${PLACEBO_KAPPA} tier=${MAX_SOFT_TOKENS}"
echo "[rl] gpus=${GPU_IDS} bs=${BATCH_SIZE} x ga=${GRAD_ACCUM_STEPS} x ${NPROC} = $((BATCH_SIZE*GRAD_ACCUM_STEPS*NPROC))  out=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node="${NPROC}" \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}" \
    qwen-vl-finetune/qwenvl/train/region_level_grpo/train_phase_b1.py \
    --model_name_or_path "${PHASE_A_CKPT}" \
    --filtered_jsonl "${FILTERED_JSONL}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES:-0}" \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}} \
    --bf16 --output_dir "${OUTPUT_DIR}" --run_name "${RUN_NAME}" --report_to tensorboard \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size "${BATCH_SIZE}" --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate 1.5e-5 --lr_scheduler_type cosine --warmup_steps 0.2 --weight_decay 0 --max_grad_norm 1 \
    --gradient_checkpointing False --model_max_length 2048 --dataloader_num_workers 2 \
    --ddp_find_unused_parameters False \
    --eval_strategy no --save_strategy no --save_total_limit 1 --logging_steps 1 \
    --remove_unused_columns False --seed 42 \
    --enable_twig True --twig_K "${TWIG_K}" --twig_T 3 --roi_loss bce --roi_multi_head True \
    --max_soft_tokens "${MAX_SOFT_TOKENS}" \
    --lambda_kl 0.5 \
    --placebo_kappa "${PLACEBO_KAPPA}" \
    "$@"
