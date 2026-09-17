#!/usr/bin/env bash
# =============================================================================
# SD-RPN online pseudo-label training (the supervised RoI-predictor stage).
#
# Trains the T=3 twig blocks attached after block K of a frozen Qwen3.5 MLLM
# to predict a query-relevant RoI heatmap. Supervision is generated ONLINE at
# every step from the frozen model's own response->image attention on the
# training corpus (no boxes, no offline label cache):
#   * textvqa / gqa rows        -> label version v1 (mean-over-response-tokens map)
#   * docvqa / infographics rows -> label version v2 (single-region peak-ratio union)
# The per-sample label version is derived from the dataset tag of each row
# (override with LABEL_VERSION_MAP="gqa=v1,docvqa=v2,...").
#
# Paper recipe (both scales): lr 1e-4 cosine, warm-up 0.03, 1 epoch,
# effective batch 128, bf16 + DeepSpeed ZeRO-2, only twig parameters trainable.
#
# Usage:
#   MODEL=qwen3_5-4b bash scripts/train_sdrpn_online.sh
#   MODEL=qwen3_5-9b GPU_IDS=0,1,2,3 bash scripts/train_sdrpn_online.sh
#
# Required inputs (see the "Data" section of README.md):
#   ROI_DATA_PATH   jsonl corpus with fields {dataset, image, question, response}
#                   (the model's own responses on the VisualCoT training split).
#                   Released as sdrpn_corpora/qwen3_5_{4b,9b}_response_corpus.jsonl,
#                   or regenerate it with data_prep/build_corpus_qwen3_5.sh.
#   DATASET_ROOT    image roots for the per-dataset image folders
# =============================================================================
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"

MODEL=${MODEL:-qwen3_5-4b}
case "${MODEL}" in
  qwen3_5-4b)
    LLM=${LLM:-Qwen/Qwen3.5-4B}
    FAMILY=${ONLINE_PSEUDO_LABEL_FAMILY:-qwen3_5_4b}
    BATCH_SIZE=${BATCH_SIZE:-8};  GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
    ROI_DATA_PATH=${ROI_DATA_PATH:-data/VisionRL2-data/sdrpn_corpora/qwen3_5_4b_response_corpus.jsonl}
    ;;
  qwen3_5-9b)
    LLM=${LLM:-Qwen/Qwen3.5-9B}
    FAMILY=${ONLINE_PSEUDO_LABEL_FAMILY:-qwen3_5_9b}
    BATCH_SIZE=${BATCH_SIZE:-4};  GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
    ROI_DATA_PATH=${ROI_DATA_PATH:-data/VisionRL2-data/sdrpn_corpora/qwen3_5_9b_response_corpus.jsonl}
    ;;
  *) echo "MODEL must be qwen3_5-4b or qwen3_5-9b (got '${MODEL}')"; exit 1 ;;
esac

# ---- hardware / launch --------------------------------------------------------
GPU_IDS=${GPU_IDS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(echo "${GPU_IDS}" | awk -F',' '{print NF}')}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-${CODE_ROOT}/qwen-vl-finetune/scripts/zero2_safe.json}
export ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=${TORCH_NCCL_WATCHDOG_TIMEOUT_SEC:-3600}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}

# ---- SD-RPN architecture ------------------------------------------------------
TWIG_K=${TWIG_K:-21}          # attach after block K
TWIG_T=${TWIG_T:-3}           # T twig blocks (initialised from blocks K+1..K+T)

# ---- data / image budgets (paper values) -------------------------------------
export DATASET_ROOT=${DATASET_ROOT:-datasets}   # per-dataset image folders live under here
export MIN_PIXELS=${MIN_PIXELS:-262144}         # 256 tokens
export MAX_PIXELS=${MAX_PIXELS:-589824}         # 576 tokens (v2 rows: docvqa / infographics)
export MAX_PIXELS_BY_VERSION=${MAX_PIXELS_BY_VERSION:-1}
export V1_MAX_PIXELS=${V1_MAX_PIXELS:-1048576}  # 1024 tokens for v1 rows (gqa / textvqa)
export EXPAND2SQUARE=${EXPAND2SQUARE:-0}        # v2 rows: no square padding
export EXPAND2SQUARE_BY_VERSION=${EXPAND2SQUARE_BY_VERSION:-1}   # v1 rows: square padding ON
export ONLINE_SINGLE_REGION=${ONLINE_SINGLE_REGION:-1}           # fallback for rows without a version tag
export STRIP_TASK_SUFFIX_PROB=${STRIP_TASK_SUFFIX_PROB:-1.0}     # strip task-instruction suffixes from prompts
export STRIP_TASK_SUFFIX_SEED=${STRIP_TASK_SUFFIX_SEED:-12345}
LR=${LR:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
GRAD_CKPT=${GRAD_CKPT:-True}

# ---- output -------------------------------------------------------------------
RUN_NAME=${RUN_NAME:-${MODEL}-sdrpn-K${TWIG_K}T${TWIG_T}-online}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT:-output}/sdrpn/${RUN_NAME}}
mkdir -p "${OUTPUT_DIR}"

echo "[sdrpn] model=${LLM} family=${FAMILY} K=${TWIG_K} T=${TWIG_T}"
echo "[sdrpn] data=${ROI_DATA_PATH}  dataset_root=${DATASET_ROOT}"
echo "[sdrpn] gpus=${GPU_IDS} nproc=${NPROC_PER_NODE} bs=${BATCH_SIZE} ga=${GRAD_ACCUM_STEPS} (eff. $((BATCH_SIZE*GRAD_ACCUM_STEPS*NPROC_PER_NODE)))"
echo "[sdrpn] pixels: v2 [${MIN_PIXELS}, ${MAX_PIXELS}] | v1 max ${V1_MAX_PIXELS}"
echo "[sdrpn] out=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  qwen-vl-finetune/qwenvl/train/train_qwen.py \
    --deepspeed "${DEEPSPEED_CFG}" \
    --model_name_or_path "${LLM}" \
    --dataset_use my_roi_dataset \
    --roi_data_path "${ROI_DATA_PATH}" \
    --remove_unused_columns False \
    --data_flatten False \
    --tune_mm_vision False --tune_mm_mlp True --tune_mm_llm True \
    --bf16 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1.0 \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}} \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --max_pixels "${MAX_PIXELS}" --min_pixels "${MIN_PIXELS}" \
    --eval_strategy no --save_strategy no --save_total_limit 1 \
    --learning_rate "${LR}" --weight_decay 0 --warmup_ratio "${WARMUP_RATIO}" \
    --max_grad_norm 1 --lr_scheduler_type cosine \
    --logging_steps 1 --report_to tensorboard --run_name "${RUN_NAME}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRAD_CKPT}" \
    --dataloader_num_workers 4 \
    --enable_twig True --twig_K "${TWIG_K}" --twig_T "${TWIG_T}" --twig_init True \
    --roi_loss bce --roi_multi_head True \
    --bg_coff 0.05 --roi_binary_coeff 0.25 \
    --online_pseudo_label True \
    --online_pseudo_label_family "${FAMILY}" \
    --online_pseudo_label_mode auto \
    --online_single_region "$([ "${ONLINE_SINGLE_REGION}" = 1 ] && echo True || echo False)"
