#!/usr/bin/env bash
# =============================================================================
# Main-table evaluation (paper Table 1): generation + scoring.
#
# Protocol: 16,384-token source-image limit, option-list prompts with free-form
# responses (no short-answer suffix), 2,048 new tokens; scoring = rule pass +
# Qwen3.5-9B LLM judge (scripts/main_table_judge.py).
# Benchmarks: V* Bench, ZoomBench, HR-Bench 4K / 8K, MME-RealWorld EN / CN.
#
#   MODEL=qwen3_5   CHECKPOINT=<rl ckpt dir>       bash scripts/main_eval.sh   # ours
#   MODEL=qwen3_5   CHECKPOINT=Qwen/Qwen3.5-4B BASE=1 bash scripts/main_eval.sh   # base model row
#   MODEL=qwen2_5_vl CHECKPOINT=<rl ckpt dir>      bash scripts/main_eval.sh
#
# Ours = two-stage RoI inference (peak-ratio region gate, crop budget 384 tokens,
# sparse visual encoding of the crop); BASE=1 = the frozen MLLM, single pass.
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

MODEL=${MODEL:-qwen3_5}                    # qwen3_5 | qwen2_5_vl
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT=<dir or HF id>}
BASE=${BASE:-0}
GPU_IDS=${GPU_IDS:-0,1,2,3}
TASKS=${TASKS:-"vstar_bench_vopd zoombench_vopd hrbench4k_vopd hrbench8k_vopd mme_realworld mme_realworld_cn"}
OUT_ROOT=${OUT_ROOT:-logs/main_eval/$(basename "${CHECKPOINT}")$([ "${BASE}" = 1 ] && echo _base)}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-9B}

case "${MODEL}" in
  qwen3_5)    PPT=1024; ROPE=",roi_infer_with_rope=True" ;;
  qwen2_5_vl) PPT=784;  ROPE="" ;;
  *) echo "MODEL must be qwen3_5 or qwen2_5_vl"; exit 1 ;;
esac
MAX_PIXELS=$((16384 * PPT))
export DISABLE_SHORT_ANSWER_SUFFIX=1

if [ "${BASE}" = 1 ]; then
  MODEL_ARGS="pretrained=${CHECKPOINT},device_map=auto,two_stage_roi=False,min_pixels=262144,max_pixels=${MAX_PIXELS},attn_implementation=flash_attention_2"
else
  export ROI_EVAL_SMOOTH_SIGMA=auto2 ROI_MIN_TOKENS_AUTO=1 ROI_MIN_PIXEL_BASE=262144
  export PROBE_CROP_TARGET_TOK=384 PROBE_CROP_MAX_UPSCALE_EDGE=3 PROBE_CROP_SRC_CAP_DIV=2
  MODEL_ARGS="pretrained=${CHECKPOINT},device_map=auto,two_stage_roi=True,roi_conf_thresh=0.0,dynamic_conf_mode=peak_ratio,dynamic_ratio_thresh=3.0,dynamic_peak_fraction=0.3,dynamic_min_gate=0.03,min_pixels=4096,max_pixels=${MAX_PIXELS},attn_implementation=flash_attention_2${ROPE},window_sparse_mode=token_budget,window_sparse_dilation=1,window_sparse_k_max=3.0"
fi

mkdir -p "${OUT_ROOT}"
NUM_PROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
for TASK in ${TASKS}; do
  CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --num_processes="${NUM_PROC}" \
      --main_process_port="${MAIN_PROCESS_PORT:-29500}" -m lmms_eval \
      --model "${MODEL}" --model_args "${MODEL_ARGS}" --tasks "${TASK}" \
      --batch_size 1 --log_samples --log_samples_suffix "${TASK}" --output_path "${OUT_ROOT}/" \
    || echo "[main_eval] WARNING: ${TASK} returned non-zero"
done

# scoring: rule pass + LLM judge on one GPU
CUDA_VISIBLE_DEVICES=${JUDGE_GPU:-${GPU_IDS%%,*}} python scripts/main_table_judge.py "${OUT_ROOT}" --judge-model "${JUDGE_MODEL}"
