#!/usr/bin/env bash
# =============================================================================
# Training-aligned evaluation for Gemma-4-12B-it (encoder-free backbone).
#
# Same protocol as scripts/aligned_eval.sh (short-answer prompts, benchmark-native
# rule metrics, no judge), but the visual budget is a discrete soft-token TIER
# instead of a pixel range. Evaluation runs at tier 1120; the RoI crop is
# re-encoded at the tier the constant-target rule asks for (target CROP_TOK
# tokens, edge cap 3x, source cap min(3T, src/2), quantized up to the next tier).
#
# Arms:
#   BASE=1                    -> timing_mode=baseline:   frozen Gemma, single pass
#   ROI_MODE=dense            -> timing_mode=roi_dense:   SD-RPN, dense bbox crop
#   ROI_MODE=sparse (default) -> timing_mode=roi_sparse:  Vision-RL2, sparse crop
#
#   CHECKPOINT=<sd-rpn full ckpt> ROI_MODE=dense bash scripts/aligned_eval_gemma4.sh
#   CHECKPOINT=<rl ckpt>                         bash scripts/aligned_eval_gemma4.sh
#   CHECKPOINT=<sd-rpn full ckpt> BASE=1         bash scripts/aligned_eval_gemma4.sh
#
# The base arm only disables the twig, so any Gemma-4 checkpoint works for it.
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT=<dir or HF id>}
BASE=${BASE:-0}
ROI_MODE=${ROI_MODE:-sparse}                # dense (SD-RPN) | sparse (Vision-RL2)
TIER=${TIER:-1120}                          # source-image soft-token tier
CROP_TOK=${CROP_TOK:-256}                   # RoI crop token target
GPU_IDS=${GPU_IDS:-0,1}
TASKS=${TASKS:-"vstar_bench zoombench hrbench mmerealworld_lite infovqa_val"}
ARM=$([ "${BASE}" = 1 ] && echo base || echo "${ROI_MODE}")
OUT_ROOT=${OUT_ROOT:-logs/aligned_eval/$(basename "${CHECKPOINT}")_${ARM}_tier${TIER}}

# per-sample stage timing rows (prefill / RoI / pass-2), one jsonl per task
export GEMMA_STAGE_TIMING=${GEMMA_STAGE_TIMING:-1}

if [ "${BASE}" = 1 ]; then
  MODE=baseline
  EXTRA=""
else
  case "${ROI_MODE}" in
    dense)  MODE=roi_dense ;;
    sparse) MODE=roi_sparse ;;
    *) echo "ROI_MODE must be dense or sparse"; exit 1 ;;
  esac
  EXTRA=",roi_recipe=q35_peak,roi_conf_thresh=0.0,roi_crop_target_tok=${CROP_TOK}"
fi
MODEL_ARGS="pretrained=${CHECKPOINT},max_soft_tokens=${TIER},attn_implementation=sdpa,two_stage_roi=True,timing_mode=${MODE}${EXTRA}"

mkdir -p "${OUT_ROOT}"
NUM_PROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
echo "[aligned_eval_gemma4] arm=${ARM} tier=${TIER} crop=${CROP_TOK} -> ${OUT_ROOT}"
for TASK in ${TASKS}; do
  GEMMA_TIMING_OUT="${OUT_ROOT}/timing_${TASK}.jsonl" \
  CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --num_processes="${NUM_PROC}" \
      --main_process_port="${MAIN_PROCESS_PORT:-29611}" -m lmms_eval \
      --model gemma4 --model_args "${MODEL_ARGS}" --tasks "${TASK}" \
      --batch_size 1 --log_samples --log_samples_suffix "${TASK}" --output_path "${OUT_ROOT}/" \
    || echo "[aligned_eval_gemma4] WARNING: ${TASK} returned non-zero"
done
echo "[aligned_eval_gemma4] results: ${OUT_ROOT}/*/*results.json   timing: ${OUT_ROOT}/timing_<task>.jsonl"
