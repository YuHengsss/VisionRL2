#!/usr/bin/env bash
# =============================================================================
# Evaluation launcher (lmms-eval) for the SD-RPN / Vision-RL2 checkpoints.
#
# Two protocols are supported (see docs/PROTOCOLS.md):
#
#   PROTOCOL=main      Paper Table 1. 16,384-token source-image limit, option-list
#                      prompts with free-form responses, no short-answer suffix.
#                      Tasks: vstar_bench_vopd zoombench_vopd hrbench4k_vopd
#                      hrbench8k_vopd mme_realworld mme_realworld_cn.
#                      Scoring = in-run rule parse, then the LLM judge
#                      (scripts/judge/, Qwen3.5-9B) on the remaining cases.
#
#   PROTOCOL=aligned   Training-aligned protocol (ablations, token/latency curves,
#                      Fig. 5). Source limit CAP in {576,1024,2048,4096} tokens,
#                      short-answer prompting, rule-based metrics only.
#                      Tasks: vstar_bench zoombench hrbench mmerealworld_lite infovqa_val.
#
# Rows:
#   ROWS=ours (default)  two-stage RoI inference with the RL/SD-RPN predictor
#                        + sparse visual encoding of the crop.
#   ROWS=base            the frozen base MLLM, single pass, no RoI (TWO_STAGE_ROI=False).
#
# Examples:
#   MODEL=qwen3_5 CHECKPOINT=output/rl/qwen3_5-4b PROTOCOL=main bash scripts/eval.sh
#   MODEL=qwen3_5 CHECKPOINT=output/rl/qwen3_5-9b PROTOCOL=aligned CAP=1024 bash scripts/eval.sh
#   MODEL=qwen3_5 CHECKPOINT=Qwen/Qwen3.5-4B ROWS=base PROTOCOL=aligned CAP=4096 bash scripts/eval.sh
#   MODEL=qwen2_5_vl CHECKPOINT=output/rl/qwen2_5vl-7b PROTOCOL=main bash scripts/eval.sh
# =============================================================================
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

MODEL=${MODEL:-qwen3_5}                 # qwen3_5 | qwen2_5_vl
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT=<dir or HF id>}
PROTOCOL=${PROTOCOL:-aligned}           # main | aligned
ROWS=${ROWS:-ours}                      # ours | base
CAP=${CAP:-576}                         # aligned protocol only: 576 | 1024 | 2048 | 4096
GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_PROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
EVAL_ATTN=${EVAL_ATTN:-flash_attention_2}
OUT_ROOT=${OUT_ROOT:-logs/eval/$(basename "${CHECKPOINT}")_${PROTOCOL}_${ROWS}$([ "${PROTOCOL}" = aligned ] && echo "_cap${CAP}")}
mkdir -p "${OUT_ROOT}"

# pixels per visual token: 32^2 for Qwen3.5, 28^2 for Qwen2.5-VL
case "${MODEL}" in
  qwen3_5)    PPT=1024; ROPE_ARG=",roi_infer_with_rope=True" ;;
  qwen2_5_vl) PPT=784;  ROPE_ARG="" ;;
  *) echo "MODEL must be qwen3_5 or qwen2_5_vl"; exit 1 ;;
esac

# ---- protocol presets ----------------------------------------------------------
if [ "${PROTOCOL}" = main ]; then
  MAX_PIXELS=$((16384 * PPT))
  TASKS=${TASKS:-"vstar_bench_vopd zoombench_vopd hrbench4k_vopd hrbench8k_vopd mme_realworld mme_realworld_cn"}
  export DISABLE_SHORT_ANSWER_SUFFIX=1   # free-form responses, judged afterwards
  PROBE_CROP_TARGET_TOK=${PROBE_CROP_TARGET_TOK:-384}
  SMOOTH=${ROI_EVAL_SMOOTH_SIGMA:-auto2}
elif [ "${PROTOCOL}" = aligned ]; then
  MAX_PIXELS=$((CAP * PPT))
  TASKS=${TASKS:-"vstar_bench zoombench hrbench mmerealworld_lite infovqa_val"}
  case "${CAP}" in
    576)  PROBE_CROP_TARGET_TOK=${PROBE_CROP_TARGET_TOK:-160} ;;
    1024) PROBE_CROP_TARGET_TOK=${PROBE_CROP_TARGET_TOK:-256} ;;
    *)    PROBE_CROP_TARGET_TOK=${PROBE_CROP_TARGET_TOK:-384} ;;
  esac
  SMOOTH=${ROI_EVAL_SMOOTH_SIGMA:-auto2}   # identical to 'auto' for caps <= 2048
else
  echo "PROTOCOL must be main or aligned"; exit 1
fi

# ---- ours vs base ----------------------------------------------------------------
if [ "${ROWS}" = ours ]; then
  MIN_PIXELS=${MIN_PIXELS:-4096}          # source floor is set by ROI_MIN_PIXEL_BASE below
  export ROI_EVAL_SMOOTH_SIGMA=${SMOOTH}
  export ROI_MIN_TOKENS_AUTO=${ROI_MIN_TOKENS_AUTO:-1}
  export ROI_MIN_PIXEL_BASE=${ROI_MIN_PIXEL_BASE:-262144}
  export PROBE_CROP_TARGET_TOK
  export PROBE_CROP_MAX_UPSCALE_EDGE=${PROBE_CROP_MAX_UPSCALE_EDGE:-3}
  export PROBE_CROP_SRC_CAP_DIV=${PROBE_CROP_SRC_CAP_DIV:-2}
  # region gate = peak-ratio on the twig heatmap (pf 0.3, ratio 3.0, min gate 0.03)
  ROI_ARGS=",two_stage_roi=True,roi_conf_thresh=0.0,dynamic_conf_mode=peak_ratio,dynamic_ratio_thresh=3.0,dynamic_peak_fraction=0.3,dynamic_min_gate=0.03${ROPE_ARG}"
  ROI_ARGS="${ROI_ARGS},window_sparse_mode=${WINDOW_SPARSE_MODE:-token_budget},window_sparse_dilation=${WINDOW_SPARSE_DILATION:-1},window_sparse_k_max=${WINDOW_SPARSE_K_MAX:-3.0}"
else
  MIN_PIXELS=${MIN_PIXELS:-262144}
  ROI_ARGS=",two_stage_roi=False"
fi

MODEL_ARGS="pretrained=${CHECKPOINT},device_map=auto,min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},attn_implementation=${EVAL_ATTN}${ROI_ARGS}"
echo "[eval] model=${MODEL} protocol=${PROTOCOL} rows=${ROWS} max_pixels=${MAX_PIXELS} tasks=${TASKS}"
echo "[eval] model_args=${MODEL_ARGS}"
echo "[eval] out=${OUT_ROOT}"

for TASK in ${TASKS}; do
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --num_processes="${NUM_PROC}" \
      --main_process_port="${MAIN_PROCESS_PORT}" -m lmms_eval \
      --model "${MODEL}" --model_args "${MODEL_ARGS}" \
      --tasks "${TASK}" --batch_size 1 --log_samples \
      --log_samples_suffix "${TASK}" --output_path "${OUT_ROOT}/" \
    || echo "[eval] WARNING: ${TASK} returned non-zero"
  echo "[eval] ${TASK} done in $(( $(date +%s) - t0 ))s"
done

if [ "${PROTOCOL}" = main ]; then
  cat <<EOF
[eval] Generation finished. Main-protocol scoring = rule parse (already applied in-run) + LLM judge
       on the unresolved cases. Run the judge on the sample logs:
         bash scripts/judge/run_judge.sh ${OUT_ROOT}
       (starts the Qwen3.5-9B judge shim, re-judges V*/HRBench as mcq and ZoomBench as zoom,
        converts MME-RealWorld to the Vision-OPD scoring format; see scripts/judge/README.md)
EOF
fi
