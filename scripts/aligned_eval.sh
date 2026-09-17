#!/usr/bin/env bash
# =============================================================================
# Training-aligned evaluation (ablations and the token / latency curves, Fig. 5).
#
# Protocol: source-image limit CAP tokens, short-answer prompting, benchmark-native
# rule metrics (no judge). The ablation tables use CAP=576 (the training limit);
# the Fig. 5 curves sweep CAP in {576, 1024, 2048, 4096}.
# Benchmarks: V* Bench, ZoomBench, HR-Bench (4K + 8K), MME-RealWorld-Lite, InfoVQA val.
#
#   MODEL=qwen3_5 CHECKPOINT=<rl ckpt dir> CAP=576 bash scripts/aligned_eval.sh    # ours
#   MODEL=qwen3_5 CHECKPOINT=Qwen/Qwen3.5-4B BASE=1 CAP=576 bash scripts/aligned_eval.sh
#
# Gemma-4-12B (encoder-free tiers) uses scripts/aligned_eval_gemma4.sh instead.
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

MODEL=${MODEL:-qwen3_5}                    # qwen3_5 | qwen2_5_vl
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT=<dir or HF id>}
BASE=${BASE:-0}
CAP=${CAP:-576}
GPU_IDS=${GPU_IDS:-0,1,2,3}
TASKS=${TASKS:-"vstar_bench zoombench hrbench mmerealworld_lite infovqa_val"}
OUT_ROOT=${OUT_ROOT:-logs/aligned_eval/$(basename "${CHECKPOINT}")$([ "${BASE}" = 1 ] && echo _base)_cap${CAP}}

case "${MODEL}" in
  qwen3_5)    PPT=1024; ROPE=",roi_infer_with_rope=True" ;;
  qwen2_5_vl) PPT=784;  ROPE="" ;;
  *) echo "MODEL must be qwen3_5 or qwen2_5_vl"; exit 1 ;;
esac
MAX_PIXELS=$((CAP * PPT))
case "${CAP}" in 576) CROP_TOK=160 ;; 1024) CROP_TOK=256 ;; *) CROP_TOK=384 ;; esac   # crop budget per cap

if [ "${BASE}" = 1 ]; then
  MODEL_ARGS="pretrained=${CHECKPOINT},device_map=auto,two_stage_roi=False,min_pixels=262144,max_pixels=${MAX_PIXELS},attn_implementation=flash_attention_2"
else
  export ROI_EVAL_SMOOTH_SIGMA=auto2 ROI_MIN_TOKENS_AUTO=1 ROI_MIN_PIXEL_BASE=262144
  export PROBE_CROP_TARGET_TOK=${CROP_TOK} PROBE_CROP_MAX_UPSCALE_EDGE=3 PROBE_CROP_SRC_CAP_DIV=2
  MODEL_ARGS="pretrained=${CHECKPOINT},device_map=auto,two_stage_roi=True,roi_conf_thresh=0.0,dynamic_conf_mode=peak_ratio,dynamic_ratio_thresh=3.0,dynamic_peak_fraction=0.3,dynamic_min_gate=0.03,min_pixels=4096,max_pixels=${MAX_PIXELS},attn_implementation=flash_attention_2${ROPE},window_sparse_mode=token_budget,window_sparse_dilation=1,window_sparse_k_max=3.0"
fi

mkdir -p "${OUT_ROOT}"
NUM_PROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
for TASK in ${TASKS}; do
  CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --num_processes="${NUM_PROC}" \
      --main_process_port="${MAIN_PROCESS_PORT:-29500}" -m lmms_eval \
      --model "${MODEL}" --model_args "${MODEL_ARGS}" --tasks "${TASK}" \
      --batch_size 1 --log_samples --log_samples_suffix "${TASK}" --output_path "${OUT_ROOT}/" \
    || echo "[aligned_eval] WARNING: ${TASK} returned non-zero"
done
echo "[aligned_eval] results: ${OUT_ROOT}/*/*results.json  (V*: vstar_overall_acc, ZoomBench: zoombench_acc, HR-Bench: average, MME-Lite: score x100, InfoVQA: anls x100; visual tokens per sample in the samples jsonl)"
