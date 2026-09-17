#!/usr/bin/env bash
# =============================================================================
# Main-table evaluation for Gemma-4-12B-it (paper Table 1 protocol).
#
# Same protocol as scripts/main_eval.sh (option-list prompts, free-form answers,
# no short-answer suffix, rule pass + Qwen3.5-9B judge), adapted to the
# encoder-free backbone: source tier 1120, RoI crop target 384 tokens, sdpa
# attention. MME-RealWorld EN / CN are long, so they are split into id shards
# (QZOOM_DOC_IDS_FILE) that run concurrently over the GPUs; the judge reads the
# whole output tree at the end.
#
# Arms:
#   BASE=1                    -> timing_mode=baseline:   frozen Gemma, single pass
#   ROI_MODE=dense            -> timing_mode=roi_dense:   SD-RPN, dense bbox crop
#   ROI_MODE=sparse (default) -> timing_mode=roi_sparse:  Vision-RL2, sparse crop
#
#   CHECKPOINT=<rl ckpt> GPU_IDS=0,1            bash scripts/main_eval_gemma4.sh
#   CHECKPOINT=<sd-rpn full ckpt> ROI_MODE=dense bash scripts/main_eval_gemma4.sh
#   CHECKPOINT=<sd-rpn full ckpt> BASE=1         bash scripts/main_eval_gemma4.sh
#
# SKIP_JUDGE=1 stops after generation; re-judge an existing tree with
#   JUDGE_ATTN=sdpa python scripts/main_table_judge.py <OUT_ROOT> --judge-model <judge>
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT=<dir or HF id>}
BASE=${BASE:-0}
ROI_MODE=${ROI_MODE:-sparse}                # dense (SD-RPN) | sparse (Vision-RL2)
TIER=${TIER:-1120}
CROP_TOK=${CROP_TOK:-384}
GPU_IDS=${GPU_IDS:-0,1}
TASKS=${TASKS:-"vstar_bench_vopd zoombench_vopd hrbench4k_vopd hrbench8k_vopd mmerealworld mmerealworld_cn"}
ARM=$([ "${BASE}" = 1 ] && echo base || echo "${ROI_MODE}")
OUT_ROOT=${OUT_ROOT:-logs/main_eval/$(basename "${CHECKPOINT}")_${ARM}}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-9B}
export JUDGE_ATTN=${JUDGE_ATTN:-sdpa}       # the gemma env has no flash-attn
export JUDGE_BATCH=${JUDGE_BATCH:-32}       # prompts per judge generate call
# MME-RealWorld id sharding: shards per language and the dataset sizes
MME_SHARDS_EN=${MME_SHARDS_EN:-8}
MME_SHARDS_CN=${MME_SHARDS_CN:-4}
MME_DOCS_EN=${MME_DOCS_EN:-23609}
MME_DOCS_CN=${MME_DOCS_CN:-5917}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}

# free-form responses over the option list, no "single word or phrase" suffix
export DISABLE_SHORT_ANSWER_SUFFIX=1
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

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NGPU=${#GPUS[@]}
SLOTS=$(( NGPU * WORKERS_PER_GPU ))
IDS_DIR="${OUT_ROOT}/_ids"
mkdir -p "${OUT_ROOT}" "${IDS_DIR}"
echo "[main_eval_gemma4] arm=${ARM} tier=${TIER} crop=${CROP_TOK} -> ${OUT_ROOT}"

make_id_shards() {   # n_docs n_shards prefix
  python "${CODE_ROOT}/scripts/_make_id_shards.py" "${IDS_DIR}" "$1" "$2" "$3"
}

run_leg() {          # task shard_tag gpu [ids_file]
  local TASK=$1 TAG=$2 GPU=$3 IDS=${4:-}
  local LEG="${OUT_ROOT}/leg_${TASK}_${TAG}"
  mkdir -p "${LEG}"
  env ${IDS:+QZOOM_DOC_IDS_FILE=${IDS}} GEMMA_TIMING_OUT="${LEG}/timing.jsonl" \
      CUDA_VISIBLE_DEVICES=${GPU} python -m lmms_eval \
      --model gemma4 --model_args "${MODEL_ARGS}" --tasks "${TASK}" \
      --batch_size 1 --log_samples --log_samples_suffix "${ARM}_${TAG}" \
      --output_path "${LEG}/" > "${LEG}/run.log" 2>&1 \
    || echo "[main_eval_gemma4] WARNING: ${TASK} ${TAG} returned non-zero (see ${LEG}/run.log)"
  # the judge keys rows by (file name, doc_id): give every shard a unique name
  find "${LEG}" -name "*samples*.jsonl" 2>/dev/null | while read -r f; do
    case "$(basename "${f}")" in
      "${ARM}_${TAG}__"*) ;;
      *) mv "${f}" "$(dirname "${f}")/${ARM}_${TAG}__$(basename "${f}")" ;;
    esac
  done
}

for TASK in ${TASKS}; do
  case "${TASK}" in
    mmerealworld)    NSH=${MME_SHARDS_EN}; NDOC=${MME_DOCS_EN}; PRE=en_s ;;
    mmerealworld_cn) NSH=${MME_SHARDS_CN}; NDOC=${MME_DOCS_CN}; PRE=cn_s ;;
    *)               NSH=1; NDOC=0; PRE=full ;;
  esac
  if [ "${NSH}" -le 1 ]; then
    echo "[main_eval_gemma4] ${TASK} (single leg)"
    run_leg "${TASK}" full "${GPUS[0]}"
  else
    make_id_shards "${NDOC}" "${NSH}" "${PRE}"
    echo "[main_eval_gemma4] ${TASK}: ${NSH} id shards over ${SLOTS} slot(s)"
    for (( S = 0; S < NSH; S += SLOTS )); do
      for (( J = 0; J < SLOTS && S + J < NSH; J++ )); do
        run_leg "${TASK}" "${PRE}$(( S + J ))" "${GPUS[$(( J % NGPU ))]}" "${IDS_DIR}/${PRE}$(( S + J )).json" &
        sleep 20
      done
      wait
    done
  fi
done

if [ "${SKIP_JUDGE:-0}" != 1 ]; then
  CUDA_VISIBLE_DEVICES=${JUDGE_GPU:-${GPUS[0]}} python scripts/main_table_judge.py "${OUT_ROOT}" \
      --judge-model "${JUDGE_MODEL}"
  echo "[main_eval_gemma4] table: ${OUT_ROOT}/main_table.txt"
fi
