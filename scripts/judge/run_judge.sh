#!/usr/bin/env bash
# =============================================================================
# Main-protocol scoring for one eval output directory produced by
#   PROTOCOL=main bash scripts/eval.sh
#
# Pipeline (identical to the paper's "unified judge v1" rows):
#   1. serve Qwen3.5-9B as an OpenAI-compatible judge endpoint (hf_openai_shim.py)
#   2. convert each task's lmms-eval samples jsonl into Vision-OPD's model_answer
#      layout (to_vopd_answer.py for V*/ZoomBench/HR-Bench, mme_to_vopd.py for MME)
#   3. judge_unified.py: rule pass (mathruler + first-letter) then the lenient LLM
#      judge on the remaining items
#   4. Vision-OPD's cal_acc.py -> accuracy
#
# Requirements:
#   VOPD_EVAL_DIR   the eval/ dir of the Vision-OPD repo (prepared benchmark jsons:
#                   vstar.json, zoombench.json, hr_bench_4k.json, hr_bench_8k.json;
#                   model_answer/ and judge_unified/ are written under it)
#   pip install mathruler openai tqdm
#
# Usage:
#   VOPD_EVAL_DIR=/path/Vision-OPD/eval JUDGE_GPU=0 bash scripts/judge/run_judge.sh <eval_out_dir> [tag]
# =============================================================================
set -uo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
J=${CODE_ROOT}/scripts/judge
OUT=${1:?usage: run_judge.sh <eval_out_dir> [tag]}
TAG=${2:-$(basename "${OUT}")}
export VOPD_EVAL_DIR=${VOPD_EVAL_DIR:?set VOPD_EVAL_DIR=/path/to/Vision-OPD/eval}
JUDGE_MODEL_ID=${JUDGE_MODEL_ID:-Qwen/Qwen3.5-9B}
JUDGE_GPU=${JUDGE_GPU:-0}
PORT=${PORT:-8124}
RES=${OUT}/judge_results.txt

# ---- 1. judge endpoint ---------------------------------------------------------
SHIM_LOG=${OUT}/judge_shim.log
CUDA_VISIBLE_DEVICES=${JUDGE_GPU} MODEL_ID=${JUDGE_MODEL_ID} PORT=${PORT} \
  python "${J}/hf_openai_shim.py" > "${SHIM_LOG}" 2>&1 &
SHIM=$!
trap 'kill ${SHIM} 2>/dev/null' EXIT
for i in $(seq 1 180); do grep -q ready "${SHIM_LOG}" 2>/dev/null && break; sleep 5; done
grep -q ready "${SHIM_LOG}" || { echo "judge shim failed to start (see ${SHIM_LOG})"; exit 1; }
API="http://127.0.0.1:${PORT}/v1/"

# ---- 2-4. per task --------------------------------------------------------------
# lmms-eval task -> (Vision-OPD benchmark name, converter)
score_lite () {  # task bench
  local TASK=$1 BENCH=$2 M="${TAG}-${BENCH}"
  ls "${OUT}"/*/*"${TASK}"*samples*.jsonl >/dev/null 2>&1 || { echo "[skip] no samples for ${TASK}"; return; }
  python "${J}/to_vopd_answer.py" --bench "${BENCH}" --tag "${M}" --samples "${OUT}" || { echo "[fail] convert ${TASK}" | tee -a "${RES}"; return; }
  ( cd "${VOPD_EVAL_DIR}" && python "${J}/judge_unified.py" --benchmark "${BENCH}" --model "${M}" \
      --api_base "${API}" --api_key EMPTY --judge_model "$(basename "${JUDGE_MODEL_ID}")" ) | grep -a "Total:" || true
  local ACC
  ACC=$( cd "${VOPD_EVAL_DIR}" && python "${J}/third_party/vision_opd_cal_acc.py" --benchmark "${BENCH}" \
      --judge_json "judge_unified/${BENCH}/${M}_answer.jsonl" 2>&1 | grep -a "Acc" )
  echo "[${TAG}] ${BENCH}: ${ACC}" | tee -a "${RES}"
}
score_mme () {  # task bench expect
  local TASK=$1 BENCH=$2 EXPECT=$3 M="${TAG}-${BENCH}"
  ls "${OUT}"/*/*"${TASK}"*samples*.jsonl >/dev/null 2>&1 || { echo "[skip] no samples for ${TASK}"; return; }
  python "${J}/mme_to_vopd.py" --bench "${BENCH}" --tag "${M}" --expect "${EXPECT}" --samples "${OUT}" || { echo "[fail] convert ${TASK}" | tee -a "${RES}"; return; }
  ( cd "${VOPD_EVAL_DIR}" && python "${J}/judge_unified.py" --benchmark "${BENCH}" --model "${M}" \
      --api_base "${API}" --api_key EMPTY --judge_model "$(basename "${JUDGE_MODEL_ID}")" ) | grep -a "Total:" || true
  local ACC
  ACC=$( cd "${VOPD_EVAL_DIR}" && python "${J}/third_party/vision_opd_cal_acc.py" --benchmark "${BENCH}" \
      --judge_json "judge_unified/${BENCH}/${M}_answer.jsonl" 2>&1 | grep -a "Acc" )
  echo "[${TAG}] ${BENCH}: ${ACC}" | tee -a "${RES}"
}

: > "${RES}"
score_lite vstar_bench_vopd vstar
score_lite zoombench_vopd   zoombench
score_lite hrbench4k_vopd   hrbench-4k
score_lite hrbench8k_vopd   hrbench-8k
score_mme  mme_realworld_cn mme-realworld-cn 5917
score_mme  mme_realworld    mme-realworld    23609
echo "[done] results in ${RES}"
