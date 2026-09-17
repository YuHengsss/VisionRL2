#!/usr/bin/env bash
# =============================================================================
# RL pool construction for Gemma-4-12B-it (stage 2 input).
#
# The pool is built with the model's OWN SD-RPN checkpoint, so every backbone
# trains on the samples its own predictor finds informative.
#
# Stages, in order (set START=filter|compose|evidence|maps to resume):
#   filter    pre_rl_filter --mode score, sharded over the GPUs: for each
#             VisualCoT candidate it extracts the regions the trainer would see
#             (peak_ratio, peak fraction 0.3, ratio 3.0, R = 6), rolls the frozen
#             reader out over them and records the per-sample reward std
#             (gqa/chartqa dropped, gold boxes covering > 10% of the image dropped)
#   compose   aggregate at retention 0.2, then compose_pool.py picks
#             5,000 infographicsvqa + 1,000 textvqa + 1,000 docvqa rows
#   evidence  free-form evidence responses from the FROZEN base model (tier 560)
#   maps      response-to-image attention maps (layers 11/17/23/29/35/41) cached
#             to disk; writes the final pool with an `ev_maps_path` per row
#
# PYTHONHASHSEED is pinned: pre_rl_filter's stratified shuffle seeds with
# hash(source), so every shard must see the same candidate order.
#
#   PHASE_A_CKPT=output/sdrpn/gemma4-12b-roi-K27T3-stage1-v4mix-full \
#   SOURCE_JSONL=data/VisionRL2-data/rl_pools/candidates_visualcot_50k.jsonl \
#   IMAGE_ROOT=datasets \
#   GPU_IDS=0,1,2,3 bash data_prep/build_pool_gemma4.sh
#
# The resulting pool jsonl is what scripts/train_rl_gemma4_12b.sh consumes as
# FILTERED_JSONL.
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

PHASE_A_CKPT=${PHASE_A_CKPT:?assembled Gemma SD-RPN checkpoint dir}
BASE_MODEL=${BASE_MODEL:-google/gemma-4-12B-it}
SOURCE_JSONL=${SOURCE_JSONL:-data/VisionRL2-data/rl_pools/candidates_visualcot_50k.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-datasets}
export DATASET_ROOT=${DATASET_ROOT:-${IMAGE_ROOT}}
OUT_BASE=${OUT_BASE:-data/rl_pools/gemma4_12b}
RUN_NAME=${RUN_NAME:-pool}
GPU_IDS=${GPU_IDS:-0,1}
TIER=${TIER:-560}                 # RL soft-token tier
RETENTION=${RETENTION:-0.2}
GEN_BATCH=${GEN_BATCH:-16}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-1}
MASKED_PIL_WORKERS=${MASKED_PIL_WORKERS:-4}
START=${START:-filter}

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NGPU=${#GPUS[@]}
NSHARD=$(( NGPU * SHARDS_PER_GPU ))
POOL_DIR="${OUT_BASE}/${RUN_NAME}"
POOL="${POOL_DIR}/pool.jsonl"
RESP="${OUT_BASE}/pool_evidence.jsonl"
CACHE="${OUT_BASE}/ev_maps_cache"
EV_POOL=${EV_POOL:-${OUT_BASE}/rl_pool.jsonl}
mkdir -p "${OUT_BASE}"

stage_idx() { case "$1" in filter) echo 0;; compose) echo 1;; evidence) echo 2;; maps) echo 3;; *) echo -1;; esac; }
START_IDX=$(stage_idx "${START}")
[ "${START_IDX}" -ge 0 ] || { echo "START must be one of filter|compose|evidence|maps"; exit 1; }
stage_at_or_after() { [ "$(stage_idx "$1")" -ge "${START_IDX}" ]; }

FILTER_COMMON=(--input-jsonl "${SOURCE_JSONL}" --model-family gemma4 --max-soft-tokens "${TIER}"
  --drop-sources gqa,chartqa --max-gold-area-fraction 0.1
  --threshold-mode peak_ratio --peak-fraction 0.3 --ratio-thresh 3.0 --R 6)

# ------------------------------------------------------------------ 1. filter
if stage_at_or_after filter; then
  echo "[pool-gemma4] scoring ${SOURCE_JSONL} with ${PHASE_A_CKPT} over ${NSHARD} shard(s)"
  for S in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$(( S % NGPU ))]} python data_prep/pre_rl_filter.py \
        --mode score --model-path "${PHASE_A_CKPT}" "${FILTER_COMMON[@]}" \
        --output-base "${OUT_BASE}" --run-name "${RUN_NAME}" \
        --num-shards "${NSHARD}" --shard-id "${S}" --masked-pil-workers "${MASKED_PIL_WORKERS}" &
    sleep 20
  done
  wait
fi

# ----------------------------------------------------------------- 2. compose
if stage_at_or_after compose; then
  python data_prep/pre_rl_filter.py --mode aggregate --model-path "${PHASE_A_CKPT}" \
      "${FILTER_COMMON[@]}" --output-base "${OUT_BASE}" --run-name "${RUN_NAME}" \
      --target-retention "${RETENTION}"
  python data_prep/compose_pool.py --stats-dir "${POOL_DIR}" --out-name "$(basename "${POOL}")"
  echo "[pool-gemma4] pool rows: $(wc -l < "${POOL}")"
fi

# ---------------------------------------------------------------- 3. evidence
if stage_at_or_after evidence; then
  echo "[pool-gemma4] evidence responses from the frozen base model (tier ${TIER})"
  for S in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$(( S % NGPU ))]} python data_prep/gen_pool_evidence_gemma.py \
        --pool "${POOL}" --image-root "${IMAGE_ROOT}" --model-path "${BASE_MODEL}" \
        --out "${OUT_BASE}/pool_evidence_shard${S}.jsonl" \
        --shard-idx "${S}" --num-shards "${NSHARD}" --batch-size "${GEN_BATCH}" \
        --max-soft-tokens "${TIER}" &
    sleep 20
  done
  wait
  cat "${OUT_BASE}"/pool_evidence_shard*.jsonl > "${RESP}"
  echo "[pool-gemma4] evidence rows: $(wc -l < "${RESP}") / pool $(wc -l < "${POOL}")"
fi

# -------------------------------------------------------------------- 4. maps
if stage_at_or_after maps; then
  echo "[pool-gemma4] caching evidence attention maps -> ${CACHE}"
  for S in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$(( S % NGPU ))]} python data_prep/build_evidence_map_cache_gemma.py \
        --pool "${POOL}" --image-root "${IMAGE_ROOT}" --model-path "${BASE_MODEL}" \
        --responses "${RESP}" --cache-dir "${CACHE}" --out-pool "${EV_POOL}" \
        --max-soft-tokens "${TIER}" --shard-id "${S}" --shard-count "${NSHARD}" &
    sleep 20
  done
  wait
  cat "${EV_POOL}".shard* > "${EV_POOL}"
  echo "[pool-gemma4] RL pool ready: ${EV_POOL} ($(wc -l < "${EV_POOL}") rows)"
  echo "[pool-gemma4] next: PHASE_A_CKPT=${PHASE_A_CKPT} FILTERED_JSONL=${EV_POOL} \\"
  echo "                EV_MAPS_ROOT=${OUT_BASE} DATASET_ROOT=${IMAGE_ROOT} \\"
  echo "                bash scripts/train_rl_gemma4_12b.sh"
fi
