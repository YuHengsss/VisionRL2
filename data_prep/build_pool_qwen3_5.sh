#!/usr/bin/env bash
# =============================================================================
# RL pool construction for the Qwen3.5 backbones (stage-2 input).
#
# The pool is built with the model's OWN SD-RPN checkpoint, so every backbone
# trains on the samples its own predictor finds informative.
#
# Stages, in order (set START=filter|compose|evidence|maps to resume):
#   filter    pre_rl_filter --mode score, sharded over the GPUs: for each
#             VisualCoT candidate it extracts the regions the trainer would see
#             (fixed threshold 0.02, R = 6), rolls the frozen reader out over
#             them and records the per-sample reward std (gqa/chartqa dropped,
#             gold boxes covering > 10% of the image dropped)
#   compose   aggregate at retention 0.2, then compose_pool.py picks
#             5,000 infographicsvqa + 1,000 textvqa + 1,000 docvqa rows
#   evidence  "[Visual Evidence]" responses for the pool's TEXTVQA rows only -
#             the docvqa / infographicsvqa rows already have evidence-style
#             responses in the stage-1 corpus (its v2 half), which is reused
#   maps      response-to-image attention maps (layers 7/11/15/19/23/27) cached
#             to disk; writes the final pool with an `ev_maps_path` per row
#
# PYTHONHASHSEED is pinned: pre_rl_filter's stratified shuffle seeds with
# hash(source), so every shard must see the same candidate order.
#
#   PHASE_A_CKPT=output/sdrpn/qwen3_5-4b-sdrpn-K21T3-online \
#   BASE_MODEL=Qwen/Qwen3.5-4B \
#   CORPUS=data/VisionRL2-data/sdrpn_corpora/qwen3_5_4b_response_corpus.jsonl \
#   DATASET_ROOT=datasets GPU_IDS=0,1,2,3 bash data_prep/build_pool_qwen3_5.sh
#
# The resulting pool jsonl is what scripts/train_rl_qwen3_5_{4b,9b}.sh consume
# as FILTERED_JSONL; point EV_MAPS_ROOT at ${OUT_BASE} (or leave it unset - the
# cache path is also resolved relative to the pool jsonl's own directory).
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

PHASE_A_CKPT=${PHASE_A_CKPT:?SD-RPN (stage-1) checkpoint dir}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-4B}
SOURCE_JSONL=${SOURCE_JSONL:-data/VisionRL2-data/rl_pools/candidates_visualcot_50k.jsonl}
# The stage-1 corpus: its v2 (docvqa + infographicsvqa) rows are the evidence
# responses the map step teacher-forces.
CORPUS=${CORPUS:-data/VisionRL2-data/sdrpn_corpora/qwen3_5_4b_response_corpus.jsonl}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
IMAGE_ROOT=${IMAGE_ROOT:-${DATASET_ROOT}}
OUT_BASE=${OUT_BASE:-data/rl_pools/qwen3_5_4b}
RUN_NAME=${RUN_NAME:-pool}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RETENTION=${RETENTION:-0.2}
GEN_BATCH=${GEN_BATCH:-8}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-1}
MASKED_PIL_WORKERS=${MASKED_PIL_WORKERS:-4}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
# Source-image budget: 256 / 576 visual tokens at 32 px patches (= the RL run's).
MIN_PIXELS=${MIN_PIXELS:-262144}
MAX_PIXELS=${MAX_PIXELS:-589824}
LAYERS=${LAYERS:-7,11,15,19,23,27}
START=${START:-filter}

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NGPU=${#GPUS[@]}
NSHARD=$(( NGPU * SHARDS_PER_GPU ))
POOL_DIR="${OUT_BASE}/${RUN_NAME}"
POOL="${POOL_DIR}/pool.jsonl"
TEXTVQA_RESP="${OUT_BASE}/pool_textvqa_evidence.jsonl"
CACHE="${OUT_BASE}/ev_maps_cache"
EV_POOL=${EV_POOL:-${OUT_BASE}/rl_pool.jsonl}
mkdir -p "${OUT_BASE}"

stage_idx() { case "$1" in filter) echo 0;; compose) echo 1;; evidence) echo 2;; maps) echo 3;; *) echo -1;; esac; }
START_IDX=$(stage_idx "${START}")
[ "${START_IDX}" -ge 0 ] || { echo "START must be one of filter|compose|evidence|maps"; exit 1; }
stage_at_or_after() { [ "$(stage_idx "$1")" -ge "${START_IDX}" ]; }

FILTER_COMMON=(--input-jsonl "${SOURCE_JSONL}" --model-family qwen3_5
  --min-pixels "${MIN_PIXELS}" --max-pixels "${MAX_PIXELS}" --attn-impl "${ATTN_IMPL}"
  --drop-sources gqa,chartqa --max-gold-area-fraction 0.1
  --threshold-mode fixed --fixed-threshold 0.02 --R 6)

# ------------------------------------------------------------------ 1. filter
if stage_at_or_after filter; then
  echo "[pool-q35] scoring ${SOURCE_JSONL} with ${PHASE_A_CKPT} over ${NSHARD} shard(s)"
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
  echo "[pool-q35] pool rows: $(wc -l < "${POOL}")"
fi

# ---------------------------------------------------------------- 3. evidence
if stage_at_or_after evidence; then
  echo "[pool-q35] evidence responses for the pool's textvqa rows"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} python data_prep/gen_pool_textvqa_evidence.py \
      --pool "${POOL}" --model-path "${BASE_MODEL}" \
      --image-root "${IMAGE_ROOT}/textvqa/train_images" \
      --out "${TEXTVQA_RESP}" --batch-size "${GEN_BATCH}"
  echo "[pool-q35] textvqa evidence rows: $(wc -l < "${TEXTVQA_RESP}")"
fi

# -------------------------------------------------------------------- 4. maps
if stage_at_or_after maps; then
  [ -f "${CORPUS}" ] || { echo "[pool-q35] missing stage-1 corpus: ${CORPUS}"; exit 1; }
  echo "[pool-q35] caching evidence attention maps -> ${CACHE}"
  # The textvqa evidence file comes FIRST: the response index keeps the first
  # hit per (dataset, image, question), and the corpus's textvqa rows carry
  # short-answer responses, not evidence ones.
  for S in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$(( S % NGPU ))]} python data_prep/build_evidence_map_cache.py \
        --pool "${POOL}" --image-root "${IMAGE_ROOT}" --model-path "${BASE_MODEL}" \
        --responses "${TEXTVQA_RESP},${CORPUS}" --layers "${LAYERS}" \
        --cache-dir "${CACHE}" --out-pool "${EV_POOL}" \
        --shard-id "${S}" --shard-count "${NSHARD}" &
    sleep 20
  done
  wait
  cat "${EV_POOL}".shard* > "${EV_POOL}"
  echo "[pool-q35] RL pool ready: ${EV_POOL} ($(wc -l < "${EV_POOL}") rows)"
  echo "[pool-q35] next: PHASE_A_CKPT=${PHASE_A_CKPT} FILTERED_JSONL=${EV_POOL} \\"
  echo "                EV_MAPS_ROOT=${OUT_BASE} DATASET_ROOT=${DATASET_ROOT} \\"
  echo "                bash scripts/train_rl_qwen3_5_4b.sh"
fi
