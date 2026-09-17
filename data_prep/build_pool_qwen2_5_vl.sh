#!/usr/bin/env bash
# =============================================================================
# RL pool construction for Qwen2.5-VL-7B (stage-2 input).
#
# Same four stages as the Qwen3.5 / Gemma-4 builders, with the Qwen2.5-VL
# specifics: 28 px patches (so the same 256 / 576 token budget is a different
# pixel count), a peak-ratio heatmap gate, and evidence responses regenerated
# for EVERY pool row (the stage-1 corpus of this family is not reused).
#
# Stages, in order (set START=filter|compose|evidence|maps to resume):
#   filter    pre_rl_filter --mode score, sharded over the GPUs: peak-ratio
#             gate (ratio 3.0, peak fraction 0.1), R = 6, gqa/chartqa dropped,
#             gold boxes covering > 10% of the image dropped
#   compose   aggregate at retention 0.2, then compose_pool.py picks
#             5,000 infographicsvqa + 1,000 textvqa + 1,000 docvqa rows
#   evidence  "[Visual Evidence]" responses from the FROZEN base model for all
#             pool rows
#   maps      response-to-image attention maps (layers 18..23) cached to disk;
#             writes the final pool with an `ev_maps_path` per row
#
#   PHASE_A_CKPT=output/sdrpn/qwen2_5vl-7b-sdrpn-K18T3 \
#   DATASET_ROOT=datasets GPU_IDS=0,1,2,3 bash data_prep/build_pool_qwen2_5_vl.sh
#
# NOTE on the released pool: `rl_pool_qwen2_5_vl_7b.jsonl` reuses the
# Qwen3.5-4B row selection (identical 7,000 sample_ids and stage-1 statistics)
# and only its evidence maps are Qwen2.5-VL-7B's own. Running this script from
# START=filter builds a genuinely 7B-selected pool instead, which is the
# stricter "own SD-RPN" recipe; to reproduce the released file exactly, start
# from START=evidence with POOL pointed at the Qwen3.5-4B pool jsonl.
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

PHASE_A_CKPT=${PHASE_A_CKPT:?SD-RPN (stage-1) checkpoint dir}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
SOURCE_JSONL=${SOURCE_JSONL:-data/VisionRL2-data/rl_pools/candidates_visualcot_50k.jsonl}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
IMAGE_ROOT=${IMAGE_ROOT:-${DATASET_ROOT}}
OUT_BASE=${OUT_BASE:-data/rl_pools/qwen2_5_vl_7b}
RUN_NAME=${RUN_NAME:-pool}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RETENTION=${RETENTION:-0.2}
GEN_BATCH=${GEN_BATCH:-8}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-1}
MASKED_PIL_WORKERS=${MASKED_PIL_WORKERS:-4}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
# Source-image budget: 256 / 576 visual tokens at 28 px patches.
MIN_PIXELS=${MIN_PIXELS:-200704}
MAX_PIXELS=${MAX_PIXELS:-451584}
LAYERS=${LAYERS:-18,19,20,21,22,23}
# Greedy decoding trips a stopping-criteria bug in transformers 4.51 for this
# model, so the evidence pass samples at temperature 1.0 (as the paper run did).
GEN_TEMPERATURE=${GEN_TEMPERATURE:-1.0}
START=${START:-filter}

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NGPU=${#GPUS[@]}
NSHARD=$(( NGPU * SHARDS_PER_GPU ))
POOL_DIR="${OUT_BASE}/${RUN_NAME}"
POOL=${POOL:-${POOL_DIR}/pool.jsonl}
RESP="${OUT_BASE}/pool_evidence.jsonl"
CACHE="${OUT_BASE}/ev_maps_cache"
EV_POOL=${EV_POOL:-${OUT_BASE}/rl_pool.jsonl}
mkdir -p "${OUT_BASE}"

stage_idx() { case "$1" in filter) echo 0;; compose) echo 1;; evidence) echo 2;; maps) echo 3;; *) echo -1;; esac; }
START_IDX=$(stage_idx "${START}")
[ "${START_IDX}" -ge 0 ] || { echo "START must be one of filter|compose|evidence|maps"; exit 1; }
stage_at_or_after() { [ "$(stage_idx "$1")" -ge "${START_IDX}" ]; }

FILTER_COMMON=(--input-jsonl "${SOURCE_JSONL}" --model-family qwen2_5_vl
  --min-pixels "${MIN_PIXELS}" --max-pixels "${MAX_PIXELS}" --attn-impl "${ATTN_IMPL}"
  --drop-sources gqa,chartqa --max-gold-area-fraction 0.1
  --threshold-mode peak_ratio --peak-fraction 0.1 --ratio-thresh 3.0 --R 6)

# ------------------------------------------------------------------ 1. filter
if stage_at_or_after filter; then
  echo "[pool-q25] scoring ${SOURCE_JSONL} with ${PHASE_A_CKPT} over ${NSHARD} shard(s)"
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
  echo "[pool-q25] pool rows: $(wc -l < "${POOL}")"
fi

# ---------------------------------------------------------------- 3. evidence
if stage_at_or_after evidence; then
  echo "[pool-q25] evidence responses from the frozen base model"
  for S in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$(( S % NGPU ))]} python data_prep/gen_pool_evidence_q25_7b.py \
        --pool "${POOL}" --image-root "${IMAGE_ROOT}" --model-path "${BASE_MODEL}" \
        --out "${OUT_BASE}/pool_evidence_shard${S}.jsonl" \
        --shard-idx "${S}" --num-shards "${NSHARD}" --batch-size "${GEN_BATCH}" \
        --max-new-tokens 512 --temperature "${GEN_TEMPERATURE}" &
    sleep 20
  done
  wait
  cat "${OUT_BASE}"/pool_evidence_shard*.jsonl > "${RESP}"
  echo "[pool-q25] evidence rows: $(wc -l < "${RESP}") / pool $(wc -l < "${POOL}")"
fi

# -------------------------------------------------------------------- 4. maps
if stage_at_or_after maps; then
  echo "[pool-q25] caching evidence attention maps -> ${CACHE}"
  for S in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$(( S % NGPU ))]} python data_prep/build_evidence_map_cache_q25_7b.py \
        --pool "${POOL}" --image-root "${IMAGE_ROOT}" --model-path "${BASE_MODEL}" \
        --responses "${RESP}" --layers "${LAYERS}" \
        --cache-dir "${CACHE}" --out-pool "${EV_POOL}" \
        --shard-id "${S}" --shard-count "${NSHARD}" &
    sleep 20
  done
  wait
  cat "${EV_POOL}".shard* > "${EV_POOL}"
  echo "[pool-q25] RL pool ready: ${EV_POOL} ($(wc -l < "${EV_POOL}") rows)"
  echo "[pool-q25] next: PHASE_A_CKPT=${PHASE_A_CKPT} FILTERED_JSONL=${EV_POOL} \\"
  echo "                EV_MAPS_ROOT=${OUT_BASE} DATASET_ROOT=${DATASET_ROOT} \\"
  echo "                bash scripts/train_rl_qwen2_5vl_7b.sh"
fi
