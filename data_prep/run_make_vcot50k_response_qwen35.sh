#!/usr/bin/env bash
# Generate vcot50k responses with Qwen3.5-VL using HF transformers
# (vLLM does not support Qwen3.5-VL yet). One process per GPU; the input
# jsonl is sharded round-robin so each process owns a disjoint half. After
# both shards finish, the script concatenates them into the canonical
# output file.
#
# Args (positional):
#   $1 GPU_IDS      e.g. "2,3" (default 2,3)
#   $2 MODEL_PATH   default Qwen/Qwen3.5-9B
#   $3 LIMIT        truncate to N samples for smoke (default empty = full)
#
# Env overrides:
#   MIN_TOKENS, MAX_TOKENS (default 256 / 1024)
#   MAX_NEW_TOKENS (default 512)
#   BATCH_SIZE     (default 1; >1 needs left-padding ok in HF)
#   ATTN_IMPL      (default flash_attention_2)
#   INPUT_JSONL, OUTPUT_PREFIX
#   PATCH_SIZE     (default 32)
#
# Output:
#   ${OUTPUT_PREFIX}.shard{0,1}.jsonl         per-shard files
#   ${OUTPUT_PREFIX}.jsonl                    concat (only if all shards
#                                             finished cleanly)

set -euo pipefail

GPU_IDS="${1:-2,3}"
MODEL_PATH="${2:-Qwen/Qwen3.5-9B}"
LIMIT_RAW="${3:-}"

MIN_TOKENS="${MIN_TOKENS:-256}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# flash_attention_2 is ABI-broken in the qwen35 env (transformers 5.6.2),
# so default to SDPA. Set ATTN_IMPL=flash_attention_2 to override once the
# env is fixed.
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
PATCH_SIZE="${PATCH_SIZE:-32}"
INPUT_JSONL="${INPUT_JSONL:-/home/yuheng/datasets/visual_cot_jsonl/vcot50k_source.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/home/yuheng/datasets/visual_cot_jsonl/vcot50k_response_qwen35_9b}"
LOG_DIR="${LOG_DIR:-/home/yuheng/code/Qwen2.5-VL/logs/make_vcot50k_qwen35}"

IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
SHARD_COUNT="${#GPU_ARR[@]}"

mkdir -p "${LOG_DIR}"

echo "[launch] GPUs=${GPU_IDS}  shards=${SHARD_COUNT}  model=${MODEL_PATH}"
echo "[launch] tokens=[${MIN_TOKENS}, ${MAX_TOKENS}]  bs=${BATCH_SIZE}  attn=${ATTN_IMPL}"
echo "[launch] input=${INPUT_JSONL}"
echo "[launch] output_prefix=${OUTPUT_PREFIX}"
echo "[launch] log_dir=${LOG_DIR}"

cd "$(dirname "$0")/.."

LIMIT_FLAG=""
if [[ -n "${LIMIT_RAW}" ]]; then
    LIMIT_FLAG="--limit ${LIMIT_RAW}"
fi

PIDS=()
for shard_id in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$shard_id]}"
    out="${OUTPUT_PREFIX}.shard${shard_id}.jsonl"
    log="${LOG_DIR}/shard${shard_id}_gpu${gpu}.log"
    echo "[launch] shard ${shard_id} on GPU ${gpu} → ${out}  (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        python make_data/make_vcot50k_response_qwen35.py \
        --input-jsonl "${INPUT_JSONL}" \
        --output-jsonl "${out}" \
        --model-path "${MODEL_PATH}" \
        --patch-size "${PATCH_SIZE}" \
        --min-tokens "${MIN_TOKENS}" \
        --max-tokens "${MAX_TOKENS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --batch-size "${BATCH_SIZE}" \
        --attn-impl "${ATTN_IMPL}" \
        --shard-id "${shard_id}" \
        --shard-count "${SHARD_COUNT}" \
        ${LIMIT_FLAG} \
        > "${log}" 2>&1 &
    PIDS+=($!)
done

echo "[launch] PIDs: ${PIDS[*]}"
echo "[launch] tail logs with: tail -f ${LOG_DIR}/shard*.log"

# Wait for all shards. If any fails, surface the failure but let the
# others keep running; we'll return non-zero so the caller knows.
fail=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "[wait] PID ${pid} exited non-zero"
        fail=1
    fi
done

if [[ "${fail}" -ne 0 ]]; then
    echo "[error] one or more shards failed; not concatenating"
    exit 1
fi

# Concat shards
final="${OUTPUT_PREFIX}.jsonl"
echo "[concat] → ${final}"
: > "${final}"
for shard_id in "${!GPU_ARR[@]}"; do
    cat "${OUTPUT_PREFIX}.shard${shard_id}.jsonl" >> "${final}"
done
n_total=$(wc -l < "${final}")
echo "[done] ${n_total} lines in ${final}"
