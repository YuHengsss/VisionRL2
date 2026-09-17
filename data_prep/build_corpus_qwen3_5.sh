#!/usr/bin/env bash
# =============================================================================
# SD-RPN stage-1 corpus for the Qwen3.5 backbones (the released
# ``qwen3_5_{4b,9b}_response_corpus.jsonl``).
#
# The corpus is the model talking to itself: the FROZEN backbone answers the
# 50k VisualCoT candidates, and its own response-to-image attention becomes the
# pseudo-label at training time. Two prompt styles are generated over disjoint
# halves of the candidates and then merged:
#
#   v1   gqa + textvqa            task suffixes (bounding boxes / single word
#                                 or phrase), square-padded images,
#                                 visual budget 256..1024 tokens
#   v2   docvqa + infographicsvqa the "[Visual Evidence] ... [Answer]" prompt,
#                                 --no-expand2square, budget 256..576 tokens
#
# Both passes use greedy decoding and 512 new tokens. Rows with an empty
# response are dropped at the merge step (~49.5k of 50k survive).
#
# Stages, in order (set START=split|gen|merge to resume):
#   split   partition the candidates by dataset into the v1 / v2 halves
#   gen     one generator process per GPU per half (sharded, resume-safe)
#   merge   concatenate the shards, tag each row with its prompt style and
#           drop empty responses -> ${OUT_JSONL}
#
# Usage:
#   MODEL_PATH=Qwen/Qwen3.5-4B OUT_JSONL=data/sdrpn/qwen3_5_4b_response_corpus.jsonl \
#   DATASET_ROOT=datasets GPU_IDS=0,1,2,3 bash data_prep/build_corpus_qwen3_5.sh
#
# The result is what scripts/train_sdrpn_online.sh consumes as ROI_DATA_PATH.
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
CANDIDATES=${CANDIDATES:-data/VisionRL2-data/rl_pools/candidates_visualcot_50k.jsonl}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
CORPUS_DIR=${CORPUS_DIR:-data/sdrpn/qwen3_5_corpus}
OUT_JSONL=${OUT_JSONL:-${CORPUS_DIR}/response_corpus.jsonl}
GPU_IDS=${GPU_IDS:-0,1,2,3}
BATCH_SIZE=${BATCH_SIZE:-8}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
PATCH_SIZE=${PATCH_SIZE:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
# Visual-token budgets per prompt style (the paper values).
MIN_TOKENS=${MIN_TOKENS:-256}
V1_MAX_TOKENS=${V1_MAX_TOKENS:-1024}
V2_MAX_TOKENS=${V2_MAX_TOKENS:-576}
START=${START:-split}

IN_V1=${IN_V1:-${CORPUS_DIR}/candidates_v1.jsonl}
IN_V2=${IN_V2:-${CORPUS_DIR}/candidates_v2.jsonl}

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NPROC=${#GPUS[@]}
mkdir -p "${CORPUS_DIR}" "$(dirname "${OUT_JSONL}")"

stage_idx() { case "$1" in split) echo 0;; gen) echo 1;; merge) echo 2;; *) echo -1;; esac; }
START_IDX=$(stage_idx "${START}")
[ "${START_IDX}" -ge 0 ] || { echo "START must be one of split|gen|merge"; exit 1; }
stage_at_or_after() { [ "$(stage_idx "$1")" -ge "${START_IDX}" ]; }

# -------------------------------------------------------------------- 1. split
if stage_at_or_after split; then
  echo "[corpus] splitting ${CANDIDATES} by prompt style"
  python data_prep/split_candidates.py \
      --input "${CANDIDATES}" --out-v1 "${IN_V1}" --out-v2 "${IN_V2}"
fi

# ---------------------------------------------------------------------- 2. gen
if stage_at_or_after gen; then
  for V in v1 v2; do
    if [ "${V}" = v1 ]; then
      IN="${IN_V1}"; MAXTOK="${V1_MAX_TOKENS}"; PAD=""
    else
      IN="${IN_V2}"; MAXTOK="${V2_MAX_TOKENS}"; PAD="--no-expand2square"
    fi
    [ -f "${IN}" ] || { echo "[corpus] missing ${V} candidates: ${IN}"; exit 1; }
    echo "[corpus] generating ${V} (budget ${MIN_TOKENS}..${MAXTOK} tokens) over ${NPROC} GPU(s)"
    for S in $(seq 0 $((NPROC - 1))); do
      CUDA_VISIBLE_DEVICES=${GPUS[$S]} python data_prep/make_vcot50k_response_qwen35.py \
          --input-jsonl "${IN}" \
          --output-jsonl "${CORPUS_DIR}/response_${V}.shard${S}.jsonl" \
          --model-path "${MODEL_PATH}" --prompt-style "${V}" \
          --patch-size "${PATCH_SIZE}" \
          --min-tokens "${MIN_TOKENS}" --max-tokens "${MAXTOK}" \
          --max-new-tokens "${MAX_NEW_TOKENS}" --temperature 0.0 \
          --batch-size "${BATCH_SIZE}" --attn-impl "${ATTN_IMPL}" ${PAD} \
          --shard-id "${S}" --shard-count "${NPROC}" \
          > "${CORPUS_DIR}/gen_${V}_shard${S}.log" 2>&1 &
    done
    wait
  done
fi

# -------------------------------------------------------------------- 3. merge
if stage_at_or_after merge; then
  echo "[corpus] merging shards -> ${OUT_JSONL}"
  python - "${CORPUS_DIR}" "${OUT_JSONL}" <<'PY'
import collections, glob, json, os, sys

corpus_dir, out = sys.argv[1], sys.argv[2]
# Prompt style per generated file, recovered from the shard name.
rows, n_in, n_empty = {}, 0, 0
for version in ("v1", "v2"):
    for path in sorted(glob.glob(os.path.join(
            corpus_dir, f"response_{version}.shard*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            if not str(rec.get("response") or "").strip():
                n_empty += 1
                continue
            rec.pop("_response_meta", None)
            rec["version"] = version
            key = (rec.get("dataset"),
                   os.path.basename(str(rec.get("image", ""))),
                   str(rec.get("question", "")).strip())
            rows.setdefault(key, rec)

with open(out, "w", encoding="utf-8") as f:
    for rec in rows.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

cnt = collections.Counter((r.get("dataset"), r.get("version")) for r in rows.values())
print(f"[merge] {n_in} rows read, {n_empty} empty responses dropped "
      f"-> {len(rows)} unique")
for k, v in sorted(cnt.items(), key=lambda kv: str(kv[0])):
    print(f"[merge]   {str(k[0]):18s} {k[1]}  {v}")
PY
  echo "[corpus] ready: ${OUT_JSONL} ($(wc -l < "${OUT_JSONL}") rows)"
  echo "[corpus] next: MODEL=qwen3_5-4b ROI_DATA_PATH=${OUT_JSONL} \\"
  echo "                DATASET_ROOT=${DATASET_ROOT} bash scripts/train_sdrpn_online.sh"
fi
