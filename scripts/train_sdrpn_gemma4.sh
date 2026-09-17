#!/usr/bin/env bash
# =============================================================================
# SD-RPN (stage 1) for Gemma-4-12B-it - the "v4mix" corpus recipe.
#
# Gemma-4 is encoder-free: the processor turns an image into a fixed number of
# soft visual tokens, picked from the discrete tiers {70, 140, 280, 560, 1120}
# (the tier is always filled). Stage 1 runs at tier 560 and clones layers
# 27/28/29 of the frozen backbone into the twig (K = 27, T = 3).
#
# Stages, in order (set START=gen|merge|train|assemble to resume):
#   gen       regenerate the stage-1 corpus with the FROZEN base model
#               v1 rows (gqa + textvqa): expand2square padding, 64 new tokens
#               v2 rows (docvqa + infographicsvqa): --no-expand2square, 512 new
#                       tokens (evidence-style responses -> single-region labels)
#   merge     de-duplicate the v1 + v2 shards into one training jsonl
#   train     twig training: lr 1e-4, effective batch 128 (micro 2 x accum 32 on
#             2 GPUs), cosine, 3% warmup, 1 epoch, keep-layers 30
#   assemble  twig delta + base snapshot -> a full, loadable checkpoint
#
# Input corpora (INPUT_V1 / INPUT_V2): jsonl, one row per QA with
#   {"question_id", "dataset", "image", "question", "version"}
#   and optionally "prompted_question" (used verbatim when present).
# "dataset" selects the image root (see IMAGE_ROOT below and the
# DEFAULT_IMAGE_ROOTS map in qwen_src/gemma4_unified/gemma_stage1_gen.py).
#
#   IMAGE_ROOT=datasets INPUT_V1=data/sdrpn/gemma4_v4mix_input_v1.jsonl \
#   INPUT_V2=data/sdrpn/gemma4_v4mix_input_v2.jsonl GPU_IDS=0,1 \
#     bash scripts/train_sdrpn_gemma4.sh
# =============================================================================
set -euo pipefail
CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export QZOOM_REPO="${CODE_ROOT}"
export PYTHONUNBUFFERED=1
G="${CODE_ROOT}/qwen_src/gemma4_unified"

BASE_MODEL=${BASE_MODEL:-google/gemma-4-12B-it}
IMAGE_ROOT=${IMAGE_ROOT:-datasets}                 # parent of the per-dataset image folders
CORPUS_DIR=${CORPUS_DIR:-data/sdrpn/gemma4_v4mix}  # generated responses land here
INPUT_V1=${INPUT_V1:-${CORPUS_DIR}/input_v1.jsonl}
INPUT_V2=${INPUT_V2:-${CORPUS_DIR}/input_v2.jsonl}
TRAIN_JSONL=${TRAIN_JSONL:-${CORPUS_DIR}/gemma12b_it_560_v4mix_train.jsonl}
OUT_DIR=${OUT_DIR:-output/sdrpn/gemma4-12b-roi-K27T3-stage1-v4mix}
OUT_FULL=${OUT_FULL:-${OUT_DIR}-full}
GPU_IDS=${GPU_IDS:-0,1}
TIER=${TIER:-560}                                  # stage-1 soft-token tier
TWIG_K=${TWIG_K:-27}; TWIG_T=${TWIG_T:-3}; KEEP_LAYERS=${KEEP_LAYERS:-30}
LR=${LR:-1e-4}; EPOCHS=${EPOCHS:-1}; EFF_BATCH=${EFF_BATCH:-128}
MICRO_BATCH=${MICRO_BATCH:-2}
GEN_BATCH=${GEN_BATCH:-16}
START=${START:-gen}
NPROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
ACCUM=${ACCUM:-$(( EFF_BATCH / MICRO_BATCH / NPROC ))}
mkdir -p "${CORPUS_DIR}" "${OUT_DIR}"

stage_idx() { case "$1" in gen) echo 0;; merge) echo 1;; train) echo 2;; assemble) echo 3;; *) echo -1;; esac; }
START_IDX=$(stage_idx "${START}")
[ "${START_IDX}" -ge 0 ] || { echo "START must be one of gen|merge|train|assemble"; exit 1; }
stage_at_or_after() { [ "$(stage_idx "$1")" -ge "${START_IDX}" ]; }

# --------------------------------------------------------------- 1. generation
if stage_at_or_after gen; then
  IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
  MAP="gqa=${IMAGE_ROOT},textvqa=${IMAGE_ROOT},docvqa=${IMAGE_ROOT},infographicsvqa=${IMAGE_ROOT}"
  # v1 = short-answer / bbox prompts on square-padded images; v2 = evidence
  # responses on the unpadded image (the labels come from the response text).
  for V in v1 v2; do
    IN=$([ "${V}" = v1 ] && echo "${INPUT_V1}" || echo "${INPUT_V2}")
    [ -f "${IN}" ] || { echo "[sdrpn-gemma4] missing ${V} corpus: ${IN}"; exit 1; }
    NEW_TOK=$([ "${V}" = v1 ] && echo 64 || echo 512)
    PAD=$([ "${V}" = v1 ] && echo "" || echo "--no-expand2square")
    echo "[sdrpn-gemma4] generating ${V} (${NEW_TOK} new tokens, tier ${TIER}) over ${NPROC} GPU(s)"
    for S in $(seq 0 $((NPROC - 1))); do
      CUDA_VISIBLE_DEVICES=${GPUS[$S]} python "${G}/gemma_stage1_gen.py" \
          --input-jsonl "${IN}" --output-jsonl "${CORPUS_DIR}/gemma12b_it_${TIER}_${V}_s${S}.jsonl" \
          --model-path "${BASE_MODEL}" --image-root-map "${MAP}" \
          --max-soft-tokens "${TIER}" --max-new-tokens "${NEW_TOK}" --batch-size "${GEN_BATCH}" ${PAD} \
          --shard-count "${NPROC}" --shard-id "${S}" &
    done
    wait
  done
fi

# ------------------------------------------------------------------- 2. merge
if stage_at_or_after merge; then
  echo "[sdrpn-gemma4] merging shards -> ${TRAIN_JSONL}"
  python - "${CORPUS_DIR}" "${TRAIN_JSONL}" <<'PY'
import collections, glob, json, os, sys
corpus, out = sys.argv[1], sys.argv[2]
seen, n_in = {}, 0
for p in sorted(glob.glob(os.path.join(corpus, "gemma12b_it_*_v[12]*.jsonl"))):
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_in += 1
        if not (r.get("response") or "").strip():
            continue
        seen.setdefault(r["question_id"], r)
with open(out, "w", encoding="utf-8") as f:
    for r in seen.values():
        r.pop("_response_meta", None)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
cnt = collections.Counter((r.get("dataset"), r.get("version")) for r in seen.values())
print(f"[merge] {n_in} rows read -> {len(seen)} unique with a non-empty response")
for k, v in sorted(cnt.items(), key=lambda kv: str(kv[0])):
    print(f"[merge]   {str(k[0]):18s} {k[1]}  {v}")
PY
fi

# ------------------------------------------------------------------- 3. train
if stage_at_or_after train; then
  [ -f "${TRAIN_JSONL}" ] || { echo "[sdrpn-gemma4] missing ${TRAIN_JSONL}"; exit 1; }
  # online pseudo-label knobs (identical to the paper run)
  export ONLINE_PSEUDO_DEBUG=${ONLINE_PSEUDO_DEBUG:-0} SKIP_POST_BRANCH=${SKIP_POST_BRANCH:-1}
  export STRIP_TASK_SUFFIX_PROB=${STRIP_TASK_SUFFIX_PROB:-1.0}
  export STRIP_GQA_SUFFIX_PROB=${STRIP_GQA_SUFFIX_PROB:-0.95}
  export STRIP_TEXTVQA_SUFFIX_PROB=${STRIP_TEXTVQA_SUFFIX_PROB:-0.20}
  export STRIP_EVIDENCE_SUFFIX_PROB=${STRIP_EVIDENCE_SUFFIX_PROB:-1.0}
  export ROI_BINARY_COEFF=${ROI_BINARY_COEFF:-0.25} BG_COFF=${BG_COFF:-0.05}
  N=$(wc -l < "${TRAIN_JSONL}"); STEPS=$(( N / EFF_BATCH )); WARMUP=${WARMUP:-$(( STEPS * 3 / 100 ))}
  echo "[sdrpn-gemma4] train: ${N} rows, ~${STEPS} steps, warmup ${WARMUP}, micro ${MICRO_BATCH} x accum ${ACCUM} x ${NPROC} = $((MICRO_BATCH*ACCUM*NPROC))"
  CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --standalone --nproc_per_node="${NPROC}" \
      "${G}/train_stage1_full.py" \
      --jsonl "${TRAIN_JSONL}" --image-root "${IMAGE_ROOT}" --out-dir "${OUT_DIR}" \
      --base-ckpt "${BASE_MODEL}" \
      --micro-batch "${MICRO_BATCH}" --accum "${ACCUM}" --lr "${LR}" --warmup "${WARMUP}" \
      --scheduler cosine --epochs "${EPOCHS}" --twig-K "${TWIG_K}" --twig-T "${TWIG_T}" \
      --keep-layers "${KEEP_LAYERS}" --save-every 100 --log-every 10 --num-workers 6 --resume auto
fi

# ---------------------------------------------------------------- 4. assemble
if stage_at_or_after assemble; then
  [ -f "${OUT_DIR}/twig_delta_final.pt" ] || { echo "[sdrpn-gemma4] missing ${OUT_DIR}/twig_delta_final.pt"; exit 1; }
  python "${G}/assemble_full_checkpoint.py" --delta "${OUT_DIR}/twig_delta_final.pt" --out "${OUT_FULL}"
  echo "[sdrpn-gemma4] SD-RPN checkpoint -> ${OUT_FULL}"
fi
