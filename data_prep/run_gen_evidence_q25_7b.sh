#!/usr/bin/env bash
# Generate [Visual Evidence] responses for the qwen2.5-VL-7B RL pool — which
# REUSES the qwen3.5-VL-4B 7k pool filtered_v2.jsonl (6897 unique over
# infographicsvqa/textvqa/docvqa), data-parallel sharded across GPUs 0,1,2,3.
# Overlaps the two vis servers on GPU 0,1 (leaves them running — there is
# enough free memory). Base teacher =
# Qwen2.5-VL-7B-Instruct, patch-28 budget, temperature 1.0 (greedy trips a
# stopping-criteria bug in this transformers build).
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh
conda activate qwen
CODE=/home/yuheng/code/Qwen2.5-VL; cd "$CODE"
mkdir -p logs/region_level_grpo
# qwen2.5-VL-7B REUSES the qwen3.5-VL-4B 7k pool (filtered_v2.jsonl, 6897 unique
# over infographicsvqa/textvqa/docvqa) — same pool the q25-7b Exp A baseline RL
# trains on, NOT the q25 own 10796 pool.
POOL=output/region_level_grpo/qwen3_5-4b-roi-K21T3-stage1-online-stripped-prompt/filtered_v2.jsonl
OUTDIR=output/region_level_grpo/phase_a_v2_responses
mkdir -p "$OUTDIR"
rm -f logs/region_level_grpo/gen_ev_q25_7b.done

GPUS=(0 1 2 3)
PIDS=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python make_data/gen_pool_evidence_q25_7b.py \
    --pool "$POOL" --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --out "$OUTDIR/pool_evidence_q25_7b_shard${i}.jsonl" \
    --shard-idx "$i" --num-shards 4 --batch-size 8 \
    --max-new-tokens 512 --temperature 1.0 \
    > "$CODE/logs/region_level_grpo/gen_ev_q25_7b_shard${i}.log" 2>&1 &
  PIDS+=($!)
  sleep 8   # stagger model loads
done
echo "[driver] launched shards on GPU ${GPUS[*]}: pids ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
echo "[driver] all shards finished (fail=$FAIL)"

# Merge shards -> single deduped jsonl.
python - <<'PY'
import json, glob, os
outdir = "output/region_level_grpo/phase_a_v2_responses"
merged = os.path.join(outdir, "pool_evidence_q25_7b.jsonl")
seen, n = set(), 0
with open(merged, "w", encoding="utf-8") as w:
    for sh in sorted(glob.glob(os.path.join(outdir, "pool_evidence_q25_7b_shard*.jsonl"))):
        for l in open(sh, encoding="utf-8"):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            k = (d.get("dataset"), os.path.basename(str(d["image"])), str(d["question"]).strip())
            if k in seen:
                continue
            seen.add(k)
            w.write(l if l.endswith("\n") else l + "\n")
            n += 1
print(f"[merge] {n} unique -> {merged}", flush=True)
PY

touch "$CODE/logs/region_level_grpo/gen_ev_q25_7b.done"
echo "[driver] DONE"
