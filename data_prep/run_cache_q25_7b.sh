#!/usr/bin/env bash
# Build the response->image ev_maps cache (layers 18-23 -> 6 maps) for the
# qwen2.5-VL-7B source-map line, over the reused qwen3.5-4B 7k pool
# (filtered_v2.jsonl), using the q25-7B evidence responses. 4-GPU sharded;
# overlaps the two vis servers on GPU 0,1 (enough free memory). Merges the
# shard out-pools -> filtered_v2_evmaps_q25_7b.jsonl (one ev_maps_path/row).
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh
conda activate qwen
CODE=/home/yuheng/code/Qwen2.5-VL; cd "$CODE"
mkdir -p logs/region_level_grpo
POOL=output/region_level_grpo/qwen3_5-4b-roi-K21T3-stage1-online-stripped-prompt/filtered_v2.jsonl
RESP=output/region_level_grpo/phase_a_v2_responses/pool_evidence_q25_7b.jsonl
CACHE=output/region_level_grpo/ev_maps_cache_q25_7b
OUTPOOL=output/region_level_grpo/filtered_v2_evmaps_q25_7b.jsonl
rm -f logs/region_level_grpo/build_cache_q25_7b.done

GPUS=(0 1 2 3)
PIDS=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python excluded/multi_group/build_evidence_map_cache_q25_7b.py \
    --pool "$POOL" --model-path Qwen/Qwen2.5-VL-7B-Instruct --responses "$RESP" \
    --cache-dir "$CACHE" --out-pool "$OUTPOOL" \
    --shard-id "$i" --shard-count 4 \
    > "$CODE/logs/region_level_grpo/build_cache_q25_7b_shard${i}.log" 2>&1 &
  PIDS+=($!)
  sleep 8
done
echo "[cache-driver] launched on GPU ${GPUS[*]}: pids ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
echo "[cache-driver] all shards finished (fail=$FAIL)"

# Merge shard out-pools (shards partition rows disjointly -> plain concat).
python - <<'PY'
import json, glob, os
op = "output/region_level_grpo/filtered_v2_evmaps_q25_7b.jsonl"
n = 0
with open(op, "w", encoding="utf-8") as w:
    for sh in sorted(glob.glob(op + ".shard*")):
        for l in open(sh, encoding="utf-8"):
            if l.strip():
                w.write(l if l.endswith("\n") else l + "\n")
                n += 1
print(f"[merge] {n} rows -> {op}", flush=True)
PY

touch "$CODE/logs/region_level_grpo/build_cache_q25_7b.done"
echo "[cache-driver] DONE"
