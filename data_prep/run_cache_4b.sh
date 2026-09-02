#!/usr/bin/env bash
# Build the 4B multi-layer response2image cache (7k pool), sharded over
# GPUs 1,2,3 (GPU0 reserved for the attn-viz server). Concats the
# per-shard pool jsonls into filtered_v2_evmaps_4b.jsonl when done.
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
cd /home/yuheng/code/Qwen2.5-VL

POOL=output/region_level_grpo/qwen3_5-4b-roi-K21T3-stage1-online-stripped-prompt/filtered_v2.jsonl
RESP=output/region_level_grpo/phase_a_v2_responses/qwen35_4b_vcot50k_v2.jsonl,output/region_level_grpo/phase_a_v2_responses/pool_textvqa_evidence_4b.jsonl
CACHE=output/region_level_grpo/ev_maps_cache_4b
OUTPOOL=output/region_level_grpo/filtered_v2_evmaps_4b.jsonl
GPUS=(0 1 2 3)   # GPU0 overlaps the attn-viz server (has headroom)

pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python excluded/multi_group/build_evidence_map_cache.py \
    --pool "${POOL}" --model-path Qwen/Qwen3.5-4B --responses "${RESP}" \
    --cache-dir "${CACHE}" --out-pool "${OUTPOOL}" \
    --shard-id "$i" --shard-count "${#GPUS[@]}" \
    > excluded/multi_group/cache_4b_shard${i}.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

cat "${OUTPOOL}".shard* > "${OUTPOOL}"
echo "[cache-4b] rc=$rc total $(wc -l < ${OUTPOOL}) rows -> ${OUTPOOL}"
grep -hE '\[done s' excluded/multi_group/cache_4b_shard*.log
touch excluded/multi_group/cache_4b.done
