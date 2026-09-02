#!/bin/bash
# ---------------------------------------------------------------------------
# ZwZ-8B reproduced through inclusionAI/Zooming-without-Zooming's OWN eval code
# (mm-eval/infer_without_tool.py), released weights inclusionAI/ZwZ-8B.
#
# Benchmarks: vstar, zoom-bench, hrbench-4k, hrbench-8k. The benchmark jsons are
# symlinked from the Vision-OPD reproduction's prepared data -- same schema
# ({images, query, response}), same source datasets, so both competitor rows
# consume byte-identical inputs. (ZwZ's own utils/convert_benchmark.py reads
# columns 'prompt'/'answer' that the released ZoomBench no longer has; the
# equivalent 'query'/'response' conversion is what Vision-OPD's prep produced.)
#
# BACKEND=vllm : their script verbatim (SamplingParams temp .7 / top_p .8 /
#                top_k 20 / presence_penalty 1.5 / seed 42 / max_tokens 8192)
# BACKEND=hf   : infer_without_tool_hf.py, identical prompt+sampling on HF
#                transformers, for when vLLM cannot run on this driver.
#
# Generation only; judging is a separate phase (needs the judge shim's GPU).
# Resumable (their script skips images[0]+query already in the answer file);
# artifact-gated on the per-benchmark line count.
# ---------------------------------------------------------------------------
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh

Z=/home/yuheng/code/ZwZ/mm-eval
QZ=/home/yuheng/code/Qwen2.5-VL
MT=$QZ/logs/region_level_grpo/main_table_repro
LOG=$MT/zwz_theircode; mkdir -p "$LOG"
S=$MT/zwz_theircode.txt; touch "$S"

BACKEND=${BACKEND:-vllm}
GPUS=${GPUS:-1,3}
MODEL=${MODEL:-ZwZ-8B}
MP=${MP:-inclusionAI/ZwZ-8B}
NGPU=$(echo "$GPUS" | awk -F',' '{print NF}')
BENCHES=${BENCHES:-"vstar zoom-bench hrbench-4k hrbench-8k"}

case $BACKEND in
  vllm) conda activate zwzvllm ;;
  hf)   conda activate qwen3 ;;
  *)    echo "bad BACKEND $BACKEND" >> "$S"; exit 1 ;;
esac

nexp () { case $1 in vstar) echo 191 ;; zoom-bench) echo 845 ;; mme-realworld) echo 23609 ;; mme-realworld-cn) echo 5917 ;; *) echo 800 ;; esac; }
nlines () { [ -f "$1" ] && wc -l < "$1" || echo 0; }

cd "$Z" || exit 1
echo "[zwz-code] start backend=$BACKEND gpus=$GPUS $(date)" >> "$S"

for B in $BENCHES; do
  WANT=$(nexp "$B")
  OUT=$Z/model_answer/$B/${MODEL}_seed42_answer.json
  HAVE=$(nlines "$OUT")
  if [ "$HAVE" -ge "$WANT" ]; then
    echo "[zwz-code] $B SKIP (have $HAVE/$WANT)" >> "$S"; continue
  fi
  T0=$(date +%s)
  echo "[zwz-code] $B start have=$HAVE/$WANT $(date)" >> "$S"

  if [ "$BACKEND" = "vllm" ]; then
    CUDA_VISIBLE_DEVICES=$GPUS VLLM_WORKER_MULTIPROC_METHOD=spawn \
      python infer_without_tool.py --benchmark "$B" --model "$MODEL" \
        --model_path "$MP" --gpus "$NGPU" > "$LOG/gen_${B}.log" 2>&1
  else
    # data-parallel shards, one per GPU, merged into the canonical file
    k=0
    for G in $(echo "$GPUS" | tr ',' ' '); do
      CUDA_VISIBLE_DEVICES=$G python infer_without_tool_hf.py --benchmark "$B" \
        --model "$MODEL" --model_path "$MP" --shard $k --num-shards $NGPU \
        > "$LOG/gen_${B}_shard${k}.log" 2>&1 &
      k=$((k+1))
    done
    wait
    cat $Z/model_answer/$B/${MODEL}_seed42_shard*_answer.json > "$OUT" 2>/dev/null
  fi

  T1=$(date +%s)
  HAVE=$(nlines "$OUT")
  if [ "$HAVE" -ge "$WANT" ]; then
    echo "[zwz-code] $B DONE $HAVE/$WANT $((T1-T0))s $(date)" >> "$S"
  else
    echo "[zwz-code] $B FAILED $HAVE/$WANT $((T1-T0))s $(date)" >> "$S"
  fi
done

echo "[zwz-code] ALL GEN DONE $(date)" >> "$S"
touch "$MT/zwz_theircode_gen.done"
