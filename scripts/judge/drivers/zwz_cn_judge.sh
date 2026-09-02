#!/bin/bash
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh; conda activate qwen35
QZ=/home/yuheng/code/Qwen2.5-VL
Z=/home/yuheng/code/ZwZ/mm-eval
V=$QZ/experiments/qwen3_5/vopd
PY=/home/yuheng/miniconda3/envs/qwen35/bin/python
RES=$QZ/logs/region_level_grpo/main_table_repro/zwzcn_judge2.res
SL=$QZ/logs/region_level_grpo/main_table_repro/zwzcn_judge2_shim.log
: > "$RES"
CUDA_VISIBLE_DEVICES=0 MODEL_ID=Qwen/Qwen3.5-9B PORT=8156 nohup $PY $V/hf_openai_shim.py > "$SL" 2>&1 &
SHIM=$!
for i in $(seq 1 120); do grep -q ready "$SL" 2>/dev/null && break; sleep 5; done
grep -q ready "$SL" || { echo "shim FAILED" >> "$RES"; kill -9 $SHIM 2>/dev/null; exit 1; }
echo "shim ready $(date +%H:%M)" >> "$RES"
cd "$Z"
for m in ZwZ-8B ZwZ-7B; do
  echo "--- $m CN ---" >> "$RES"
  $PY zwz_judge_api.py --benchmark mme-realworld-cn --model ${m}_seed42 \
     --api_base http://127.0.0.1:8156/v1/ --api_key EMPTY --judge_model Qwen3.5-9B \
     2>&1 | grep -aE "RESULT" >> "$RES"
done
kill -9 $SHIM 2>/dev/null
echo "ZWZCN_JUDGE2_DONE $(date)" >> "$RES"
