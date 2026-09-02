#!/bin/bash
# ZwZ competitor-row judging: ZwZ's OWN judge logic (zwz_judge_api.py) with the
# LLM step served by a Qwen3.5-9B shim -- same judge model + serving VOPD used.
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh; conda activate qwen35
QZ=/home/yuheng/code/Qwen2.5-VL
Z=/home/yuheng/code/ZwZ/mm-eval
V=$QZ/experiments/qwen3_5/vopd
PY=/home/yuheng/miniconda3/envs/qwen35/bin/python
GPU=${1:-2}; PORT=${2:-8150}; MTAG=${3:-ZwZ-8B}
RES=$QZ/logs/region_level_grpo/main_table_repro/zwz_judge_${MTAG}.res
LOGD=$QZ/logs/region_level_grpo/main_table_repro
SL=$LOGD/zwz_judge_shim_$PORT.log
: > "$RES"

echo "[zwzjudge] start shim gpu=$GPU port=$PORT $(date)" | tee -a "$RES"
CUDA_VISIBLE_DEVICES=$GPU MODEL_ID=Qwen/Qwen3.5-9B PORT=$PORT nohup $PY $V/hf_openai_shim.py > "$SL" 2>&1 &
SHIM=$!
for i in $(seq 1 120); do grep -q ready "$SL" 2>/dev/null && break; sleep 5; done
grep -q ready "$SL" || { echo "[zwzjudge] shim FAILED"; tail -5 "$SL" | tee -a "$RES"; kill -9 $SHIM 2>/dev/null; exit 1; }
echo "[zwzjudge] shim ready" | tee -a "$RES"

cd "$Z"
for B in vstar zoom-bench hrbench-4k hrbench-8k; do
  echo "[zwzjudge] judging $B $(date +%H:%M:%S)" | tee -a "$RES"
  $PY zwz_judge_api.py --benchmark "$B" --model ${MTAG}_seed42 \
      --api_base "http://127.0.0.1:${PORT}/v1/" --api_key EMPTY --judge_model Qwen3.5-9B \
      2>&1 | grep -aE "RESULT|Error|Traceback" | tee -a "$RES"
done
kill -9 $SHIM 2>/dev/null
echo "[zwzjudge] ALL DONE $(date)" | tee -a "$RES"
