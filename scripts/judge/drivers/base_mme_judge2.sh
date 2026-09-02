#!/bin/bash
# base_mme_judge2.sh LEG GPU_A PORT_A GPU_B PORT_B
# Two-shim accelerated judging of ONE base MME leg with frozen judge_unified v1.
# Flow: convert-once -> split answer file in half -> judge each half on its own
# 9B shim (GPU_A/GPU_B) in parallel -> merge judged JSON ARRAYS -> cal_acc.
# NOTE: judge_unified writes json.dump(list, indent=4) (a pretty JSON ARRAY),
# NOT jsonl -- so halves MUST be merged as arrays (json load+concat+dump),
# never cat'd.
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh; conda activate qwen35
QZ=/home/yuheng/code/Qwen2.5-VL
EV=/home/yuheng/code/Vision-OPD/eval
C=$QZ/experiments/qwen3_5/main_table_repro/mme_to_vopd.py
SROOT=$QZ/experiments/qwen3_5/main_table_repro/base_mme/nci_jsonls
RES=$QZ/experiments/qwen3_5/main_table_repro/base_mme/base_mme.res
LOGD=$QZ/logs/region_level_grpo/main_table_repro
V=$QZ/experiments/qwen3_5/vopd
PY=/home/yuheng/miniconda3/envs/qwen35/bin/python
LEG=${1:?leg}; GA=${2:?gpuA}; PA=${3:?portA}; GB=${4:?gpuB}; PB=${5:?portB}

case $LEG in *_en) B=mme-realworld; EXP=23609;; *_cn) B=mme-realworld-cn; EXP=5917;; esac
case $LEG in base4b_en) TAG=mme_en_base4b; SRC=$SROOT/mtstd_base4b_en;;
             base4b_cn) TAG=mme_cn_base4b; SRC=$SROOT/mtstd_base4b_cn;;
             base9b_en) TAG=mme_en_base9b; SRC=$SROOT/mtstd_base9b_en;;
             base9b_cn) TAG=mme_cn_base9b; SRC=$SROOT/mtstd_base9b_cn;; esac

grep -q 'JUDGE_UNIFIED_VERSION = 2' $EV/judge_unified.py \
  && { echo "FATAL: judge is v2"; exit 1; } || echo "judge = v1 OK"

NL=$(find "$SRC" -name '*samples*.jsonl' 2>/dev/null | xargs cat 2>/dev/null | grep -c .)
[ "${NL:-0}" -eq "$EXP" ] || { echo "[base2] SKIP $LEG (lines=${NL:-0} expect=$EXP)" | tee -a "$RES"; exit 0; }

cd $EV
echo "[base2] CONVERT $LEG -> UNI-$TAG ($(date +%H:%M:%S))" | tee -a "$RES"
CV=$($PY $C --bench "$B" --tag "UNI-$TAG" --expect "$EXP" --samples "$SRC" 2>&1 | tail -2)
echo "[conv] $LEG $CV" | tee -a "$RES"
case "$CV" in *FAIL*|*Traceback*) echo "[base2] $LEG CONVERT-FAILED" | tee -a "$RES"; exit 1 ;; esac
AF=model_answer/$B/UNI-${TAG}_answer.jsonl
TOT=$(grep -c . "$AF"); HALF=$(( (TOT + 1) / 2 ))
head -n "$HALF"          "$AF" > model_answer/$B/UNI-${TAG}_h0_answer.jsonl
tail -n +"$((HALF + 1))" "$AF" > model_answer/$B/UNI-${TAG}_h1_answer.jsonl
echo "[base2] split $TOT -> h0=$(grep -c . model_answer/$B/UNI-${TAG}_h0_answer.jsonl) h1=$(grep -c . model_answer/$B/UNI-${TAG}_h1_answer.jsonl) (line-count; converter emits jsonl)" | tee -a "$RES"

start_shim () {
  local SL=$LOGD/base_mme_shim_$2.log
  CUDA_VISIBLE_DEVICES=$1 MODEL_ID=Qwen/Qwen3.5-9B PORT=$2 \
    nohup $PY $V/hf_openai_shim.py > $SL 2>&1 &
  echo $!
}
PIDA=$(start_shim $GA $PA); PIDB=$(start_shim $GB $PB)
for P in $PA $PB; do
  for i in $(seq 1 120); do grep -q ready $LOGD/base_mme_shim_$P.log 2>/dev/null && break; sleep 5; done
  grep -q ready $LOGD/base_mme_shim_$P.log || { echo "shim $P FAILED" | tee -a "$RES"; kill -9 $PIDA $PIDB 2>/dev/null; exit 1; }
done
echo "[base2] shims ready A=$PA(gpu$GA) B=$PB(gpu$GB)" | tee -a "$RES"

judge_half () {
  cd $EV && $PY judge_unified.py --benchmark "$B" --model "UNI-${TAG}_$1" \
      --api_base "http://127.0.0.1:$2/v1/" --api_key EMPTY --judge_model Qwen3.5-9B \
      > $LOGD/base_judge_${TAG}_$1.log 2>&1
}
judge_half h0 $PA & JH0=$!
judge_half h1 $PB & JH1=$!
wait $JH0; wait $JH1
kill -9 $PIDA $PIDB 2>/dev/null

# merge JSON ARRAYS (json_unified writes indent=4 arrays, NOT jsonl -> never cat)
MG=$($PY -c "
import json
a=json.load(open('judge_unified/$B/UNI-${TAG}_h0_answer.jsonl'))
b=json.load(open('judge_unified/$B/UNI-${TAG}_h1_answer.jsonl'))
m=a+b
json.dump(m,open('judge_unified/$B/UNI-${TAG}_answer.jsonl','w'),ensure_ascii=False,indent=4)
print(len(m))
")
O0=$(grep -a "Total:" $LOGD/base_judge_${TAG}_h0.log | tail -1)
O1=$(grep -a "Total:" $LOGD/base_judge_${TAG}_h1.log | tail -1)
A=$(cd $EV && $PY cal_acc.py --benchmark "$B" --judge_json judge_unified/$B/UNI-${TAG}_answer.jsonl 2>&1 | grep -a "Acc:")
echo "[base2] $LEG merged=$MG/$EXP | h0: $O0 | h1: $O1 | $A" | tee -a "$RES"
echo "[base2] $LEG DONE $(date)" | tee -a "$RES"
