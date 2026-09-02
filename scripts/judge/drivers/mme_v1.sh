#!/bin/bash
# 1. Cancel the judge-model sensitivity study; free the Qwen3.5-4B shim.
# 2. MME unified-v1 scoring of the six staged sets.
#
# JUDGING ONLY. No generation is re-run: the inputs are the already-staged
# NCI jsonls (nci_mme_uni/*) and the local q25 dirs. Only the scoring pass
# restarts, because the earlier one began while judge_unified.py was being
# upgraded v1->v2 and would have straddled two rule versions.
# judge_unified.py is now restored to v1 (frozen) and writes to judge_unified/.
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh; conda activate qwen35
QZ=/home/yuheng/code/Qwen2.5-VL
EV=/home/yuheng/code/Vision-OPD/eval
L=$QZ/logs/iter_pol_kl
M=$QZ/logs/region_level_grpo/main_table_repro
V=$QZ/experiments/qwen3_5/vopd
C=$QZ/experiments/qwen3_5/main_table_repro/mme_to_vopd.py
PY=/home/yuheng/miniconda3/envs/qwen35/bin/python
cd $QZ

echo "=== 1. cancel judge-model study ==="
pkill -9 -f '[r]ejudge_altmodel.py'; sleep 2
# free the 4B judge shim (port 8130) by PID from its own log-bearing cmdline
for p in $(pgrep -f '[h]f_openai_shim'); do
  if tr '\0' ' ' < /proc/$p/environ 2>/dev/null | grep -q 'PORT=8130'; then
    echo "  killing 4B shim pid=$p"; kill -9 $p
  fi
done
sleep 6
pgrep -af '[r]ejudge_altmodel' || echo "  study cancelled"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== 2. verify active judge is v1 ==="
grep -q 'JUDGE_UNIFIED_VERSION = 2' $EV/judge_unified.py \
  && { echo "  FATAL: active judge is v2"; exit 1; } || echo "  active judge = v1 OK"

echo "=== 3. shims for MME (9B judge) ==="
start_shim () {  # GPU PORT
  local SL=$M/mme_shim_$2.log
  CUDA_VISIBLE_DEVICES=$1 MODEL_ID=Qwen/Qwen3.5-9B PORT=$2 \
    nohup $PY $V/hf_openai_shim.py > $SL 2>&1 &
  for i in $(seq 1 120); do grep -q ready $SL 2>/dev/null && break; sleep 5; done
  grep -q ready $SL && echo "  shim $2 ready (gpu $1)" || echo "  shim $2 FAILED"
}
start_shim 0 8124
start_shim 1 8125

sh8 () { for i in 0 1 2 3 4 5 6 7; do echo -n " --samples $L/nci_mme_uni/$1_s$i"; done; }

run () {  # PORT TAG BENCH EXPECT SAMPLEARGS...
  local PORT=$1 TAG=$2 B=$3 EXP=$4; shift 4
  local R=$M/mme_v1.res
  local CV=$($PY $C --bench "$B" --tag "UNI-$TAG" --expect "$EXP" "$@" 2>&1 | tail -2)
  echo "[conv] $TAG $CV" >> $R
  case "$CV" in *FAIL*|*Traceback*) return ;; esac
  local O=$(cd $EV && $PY judge_unified.py --benchmark "$B" --model "UNI-$TAG" \
      --api_base "http://127.0.0.1:${PORT}/v1/" --api_key EMPTY \
      --judge_model Qwen3.5-9B 2>&1 | grep -a "Total:")
  local A=$(cd $EV && $PY cal_acc.py --benchmark "$B" \
      --judge_json judge_unified/$B/UNI-${TAG}_answer.jsonl 2>&1 | grep -a "Acc:")
  echo "[mme-v1] $TAG | $O | $A" >> $R
}

: > $M/mme_v1.res
(
  run 8124 mme_en_ours4b mme-realworld    23609 $(sh8 en4b16k)
  run 8124 mme_cn_ours4b mme-realworld-cn  5917 $(sh8 cn4b16k)
  run 8124 mme_cn_q25    mme-realworld-cn  5917 --samples $L/q25cn16k
  touch $M/mme_v1_a.done
) &
(
  run 8125 mme_en_ours9b mme-realworld    23609 $(sh8 en9b16k)
  run 8125 mme_cn_ours9b mme-realworld-cn  5917 $(sh8 cn9bstd)
  run 8125 mme_en_q25    mme-realworld    23609 --samples $L/q25en16kv2_s0 --samples $L/q25en16kv2_s1
  touch $M/mme_v1_b.done
) &
wait
echo "[mme-v1] ALL DONE $(date)" >> $M/mme_v1.res
touch $M/mme_v1.done
