#!/bin/bash
# VOPD their-code lite generations: ZB + HR4K + HR8K x {VOPD-4B-conv, VOPD-9B-conv}
# Mirrors experiments/qwen3_5/vopd_curve/base_repro.sh exactly (the completed
# V* repro pattern): make_shards stride-2 split, one shim per (model, shard),
# their prepare/infer, merge, then THEIR judge_qwenlm + cal_acc.
#
# Judging uses THEIR judge (judge_qwenlm.py), NOT judge_unified -- provenance
# policy: their-code rows are scored end-to-end by their code.
#
# CONVENTION RECORDED: infer.py is invoked with --max_tokens 32768, but the
# shim caps generation at MAX_NEW_CAP=1024 (its default). 1024 is therefore the
# effective their-code decode budget and is KEPT, because that is exactly what
# the completed V* / base repro runs used. The shim now reports a real
# finish_reason (stop vs length), so cap-truncation is visible in the logs
# rather than silently reported as "stop".
#
# Gated: waits for the 9B-HR redo and the MME v1 scoring to release the GPUs.
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
QZ=/home/yuheng/code/Qwen2.5-VL
EV=/home/yuheng/code/Vision-OPD/eval
V=$QZ/experiments/qwen3_5/vopd
W=$QZ/experiments/qwen3_5/vopd_curve
M=$QZ/logs/region_level_grpo/main_table_repro
LOG=$M/vopd_lite; mkdir -p $LOG
PROG=$M/vopd_lite.res; touch $PROG
BENCHES="zoombench hrbench-4k hrbench-8k"
declare -A BN=( [zoombench]=845 [hrbench-4k]=800 [hrbench-8k]=800 )
declare -A BJ=( [zoombench]=zoombench.json [hrbench-4k]=hr_bench_4k.json [hrbench-8k]=hr_bench_8k.json )
nlines () { [ -f "$1" ] && wc -l < "$1" || echo 0; }

echo "[vopd-lite] gated: waiting for HR9B redo + MME v1 $(date)" >> $PROG
until [ -f $M/rc16k_hr9b_redo.done ] && [ -f $M/mme_v1.done ]; do sleep 60; done
echo "[vopd-lite] gates released $(date)" >> $PROG
# release any judge shims still holding memory (recorded ports only)
for p in $(pgrep -f '[h]f_openai_shim'); do
  E=$(tr '\0' ' ' < /proc/$p/environ 2>/dev/null)
  case "$E" in *PORT=8124*|*PORT=8125*) echo "  freeing shim pid=$p" >> $PROG; kill -9 $p ;; esac
done
sleep 15

python3 $W/make_shards.py >> $LOG/shards.log 2>&1
echo "[vopd-lite] shards ready $(date)" >> $PROG

start_shim () { # GPU MODEL PORT NAME
  CUDA_VISIBLE_DEVICES=$1 MODEL_ID=$2 PORT=$3 \
    nohup python $V/hf_openai_shim.py > $LOG/$4.log 2>&1 &
  echo $! > $LOG/$4.pid
}
start_shim 0 $QZ/output/Vision-OPD-4B-conv 8140 shim_4b_s0
start_shim 1 $QZ/output/Vision-OPD-4B-conv 8141 shim_4b_s1
start_shim 2 $QZ/output/Vision-OPD-9B-conv 8142 shim_9b_s0
start_shim 3 $QZ/output/Vision-OPD-9B-conv 8143 shim_9b_s1
for L in shim_4b_s0 shim_4b_s1 shim_9b_s0 shim_9b_s1; do
  for i in $(seq 1 180); do grep -q ready $LOG/$L.log 2>/dev/null && break; sleep 10; done
  grep -q ready $LOG/$L.log || { echo "[vopd-lite] $L FAILED" >> $PROG; exit 1; }
done
echo "[vopd-lite] all 4 shims ready (MAX_NEW_CAP=1024 kept) $(date)" >> $PROG

stream () { # TAG SHARD PORT
  local TAG=$1 SH=$2 PORT=$3
  cd $EV || return 1
  for B in $BENCHES; do
    local SJ=$W/shards/${B}_${SH}.json
    local N=$(python3 -c "import json;print(len(json.load(open('$SJ'))))")
    local OUTF=$EV/model_answer/$B/${TAG}_${SH}_seed42_answer.jsonl
    for try in 1 2 3; do
      [ "$(nlines $OUTF)" -ge "$N" ] && break
      echo "[gen] $TAG $SH $B try=$try $(date)" >> $PROG
      python3 infer.py --benchmark "$B" --benchmark_json "$SJ" \
        --out_dir model_answer --model_name "${TAG}_${SH}_seed42" --seed 42 \
        --api_base "http://127.0.0.1:${PORT}/v1/" --api_key EMPTY \
        --model_id "$TAG" --max_tokens 32768 --max_retries 5 \
        --parallel_workers 4 >> $LOG/gen_${TAG}_${SH}.log 2>&1
    done
    echo "[gen] DONE $TAG $SH $B lines=$(nlines $OUTF)/$N $(date)" >> $PROG
  done
}
stream VOPD-4B-lite s0 8140 & P1=$!
stream VOPD-4B-lite s1 8141 & P2=$!
stream VOPD-9B-lite s0 8142 & P3=$!
stream VOPD-9B-lite s1 8143 & P4=$!
wait $P1 $P2 $P3 $P4
echo "[vopd-lite] generation done $(date)" >> $PROG

for TAG in VOPD-4B-lite VOPD-9B-lite; do
  for B in $BENCHES; do
    cat $EV/model_answer/$B/${TAG}_s0_seed42_answer.jsonl \
        $EV/model_answer/$B/${TAG}_s1_seed42_answer.jsonl \
        > $EV/model_answer/$B/${TAG}_seed42_answer.jsonl 2>/dev/null
    echo "[merge] $TAG $B lines=$(nlines $EV/model_answer/$B/${TAG}_seed42_answer.jsonl)/${BN[$B]}" >> $PROG
  done
done

for L in shim_4b_s0 shim_4b_s1 shim_9b_s0 shim_9b_s1; do
  [ -f $LOG/$L.pid ] && kill -9 $(cat $LOG/$L.pid) 2>/dev/null
done
sleep 15

# ---------- phase B: THEIR judge ----------
start_shim 0 Qwen/Qwen3.5-9B 8144 shim_judge
for i in $(seq 1 180); do grep -q ready $LOG/shim_judge.log 2>/dev/null && break; sleep 10; done
cd $EV
for TAG in VOPD-4B-lite VOPD-9B-lite; do
  for B in $BENCHES; do
    python3 judge_qwenlm.py --benchmark "$B" --model "${TAG}_seed42" \
      --api_base "http://127.0.0.1:8144/v1/" --api_key EMPTY \
      --judge_model Qwen3.5-9B --judge_max_tokens 2048 >> $LOG/judge_${TAG}_${B}.log 2>&1
    BJARG=""
    case $B in hrbench-4k) BJARG="--benchmark_json $EV/hr_bench_4k.json";;
               hrbench-8k) BJARG="--benchmark_json $EV/hr_bench_8k.json";; esac
    A=$(python3 cal_acc.py --benchmark "$B" \
        --judge_json judge/$B/${TAG}_seed42_answer.jsonl $BJARG 2>&1 | grep -a "Acc:")
    NL=$(grep -a "LLM used" $LOG/judge_${TAG}_${B}.log | tail -1)
    echo "[vopd-lite-score] $TAG $B | $A | $NL" >> $PROG
  done
done
[ -f $LOG/shim_judge.pid ] && kill -9 $(cat $LOG/shim_judge.pid) 2>/dev/null
echo "[vopd-lite] ALL DONE $(date)" >> $PROG
touch $M/vopd_lite.done
