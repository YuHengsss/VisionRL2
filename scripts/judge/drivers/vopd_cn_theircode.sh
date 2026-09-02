#!/bin/bash
# VOPD MME-CN reproduction on CityU (their code): VOPD-4B-conv + VOPD-9B-conv on
# mme-realworld-cn, multi-shim (2 shims/GPU x 4 = 8 streams), MAX_NEW_CAP=1024,
# then THEIR judge (judge_qwenlm.py, Qwen3.5-9B shim) + cal_acc.py.
set -uo pipefail
source /home/yuheng/miniconda3/etc/profile.d/conda.sh; conda activate qwen35
QZ=/home/yuheng/code/Qwen2.5-VL
EV=/home/yuheng/code/Vision-OPD/eval
V=$QZ/experiments/qwen3_5/vopd
PY=/home/yuheng/miniconda3/envs/qwen35/bin/python
LOGD=$QZ/logs/region_level_grpo/main_table_repro
RES=$LOGD/vopd_cn.res
SD=$LOGD/vopd_cn_shards; mkdir -p "$SD"
CNJSON=$EV/MME_RealWorld_CN.json
NSHARD=8
export MAX_NEW_CAP=1024
: > "$RES"
say(){ echo "[vopdcn] $*" | tee -a "$RES"; }

# ---- shard the CN json into NSHARD parts (stride) ----
$PY - "$CNJSON" "$SD" "$NSHARD" <<'PYS'
import json,sys
d=json.load(open(sys.argv[1])); sd=sys.argv[2]; n=int(sys.argv[3])
for k in range(n):
    json.dump(d[k::n], open(f"{sd}/cn_shard{k}.json","w"), ensure_ascii=False)
print("sharded",len(d),"into",n)
PYS
say "sharded CN into $NSHARD"

start_shim(){ local GPU=$1 PORT=$2 CK=$3 NAME=$4
  CUDA_VISIBLE_DEVICES=$GPU MODEL_ID=$CK PORT=$PORT MAX_NEW_CAP=$MAX_NEW_CAP \
    nohup $PY $V/hf_openai_shim.py > $LOGD/$NAME.log 2>&1 &
  echo $! ; }

run_model(){ # TAG CK
  local TAG=$1 CK=$2
  say "==== MODEL $TAG ($CK) $(date +%H:%M) ===="
  local PIDS=() PORTS=()
  for k in $(seq 0 $((NSHARD-1))); do
    local gpu=$((k % 4)) port=$((8300+k))
    PIDS+=($(start_shim $gpu $port "$CK" "vopdcn_shim_${TAG}_$k")); PORTS+=($port)
  done
  # wait all shims ready
  for k in $(seq 0 $((NSHARD-1))); do
    local lg=$LOGD/vopdcn_shim_${TAG}_$k.log
    for i in $(seq 1 180); do grep -q ready "$lg" 2>/dev/null && break; sleep 5; done
    grep -q ready "$lg" || { say "shim $k FAILED"; return 1; }
  done
  say "  $NSHARD shims ready"
  # infer per shard
  local JPIDS=()
  cd $EV
  for k in $(seq 0 $((NSHARD-1))); do
    ( $PY infer.py --benchmark mme-realworld-cn --benchmark_json "$SD/cn_shard${k}.json" \
        --out_dir model_answer --model_name "${TAG}-cn_shard${k}_seed42" --seed 42 \
        --api_base "http://127.0.0.1:${PORTS[$k]}/v1/" --api_key EMPTY \
        --model_id "$TAG" --max_tokens 32768 --max_retries 5 --parallel_workers 4 \
        > $LOGD/vopdcn_gen_${TAG}_$k.log 2>&1 ) &
    JPIDS+=($!)
  done
  wait "${JPIDS[@]}"
  # merge
  local M=$EV/model_answer/mme-realworld-cn/${TAG}-cn_seed42_answer.jsonl
  cat $EV/model_answer/mme-realworld-cn/${TAG}-cn_shard*_seed42_answer.jsonl > "$M" 2>/dev/null
  say "  $TAG merged: $(grep -c . "$M")/5917"
  for p in $(pgrep -f "[h]f_openai_shim"); do tr '\0' ' ' </proc/$p/environ 2>/dev/null | grep -q "MODEL_ID=$CK" && kill -9 $p; done
  sleep 5
}

run_model VOPD-4B-repro $QZ/output/Vision-OPD-4B-conv
run_model VOPD-9B-repro $QZ/output/Vision-OPD-9B-conv

# ---- judge (their judge_qwenlm.py + Qwen3.5-9B shim) ----
say "==== JUDGE $(date +%H:%M) ===="
JSHIM=$(start_shim 0 8320 Qwen/Qwen3.5-9B vopdcn_judge_shim)
for i in $(seq 1 180); do grep -q ready $LOGD/vopdcn_judge_shim.log 2>/dev/null && break; sleep 5; done
cd $EV
for TAG in VOPD-4B-repro VOPD-9B-repro; do
  $PY judge_qwenlm.py --benchmark mme-realworld-cn --model ${TAG}-cn_seed42 \
     --api_base http://127.0.0.1:8320/v1/ --api_key EMPTY --judge_model Qwen3.5-9B \
     > $LOGD/vopdcn_judge_${TAG}.log 2>&1
  A=$($PY cal_acc.py --benchmark mme-realworld-cn --judge_json judge/mme-realworld-cn/${TAG}-cn_seed42_answer.jsonl 2>&1 | grep -a "Acc:")
  say "[vopd-cn-score] $TAG | $A"
done
kill -9 $JSHIM 2>/dev/null
say "ALL DONE $(date)"
