#!/usr/bin/env bash
# Region-level RL on Qwen3.5-4B (stage 2; control-margin scale kappa = 1.25).
# Inputs: the SD-RPN checkpoint from scripts/train_sdrpn_online.sh and the 7k RL pool
# (+ its evidence-map cache); see README.
export PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/qwen3_5-4b-sdrpn-K21T3}
export FILTERED_JSONL=${FILTERED_JSONL:-data/VisionRL2-data/rl_pools/rl_pool_qwen3_5_4b.jsonl}
export EV_MAPS_ROOT=${EV_MAPS_ROOT:-data/ev_maps}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
export GPU_IDS=${GPU_IDS:-0,1,2,3}
export RUN_NAME=${RUN_NAME:-qwen3_5-4b-rl}

export TWIG_K=21
export MIN_PIXELS=262144 MAX_PIXELS=589824      # 256 / 576 visual tokens (32 px patches)
export PLACEBO_KAPPA=1.25
export BATCH_SIZE=${BATCH_SIZE:-8} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}   # effective batch 32 on 4 GPUs (bs x ga x #GPUs; keep 32 on fewer GPUs, e.g. GRAD_ACCUM_STEPS=4 on one)
export ATTN_IMPL=sdpa

bash "$(dirname "$0")/_rl_launch.sh"
