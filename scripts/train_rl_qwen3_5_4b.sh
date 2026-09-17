#!/usr/bin/env bash
# Region-level RL on Qwen3.5-4B (paper checkpoint: q35vl-4b-v4pa-placebo125-s42-full).
# Inputs: the SD-RPN checkpoint from scripts/train_sdrpn_online.sh and the 7k RL pool
# (+ its evidence-map cache); see README.
export PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/qwen3_5-4b-sdrpn-K21T3-online}
export FILTERED_JSONL=${FILTERED_JSONL:-data/VisionRL2-data/rl_pools/rl_pool_qwen3_5_4b.jsonl}
export EV_MAPS_ROOT=${EV_MAPS_ROOT:-data/ev_maps}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
export GPU_IDS=${GPU_IDS:-0,1,2,3}
export RUN_NAME=${RUN_NAME:-qwen3_5-4b-rl}

export TWIG_K=21
export MIN_PIXELS=262144 MAX_PIXELS=589824      # 256 / 576 visual tokens (32 px patches)
export PLACEBO_KAPPA=1.25
export BATCH_SIZE=8 GRAD_ACCUM_STEPS=1          # effective batch 32 on 4 GPUs
export ATTN_IMPL=sdpa

bash "$(dirname "$0")/_rl_launch.sh"
