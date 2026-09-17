#!/usr/bin/env bash
# Region-level RL on Qwen3.5-9B (paper checkpoint: q9b-placebo100-s42-full).
export PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/qwen3_5-9b-sdrpn-K21T3-online}
export FILTERED_JSONL=${FILTERED_JSONL:-data/VisionRL2-data/rl_pools/rl_pool_qwen3_5_9b.jsonl}
export EV_MAPS_ROOT=${EV_MAPS_ROOT:-data/ev_maps}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
export GPU_IDS=${GPU_IDS:-0,1,2,3}
export RUN_NAME=${RUN_NAME:-qwen3_5-9b-rl}

export TWIG_K=21
export MIN_PIXELS=262144 MAX_PIXELS=589824      # 256 / 576 visual tokens (32 px patches)
export PLACEBO_KAPPA=1.0
export BATCH_SIZE=8 GRAD_ACCUM_STEPS=1          # effective batch 32 on 4 GPUs
export ATTN_IMPL=sdpa

bash "$(dirname "$0")/_rl_launch.sh"
