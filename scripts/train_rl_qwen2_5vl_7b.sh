#!/usr/bin/env bash
# Region-level RL on Qwen2.5-VL-7B (paper checkpoint: q25vl-7b-fin-placebo100-s42).
# Uses the released SD-RPN checkpoint qwen2_5vl-7b-roi-K18T3-stage1 as init and the
# transformers-4.51 environment (requirements_qwen2_5vl.txt).
export PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/qwen2_5vl-7b-roi-K18T3-stage1}
export FILTERED_JSONL=${FILTERED_JSONL:-data/rl_pools/filtered_v2_evmaps_q25_7b.jsonl}
export DATASET_ROOT=${DATASET_ROOT:-datasets}
export GPU_IDS=${GPU_IDS:-0,1}
export RUN_NAME=${RUN_NAME:-qwen2_5vl-7b-rl}

export TWIG_K=18
export MIN_PIXELS=200704 MAX_PIXELS=451584      # 256 / 576 visual tokens (28 px patches)
export PLACEBO_KAPPA=1.0
export BATCH_SIZE=1 GRAD_ACCUM_STEPS=16         # effective batch 32 on 2 GPUs
export ATTN_IMPL=flash_attention_2

bash "$(dirname "$0")/_rl_launch.sh"
