#!/usr/bin/env bash
# =============================================================================
# Region-level GR-REINFORCE (the RL stage) on the SD-RPN twig of a frozen
# Qwen VLM.
#
# Starting from a Phase-A (SD-RPN online-pseudo-label) checkpoint, only the
# T=3 twig blocks are trained. Per sample the policy heatmap is split into
# connected regions; each singleton region-drop is rewarded by the frozen LM's
# teacher-forced log p(gold | masked crop) (clipped-logit height), the
# advantage is baselined against the keep-all action minus a per-sample
# placebo noise bar, and an additive "supplement" group pushes the heatmap up
# on regions the cached answer->image evidence maps attend to but the policy
# misses. A KL anchor to the frozen Phase-A twig (recomputed online at train
# resolution) regularises every sample.
#
# Recipe (all three models): lr 1.5e-5 cosine, warm-up 0.2, 1 epoch,
# effective batch 32, bf16 + DeepSpeed ZeRO-2, lambda_kl 0.5,
# singleton drops + std-regularised advantage (eps 1.0), logit-clip reward
# (delta 5), winnability weighting, placebo bar (kappa 1.25 for Qwen3.5-4B,
# 1.0 for Qwen3.5-9B / Qwen2.5-VL-7B), supplement group + attention-fg
# gradient mask from the evidence-map cache.
#
# Usage:
#   MODEL=qwen3_5-4b bash scripts/train_rl.sh
#   MODEL=qwen3_5-9b GPU_IDS=0,1,2,3 bash scripts/train_rl.sh
#   MODEL=qwen2_5vl-7b GPU_IDS=0,1 bash scripts/train_rl.sh
#
# Required inputs (see docs/DATA.md):
#   PHASE_A_CKPT    SD-RPN (Phase-A) checkpoint trained with scripts/train_sdrpn_online.sh
#   FILTERED_JSONL  RL pool jsonl with fields {dataset, image, question, gold_answer,
#                   ev_maps_path} (ev_maps_path -> the evidence-map cache .pt per row)
#   DATASET_ROOT    root of the per-dataset image folders (dataset.py:DS_IMAGE_SUBDIRS)
#   EV_MAPS_ROOT    (optional) root used to resolve relative ev_maps_path entries; by
#                   default they are tried as given, then relative to the pool jsonl's dir
#
# Conda env: Qwen3.5 models need the transformers-5.x env; Qwen2.5-VL-7B needs
# the transformers-4.51 env (see README).
# =============================================================================
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/qwen-vl-finetune:${PYTHONPATH:-}"
entry_file=qwen-vl-finetune/qwenvl/train/region_level_grpo/train_phase_b1.py

# ---- model presets ------------------------------------------------------------
MODEL=${MODEL:-qwen3_5-4b}
TWIG_T=${TWIG_T:-3}
case "${MODEL}" in
  qwen3_5-4b)
    TWIG_K=${TWIG_K:-21}
    PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/qwen3_5-4b-roi-K${TWIG_K}T${TWIG_T}}
    MIN_PIXELS=${MIN_PIXELS:-262144};  MAX_PIXELS=${MAX_PIXELS:-589824}   # 256 / 576 tokens (patch 32)
    PLACEBO_KAPPA=${PLACEBO_KAPPA:-1.25}
    BATCH_SIZE=${BATCH_SIZE:-8};  GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
    GPU_IDS=${GPU_IDS:-0,1,2,3}
    ATTN_IMPL=${ATTN_IMPL:-sdpa}
    ;;
  qwen3_5-9b)
    TWIG_K=${TWIG_K:-21}
    PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/qwen3_5-9b-roi-K${TWIG_K}T${TWIG_T}}
    MIN_PIXELS=${MIN_PIXELS:-262144};  MAX_PIXELS=${MAX_PIXELS:-589824}   # 256 / 576 tokens (patch 32)
    PLACEBO_KAPPA=${PLACEBO_KAPPA:-1.0}
    BATCH_SIZE=${BATCH_SIZE:-8};  GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
    GPU_IDS=${GPU_IDS:-0,1,2,3}
    ATTN_IMPL=${ATTN_IMPL:-sdpa}
    ;;
  qwen2_5vl-7b)
    TWIG_K=${TWIG_K:-18}
    PHASE_A_CKPT=${PHASE_A_CKPT:-output/sdrpn/qwen2_5vl-7b-roi-K${TWIG_K}T${TWIG_T}}
    MIN_PIXELS=${MIN_PIXELS:-200704};  MAX_PIXELS=${MAX_PIXELS:-451584}   # 256 / 576 tokens (patch 28)
    PLACEBO_KAPPA=${PLACEBO_KAPPA:-1.0}
    BATCH_SIZE=${BATCH_SIZE:-1};  GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
    GPU_IDS=${GPU_IDS:-0,1}
    ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
    ;;
  *) echo "MODEL must be qwen3_5-4b | qwen3_5-9b | qwen2_5vl-7b (got '${MODEL}')"; exit 1 ;;
esac

# ---- corpus -------------------------------------------------------------------
# one process per listed GPU (override with NPROC_PER_NODE)
NPROC_PER_NODE=${NPROC_PER_NODE:-$(echo "${GPU_IDS}" | awk -F',' '{print NF}')}
FILTERED_JSONL=${FILTERED_JSONL:-data/rl_pools/${MODEL}.jsonl}
export DATASET_ROOT=${DATASET_ROOT:-datasets}   # per-dataset image folders live under here
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}       # 0 = whole pool

# ---- run name + output --------------------------------------------------------
RUN_NAME=${RUN_NAME:-${MODEL}-rl-K${TWIG_K}T${TWIG_T}}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/rl/${RUN_NAME}}
mkdir -p "${OUTPUT_DIR}"

# ---- distributed --------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2_safe.json}
deepspeed=${CODE_ROOT}/qwen-vl-finetune/scripts/${DEEPSPEED_CFG}
if [[ "${USE_DEEPSPEED}" == "1" ]]; then
    DEEPSPEED_ARG="--deepspeed ${deepspeed}"
else
    # find_unused_parameters MUST be False: every twig_layer param flows
    # through both the policy and KL paths within a single backward, and
    # DDP's find-unused barrier double-fires the final param's reducer
    # hook in that scenario.
    DEEPSPEED_ARG="--ddp_find_unused_parameters False --ddp_broadcast_buffers False"
fi

# ---- env ----------------------------------------------------------------------
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=${TORCH_NCCL_WATCHDOG_TIMEOUT_SEC:-3600}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}
export ATTN_IMPL
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# CPU threads per rank for building the masked reward crops.
export REWARD_MASK_PIL_WORKERS=${REWARD_MASK_PIL_WORKERS:-4}

# ---- optimisation -------------------------------------------------------------
LR=${LR:-1.5e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.2}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1.0}
GRAD_CKPT=${GRAD_CKPT:-True}
SEED=${SEED:-42}
SAVE_STRATEGY=${SAVE_STRATEGY:-no}      # final weights are always written unless SKIP_FINAL_SAVE=1
SAVE_STEPS=${SAVE_STEPS:-500}
MAX_STEPS_ARG=""
if [[ -n "${MAX_STEPS:-}" ]]; then
    MAX_STEPS_ARG="--max_steps ${MAX_STEPS}"
fi

# ---- heatmap -> regions -------------------------------------------------------
THRESHOLD_MODE=${THRESHOLD_MODE:-peak_ratio}
FIXED_THRESHOLD=${FIXED_THRESHOLD:-0.02}   # also binarises the reference for the K=1 anchor
PEAK_FRACTION=${PEAK_FRACTION:-0.3}        # matches eval (dynamic_peak_fraction=0.3)
RATIO_THRESH=${RATIO_THRESH:-3.0}
MIN_GATE=${MIN_GATE:-0.03}
SMOOTH_KERNEL=${SMOOTH_KERNEL:-3}
SMOOTH_SIGMA=${SMOOTH_SIGMA:-1.0}
SCORE_P=${SCORE_P:-1.0}
R_MAX=${R_MAX:-6}
ENUMERATE_THRESHOLD=${ENUMERATE_THRESHOLD:-12}
ROLLOUT_K=${ROLLOUT_K:-4}
SINGLETON_ONLY_ACTIONS=${SINGLETON_ONLY_ACTIONS:-1}

# ---- reward / advantage -------------------------------------------------------
REWARD_LOGIT_CLIP=${REWARD_LOGIT_CLIP:-1}
REWARD_LOGIT_CLIP_DELTA=${REWARD_LOGIT_CLIP_DELTA:-5.0}
REWARD_SIZE_BETA=${REWARD_SIZE_BETA:-0.0}
REWARD_LINEAR_NCC_ALPHA=${REWARD_LINEAR_NCC_ALPHA:-0.0}
ADVANTAGE_STD_EPS=${ADVANTAGE_STD_EPS:-1.0}
LAMBDA_KL=${LAMBDA_KL:-0.5}
LAMBDA_ANCHOR_K1=${LAMBDA_ANCHOR_K1:-1.0}
# placebo-bar subtractor: reference = R(empty) - kappa * measured reward noise
SUBTRACTOR_MODE=${SUBTRACTOR_MODE:-placebo}
PLACEBO_P_THRESH=${PLACEBO_P_THRESH:-0.02}
PLACEBO_BAR_MAX=${PLACEBO_BAR_MAX:-1.0}
# winnability weight: PG loss x (max_a p(a)) / EMA
WINNABILITY_WEIGHT=${WINNABILITY_WEIGHT:-max_p}
WINNABILITY_EMA_DECAY=${WINNABILITY_EMA_DECAY:-0.99}
WINNABILITY_FLOOR=${WINNABILITY_FLOOR:-0.05}
WINNABILITY_MAX=${WINNABILITY_MAX:-3.0}

# ---- supplement group (evidence maps) + attention-fg gradient mask -------------
SOURCE_MAP_GROUP=${SOURCE_MAP_GROUP:-1}
SOURCE_MAP_POLICY_SIGMA=${SOURCE_MAP_POLICY_SIGMA:-1.0}
SOURCE_MAP_SUPP_K_MAX=${SOURCE_MAP_SUPP_K_MAX:-4}
SOURCE_MAP_SUPP_MIN_CELLS=${SOURCE_MAP_SUPP_MIN_CELLS:-1}
SOURCE_MAP_SUPP_MERGE_IOU=${SOURCE_MAP_SUPP_MERGE_IOU:-0.5}
SOURCE_MAP_SUPP_LOGIT_CLIP=${SOURCE_MAP_SUPP_LOGIT_CLIP:-1}
SOURCE_MAP_SUPP_UNIFORM_WEIGHT=${SOURCE_MAP_SUPP_UNIFORM_WEIGHT:-1}
ATTENTION_FG_GRADIENT_MASK=${ATTENTION_FG_GRADIENT_MASK:-1}
ATTENTION_FG_VOTE_THRESHOLD=${ATTENTION_FG_VOTE_THRESHOLD:-3}
ATTENTION_FG_FULLY_BELOW_FULL_GRAD=${ATTENTION_FG_FULLY_BELOW_FULL_GRAD:-1}
ATTENTION_FG_VOTE_DILATION_K=${ATTENTION_FG_VOTE_DILATION_K:-3}

# ---- scoring convention / reference / reward prompt ---------------------------
ONLINE_P_REF=${ONLINE_P_REF:-1}
ROI_SCORE_WITH_ROPE=${ROI_SCORE_WITH_ROPE:-1}
ROI_SCORE_QUERY_MODE=${ROI_SCORE_QUERY_MODE:-last_prompt}
REF_SCORE_WITH_ROPE=${REF_SCORE_WITH_ROPE:-1}
REF_SCORE_QUERY_MODE=${REF_SCORE_QUERY_MODE:-last_prompt}
REWARD_DISABLE_THINKING_PREFIX=${REWARD_DISABLE_THINKING_PREFIX:-1}
REWARD_SKIP_TRAILING_EOS=${REWARD_SKIP_TRAILING_EOS:-1}

args="
    ${DEEPSPEED_ARG} \
    --model_name_or_path ${PHASE_A_CKPT} \
    --filtered_jsonl ${FILTERED_JSONL} \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    ${MAX_STEPS_ARG} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --eval_strategy no \
    --save_strategy ${SAVE_STRATEGY} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0 \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm 1 \
    --lr_scheduler_type ${LR_SCHEDULER} \
    --logging_steps 1 \
    --model_max_length ${MODEL_MAX_LENGTH:-2048} \
    --gradient_checkpointing ${GRAD_CKPT} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-2} \
    --run_name ${RUN_NAME} \
    --report_to ${REPORT_TO:-tensorboard} \
    --remove_unused_columns False \
    --seed ${SEED} \
    --enable_twig True \
    --twig_K ${TWIG_K} \
    --twig_T ${TWIG_T} \
    --roi_loss bce \
    --roi_multi_head True \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --threshold_mode ${THRESHOLD_MODE} \
    --fixed_threshold ${FIXED_THRESHOLD} \
    --peak_fraction ${PEAK_FRACTION} \
    --ratio_thresh ${RATIO_THRESH} \
    --min_gate ${MIN_GATE} \
    --smooth_kernel ${SMOOTH_KERNEL} \
    --smooth_sigma ${SMOOTH_SIGMA} \
    --score_p ${SCORE_P} \
    --R_max ${R_MAX} \
    --enumerate_threshold ${ENUMERATE_THRESHOLD} \
    --rollout_K ${ROLLOUT_K} \
    --singleton_only_actions ${SINGLETON_ONLY_ACTIONS} \
    --reward_logit_clip ${REWARD_LOGIT_CLIP} \
    --reward_logit_clip_delta ${REWARD_LOGIT_CLIP_DELTA} \
    --reward_size_beta ${REWARD_SIZE_BETA} \
    --reward_linear_ncc_alpha ${REWARD_LINEAR_NCC_ALPHA} \
    --advantage_std_eps ${ADVANTAGE_STD_EPS} \
    --lambda_kl ${LAMBDA_KL} \
    --lambda_anchor_k1 ${LAMBDA_ANCHOR_K1} \
    --subtractor_mode ${SUBTRACTOR_MODE} \
    --placebo_kappa ${PLACEBO_KAPPA} \
    --placebo_p_thresh ${PLACEBO_P_THRESH} \
    --placebo_bar_max ${PLACEBO_BAR_MAX} \
    --winnability_weight ${WINNABILITY_WEIGHT} \
    --winnability_ema_decay ${WINNABILITY_EMA_DECAY} \
    --winnability_floor ${WINNABILITY_FLOOR} \
    --winnability_max ${WINNABILITY_MAX} \
    --source_map_group ${SOURCE_MAP_GROUP} \
    --source_map_policy_sigma ${SOURCE_MAP_POLICY_SIGMA} \
    --source_map_supp_additive True \
    --source_map_supp_multilayer True \
    --source_map_supp_k_max ${SOURCE_MAP_SUPP_K_MAX} \
    --source_map_supp_min_cells ${SOURCE_MAP_SUPP_MIN_CELLS} \
    --source_map_supp_merge_iou ${SOURCE_MAP_SUPP_MERGE_IOU} \
    --source_map_supp_logit_clip ${SOURCE_MAP_SUPP_LOGIT_CLIP} \
    --source_map_supp_uniform_weight ${SOURCE_MAP_SUPP_UNIFORM_WEIGHT} \
    --attention_fg_gradient_mask ${ATTENTION_FG_GRADIENT_MASK} \
    --attention_fg_vote_threshold ${ATTENTION_FG_VOTE_THRESHOLD} \
    --attention_fg_fully_below_full_grad ${ATTENTION_FG_FULLY_BELOW_FULL_GRAD} \
    --attention_fg_vote_dilation_k ${ATTENTION_FG_VOTE_DILATION_K} \
    --online_p_ref ${ONLINE_P_REF} \
    --roi_score_with_rope ${ROI_SCORE_WITH_ROPE} \
    --roi_score_query_mode ${ROI_SCORE_QUERY_MODE} \
    --ref_score_with_rope ${REF_SCORE_WITH_ROPE} \
    --ref_score_query_mode ${REF_SCORE_QUERY_MODE} \
    --reward_disable_thinking_prefix ${REWARD_DISABLE_THINKING_PREFIX} \
    --reward_skip_trailing_eos ${REWARD_SKIP_TRAILING_EOS}
    "

echo "[rl] model=${MODEL}  phase_a_ckpt=${PHASE_A_CKPT}  twig K=${TWIG_K} T=${TWIG_T}"
echo "[rl] pool=${FILTERED_JSONL}  dataset_root=${DATASET_ROOT}"
echo "[rl] GPU_IDS=${GPU_IDS}  NPROC=${NPROC_PER_NODE}  bs=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM_STEPS}  attn=${ATTN_IMPL}"
echo "[rl] pixels [${MIN_PIXELS}, ${MAX_PIXELS}]  lr=${LR}  lambda_kl=${LAMBDA_KL}  placebo_kappa=${PLACEBO_KAPPA}"
echo "[rl] out=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${entry_file} ${args} 2>&1 | tee "${OUTPUT_DIR}/train_console.log"
