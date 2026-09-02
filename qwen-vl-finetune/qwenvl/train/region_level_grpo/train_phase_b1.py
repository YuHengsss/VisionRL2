"""Entrypoint for Phase-B.1 region-level GR-REINFORCE training.

Mirrors the qwen3.5 branch of ``qwen-vl-finetune/qwenvl/train/train_qwen.py``
but swaps in:

  * :class:`RegionLevelGRPODataset` + :class:`RegionLevelGRPOCollator`
    (loaded from the filtered RL pool jsonl).
  * :class:`RegionLevelGRPOTrainer` (HF Trainer subclass).
  * :class:`PhaseB1Config` for the loss hyperparameters.

Supported backbones: Qwen3.5-VL (4B / 9B) and Qwen2.5-VL (7B), auto-detected
from the Phase-A checkpoint config.

Standard SD-RPN flags (``--enable_twig``, ``--twig_K``, ``--twig_T``,
``--roi_loss``) are reused from :mod:`qwenvl.train.argument` so the launch
script looks similar to a Phase-A SFT script.
"""

from __future__ import annotations

import logging
import os

# --- Q-Zoom centralized env-knob accessor (Phase A) ---
try:
    from qwen_src.qzoom_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from qzoom_config import getenv as qz_getenv
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

# Repo root + qwen-vl-finetune on path so qwen_src/ + qwenvl/ resolve.
# parents: [0]=region_level_grpo, [1]=train, [2]=qwenvl,
#          [3]=qwen-vl-finetune, [4]=repo root
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[4]
for p in (_REPO_ROOT, _REPO_ROOT / "qwen-vl-finetune"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qwenvl.train.argument import ModelArguments, TrainingArguments

# Defer the modeling-fork import until after argparse so import errors
# surface with a clean error rather than at module load time.

from qwenvl.train.region_level_grpo.dataset import (  # noqa: E402
    DS_IMAGE_ROOTS,
    RegionLevelGRPOCollator,
    RegionLevelGRPODataset,
)
from qwenvl.train.region_level_grpo.trainer import (  # noqa: E402
    PhaseB1Config,
    RegionLevelGRPOTrainer,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase-B.1 specific arguments
# ---------------------------------------------------------------------------

@dataclass
class PhaseB1Arguments:
    """Phase-B.1 dataset path + GR-REINFORCE hyperparameters.

    Defaults are the released recipe; see ``scripts/train_rl.sh`` for the
    per-model presets (pixel budget, ``placebo_kappa``, batch shape).
    """

    # ---- corpus ----
    filtered_jsonl: str = field(
        default="",
        metadata={"help": "RL pool jsonl (required). Rows: {dataset, image, "
                          "question, gold_answer, ev_maps_path, ...}."},
    )
    max_train_samples: int = field(
        default=0,
        metadata={"help": "Cap on dataset size (0 = use all)."},
    )

    # ---- pixel budget (applied to the processor's image_processor) ----
    min_pixels: int = field(default=262144)
    max_pixels: int = field(default=589824)

    # ---- heatmap → components ----
    threshold_mode: str = field(
        default="peak_ratio",
        metadata={"help": "'peak_ratio' (dynamic, default) | 'fixed'."},
    )
    fixed_threshold: float = field(
        default=0.02,
        metadata={"help": "Absolute threshold for threshold_mode=fixed; also "
                          "binarizes the reference heatmap for the K=1 anchor."},
    )
    peak_fraction: float = field(default=0.3)   # peak_ratio: thr = peak·this
    ratio_thresh: float = field(default=3.0)    # peak_ratio: reject peak/mean < this
    min_gate: float = field(
        default=0.03,
        metadata={"help": "peak_ratio: reject the heatmap (no foreground) "
                          "when its peak probability is below this."},
    )
    smooth_kernel: int = field(default=3)
    smooth_sigma: float = field(default=1.0)
    score_p: float = field(default=1.0)
    score_beta: float = field(default=1.0)
    score_gamma: float = field(default=0.6)

    # ---- source-map supplement group ----
    source_map_group: bool = field(
        default=True,
        metadata={"help": "Add the additive supplement group driven by the "
                          "cached answer→image evidence maps (ev_maps_path in "
                          "the pool jsonl), averaged 1/2-1/2 with the "
                          "region-drop group."},
    )
    source_map_policy_sigma: float = field(default=1.0)
    source_map_supp_additive: bool = field(default=True)   # must be True
    source_map_supp_k_max: int = field(default=4)
    source_map_supp_min_cells: int = field(default=1)
    source_map_supp_multilayer: bool = field(default=True)  # must be True
    source_map_supp_merge_iou: float = field(default=0.5)
    source_map_supp_logit_clip: bool = field(default=True)
    source_map_supp_uniform_weight: bool = field(default=True)

    # ---- attention-fg gradient mask ----
    attention_fg_gradient_mask: bool = field(default=True)
    attention_fg_vote_threshold: int = field(default=3)
    attention_fg_fully_below_full_grad: bool = field(default=True)
    attention_fg_vote_dilation_k: int = field(default=3)

    # ---- top-R + action set ----
    R_max: int = field(default=6)
    enumerate_threshold: int = field(default=12)
    rollout_K: int = field(default=4)
    singleton_only_actions: bool = field(
        default=True,
        metadata={"help": "Action set = ∅ + K singleton drops (no pair drops)."},
    )

    # ---- policy ----
    softmax_T: float = field(default=1.0)
    removal_beta_keep: float = field(default=1.0)
    removal_gamma_keep: float = field(default=0.6)

    # ---- reward shaping ----
    reward_size_beta: float = field(
        default=0.0,
        metadata={"help": "Linear size penalty R = h_a − β·keep_frac."},
    )
    reward_linear_ncc_alpha: float = field(
        default=0.0,
        metadata={"help": "Add −α·N_cc(Y_keep) to each rollout's reward "
                          "(constant marginal bonus per dropped component)."},
    )
    reward_logit_clip: bool = field(
        default=True,
        metadata={"help": "Reward height = g0 + clip(logit(p)−g0, ±δ), "
                          "g0 = logit(p_∅) — difficulty-linear, saturation-free."},
    )
    reward_logit_clip_delta: float = field(default=5.0)

    # ---- loss weights ----
    lambda_kl: float = field(default=0.5)
    lambda_anchor_k1: float = field(default=1.0)
    advantage_std_eps: float = field(
        default=1.0,
        metadata={"help": "Constant added to the advantage denominator "
                          "(R − baseline) / (std + eps)."},
    )

    # ---- online Phase-A reference twig for the KL anchor ----
    online_p_ref: bool = field(
        default=True,
        metadata={"help": "Build a frozen Phase-A twig copy (RefTwigModule) at "
                          "init; at every step run it on the same pre-twig "
                          "hidden states as the policy twig to produce a "
                          "resolution-matched p_ref. False = cached p_ref_path."},
    )

    # ---- winnability-weighted PG ----
    winnability_weight: str = field(
        default="max_p",
        metadata={"help": "off | max_p. Weight each sample's PG loss by "
                          "w = max_a p(a), EMA-normalized to mean≈1."},
    )
    winnability_ema_decay: float = field(default=0.99)
    winnability_floor: float = field(default=0.05)
    winnability_max: float = field(default=3.0)

    # ---- placebo-bar subtractor ----
    subtractor_mode: str = field(
        default="placebo",
        metadata={"help": "placebo | none. placebo: ∅-baselined advantage "
                          "against R(∅) − bar, bar = kappa·(measured reward "
                          "noise on 2 low-evidence null probes)."},
    )
    placebo_kappa: float = field(
        default=1.25,
        metadata={"help": "Bar multiplier (1.25 Qwen3.5-4B; 1.0 9B / Qwen2.5-7B)."},
    )
    placebo_p_thresh: float = field(default=0.02)
    placebo_bar_max: float = field(default=1.0)

    # ---- SD-RPN scoring convention ----
    roi_score_with_rope: bool = field(
        default=True,
        metadata={"help": "Apply RoPE to the SD-RPN training-time scoring "
                          "(policy side)."},
    )
    roi_score_query_mode: str = field(
        default="last_prompt",
        metadata={"help": "'last_prompt' (position just before the first "
                          "response token; matches inference) | "
                          "'response_tokens' (mean over gold-answer rows)."},
    )
    ref_score_with_rope: bool = field(default=True)
    ref_score_query_mode: str = field(default="last_prompt")

    # ---- reward LM prompt format ----
    reward_disable_thinking_prefix: bool = field(
        default=True,
        metadata={"help": "Prepend the official Qwen3.5-VL enable_thinking=False "
                          "prefix to the reward-LM assistant turn (family-gated: "
                          "no-op for Qwen2.5-VL)."},
    )
    reward_skip_trailing_eos: bool = field(
        default=True,
        metadata={"help": "Exclude the closing <|im_end|> from the reward-LM "
                          "answer mask."},
    )


def _build_phase_b1_config(args: PhaseB1Arguments) -> PhaseB1Config:
    if args.source_map_group and not (
            args.source_map_supp_additive and args.source_map_supp_multilayer):
        raise ValueError(
            "source_map_group requires source_map_supp_additive=True and "
            "source_map_supp_multilayer=True (the whole-map and single-layer "
            "variants are not part of this release)."
        )
    if args.winnability_weight not in ("off", "max_p"):
        raise ValueError(
            f"winnability_weight must be 'off' or 'max_p'; got {args.winnability_weight!r}")
    if args.subtractor_mode not in ("placebo", "none"):
        raise ValueError(
            f"subtractor_mode must be 'placebo' or 'none'; got {args.subtractor_mode!r}")
    return PhaseB1Config(
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
        peak_fraction=args.peak_fraction,
        ratio_thresh=args.ratio_thresh,
        min_gate=float(args.min_gate),
        smooth_kernel=args.smooth_kernel,
        smooth_sigma=args.smooth_sigma,
        source_map_group=bool(args.source_map_group),
        source_map_policy_sigma=float(args.source_map_policy_sigma),
        source_map_supp_additive=bool(args.source_map_supp_additive),
        source_map_supp_k_max=int(args.source_map_supp_k_max),
        source_map_supp_min_cells=int(args.source_map_supp_min_cells),
        source_map_supp_multilayer=bool(args.source_map_supp_multilayer),
        source_map_supp_merge_iou=float(args.source_map_supp_merge_iou),
        source_map_supp_logit_clip=bool(args.source_map_supp_logit_clip),
        source_map_supp_uniform_weight=bool(args.source_map_supp_uniform_weight),
        attention_fg_gradient_mask=bool(args.attention_fg_gradient_mask),
        attention_fg_vote_threshold=int(args.attention_fg_vote_threshold),
        attention_fg_fully_below_full_grad=bool(
            args.attention_fg_fully_below_full_grad),
        attention_fg_vote_dilation_k=int(args.attention_fg_vote_dilation_k),
        score_p=args.score_p,
        score_beta=args.score_beta,
        score_gamma=args.score_gamma,
        R_max=args.R_max,
        enumerate_threshold=args.enumerate_threshold,
        rollout_K=args.rollout_K,
        softmax_T=args.softmax_T,
        removal_beta_keep=args.removal_beta_keep,
        removal_gamma_keep=args.removal_gamma_keep,
        reward_size_beta=args.reward_size_beta,
        reward_linear_ncc_alpha=args.reward_linear_ncc_alpha,
        reward_logit_clip=args.reward_logit_clip,
        reward_logit_clip_delta=args.reward_logit_clip_delta,
        lambda_kl=args.lambda_kl,
        lambda_anchor_k1=args.lambda_anchor_k1,
        online_p_ref=args.online_p_ref,
        winnability_weight=args.winnability_weight,
        winnability_ema_decay=args.winnability_ema_decay,
        winnability_floor=args.winnability_floor,
        winnability_max=args.winnability_max,
        singleton_only_actions=args.singleton_only_actions,
        advantage_std_eps=args.advantage_std_eps,
        subtractor_mode=str(args.subtractor_mode),
        placebo_kappa=float(args.placebo_kappa),
        placebo_p_thresh=float(args.placebo_p_thresh),
        placebo_bar_max=float(args.placebo_bar_max),
        roi_score_with_rope=args.roi_score_with_rope,
        roi_score_query_mode=args.roi_score_query_mode,
        ref_score_with_rope=args.ref_score_with_rope,
        ref_score_query_mode=args.ref_score_query_mode,
        reward_disable_thinking_prefix=args.reward_disable_thinking_prefix,
        reward_skip_trailing_eos=args.reward_skip_trailing_eos,
    )


# ---------------------------------------------------------------------------
# Model loading + freezing
# ---------------------------------------------------------------------------

def _load_policy_with_twig(model_args: ModelArguments,
                           training_args: TrainingArguments,
                           attn_implementation: str = "flash_attention_2"):
    """Load the Phase-A checkpoint with SD-RPN flags propagated.

    Family is auto-detected from the checkpoint config: Qwen3.5-VL
    (``Qwen3_5Config``, transformers 5.x) or Qwen2.5-VL (``Qwen2_5_VLConfig``,
    transformers 4.51).
    """
    original_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    _cfg_name = type(original_config).__name__
    if "Qwen2_5" in _cfg_name:
        # Qwen2.5-VL-7B (tf4.51): non-gated twig + mrope. The batch fork
        # exposes per_head_scores for RL.
        from qwen_src.qwen2_5_vl.modeling_qwen2_5_vl_batch import (
            Qwen2_5_VLForConditionalGeneration as _PhaseAModel,
        )
    else:
        from qwen_src.qwen3_5.modeling_qwen3_5_batch import (
            Qwen3_5ForConditionalGeneration as _PhaseAModel,
        )
    config = type(original_config).from_dict(original_config.to_dict())

    # Mirror the Phase-A flag set we trained with — most are no-ops at
    # inference but the model code reads them defensively. The SD-RPN
    # architecture knobs are pinned to the released recipe.
    config.enable_twig = True
    config.twig_K = model_args.twig_K
    config.twig_T = model_args.twig_T
    config.roi_source = "qk"
    config.roi_loss = model_args.roi_loss
    config.roi_super_type = "v1"
    config.roi_multi_head = True
    config.roi_skip_ffn = False
    config.roi_keep_ffn_mod_ratio = 1000
    config.enable_high_res = False
    config.roi_post_training = False
    config.online_pseudo_label = False
    config.use_roi_prompt = False
    config.reuse_src_pos = False
    config.return_per_head_score = True   # critical: enable heatmap capture
    config.roi_per_head_sft = False        # we want raw mean-over-heads Z_theta

    # Qwen3_5TextModel reads enable_twig / twig_K / twig_T off
    # ``config.text_config`` -- propagate.
    if hasattr(config, "text_config") and config.text_config is not None:
        for attr in (
            "enable_twig", "twig_K", "twig_T",
            "roi_loss", "return_per_head_score",
        ):
            if hasattr(config, attr):
                setattr(config.text_config, attr, getattr(config, attr))

    # tf4.51 (qwen2.5) takes the legacy ``torch_dtype`` kwarg; tf5.x (qwen3.5)
    # takes the newer ``dtype`` alias and would error on ``torch_dtype``.
    _dtype_val = torch.bfloat16 if training_args.bf16 else None
    _dtype_kw = ({"torch_dtype": _dtype_val} if "Qwen2_5" in _cfg_name
                 else {"dtype": _dtype_val})
    model = _PhaseAModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        config=config,
        **_dtype_kw,
    )
    if not hasattr(model, "visual") and hasattr(model.model, "visual"):
        object.__setattr__(model, "visual", model.model.visual)
    return model


def _freeze_for_phase_b1(model) -> None:
    """Freeze everything except the twig layers.

    Phase-B.1 only updates the SD-RPN twig heads. The LLM backbone,
    vision tower, and lm_head all stay at the Phase-A weights so the
    reward forward (which shares the model) is consistent across
    training steps.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Both families lay out the LLM at model.model.language_model; fall
    # back to a module search if a future layout differs.
    llm = getattr(getattr(model, "model", model), "language_model", None)
    if llm is None or not hasattr(llm, "twig_layers"):
        llm = None
        for _n, _m in model.named_modules():
            if hasattr(_m, "twig_layers"):
                llm = _m
                break
    if llm is None or not hasattr(llm, "twig_layers"):
        raise RuntimeError(
            "no module with twig_layers found; was the checkpoint trained "
            "with --enable_twig? Check the model path."
        )
    for layer in llm.twig_layers:
        for p in layer.parameters():
            if p.is_floating_point() or p.is_complex():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(attn_implementation: str = "flash_attention_2") -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, PhaseB1Arguments)
    )
    model_args, training_args, phase_b1_args = parser.parse_args_into_dataclasses()
    if not phase_b1_args.filtered_jsonl:
        raise ValueError("--filtered_jsonl is required (path to the RL pool jsonl)")
    training_args.save_total_limit = training_args.save_total_limit or 1
    training_args.remove_unused_columns = False  # we attach extras to batch
    os.makedirs(training_args.output_dir, exist_ok=True)
    local_rank = training_args.local_rank

    if local_rank in (0, -1):
        print(f"[phase_b1] model_path = {model_args.model_name_or_path}")
        print(f"[phase_b1] filtered   = {phase_b1_args.filtered_jsonl}")
        print(f"[phase_b1] output_dir = {training_args.output_dir}")
        # ---- EFFECTIVE REWARD CONFIG ----------------------------------
        # Single source of truth for the *parsed* reward/group knobs, so a
        # set-but-never-forwarded env var shows up here as its default.
        _pa = phase_b1_args
        print("[phase_b1] ===== EFFECTIVE REWARD CONFIG =====")
        print(f"[phase_b1]   reward_logit_clip          = {_pa.reward_logit_clip} (delta={_pa.reward_logit_clip_delta})")
        print(f"[phase_b1]   reward_size_beta           = {_pa.reward_size_beta} (linear, no gamma)")
        print(f"[phase_b1]   reward_linear_ncc_alpha    = {_pa.reward_linear_ncc_alpha}")
        print(f"[phase_b1]   lambda_kl                  = {_pa.lambda_kl}")
        print(f"[phase_b1]   singleton_only_actions     = {_pa.singleton_only_actions}")
        print(f"[phase_b1]   advantage_std_eps          = {_pa.advantage_std_eps}")
        print(f"[phase_b1]   subtractor_mode            = {_pa.subtractor_mode} (kappa={_pa.placebo_kappa}, p_thresh={_pa.placebo_p_thresh}, bar_max={_pa.placebo_bar_max})")
        print(f"[phase_b1]   winnability_weight         = {_pa.winnability_weight} (ema={_pa.winnability_ema_decay}, floor={_pa.winnability_floor}, max={_pa.winnability_max})")
        print(f"[phase_b1]   source_map_group           = {_pa.source_map_group} (sigma={_pa.source_map_policy_sigma})")
        print(f"[phase_b1]   source_map_supp_additive   = {_pa.source_map_supp_additive} (multilayer={_pa.source_map_supp_multilayer}, k_max={_pa.source_map_supp_k_max}, min_cells={_pa.source_map_supp_min_cells}, merge_iou={_pa.source_map_supp_merge_iou})")
        print(f"[phase_b1]   source_map_supp_logit_clip = {_pa.source_map_supp_logit_clip}")
        print(f"[phase_b1]   source_map_supp_uniform_weight = {_pa.source_map_supp_uniform_weight}")
        print(f"[phase_b1]   attention_fg_gradient_mask = {_pa.attention_fg_gradient_mask} (vote_thr={_pa.attention_fg_vote_threshold}, fully_below_full_grad={_pa.attention_fg_fully_below_full_grad}, dilation_k={_pa.attention_fg_vote_dilation_k})")
        print(f"[phase_b1]   threshold_mode             = {_pa.threshold_mode} (peak_fraction={_pa.peak_fraction}, ratio_thresh={_pa.ratio_thresh}, min_gate={_pa.min_gate}, fixed={_pa.fixed_threshold})")
        print(f"[phase_b1]   score_p                    = {_pa.score_p}")
        print(f"[phase_b1]   online_p_ref               = {_pa.online_p_ref} (ref rope={_pa.ref_score_with_rope}, ref query={_pa.ref_score_query_mode})")
        print(f"[phase_b1]   roi_score                  = rope={_pa.roi_score_with_rope}, query={_pa.roi_score_query_mode}")
        print(f"[phase_b1]   reward prompt              = think_prefix={_pa.reward_disable_thinking_prefix}, skip_trailing_eos={_pa.reward_skip_trailing_eos}")
        print("[phase_b1] ===================================")

    # ---- model + processor ----
    model = _load_policy_with_twig(model_args, training_args,
                                   attn_implementation=attn_implementation)
    model.config.use_cache = False
    # Gate RoPE inside the SD-RPN training-time scoring. Modeling code reads
    # ``self.config.roi_score_with_rope``; propagate to text_config so the
    # language-model fork sees it too.
    model.config.roi_score_with_rope = bool(phase_b1_args.roi_score_with_rope)
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.roi_score_with_rope = bool(phase_b1_args.roi_score_with_rope)
    # Same plumbing for the query-mode selection.
    model.config.roi_score_query_mode = str(phase_b1_args.roi_score_query_mode)
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.roi_score_query_mode = str(phase_b1_args.roi_score_query_mode)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _hook(module, _inp, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_hook)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    # Apply the pixel budget to the image processor (no per-call kwarg
    # in Qwen's processor; the image_processor reads these attributes).
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = phase_b1_args.min_pixels
        processor.image_processor.max_pixels = phase_b1_args.max_pixels

    # online_p_ref: build the frozen Phase-A reference twig BEFORE freezing
    # the policy twig layers — the deep-copy captures their initial state
    # from the freshly-loaded Phase-A checkpoint. After this, the policy
    # twig can train freely; the snapshot stays at Phase-A weights for the
    # entire run.
    ref_twig_module = None
    if phase_b1_args.online_p_ref:
        from qwenvl.train.region_level_grpo.ref_twig import (
            load_ref_twig_from_policy,
        )
        ref_twig_module = load_ref_twig_from_policy(model)
        if local_rank in (0, -1):
            n_params = sum(p.numel() for p in ref_twig_module.parameters())
            print(
                f"[phase_b1] online_p_ref=True → RefTwigModule built "
                f"({n_params / 1e6:.2f}M params, frozen)"
            )

    _freeze_for_phase_b1(model)

    if local_rank in (0, -1):
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"[phase_b1] trainable params: {n_train / 1e6:.2f}M / "
            f"{n_total / 1e6:.2f}M total ({100 * n_train / n_total:.2f}%)"
        )
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"  trainable: {n}  shape={tuple(p.shape)}")

    # ---- dataset + collator ----
    dataset = RegionLevelGRPODataset(
        filtered_jsonl=phase_b1_args.filtered_jsonl,
        image_roots=DS_IMAGE_ROOTS,
        max_samples=phase_b1_args.max_train_samples,
        online_p_ref=bool(phase_b1_args.online_p_ref),
    )
    if local_rank in (0, -1):
        print(f"[phase_b1] dataset size = {len(dataset)}")
    collator = RegionLevelGRPOCollator(
        processor=processor,
        fixed_threshold=phase_b1_args.fixed_threshold,
    )

    # ---- trainer ----
    phase_b1_cfg = _build_phase_b1_config(phase_b1_args)
    trainer = RegionLevelGRPOTrainer(
        model=model,
        processing_class=processor,  # processor exposes tokenizer too
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        phase_b1_config=phase_b1_cfg,
        ref_twig_module=ref_twig_module,
    )

    # ---- run ----
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # SKIP_FINAL_SAVE=1 avoids writing the full model + processor at the
    # end of training (e.g. sweeps that only want the TensorBoard logs).
    if int(qz_getenv("SKIP_FINAL_SAVE", "0")):
        if local_rank in (0, -1):
            print(
                f"[phase_b1] SKIP_FINAL_SAVE=1: not writing model or "
                f"processor to {training_args.output_dir}",
                flush=True,
            )
        return
    processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    _attn = qz_getenv("ATTN_IMPL", "flash_attention_2")
    train(attn_implementation=_attn)
