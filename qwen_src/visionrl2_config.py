"""Centralized environment-knob registry.

This module is a *behavior-neutral* front door for the project's
environment-variable "knobs". It does two things:

1. Documents every project knob in a single registry (:data:`KNOBS`).
2. Provides :func:`getenv`, a drop-in for ``os.environ.get`` whose only
   addition is a registry-level default for registered knobs when the call
   site passes no default.

Design guarantees:
- :func:`getenv` never parses, converts, strips, or otherwise transforms values.
  It returns raw strings exactly like ``os.environ.get`` would.
- It never raises for unknown knobs.
- The precedence is IDENTICAL to ``os.environ.get(name, default)``:
      env value  >  caller-passed default  >  registry default  >  None

The module imports only the standard library so it can be imported from anywhere
(qwen_src, qwenvl, lmms-eval) without creating import cycles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Knob:
    """A single registered environment knob.

    Attributes:
        name:    The environment variable name (ALL-CAPS).
        default: The registry-level default, as a RAW string exactly as the call
                 sites pass to ``os.environ.get`` -- or ``None`` when the call site
                 passes no literal default (or passes a non-literal expression, in
                 which case the site default is preserved verbatim at the call site
                 and this registry default stays ``None``).
        doc:     One-line human description.
        scope:   One of {'eval', 'train', 'data', 'debug'}.
    """

    name: str
    default: Optional[str]
    doc: str
    scope: str


def _mk(name: str, default: Optional[str], doc: str, scope: str) -> Knob:
    assert scope in ("eval", "train", "data", "debug"), scope
    return Knob(name=name, default=default, doc=doc, scope=scope)


# NOTE: defaults below are the RAW strings the call sites pass. Where a call site
# passes a non-literal expression (e.g. ``ori_min_pixel``, ``str(...)``, or a
# per-class default), the registry default is ``None`` and the site default is
# kept verbatim at the call site (site default wins when the env is unset).
_KNOB_LIST: List[Knob] = [
    # ---- Eval-time RoI crop geometry ------------------------------------------
    _mk("ROI_EVAL_SMOOTH_SIGMA", "1.0", "Gaussian sigma for smoothing the RoI heatmap at eval time; 'auto' (1.0@<=512 tok -> 2.0@>=2048 tok) or 'auto2' (auto, extended past 2048 tok, cap 4.0).", "eval"),
    _mk("DISABLE_SHORT_ANSWER_SUFFIX", "0", "If '1', DO NOT append the single-word-answer suffix to eval prompts (verbose-CoT / judge protocol).", "eval"),
    _mk("ROI_MIN_PIXEL_BASE", None, "Base min-pixel budget for the RoI crop; site default is the processor's min_pixels (deployment: 262144 = 256 tokens).", "eval"),
    _mk("ROI_MIN_TOKENS_AUTO", "0", "If '1', clamp the RoI crop min budget to min(ROI_MIN_PIXEL_BASE, src_tokens/4) per sample.", "eval"),
    _mk("PROBE_CROP_TARGET_TOK", "0", "If >0, constant-target crop rule: kept tokens = max(native, min(edge^2*native, T)) capped by min(3T, src_cap/DIV); 0 = crop shares the source cap.", "eval"),
    _mk("PROBE_CROP_MAX_UPSCALE_EDGE", "3", "Max per-edge upscale factor of the RoI crop under PROBE_CROP_TARGET_TOK.", "eval"),
    _mk("PROBE_CROP_SRC_CAP_DIV", "2", "Divisor on the source-cap term of the crop cap under PROBE_CROP_TARGET_TOK.", "eval"),
    # ---- Eval-time instrumentation --------------------------------------------
    _mk("VISIONRL2_STAGE_TIMING", "", "If set and not '0'/'false', record per-sample stage latencies (cuda-synchronized) to a JSONL sidecar.", "eval"),
    _mk("VISIONRL2_STAGE_TIMING_DIR", "./logs/stage_timing", "Output dir for the stage-timing sidecars.", "eval"),
    _mk("VISIONRL2_STAGE_TIMING_WARMUP", "2", "Number of leading samples excluded from stage-timing aggregation.", "eval"),
    _mk("VISIONRL2_RUN_TAG", None, "Tag stamped into stage-timing sidecar filenames; site default is a timestamp.", "eval"),
    _mk("VISIONRL2_EVAL_PREFETCH", "4", "Worker threads prefetching the CPU stage (image load + processor) in the lmms-eval chat wrappers; 0 = synchronous.", "eval"),
    # ---- Model-forward / online-supervision behavior -------------------------
    _mk("SKIP_POST_BRANCH", "1", "If '1', skip the post-twig LM layers when only the twig heatmap is needed (RL policy forward); the zero-loss fallback routes via twig_hidden.", "train"),
    _mk("SINGLE_REGION_EXTRACT_SIGMA", "0.0", "Gaussian sigma used when extracting the single-region (v2) online label from the attention heatmap; 0 = no smoothing.", "train"),
    # ---- Data pipeline -------------------------------------------------------
    _mk("CONVERT_GQA_TO_SIMPLE_PROB", None, "Probability of converting GQA samples to the simple-prompt form; site default is str(_DEFAULT_CONVERT_GQA_TO_SIMPLE_PROB).", "data"),
    _mk("STRIP_TASK_SUFFIX_PROB", "1.0", "Probability of stripping the task suffix from the prompt.", "data"),
    _mk("STRIP_TASK_SUFFIX_SEED", "12345", "RNG seed for task-suffix stripping.", "data"),
    _mk("EXPAND2SQUARE", None, "If set, pad training images to square (expand2square). Unset -> None.", "data"),
    _mk("EXPAND2SQUARE_BY_VERSION", "0", "If '1', apply expand2square selectively by label version.", "data"),
    _mk("MAX_PIXELS_BY_VERSION", "0", "If '1', apply per-label-version max-pixels overrides.", "data"),
    _mk("V1_MAX_PIXELS", "1048576", "Max-pixels budget applied to v1-label samples.", "data"),
    _mk("DS_IMAGE_ROOTS", "", "Optional per-dataset image-root overrides (parsed downstream).", "data"),
    _mk("LABEL_VERSION_MAP", "", "Optional dataset->label-version map (parsed downstream).", "data"),
    # ---- Training driver -----------------------------------------------------
    _mk("ATTN_IMPL", "flash_attention_2", "attn_implementation passed to model load.", "train"),
    _mk("VISIONRL2_DISABLE_CAUSAL_CONV1D", "0", "If '1', force the torch fallback for the qwen3.5 linear-attention conv path even when causal-conv1d is installed (reference training kernel).", "train"),
    _mk("VISIONRL2_DISABLE_REWARD_MASK_IP_CAP", "0", "If '1', skip registering the reward image_processor cap for _build_masked_pils (masked PILs built from the uncapped source image).", "train"),
    # ---- Region-level RL -----------------------------------------------------
    _mk("RLG_DATA_BASE", None, "Base dir for region-level RL data (unset -> None).", "data"),
    _mk("REWARD_CE_CHUNK", "1", "Chunk size for the reward-model CE forward.", "train"),
    _mk("REWARD_MASK_PIL_WORKERS", "16", "Worker count for building masked reward PILs.", "train"),
    _mk("PHASE_B1_DEBUG", "0", "If '1', emit RL trainer debug output.", "debug"),
    _mk("SKIP_FINAL_SAVE", "0", "If '1', skip the final checkpoint save (debug).", "train"),
]

KNOBS: Dict[str, Knob] = {k.name: k for k in _KNOB_LIST}


_UNSET = object()


def getenv(name: str, default: Any = _UNSET) -> Any:
    """Drop-in for ``os.environ.get``.

    Precedence, identical to ``os.environ.get(name, default)``:
        env value  >  caller-passed default  >  registry default  >  None

    Does NOT parse/convert the value.
    """
    if name in os.environ:
        return os.environ[name]
    if default is not _UNSET:
        return default
    knob = KNOBS.get(name)
    return knob.default if knob is not None else None
