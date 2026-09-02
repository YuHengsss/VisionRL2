"""On-the-fly pseudo-label generator for Qwen3.5 SD-RPN supervision.

Couples ``online_attention`` (which captures hidden states + recomputes
attention) with the existing ``create_pseudo_labels`` (in
``qwen_src/roi/heatmap.py``) so a training step can produce a fresh
``roi_target_map`` per sample without relying on a precomputed pkl.

Per-sample head-config dispatch by dataset (matches Qwen3-VL):
    - ``textvqa`` / ``docvqa`` / ``infographicsvqa`` → ``textual``
    - ``gqa``                                       → ``natural``

Auto-detect from prompted question text when dataset isn't threaded
through (``GQA_BBOX_SUFFIX`` ⇒ natural; otherwise textual).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from qwen_src.roi.heatmap import create_pseudo_labels

from .online_attention import (
    GroundingCacheEntry,
    compute_response_to_image_attention,
)


# ---- Head presets for Qwen3.5
# Qwen3.5-9B has 32 layers (8 full_attention at idx 3,7,11,15,19,23,27,31)
# with num_attention_heads=16 (so head idxs 0..15 valid). Layer 19 is the
# 5th full-attention block. Heads 14 and 15 were identified as the
# response→image grounding heads via the attn_viz_demo.
#
# No sink heads identified for Qwen3.5 yet — leaving empty so
# create_pseudo_labels gets sink_attn = zeros, which still produces a
# usable {0,1} foreground mask from grounding alone.
HEAD_CONFIGS: Dict[str, Dict[str, Tuple[Dict[int, List[int]], Dict[int, List[int]]]]] = {
    "textual": {
        "qwen3_5_9b": ({19: [14, 15]}, {}),
        # Qwen3.5-4B uses the same response→image grounding heads as
        # the 9B (layer 19 heads {14, 15}) — picked via the attn-viz
        # panel for textvqa-style prompts on the public Qwen3.5-4B.
        "qwen3_5_4b": ({19: [14, 15]}, {}),
    },
    "natural": {
        "qwen3_5_9b": ({19: [14, 15]}, {}),
        "qwen3_5_4b": ({19: [14, 15]}, {}),
    },
}

DATASET_TO_MODE = {
    "textvqa": "textual",
    "docvqa": "textual",
    "infographicsvqa": "textual",
    "gqa": "natural",
}

NATURAL_PROMPT_MARKER = "Output the grounding bounding boxes of Region of Interests"


def detect_mode(dataset: Optional[str], prompted_question: Optional[str]) -> str:
    if dataset and dataset in DATASET_TO_MODE:
        return DATASET_TO_MODE[dataset]
    if prompted_question and NATURAL_PROMPT_MARKER in prompted_question:
        return "natural"
    return "textual"


def get_head_config(
    model_family: str, mode: str,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    if mode not in HEAD_CONFIGS:
        raise ValueError(f"unknown mode {mode!r}")
    if model_family not in HEAD_CONFIGS[mode]:
        raise ValueError(
            f"no head config for model_family={model_family!r}, mode={mode!r}"
        )
    return HEAD_CONFIGS[mode][model_family]


def grounding_layer_set(
    model_family: str, modes: Sequence[str] = ("textual", "natural"),
) -> List[int]:
    """Union of every layer used by any head config for ``model_family``
    across the listed modes — passed to ``register_grounding_cache`` so
    the pre-hook captures every layer the runtime might query."""
    out = set()
    for mode in modes:
        g, s = get_head_config(model_family, mode)
        out.update(g.keys())
        out.update(s.keys())
    return sorted(out)


def _slice_cache_entry(entry, sample_idx: int):
    """Slice each cached tensor to a single batch row."""
    cos, sin = entry.position_embeddings
    return GroundingCacheEntry(
        hidden_states=entry.hidden_states[sample_idx:sample_idx + 1],
        position_embeddings=(
            cos[sample_idx:sample_idx + 1] if cos.dim() >= 3 and cos.shape[0] != 3 else cos,
            sin[sample_idx:sample_idx + 1] if sin.dim() >= 3 and sin.shape[0] != 3 else sin,
        ),
        attention_mask=(
            entry.attention_mask[sample_idx:sample_idx + 1]
            if entry.attention_mask is not None else None
        ),
        position_ids=(
            entry.position_ids[..., sample_idx:sample_idx + 1, :]
            if entry.position_ids is not None and entry.position_ids.dim() >= 2
            else entry.position_ids
        ),
    )


def online_pseudo_label_for_sample(
    *,
    model: torch.nn.Module,
    sample_idx: int,
    response_rows: slice,
    image_cols: slice,
    feat_hw: Tuple[int, int],
    original_image_size: Tuple[int, int],
    dataset: Optional[str] = None,
    prompted_question: Optional[str] = None,
    model_family: str = "qwen3_5_9b",
    mode_override: Optional[str] = None,
    pseudo_label_mode: str = "aggregated",
    create_kwargs: Optional[dict] = None,
) -> dict:
    """Generate one sample's pseudo-label using the captured grounding
    cache. Returns the ``create_pseudo_labels`` output (``aggregated``
    mode: grounding heads averaged into one map)."""
    mode = mode_override or detect_mode(dataset, prompted_question)
    grounding_heads, sink_heads = get_head_config(model_family, mode)
    needed_layers = sorted(set(grounding_heads.keys()) | set(sink_heads.keys()))

    cache = getattr(model, "_grounding_cache", None)
    if cache is None:
        raise RuntimeError(
            "model._grounding_cache missing — wrap forward in "
            "register_grounding_cache."
        )
    layers_module = model.model.language_model.layers
    h_feat, w_feat = feat_hw

    layer_image_attn: Dict[int, torch.Tensor] = {}
    for li in needed_layers:
        entry = cache.get(int(li))
        if entry is None:
            continue
        cos, sin = entry.position_embeddings
        sliced_entry = GroundingCacheEntry(
            hidden_states=entry.hidden_states[sample_idx:sample_idx + 1],
            position_embeddings=(
                cos[sample_idx:sample_idx + 1] if cos.dim() >= 3 and cos.shape[0] != 3 else cos,
                sin[sample_idx:sample_idx + 1] if sin.dim() >= 3 and sin.shape[0] != 3 else sin,
            ),
            attention_mask=(
                entry.attention_mask[sample_idx:sample_idx + 1]
                if entry.attention_mask is not None else None
            ),
            position_ids=entry.position_ids,
        )
        resp_to_img = compute_response_to_image_attention(
            layers_module[li], sliced_entry, response_rows, image_cols,
        )                                                # [1, H, T_resp, T_img]
        layer_image_attn[int(li)] = resp_to_img.mean(dim=2).squeeze(0)  # [H, T_img]

    create_kwargs = dict(create_kwargs or {})
    create_kwargs.setdefault("original_image_size", original_image_size)

    def _agg_heads(heads_cfg: Dict[int, List[int]]) -> torch.Tensor:
        acc = None
        total = 0
        for lid, hs in heads_cfg.items():
            la = layer_image_attn.get(int(lid))
            if la is None:
                continue
            for h in hs:
                row = la[h].float()
                acc = row if acc is None else acc + row
                total += 1
        if acc is None or total == 0:
            return torch.zeros(h_feat, w_feat, dtype=torch.float32)
        return (acc / total).reshape(h_feat, w_feat)

    if pseudo_label_mode == "aggregated":
        grounding_2d = _agg_heads(grounding_heads)
        sink_2d = _agg_heads(sink_heads)
        return create_pseudo_labels(
            sink_attn=sink_2d.cpu().numpy(),
            grounding_attn_o2i=grounding_2d.cpu().numpy(),
            **create_kwargs,
        )

    raise ValueError(f"unknown pseudo_label_mode {pseudo_label_mode!r}")


# "[Visual Evidence]\n" under the Qwen3.5 tokenizer (verified identical for
# the 4B and 9B). Used by the model forward to decide the per-sample prefix
# skip for single-region labels (textvqa / instruction-skips → no prefix).
PREFIX_IDS_VISUAL_EVIDENCE: List[int] = [58, 9308, 42249, 60, 198]


@torch.no_grad()
def online_single_region_label_for_sample(
    *,
    model: torch.nn.Module,
    sample_idx: int,
    response_rows: slice,
    image_cols: slice,
    feat_hw: Tuple[int, int],
    model_family: str = "qwen3_5_4b",
    mode: str = "textual",
    skip_leading_tokens: int = 0,
    max_label_tokens: int = 256,
    ratio_thresh: float = 3.0,
    peak_fraction: float = 0.3,
    extract_smooth_sigma: float = 0.0,
    halo_smooth_sigma: float = 1.0,
    halo_thresh_frac: float = 0.3,
    empty_sample_ignore: bool = True,
) -> Optional[dict]:
    """Single-region online ROI label for one sample (v2 path).

    Unlike :func:`online_pseudo_label_for_sample` (which means the grounding
    heads' attention over ALL response tokens then runs
    ``create_pseudo_labels``), this aggregates the grounding heads PER
    RESPONSE TOKEN (default layer 19 heads {14, 15} via ``get_head_config``)
    and runs the single-region (peak/mean ratio gate + 1-CC) per-token union
    extractor from ``online_single_region_label``.

    The ``[Visual Evidence]\\n`` prefix is skipped via ``skip_leading_tokens``
    (the caller passes 5 for evidence-following samples, 0 for textvqa /
    direct responses). Tokens are then capped to ``max_label_tokens`` to
    bound the per-token CC cost.

    Returns ``{"labels": np.ndarray[h, w]}`` with values ``{1, 0, -100}`` to
    match the consumption site in the model forward, or ``None`` on failure
    (caller keeps the existing target).
    """
    from .online_single_region_label import single_region_roi_label

    grounding_heads, _sink = get_head_config(model_family, mode)
    needed_layers = sorted(grounding_heads.keys())
    cache = getattr(model, "_grounding_cache", None)
    if cache is None:
        raise RuntimeError(
            "model._grounding_cache missing — wrap forward in "
            "register_grounding_cache."
        )
    layers_module = model.model.language_model.layers
    h_feat, w_feat = feat_hw

    # Per-token attention aggregated over the configured grounding heads:
    # [T_resp, T_img] where T_img == h_feat * w_feat.
    acc = None
    total = 0
    for li in needed_layers:
        entry = cache.get(int(li))
        if entry is None:
            continue
        sliced = _slice_cache_entry(entry, sample_idx)
        resp_to_img = compute_response_to_image_attention(
            layers_module[li], sliced, response_rows, image_cols,
        )                                            # [1, H, T_resp, T_img]
        a = resp_to_img.squeeze(0)                   # [H, T_resp, T_img]
        for h in grounding_heads[int(li)]:
            row = a[h].float()                       # [T_resp, T_img]
            acc = row if acc is None else acc + row
            total += 1
    if acc is None or total == 0:
        return None
    per_tok = acc / total                            # [T_resp, T_img]

    # Drop the [Visual Evidence] prefix, then cap to bound CC cost.
    if skip_leading_tokens > 0:
        per_tok = per_tok[int(skip_leading_tokens):]
    if max_label_tokens and per_tok.shape[0] > int(max_label_tokens):
        per_tok = per_tok[: int(max_label_tokens)]
    if per_tok.shape[0] == 0:
        return None
    per_tok = per_tok.reshape(per_tok.shape[0], h_feat, w_feat)

    lbl = single_region_roi_label(
        per_tok,
        skip_leading_tokens=0,                       # already skipped above
        ratio_thresh=ratio_thresh,
        peak_fraction=peak_fraction,
        extract_smooth_sigma=extract_smooth_sigma,
        halo_smooth_sigma=halo_smooth_sigma,
        halo_thresh_frac=halo_thresh_frac,
        empty_sample_ignore=empty_sample_ignore,
    )                                                # [h, w] long {1, 0, -100}

    # Stats + a mean attention map (same output contract as the v1 path).
    fg = (lbl == 1)
    n_fg = int(fg.sum())
    fg_bbox = None
    if n_fg > 0:
        ys, xs = torch.nonzero(fg, as_tuple=True)
        fg_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "labels": lbl.detach().cpu().numpy(),
        "grounding_attn": per_tok.mean(dim=0).detach().cpu().numpy(),
        "stats": {
            "num_fg": n_fg,
            "num_bg": int((lbl == 0).sum()),
            "num_ignore": int((lbl == -100).sum()),
            "n_tokens": int(per_tok.shape[0]),
            "fg_bbox": fg_bbox,
        },
    }
