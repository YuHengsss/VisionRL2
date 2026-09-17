"""On-the-fly pseudo-label generator for Gemma-4-12B-IT SD-RPN supervision.

Port of qwen_src/qwen3_5/online_pseudo_label.py (the v1 mean-over-response-
tokens path) with the Gemma-specific adaptations:

  * Attention capture is HOOK-based on eager attention weights (the
    attn_export_prototype.py / vis50_it.py pattern): forward hooks on
    ``model.model.language_model.layers[i].self_attn`` store the softmax
    weights returned by ``Gemma4UnifiedTextAttention.forward`` during the
    MAIN training forward. This requires ``attn_implementation="eager"``
    (sdpa/flash return attn_weights=None). No q/k recompute is needed.
  * The token grid comes from ``image_position_ids`` (x, y) pairs — NOT
    from an image_grid_thw. Under expand2square at the 560 tier the grid is
    square 23x23 (529 tokens), but the scatter always goes through
    ``flat_idx = y * gw + x`` rather than a blind reshape.
  * Sink handling: Gemma has no sink-head detection. Per head_config.py,
    the OUTERMOST RING of the token grid is the sink band — expressed by
    passing a {0,1} ring mask as ``sink_attn`` (>= sink_thresh marks sink)
    into the unchanged ``create_pseudo_labels``.
  * Head configs (finalized 2026-08-13, human-reviewed):
      textual (ocrvqa/docvqa): L17 heads {0,15} + L29 heads {3,15}
      natural (gqa):           L29 heads {0,1}
    Both layers are global (full_attention) layers — sliding layers cannot
    see the image span once the response is >1024 tokens away.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from qwen_src.gemma4_unified.pseudo_label_core import create_pseudo_labels
from qwen_src.gemma4_unified.head_config import (
    EXPAND2SQUARE,
    HEAD_CONFIGS,
    MODEL_ID,
    SINK_RULE,
    SOFT_TOKEN_TIER,
    outer_ring_mask,
)

MODEL_FAMILY = "gemma4_12b"

DATASET_TO_MODE = {
    "textvqa": "textual",
    "docvqa": "textual",
    "infographicsvqa": "textual",
    "ocrvqa": "textual",
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
    mode: str, model_family: str = MODEL_FAMILY,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    if mode not in HEAD_CONFIGS:
        raise ValueError(f"unknown mode {mode!r}")
    if model_family not in HEAD_CONFIGS[mode]:
        raise ValueError(
            f"no head config for model_family={model_family!r}, mode={mode!r}"
        )
    return HEAD_CONFIGS[mode][model_family]


def grounding_layer_set(
    model_family: str = MODEL_FAMILY,
    modes: Sequence[str] = ("textual", "natural"),
) -> List[int]:
    """Union of every layer used by any head config — the capture context
    hooks every layer the per-sample dispatch might query."""
    out = set()
    for mode in modes:
        g, s = get_head_config(mode, model_family)
        out.update(g.keys())
        out.update(s.keys())
    return sorted(out)


class _AttnCaptureCtx:
    """Forward hooks on selected layers' self_attn that stash the eager
    softmax attention weights into ``model._opl_attn_cache[layer_idx]``.

    Weights are kept on-GPU, detached, in their native dtype — one
    [B, 16, S, S] tensor per hooked layer per microbatch (~10-20 MB at
    stage1 sequence lengths)."""

    def __init__(self, model: torch.nn.Module, layer_indices: Sequence[int]):
        self.model = model
        self.layer_indices = [int(i) for i in layer_indices]
        self.hooks = []

    def __enter__(self):
        layers = self.model.model.language_model.layers
        cache: Dict[int, torch.Tensor] = {}
        self.model._opl_attn_cache = cache

        def make_hook(idx):
            def _hook(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    cache[idx] = out[1].detach()
                return out
            return _hook

        for i in self.layer_indices:
            self.hooks.append(
                layers[i].self_attn.register_forward_hook(make_hook(i))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        return False


def register_attention_capture(
    model: torch.nn.Module, layer_indices: Sequence[int],
) -> _AttnCaptureCtx:
    return _AttnCaptureCtx(model, layer_indices)


@torch.no_grad()
def online_pseudo_label_for_sample(
    *,
    model: torch.nn.Module,
    sample_idx: int,
    response_rows: slice,
    image_cols: slice,
    flat_idx: torch.Tensor,
    grid_hw: Tuple[int, int],
    original_image_size: Optional[Tuple[int, int]] = None,
    mode: str = "textual",
    model_family: str = MODEL_FAMILY,
    create_kwargs: Optional[dict] = None,
) -> Optional[dict]:
    """v1 (mean-over-response-tokens) online pseudo-label for one sample.

    Reads ``model._opl_attn_cache`` (filled during the main forward by
    ``register_attention_capture``), means the configured grounding heads'
    resp→image attention over all response rows, scatters the image-token
    columns into the (gh, gw) grid via ``flat_idx``, and runs the shared
    ``create_pseudo_labels`` thresholding with the outer-ring sink band.

    Returns ``{"labels": [gh, gw], "labels_tokens": [N_img], "stats": ...}``
    with values in {1, 0, -100}, or ``None`` when no attention is available.
    """
    cache = getattr(model, "_opl_attn_cache", None)
    if cache is None:
        raise RuntimeError(
            "model._opl_attn_cache missing — wrap the forward in "
            "register_attention_capture(...)."
        )
    grounding_heads, sink_heads = get_head_config(mode, model_family)
    gh, gw = grid_hw
    n_img = image_cols.stop - image_cols.start
    flat_np = flat_idx.detach().cpu().numpy()

    def _agg_heads(heads_cfg: Dict[int, List[int]]) -> Optional[np.ndarray]:
        acc = None
        total = 0
        for lid, heads in heads_cfg.items():
            attn = cache.get(int(lid))
            if attn is None:
                continue
            # [B, H, S, S] → resp rows x image cols, mean over resp rows
            rows = attn[sample_idx, :, response_rows, image_cols].float()
            rows = rows.mean(dim=1)                        # [H, N_img]
            for h in heads:
                r = rows[h]
                acc = r if acc is None else acc + r
                total += 1
        if acc is None or total == 0:
            return None
        tok = (acc / total).cpu().numpy()                  # [N_img]
        grid = np.zeros(gh * gw, dtype=np.float32)
        grid[flat_np] = tok
        return grid.reshape(gh, gw)

    grounding_2d = _agg_heads(grounding_heads)
    if grounding_2d is None:
        return None

    # Sink band: outer ring of the grid (SINK_RULE == "outer_ring"),
    # optionally unioned with configured sink heads (none for Gemma).
    if SINK_RULE == "outer_ring":
        sink_2d = outer_ring_mask(gh, gw).astype(np.float32)
    else:
        sink_2d = np.zeros((gh, gw), dtype=np.float32)
    sink_from_heads = _agg_heads(sink_heads) if sink_heads else None
    if sink_from_heads is not None:
        sink_2d = np.maximum(sink_2d, sink_from_heads)

    ck = dict(create_kwargs or {})
    ck.setdefault("original_image_size", original_image_size)
    result = create_pseudo_labels(
        sink_attn=sink_2d,
        grounding_attn_o2i=grounding_2d,
        **ck,
    )
    labels_grid = np.asarray(result["labels"]).reshape(gh, gw)
    labels_tokens = labels_grid.reshape(-1)[flat_np]       # image-token order
    if labels_tokens.shape[0] != n_img:
        return None
    stats = dict(result.get("stats", {}))
    stats["grd_max"] = float(grounding_2d.max())
    return {
        "labels": labels_grid,
        "labels_tokens": labels_tokens,
        "grounding_attn": grounding_2d,
        "stats": stats,
    }


@torch.no_grad()
def online_single_region_label_for_sample(
    *,
    model: torch.nn.Module,
    sample_idx: int,
    response_rows: slice,
    image_cols: slice,
    flat_idx: torch.Tensor,
    grid_hw: Tuple[int, int],
    mode: str = "textual",
    model_family: str = MODEL_FAMILY,
    skip_leading_tokens: int = 0,
    max_label_tokens: int = 256,
    ratio_thresh: float = 3.0,
    peak_fraction: float = 0.3,
    extract_smooth_sigma: float = 0.0,
    halo_smooth_sigma: float = 1.0,
    halo_thresh_frac: float = 0.3,
    empty_sample_ignore: bool = True,
) -> Optional[dict]:
    """v2 (per-token single-region union) online label for one sample.

    Port of ``qwen_src.qwen3_5.online_pseudo_label.online_single_region_label_for_sample``
    to the Gemma-4 attention cache: the configured grounding heads' resp->image
    attention is averaged PER RESPONSE TOKEN (no mean over tokens), the
    ``[Visual Evidence]\\n`` prefix is skipped, tokens are capped, each token's
    map is scattered into the (gh, gw) grid via ``flat_idx`` and the shared
    single-region extractor (peak/mean ratio gate + threshold + 1 connected
    component per token, union, Gaussian halo = ignore) builds the label.

    Returns ``{"labels": [gh, gw], "labels_tokens": [N_img], "stats": ...}``
    with values in {1, 0, -100}, or ``None`` when no attention is available.
    """
    from qwen_src.gemma4_unified.single_region_label import single_region_roi_label

    cache = getattr(model, "_opl_attn_cache", None)
    if cache is None:
        raise RuntimeError(
            "model._opl_attn_cache missing — wrap the forward in "
            "register_attention_capture(...)."
        )
    grounding_heads, _sink = get_head_config(mode, model_family)
    gh, gw = grid_hw
    n_img = image_cols.stop - image_cols.start
    flat = flat_idx.detach().long()

    acc = None
    total = 0
    for lid, heads in grounding_heads.items():
        attn = cache.get(int(lid))
        if attn is None:
            continue
        rows = attn[sample_idx, :, response_rows, image_cols].float()  # [H, T, N_img]
        for h in heads:
            r = rows[h]
            acc = r if acc is None else acc + r
            total += 1
    if acc is None or total == 0:
        return None
    per_tok = acc / total                                   # [T, N_img]
    if skip_leading_tokens > 0:
        per_tok = per_tok[int(skip_leading_tokens):]
    if max_label_tokens and per_tok.shape[0] > int(max_label_tokens):
        per_tok = per_tok[: int(max_label_tokens)]
    if per_tok.shape[0] == 0 or per_tok.shape[1] != n_img:
        return None
    grid = per_tok.new_zeros((per_tok.shape[0], gh * gw))
    grid[:, flat.to(grid.device)] = per_tok
    grid = grid.view(per_tok.shape[0], gh, gw)

    lbl = single_region_roi_label(
        grid,
        skip_leading_tokens=0,
        ratio_thresh=ratio_thresh,
        peak_fraction=peak_fraction,
        extract_smooth_sigma=extract_smooth_sigma,
        halo_smooth_sigma=halo_smooth_sigma,
        halo_thresh_frac=halo_thresh_frac,
        empty_sample_ignore=empty_sample_ignore,
    )                                                       # [gh, gw] long
    labels_grid = lbl.detach().cpu().numpy()
    labels_tokens = labels_grid.reshape(-1)[flat.cpu().numpy()]
    fg = labels_grid == 1
    n_fg = int(fg.sum())
    fg_bbox = None
    if n_fg > 0:
        ys, xs = np.nonzero(fg)
        fg_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "labels": labels_grid,
        "labels_tokens": labels_tokens,
        "grounding_attn": grid.mean(dim=0).detach().cpu().numpy(),
        "stats": {
            "num_fg": n_fg,
            "num_bg": int((labels_grid == 0).sum()),
            "num_ignore": int((labels_grid == -100).sum()),
            "n_tokens": int(per_tok.shape[0]),
            "fg_bbox": fg_bbox,
            "grd_max": float(grid.max()),
        },
    }
