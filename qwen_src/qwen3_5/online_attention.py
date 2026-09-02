"""On-the-fly response→image attention for Qwen3.5 (hybrid-attention LLM).

Two Qwen3.5-specific points:

1. **Hybrid attention**. Qwen3.5 alternates ``linear_attention`` and
   ``full_attention`` decoder layers (every 4th is full). Only
   full-attention layers expose ``self_attn`` with q/k projections — the
   linear_attention layers replace it with ``Qwen3_5GatedDeltaNet`` and
   have no Q/K we can recompute. Grounding layers must therefore be
   full-attn layer indices (e.g. 3, 7, 11, 15, 19, 23, 27, 31 in the 9B).

2. ``Qwen3_5ForConditionalGeneration`` wraps the text decoder under
   ``model.language_model``; the layer resolution helper drills down
   accordingly.

Used by:
- ``qwen_src/qwen3_5/online_pseudo_label.py`` (Phase-A online pseudo-label).
- the Phase-A training forward in ``modeling_qwen3_5_batch.py``
  (``register_grounding_cache`` around the main forward).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .modeling_qwen3_5_batch import apply_rotary_pos_emb, repeat_kv


@dataclass
class GroundingCacheEntry:
    """One layer's captured pre-forward state. Detached / clone'd, safe to
    hold across the rest of the main forward without keeping autograd live."""
    hidden_states: torch.Tensor                  # [B, T, D] — pre-input_layernorm
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]  # (cos, sin) shared
    attention_mask: Optional[torch.Tensor]       # additive causal mask, or None
    position_ids: Optional[torch.LongTensor]


def _resolve_language_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the language-model decoder layer ModuleList for Qwen3.5."""
    for path in (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
        ("layers",),
    ):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and isinstance(cur, torch.nn.ModuleList):
            return cur
    raise AttributeError(
        "Could not locate language_model.layers on the given module."
    )


def _layer_types(model: torch.nn.Module) -> Optional[List[str]]:
    """Pull layer_types from the (possibly nested) text config. Returns
    None if the model doesn't expose hybrid-attention metadata, in which
    case all layers are assumed full-attention (e.g. Qwen3-VL)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    lt = getattr(cfg, "layer_types", None)
    if lt is None and hasattr(cfg, "text_config"):
        lt = getattr(cfg.text_config, "layer_types", None)
    return list(lt) if lt is not None else None


@contextmanager
def register_grounding_cache(
    model: torch.nn.Module, layer_indices: Iterable[int],
):
    """Context manager. While active, every forward through ``model``
    populates ``model._grounding_cache: Dict[int, GroundingCacheEntry]``
    for the requested layers.

    Raises early if any requested layer is ``linear_attention`` — those
    decoder layers don't expose Q/K projections so attention recompute is
    impossible.
    """
    layers = _resolve_language_layers(model)
    layer_set = sorted({int(i) for i in layer_indices})
    if any(i < 0 or i >= len(layers) for i in layer_set):
        raise ValueError(
            f"layer_indices {layer_set} out of range "
            f"[0, {len(layers)})."
        )
    layer_types = _layer_types(model)
    if layer_types is not None:
        bad = [i for i in layer_set if layer_types[i] != "full_attention"]
        if bad:
            full_idxs = [i for i, t in enumerate(layer_types)
                         if t == "full_attention"]
            raise ValueError(
                f"Qwen3.5 SD-RPN: layers {bad} are not full_attention "
                f"(no Q/K to recompute). Available full_attention "
                f"layer indices: {full_idxs}"
            )

    cache: Dict[int, GroundingCacheEntry] = {}
    model._grounding_cache = cache  # exposed for the caller to read

    def _make_hook(idx: int):
        def hook(_module, args, kwargs):
            # Decoder-layer signature (Qwen3.5):
            # forward(hidden_states, position_embeddings, attention_mask=None,
            #         position_ids=None, ...)
            # Capture the FIRST invocation per layer per forward (matches
            # Qwen3-VL behavior — twig path may re-run the layer later on
            # a truncated sequence we don't want to overwrite the cache
            # with).
            if idx in cache:
                return
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = (
                args[1] if len(args) > 1
                else kwargs.get("position_embeddings")
            )
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            if position_embeddings is None:
                # full-attention layers always receive position_embeddings;
                # this path is defensive against a future signature drift.
                return
            cos, sin = position_embeddings
            cache[idx] = GroundingCacheEntry(
                hidden_states=hidden_states.detach(),
                position_embeddings=(cos.detach(), sin.detach()),
                attention_mask=(
                    attention_mask.detach()
                    if isinstance(attention_mask, torch.Tensor) else None
                ),
                position_ids=(
                    position_ids.detach()
                    if isinstance(position_ids, torch.Tensor) else None
                ),
            )
        return hook

    handles = []
    for i in layer_set:
        handles.append(
            layers[i].register_forward_pre_hook(_make_hook(i), with_kwargs=True)
        )

    try:
        yield cache
    finally:
        for h in handles:
            h.remove()
        # Don't drop model._grounding_cache — caller may still want to read.


@torch.no_grad()
def compute_response_to_image_attention(
    layer: torch.nn.Module,
    entry: GroundingCacheEntry,
    response_rows: slice,
    image_cols: slice,
) -> torch.Tensor:
    """Recompute post-softmax attention at one full-attention decoder
    layer (``input_layernorm → q/k_proj → q/k_norm → RoPE → softmax(QK)``),
    materialising only the response query rows.

    Returns post-softmax attention sliced to response→image,
    ``[B, num_attention_heads, T_resp, T_img]`` (fp32).
    """
    self_attn = layer.self_attn
    head_dim = self_attn.head_dim
    n_groups = self_attn.num_key_value_groups

    h = layer.input_layernorm(entry.hidden_states)
    B, T, _ = h.shape

    # Qwen3.5 q_proj is "gated": output reshapes to (..., H, 2*head_dim)
    # which chunks into (query, gate). The gate is applied post-attention,
    # so for QK^T we only need ``query``.
    h_resp = h[:, response_rows, :]
    q_full = self_attn.q_proj(h_resp).view(B, h_resp.shape[1], -1, head_dim * 2)
    q, _gate = torch.chunk(q_full, 2, dim=-1)
    q = self_attn.q_norm(q).transpose(1, 2)             # [B, H, T_resp, d]

    k = self_attn.k_proj(h).view(B, T, -1, head_dim)
    k = self_attn.k_norm(k).transpose(1, 2)             # [B, Hkv, T, d]

    cos, sin = entry.position_embeddings
    if cos.dim() == 4 and cos.shape[0] == 3:
        cos = cos[0]; sin = sin[0]
    cos_q = cos[:, response_rows, :]
    sin_q = sin[:, response_rows, :]
    q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
    _, k = apply_rotary_pos_emb(k, k, cos, sin)

    k = repeat_kv(k, n_groups)                          # [B, H, T, d]

    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self_attn.scaling
    # [B, H, T_resp, T]

    neg_inf = torch.finfo(attn.dtype).min
    rs, re = response_rows.start, response_rows.stop
    full_causal = torch.triu(
        torch.full((T, T), neg_inf, device=attn.device, dtype=attn.dtype),
        diagonal=1,
    )
    causal_resp = full_causal[rs:re, :][None, None, :, :]
    additive = causal_resp
    if entry.attention_mask is not None:
        m = entry.attention_mask
        if m.dim() == 2:
            pad = (1.0 - m.to(attn.dtype)) * neg_inf
            additive = additive + pad[:, None, None, :]
        elif m.dim() == 4:
            if m.shape[-1] >= T and m.shape[-2] >= T:
                m = m[..., rs:re, :T]
            additive = additive + m.to(attn.dtype)
    attn = attn + additive
    attn = F.softmax(attn, dim=-1)

    return attn[:, :, :, image_cols]                    # [B, H, T_resp, T_img]
