"""Frozen Phase-A SD-RPN reference twig (``online_p_ref``).

The KL anchor needs a Phase-A heatmap on the *same* grid as the live policy
heatmap. Bilinearly upsampling a cached ``p_ref`` captured at another
resolution would inject spurious smoothness that Phase-A wouldn't actually
produce at the new resolution.

This module sidesteps that by holding a frozen, deep-copied snapshot
of Phase-A's twig layers and re-running them at every training step
on the *same pre-twig hidden states the policy twig consumes*. The
result is a resolution-matched ``Z_ref`` for the KL anchor.

Design note: this class lives *outside* the modeling forks so the
per-head capture loop in the policy forward doesn't have to change. It
only depends on the model exposing a ``pre_twig_ctx`` dict and the last
twig layer's ``self_attn.q_proj / k_proj / q_norm / k_norm /
num_key_value_groups`` interface. The per-sample QK^T computation here
mirrors the model's capture loop exactly to keep the policy / ref scores
commensurate.

Supported families: Qwen3.5-VL (gated q_proj, 1D RoPE) and Qwen2.5-VL
(non-gated q_proj, multimodal RoPE).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn


def _apply_rotary(q_gated: bool, q, k, cos, sin):
    """RoPE application matching the Qwen3.5 policy modeling fork.

    Only the gated (Qwen3.5) 1D-RoPE convention is supported here; the
    Qwen2.5-VL family uses multimodal RoPE and is dispatched separately in
    :meth:`RefTwigModule.compute_per_head_scores`.
    """
    if not q_gated:
        raise RuntimeError(
            "RefTwigModule: non-gated 1D-RoPE twig (Qwen3-VL) is not supported "
            "in this release; only Qwen3.5 (gated) and Qwen2.5-VL (M-RoPE) are."
        )
    from qwen_src.qwen3_5.modeling_qwen3_5_batch import apply_rotary_pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)


class RefTwigModule(nn.Module):
    """Wraps a frozen deep-copy of Phase-A's twig layers + a helper
    to compute per-head response→image attention scores per sample,
    identical to the policy's capture loop in the modeling forward.
    """

    def __init__(self, source_twig_layers: nn.ModuleList):
        super().__init__()
        self.twig_layers = copy.deepcopy(source_twig_layers)
        for p in self.twig_layers.parameters():
            p.requires_grad = False
        # --- Family detection: gated (qwen3_5) vs non-gated (qwen2.5-VL) q_proj.
        # qwen3_5's q_proj emits ``num_heads * head_dim * 2`` (query + gate,
        # chunked); qwen2.5-VL's emits ``num_heads * head_dim`` (no gate). The
        # policy modeling forward chunks for qwen3_5 and does NOT chunk for
        # qwen2.5, so the ref twig must mirror the same convention to keep
        # the policy / ref per-head scores commensurate.
        _last = self.twig_layers[-1]
        _attn = _last.self_attn
        _hd = _attn.head_dim
        _q_out = _attn.q_proj.weight.shape[0]
        _kv_groups = getattr(_attn, "num_key_value_groups", 1)
        _n_kv = _attn.k_proj.weight.shape[0] // _hd
        _n_heads_expected = _n_kv * _kv_groups
        self._q_gated = (_q_out == 2 * _n_heads_expected * _hd)
        # Sanity: non-gated must be exactly num_heads * head_dim.
        if not self._q_gated and _q_out != _n_heads_expected * _hd:
            # Fall back to gated if the non-gated shape doesn't line up
            # either (unknown family) — preserves legacy qwen3_5 behavior.
            self._q_gated = True
        # qwen2.5-VL family: multimodal RoPE (rope_scaling["mrope_section"])
        # instead of standard 1D RoPE, and decoder layers take ``past_key_value``
        # (singular) + return a tuple. The ref twig must mirror both to stay
        # commensurate with the qwen2.5 policy.
        _rs = getattr(_attn, "rope_scaling", None) or {}
        self._mrope = isinstance(_rs, dict) and ("mrope_section" in _rs)
        # Force eager attention on the deep-copied layers. flash-attn's
        # varlen path triggers a CUDA device-side assert when called
        # from a frozen, no_grad context on layers that were originally
        # configured with ``flash_attention_2`` — likely because the
        # cu_seqlens / unpadding logic doesn't survive deep-copy cleanly.
        # Eager attention is slower but the ref twig only runs a handful
        # of layers (twig_T=3) on a single mini-batch per step, so the
        # overhead is negligible compared to the policy forward.
        for layer in self.twig_layers:
            try:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"
            except Exception:  # noqa: BLE001
                pass
        self.eval()  # disable dropout etc. — ref runs frozen

    @torch.no_grad()
    def _compute_twig_hidden(
        self,
        pre_twig_ctx: Dict[str, Any],
        **layer_kwargs,
    ) -> torch.Tensor:
        """Run the (frozen) ref twig layers on the policy-twig's input.

        Family-aware stack depth, mirroring each policy's modeling forward:

        * gated (qwen3_5): the inner model runs the FULL twig stack and
          stores that as ``twig_hidden_states``; the policy then applies
          ``last.input_layernorm`` to it. So the ref runs ALL twig layers.
        * non-gated (qwen2.5-VL): the inner model runs ``twig_layers[:-1]``
          (``rpn_hidden_states``) and the policy applies the LAST layer's
          ``input_layernorm`` + q/k projections directly to that — the last
          layer is NEVER run as a full decoder block. So the ref runs only
          ``twig_layers[:-1]``.

        In BOTH cases ``compute_per_head_scores`` then applies
        ``last.input_layernorm`` + last-layer q/k to the result, so the two
        families share the QK^T scoring tail while differing only in how many
        full twig decoder layers feed it — exactly matching the policy.
        """
        hs = pre_twig_ctx["hidden_states"]
        # The ref twig forces EAGER attention (above), and eager does
        # ``attn_weights + attention_mask`` directly -> it needs a 4D additive
        # mask. The captured ``attention_mask`` is the parent's flash mask (raw
        # 2D under padding, or None) which eager can't use. When the batch has
        # padding (bs>1 + variable image grids) build a 4D (B,1,1,S) additive
        # PADDING mask from the raw 2D mask passed via ``raw_attn_mask``.
        # Padding-only (bidirectional) matches the bs=1 / uniform path, where
        # the eager ref twig gets None (no mask).
        _raw = pre_twig_ctx.get("raw_attn_mask", None)
        amask = pre_twig_ctx["attention_mask"]
        if _raw is not None and _raw.dim() == 2 and not bool(_raw.all()):
            _B, _S = _raw.shape
            _mn = torch.finfo(hs.dtype).min
            amask = torch.zeros(
                (_B, 1, 1, _S), dtype=hs.dtype, device=hs.device
            ).masked_fill(~_raw.to(torch.bool)[:, None, None, :], _mn)
        # Pass ``cache_position`` through when the capturing model supplied it
        # (None is a safe default for both families).
        _extra = {}
        if "cache_position" in pre_twig_ctx and pre_twig_ctx["cache_position"] is not None:
            _extra["cache_position"] = pre_twig_ctx["cache_position"]
        # Non-gated (qwen2.5-VL): run all but the last twig layer. Gated
        # (qwen3_5): run the full stack.
        _layers = self.twig_layers if self._q_gated else self.twig_layers[:-1]
        for layer in _layers:
            if self._mrope:
                # qwen2.5-VL decoder layer: ``past_key_value`` (singular) +
                # ``output_attentions``; returns a tuple (hidden, ...).
                _out = layer(
                    hs,
                    attention_mask=amask,
                    position_ids=pre_twig_ctx["position_ids"],
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    position_embeddings=pre_twig_ctx["position_embeddings"],
                    **_extra,
                )
                hs = _out[0] if isinstance(_out, (tuple, list)) else _out
            else:
                hs = layer(
                    hs,
                    position_embeddings=pre_twig_ctx["position_embeddings"],
                    attention_mask=amask,
                    position_ids=pre_twig_ctx["position_ids"],
                    past_key_values=None,
                    use_cache=False,
                    **_extra,
                    **layer_kwargs,
                )
        return hs

    def _project_qk(self, normed: torch.Tensor):
        """Compute (q, k) from the frozen last twig layer, family-aware.

        Returns ``q, k`` each shaped ``(B, num_heads_or_kv, L, head_dim)``
        (k uses the KV-head count; GQA repeat is applied by the caller).
        qwen3_5 chunks the gated q_proj (query+gate) and keeps query only;
        qwen2.5-VL's q_proj is non-gated and reshaped directly. This mirrors
        the policy modeling forward exactly so policy/ref scores match.
        """
        last = self.twig_layers[-1]
        B = normed.shape[0]
        L = normed.shape[1]
        head_dim = last.self_attn.head_dim
        if self._q_gated:
            # qwen3_5: (..., H, 2*d) -> chunk into (query, gate); keep query.
            q_full = last.self_attn.q_proj(normed).view(B, L, -1, head_dim * 2)
            q, _gate = torch.chunk(q_full, 2, dim=-1)
            q = q.transpose(1, 2)
        else:
            # non-gated (..., H, d) -> straight reshape, no chunk.
            q = last.self_attn.q_proj(normed).view(B, L, -1, head_dim).transpose(1, 2)
        k = last.self_attn.k_proj(normed).view(B, L, -1, head_dim).transpose(1, 2)
        if hasattr(last.self_attn, "q_norm"):
            q = last.self_attn.q_norm(q)
        if hasattr(last.self_attn, "k_norm"):
            k = last.self_attn.k_norm(k)
        return q, k

    @torch.no_grad()
    def compute_per_head_scores(
        self,
        pre_twig_ctx: Dict[str, Any],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        image_token_id: int,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
        apply_rope: bool = False,
        query_mode: str = "response_tokens",
        **layer_kwargs,
    ) -> List[Optional[torch.Tensor]]:
        """Replicate the policy's per-sample QK^T capture using the ref twig.

        Returns: list of length B; each entry is the per-head score tensor
        ``(num_heads, k_img_b)`` for that sample (matching the policy's
        ``per_head_scores`` shape), or ``None`` for samples with no visual
        tokens or no response tokens.
        """
        twig_hidden = self._compute_twig_hidden(pre_twig_ctx, **layer_kwargs)
        B, _T, _D = twig_hidden.shape

        last = self.twig_layers[-1]
        normed = last.input_layernorm(twig_hidden)
        head_dim = last.self_attn.head_dim
        # Family-aware q/k (gated chunk for qwen3_5, straight reshape otherwise).
        q, k = self._project_qk(normed)
        # Apply RoPE — gated by caller (matches the modeling-side
        # ``config.roi_score_with_rope`` flag).
        if apply_rope:
            _pe = pre_twig_ctx.get("position_embeddings")
            if _pe is not None:
                _cos, _sin = _pe
                if self._mrope:
                    from qwen_src.qwen2_5_vl.modeling_qwen2_5_vl_batch import (
                        apply_multimodal_rotary_pos_emb,
                    )
                    q, k = apply_multimodal_rotary_pos_emb(
                        q, k, _cos, _sin, pre_twig_ctx["mrope_section"],
                    )
                else:
                    q, k = _apply_rotary(self._q_gated, q, k, _cos, _sin)

        visual_mask = (input_ids == image_token_id) if image_token_id is not None else None

        per_sample: List[Optional[torch.Tensor]] = []
        for b in range(B):
            if visual_mask is None or not visual_mask[b].any():
                per_sample.append(None)
                continue
            img_idx = visual_mask[b].nonzero(as_tuple=False).squeeze(-1)
            img_slice = slice(int(img_idx[0]), int(img_idx[-1]) + 1)
            resp_mask = (labels[b] != -100)
            if not resp_mask.any():
                per_sample.append(None)
                continue
            rdx = resp_mask.nonzero(as_tuple=False).squeeze(-1)
            # Query token set selection — mirrors the policy (modeling-side)
            # ``roi_score_query_mode`` so the ref tracks the same query the
            # policy uses. ``last_prompt`` = the single position just before
            # the first response token (the inference-path proxy query).
            if str(query_mode) == "last_prompt":
                _lp = max(0, int(rdx[0]) - 1)
                resp_slice = slice(_lp, _lp + 1)
            else:
                resp_slice = slice(int(rdx[0]), int(rdx[-1]) + 1)

            n_groups = last.self_attn.num_key_value_groups
            k_b = k[b:b + 1]
            if n_groups > 1:
                k_b = k_b.repeat_interleave(n_groups, dim=1)
            kb = k_b[:, :, img_slice, :]
            qb = q[b:b + 1, :, resp_slice, :]
            score = torch.matmul(qb, kb.transpose(-1, -2)) / math.sqrt(head_dim)
            per_sample.append(score.squeeze(0).mean(dim=1))
        return per_sample


def load_ref_twig_from_policy(model: nn.Module) -> RefTwigModule:
    """Build a frozen RefTwigModule from a freshly-loaded Phase-A policy.

    Call this BEFORE Phase-B.1 training starts modifying the policy's
    twig layers — the deep-copy captures their initial state. After this,
    the policy's twig_layers can train freely; the snapshot stays at
    Phase-A weights.

    Locates twig_layers via the standard ``model.model.language_model``
    path, with a named-modules fallback for any layout variation — mirrors
    ``_freeze_for_phase_b1``.
    """
    # Gemma-4 (encoder-free, per-layer-type rope/mask): dedicated module.
    from qwenvl.train.region_level_grpo.gemma_support import (
        is_gemma, load_gemma_ref_twig,
    )
    if is_gemma(model):
        return load_gemma_ref_twig(model)
    lm_node = getattr(model, "model", model)
    lm_node = getattr(lm_node, "language_model", lm_node)
    twig_layers = getattr(lm_node, "twig_layers", None)
    if twig_layers is None:
        # Fallback: scan every submodule for one exposing ``twig_layers``
        # (robust to nesting differences between families).
        for _n, _m in model.named_modules():
            _tl = getattr(_m, "twig_layers", None)
            if _tl is not None:
                twig_layers = _tl
                break
    if twig_layers is None:
        raise RuntimeError(
            "load_ref_twig_from_policy: no module with twig_layers found; "
            "was the checkpoint trained with --enable_twig?"
        )
    ref = RefTwigModule(twig_layers)
    ref.requires_grad_(False)
    return ref
