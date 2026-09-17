"""Gemma-4 (Unified, encoder-free) support for region-level GR-REINFORCE.

Everything that is family-specific for the Gemma-4-12B-it SD-RPN policy lives
here so the trainer / reward model / collator only need a small dispatch:

* :func:`is_gemma` — family detection from a model / config / name.
* :func:`gemma_chat_strings` — ``(text_full, text_prompt)`` for the reward
  forward and the policy collator (chat template with ``enable_thinking=False``;
  the gold answer is closed by ``<turn|>``).
* :class:`GemmaRefTwigModule` — frozen Phase-A twig snapshot that re-runs the
  twig stack on the policy's fork input (``pre_twig_ctx``) and scores the
  image tokens from the last prompt token, per head, exactly like the policy
  capture (``modeling_gemma4_unified_batch._roi_per_head_scores``).
* :class:`GemmaHeatmapRunner` — ``infer(image, question, heatmap_only=True)``
  returning ``prob_map`` / ``pred_map`` / ``feat_hw`` (the heatmap-runner
  subset the pool filter uses).

Conventions (match the Gemma inference heatmap in ``roi_inference``):
query = last prefill (prompt) position, RoPE applied, score scaled by
``self_attn.scaling`` (1.0), grid (gh, gw) and cell order from
``image_position_ids``.
"""
from __future__ import annotations

import copy
from collections import UserDict
from typing import Any, Dict, List, Optional

import torch
from torch import nn

GEMMA_TURN_END = "<turn|>"
GEMMA_DEFAULT_TIER = 560


def is_gemma(obj) -> bool:
    """True for a Gemma-4 model, config, config-class name or family string."""
    if obj is None:
        return False
    if isinstance(obj, str):
        return "gemma" in obj.lower()
    name = type(obj).__name__.lower()
    if "gemma" in name:
        return True
    cfg = getattr(obj, "config", None)
    mt = str(getattr(cfg, "model_type", "") or getattr(obj, "model_type", "")).lower()
    return "gemma" in mt


def gemma_chat_strings(processor, question: str, gold_answer: str):
    """``(text_full, text_prompt)`` for one (question, gold) pair.

    ``text_prompt`` ends with ``<|turn>model\\n<|channel>thought\\n<channel|>``
    (the ``enable_thinking=False`` generation prefix, i.e. the last prompt
    token the SD-RPN query reads). ``text_full`` appends the gold answer and
    the closing ``<turn|>`` so the reward's ``skip_trailing_eos=1`` drops it.
    """
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    text_full = text_prompt + gold_answer + GEMMA_TURN_END
    return text_full, text_prompt


def gemma_processor_call(processor, texts: List[str], images: List, **kw):
    """Run the Gemma processor with one image per sample (nested list API)."""
    return processor(
        text=list(texts), images=[[im] for im in images],
        padding=True, return_tensors="pt", **kw,
    )


def set_gemma_tier(processor, max_soft_tokens: int) -> None:
    """Fix the visual token tier on the image processor (no per-call kwargs)."""
    processor.image_processor.max_soft_tokens = int(max_soft_tokens)


class GemmaRefTwigModule(nn.Module):
    """Frozen deep-copy of the Phase-A twig layers (Gemma-4 layout).

    ``compute_per_head_scores`` has the same signature / return convention as
    :class:`qwenvl.train.region_level_grpo.ref_twig.RefTwigModule` so the
    trainer's ``online_p_ref`` path is family-agnostic.
    """

    def __init__(self, source_twig_layers: nn.ModuleList):
        super().__init__()
        self.twig_layers = copy.deepcopy(source_twig_layers)
        for p in self.twig_layers.parameters():
            p.requires_grad = False
        self._q_gated = False          # informational (trainer proof print)
        self._mrope = False
        self.eval()

    @torch.no_grad()
    def _compute_twig_hidden(self, pre_twig_ctx: Dict[str, Any]) -> torch.Tensor:
        hs = pre_twig_ctx["hidden_states"]
        pe_all = pre_twig_ctx["position_embeddings_all"]
        mask_all = pre_twig_ctx["attention_mask_all"]
        types = pre_twig_ctx["twig_layer_types"]
        shared = UserDict()            # private KV-share dict, as in the policy
        for t, layer in enumerate(self.twig_layers):
            ltype = types[t]
            hs = layer(
                hs,
                shared_kv_states=shared,
                position_embeddings=pe_all[ltype],
                attention_mask=mask_all[ltype],
                position_ids=pre_twig_ctx["position_ids"],
                past_key_values=None,
            )
        return hs

    @torch.no_grad()
    def compute_per_head_scores(
        self,
        pre_twig_ctx: Dict[str, Any],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        image_token_id: int,
        image_grid_thw=None,
        spatial_merge_size: int = 2,
        apply_rope: bool = True,
        query_mode: str = "last_prompt",
        flat_idx_list: Optional[List[torch.Tensor]] = None,
        **_unused,
    ) -> List[Optional[torch.Tensor]]:
        from qwen_src.gemma4_unified.modeling_gemma4_unified_batch import (
            apply_rotary_pos_emb, repeat_kv,
        )
        twig_hidden = self._compute_twig_hidden(pre_twig_ctx)
        B, S, _ = twig_hidden.shape
        last = self.twig_layers[-1]
        attn = last.self_attn
        head_dim = attn.head_dim
        normed = last.input_layernorm(twig_hidden)
        q = attn.q_norm(attn.q_proj(normed).view(B, S, -1, head_dim))
        k = attn.k_norm(attn.k_proj(normed).view(B, S, -1, head_dim))
        if apply_rope and pre_twig_ctx.get("position_embeddings") is not None:
            cos, sin = pre_twig_ctx["position_embeddings"]
            q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
            k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        flat_list = flat_idx_list if flat_idx_list is not None \
            else pre_twig_ctx.get("flat_idx_list")
        visual_mask = input_ids == image_token_id
        out: List[Optional[torch.Tensor]] = []
        kept = 0   # index into flat_list (policy skipped samples w/o vis/resp)
        for b in range(B):
            if not visual_mask[b].any():
                out.append(None)
                continue
            idxs = visual_mask[b].nonzero(as_tuple=False).squeeze(-1)
            img_slice = slice(int(idxs[0]), int(idxs[-1]) + 1)
            resp_mask = labels[b] != -100
            if not resp_mask.any():
                out.append(None)
                continue
            rdx = resp_mask.nonzero(as_tuple=False).squeeze(-1)
            if str(query_mode) == "last_prompt":
                lp = max(0, int(rdx[0]) - 1)
                q_slice = slice(lp, lp + 1)
            else:
                q_slice = slice(int(rdx[0]), int(rdx[-1]) + 1)
            kb = k[b:b + 1]
            if attn.num_key_value_groups > 1:
                kb = repeat_kv(kb, attn.num_key_value_groups)
            kb = kb[:, :, img_slice, :]
            qb = q[b:b + 1, :, q_slice, :]
            score = torch.matmul(qb, kb.transpose(-1, -2)) * attn.scaling
            score = score.squeeze(0).mean(dim=1)          # [H, N_img]
            if flat_list is None or kept >= len(flat_list):
                out.append(None)
                continue
            flat_idx = flat_list[kept].to(score.device)
            kept += 1
            if flat_idx.numel() != score.shape[1]:
                out.append(None)
                continue
            n_cells = int(flat_idx.max()) + 1
            grid = score.new_zeros((score.shape[0], n_cells))
            grid[:, flat_idx] = score
            out.append(grid)
        return out


def load_gemma_ref_twig(model: nn.Module) -> GemmaRefTwigModule:
    lm = model.model.language_model
    ref = GemmaRefTwigModule(lm.twig_layers)
    ref.requires_grad_(False)
    return ref


class _HeatmapResult:
    def __init__(self, prob_map, pred_map, feat_hw):
        self.prob_map = prob_map
        self.pred_map = pred_map
        self.feat_hw = feat_hw


class GemmaHeatmapRunner:
    """Minimal ``QwenHeatmapRunner``-like runner for the pool filter.

    ``infer(image, question, heatmap_only=True)`` runs the SD-RPN twig on the
    stripped prompt (question only, chat template, thinking disabled) and
    returns the mean-head last-prompt-token heatmap: ``pred_map`` = raw logits
    ``(gh, gw)``, ``prob_map`` = sigmoid, ``feat_hw`` = ``(gh, gw)``.
    """

    def __init__(self, pretrained: str, *, max_soft_tokens: int = GEMMA_DEFAULT_TIER,
                 attn_implementation: str = "sdpa", device: str = "cuda:0",
                 dtype=torch.bfloat16, **_unused):
        import transformers
        from qwen_src.gemma4_unified.modeling_gemma4_unified_batch import (
            Gemma4UnifiedForConditionalGeneration,
        )
        cfg = transformers.AutoConfig.from_pretrained(pretrained)
        tc = cfg.get_text_config()
        tc.enable_twig = True
        tc.online_pseudo_label = False
        tc.enable_high_res = False
        cfg.return_per_head_score = True
        tc.return_per_head_score = True
        self.model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            pretrained, config=cfg, dtype=dtype, device_map=device,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.model.model.language_model._heatmap_only = True
        self.processor = transformers.AutoProcessor.from_pretrained(pretrained)
        set_gemma_tier(self.processor, max_soft_tokens)
        self.processor.tokenizer.padding_side = "right"
        self.device = self.model.device

    @torch.no_grad()
    def infer(self, image, question: str, heatmap_only: bool = True, **_unused):
        # A one-token dummy "response" so the capture finds a label row; the
        # query is the last PROMPT position, so the dummy content is irrelevant.
        text_full, text_prompt = gemma_chat_strings(self.processor, question, "x")
        enc = gemma_processor_call(self.processor, [text_full], [image])
        enc = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        n_prompt = len(self.processor.tokenizer(text_prompt, add_special_tokens=False).input_ids)
        n_full = len(self.processor.tokenizer(text_full, add_special_tokens=False).input_ids)
        n_ans = max(1, n_full - n_prompt)
        labels = torch.full_like(enc["input_ids"], -100)
        valid = enc["attention_mask"][0].nonzero(as_tuple=True)[0]
        labels[0, valid[-n_ans:]] = enc["input_ids"][0, valid[-n_ans:]]
        out = self.model(**enc, labels=labels, use_cache=False)
        phs = getattr(out, "per_head_scores", None)
        if not phs:
            return _HeatmapResult(None, None, None)
        gh, gw = out.feat_hw[0]
        z = phs[0].float().mean(dim=0).view(gh, gw).cpu()
        return _HeatmapResult(torch.sigmoid(z), z, (gh, gw))
