"""In-repo SD-RPN heatmap runner for the Qwen families.

The RL pool builder (``data_prep/pre_rl_filter.py``) needs one thing from the
stage-1 checkpoint: the query-relevant RoI heatmap the RL trainer would see for
a ``(image, question)`` pair. Historically that came from a demo wrapper that
lives outside this repo; ``QwenHeatmapRunner`` provides it here instead, so the
released pipeline can regenerate the pools without extra code.

It mirrors :class:`~qwenvl.train.region_level_grpo.gemma_support.GemmaHeatmapRunner`:

* the checkpoint is loaded through :func:`train_phase_b1._load_policy_with_twig`
  (the exact flag set the RL trainer uses, including
  ``return_per_head_score=True``), so the heatmap is byte-for-byte the policy's
  own ``Z_theta`` at step 0;
* ``roi_score_query_mode="last_prompt"`` and ``roi_score_with_rope=True`` are
  the RL defaults (see ``PhaseB1Arguments``);
* one full chat input is built with a one-token dummy answer, so a
  ``labels != -100`` row exists — the modeling fork's per-head-score capture
  branch keys off it;
* :meth:`infer` returns ``prob_map`` (sigmoid), ``pred_map`` (raw logits) and
  ``feat_hw``, matching the Gemma runner's contract.

The family is auto-detected from the checkpoint config, so both Qwen3.5-VL and
Qwen2.5-VL load through the same helper.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch

# Shared tiny result container (prob_map / pred_map / feat_hw).
from qwenvl.train.region_level_grpo.gemma_support import _HeatmapResult

__all__ = ["QwenHeatmapRunner"]


# Per-family SD-RPN twig depth, used only when the checkpoint config does not
# carry ``twig_K`` / ``twig_T`` itself (it normally does).
_DEFAULT_TWIG_K = {"qwen3_5": 21, "qwen2_5_vl": 18}
_DEFAULT_TWIG_T = 3


class QwenHeatmapRunner:
    """``infer(image, question)`` -> the policy's mean-head RoI heatmap.

    Parameters mirror the pool filter's CLI: ``min_pixels`` / ``max_pixels``
    set the source-image budget the heatmap is computed at (they must match
    the RL run's budget, otherwise the cached ``feat_hw`` will not line up).
    """

    def __init__(
        self,
        pretrained: str,
        *,
        model_family: str = "qwen3_5",
        attn_implementation: str = "flash_attention_2",
        min_pixels: int = 262144,
        max_pixels: int = 589824,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        twig_K: Optional[int] = None,
        twig_T: Optional[int] = None,
        roi_loss: str = "bce",
        **_unused,
    ) -> None:
        import transformers

        from qwenvl.train.region_level_grpo.train_phase_b1 import (
            _load_policy_with_twig,
        )

        self.model_family = str(model_family)
        cfg = transformers.AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        if twig_K is None:
            twig_K = int(getattr(cfg, "twig_K", 0) or
                         _DEFAULT_TWIG_K.get(self.model_family, 21))
        if twig_T is None:
            twig_T = int(getattr(cfg, "twig_T", 0) or _DEFAULT_TWIG_T)

        model_args = SimpleNamespace(
            model_name_or_path=pretrained,
            twig_K=int(twig_K),
            twig_T=int(twig_T),
            roi_loss=str(roi_loss),
        )
        training_args = SimpleNamespace(
            cache_dir=None,
            bf16=(dtype == torch.bfloat16),
        )
        model = _load_policy_with_twig(
            model_args, training_args, attn_implementation=attn_implementation,
        )

        # RL-side scoring conventions (PhaseB1Arguments defaults): score at the
        # last prompt position, with RoPE applied inside the twig attention.
        for cfg_obj in (model.config, getattr(model.config, "text_config", None)):
            if cfg_obj is None:
                continue
            cfg_obj.roi_score_with_rope = True
            cfg_obj.roi_score_query_mode = "last_prompt"
        model.config.use_cache = False

        self.model = model.to(device).eval()
        self.model.requires_grad_(False)

        self.processor = transformers.AutoProcessor.from_pretrained(pretrained)
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.min_pixels = int(min_pixels)
            self.processor.image_processor.max_pixels = int(max_pixels)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "right"
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.device = self.model.device

        # Chat strings identical to the RL collator / reward model.
        from qwenvl.train.region_level_grpo.reward_model import RewardModel
        self._chat = RewardModel(
            model=self.model, processor=self.processor, model_family=self.model_family,
        )

    # ------------------------------------------------------------------ infer

    @torch.no_grad()
    def infer(self, image, question: str, heatmap_only: bool = True, **_unused):
        """Run one SD-RPN forward and return the mean-head heatmap."""
        # A one-token dummy answer: the query is the last PROMPT position, so
        # the content is irrelevant — it only has to produce a label row.
        text_full, text_prompt = self._chat._format_chat_strings(question, "x")
        n_full = len(self.tokenizer(text_full, add_special_tokens=False).input_ids)
        n_prompt = len(self.tokenizer(text_prompt, add_special_tokens=False).input_ids)
        n_ans = max(1, n_full - n_prompt)

        enc = self.processor(
            text=[text_full], images=[image], return_tensors="pt", padding=True,
        )
        enc = {k: (v.to(self.device) if hasattr(v, "to") else v)
               for k, v in enc.items()}

        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        labels = torch.full_like(input_ids, -100)
        if attn is not None:
            valid = attn[0].nonzero(as_tuple=True)[0]
            n_ans = min(n_ans, len(valid))
            labels[0, valid[-n_ans:]] = input_ids[0, valid[-n_ans:]]
        else:
            labels[0, -n_ans:] = input_ids[0, -n_ans:]
        enc["labels"] = labels
        # The Qwen forks only enter the per-sample capture loop when a
        # roi_target_map is supplied (its values are never read on the
        # return_per_head_score path); the RL collator attaches the same dummy.
        enc["roi_target_map"] = [torch.zeros(1, dtype=torch.float32)]

        out = self.model(**enc, use_cache=False)
        phs = getattr(out, "per_head_scores", None)
        if not phs:
            return _HeatmapResult(None, None, None)
        gh, gw = self._feat_hw(out, phs[0])
        z = phs[0].float().mean(dim=0).view(gh, gw).cpu()
        return _HeatmapResult(torch.sigmoid(z), z, (gh, gw))

    # -------------------------------------------------------------- internals

    @staticmethod
    def _feat_hw(outputs, per_head: torch.Tensor) -> Tuple[int, int]:
        feat_hw = getattr(outputs, "feat_hw", None)
        if feat_hw:
            gh, gw = feat_hw[0]
            return int(gh), int(gw)
        raise RuntimeError(
            "model output carries no feat_hw; the checkpoint was not loaded "
            "with return_per_head_score=True"
        )
