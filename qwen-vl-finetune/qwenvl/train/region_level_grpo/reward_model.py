"""Frozen-LLM teacher-forced log-prob reward computation.

The reward for region-level GR-REINFORCE is the log-probability of the
gold answer under a masked image, computed in teacher-forcing mode (no
decoding). For a batch of K masked images sharing the same prompt:

    R_k = log p(a* | I_masked_k, q)

In Phase B.1 the LLM backbone is frozen, so this module re-uses the
policy model itself for reward forward passes. The twig heads are
not invoked because reward computation only needs the LM head logits.

Usage::

    rm = RewardModel(model, processor)
    log_probs = rm.compute_logprobs(
        masked_images=[pil1, pil2, pil3, pil4],
        question="What color is the car?",
        gold_answer="red",
    )  # tensor of shape (4,)
"""

from __future__ import annotations

# --- Q-Zoom centralized env-knob accessor (Phase A) ---
try:
    from qwen_src.qzoom_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from qzoom_config import getenv as qz_getenv
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


# Matches the inference-time chat-class system message convention.
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

# Qwen vision sentinels: <|image_pad|> is what the processor expands
# to per-token image patches. Wrap with <|vision_start|>/<|vision_end|>.
_QWEN_VISION_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"


def _detect_qwen35(model, model_family=None) -> bool:
    """True iff the reward model is a Qwen3.5-family model — the ONLY family
    whose chat template carries the `<think>\\n\\n</think>\\n\\n`
    (enable_thinking=False) prefix. Qwen2.5-VL and Qwen3-VL have no thinking
    template, so the prefix is OOD for them. Prefers an explicit ``model_family``
    string; otherwise auto-detects from the model class name / config."""
    if model_family:
        f = str(model_family).lower().replace(".", "_").replace("-", "_")
        if "qwen3_5" in f or "qwen35" in f:
            return True
        if ("qwen2_5" in f or "qwen2" in f or "qwen3_vl" in f
                or "qwen3vl" in f):
            return False
    name = type(model).__name__.lower()
    mt = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    return "qwen3_5" in name or "qwen3_5" in mt


class RewardModel:
    """Wraps a frozen LLM for teacher-forced log-prob computation.

    The same model instance can be used for the policy forward and
    the reward forward (LLM is frozen during Phase B.1, so they
    share weights). Callers must ensure ``model.eval()`` and run
    inside ``torch.no_grad()`` -- :meth:`compute_logprobs` enforces
    the latter.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        image_token_template: str = _QWEN_VISION_TEMPLATE,
        disable_thinking_prefix: bool = False,
        skip_trailing_eos: bool = False,
        model_family: Optional[str] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        # Register the reward image_processor so _build_masked_pils caps source
        # images at this run's max_pixels (= max visual tokens × patch²), the
        # exact budget masked crops are processed at — lossless + minimal CPU.
        try:
            from qwenvl.train.region_level_grpo import trainer as _t
            _ip = getattr(processor, "image_processor", None)
            if _ip is not None:
                _t.set_reward_mask_image_processor(_ip)
        except Exception:
            pass
        if hasattr(processor, "tokenizer"):
            self.tokenizer = processor.tokenizer
        else:  # some processors expose tokenizer directly
            self.tokenizer = processor
        self.system_message = system_message
        self.image_token_template = image_token_template
        # Align teacher-forcing with the model's native generation format.
        # When True, prepend the official ``enable_thinking=False`` template
        # prefix (`<think>\\n\\n</think>\\n\\n`) to the assistant turn — see
        # Qwen3.5-VL's chat_template.jinja. Removes the structural mismatch
        # where the model wants to emit `<think>` first.
        self.disable_thinking_prefix = bool(disable_thinking_prefix)
        # The `<think>\\n\\n</think>\\n\\n` prefix is the Qwen3.5-VL
        # enable_thinking=False template and is ONLY in-distribution for the
        # Qwen3.5 family. Qwen2.5-VL / Qwen3-VL have NO thinking template, so
        # injecting it is OOD there. Gate the prefix on family: emit it ONLY
        # for Qwen3.5 (and only when disable_thinking_prefix is set); force
        # empty for q25 / q3-VL regardless of the flag. See _think_prefix().
        self.model_family = model_family
        self._is_qwen35 = _detect_qwen35(model, model_family)
        # Gemma-4: chat strings from the processor template (thinking off),
        # one image per sample passed as a nested list, gold closed by <turn|>.
        from qwenvl.train.region_level_grpo.gemma_support import is_gemma
        self._is_gemma = is_gemma(model_family) or is_gemma(model)
        # Exclude the closing `<|im_end|>` from the answer-mask so the log_p
        # reduction only spans the gold-answer content tokens. `<|im_end|>`
        # after a short answer carries noise about whether the model wanted
        # to keep talking, not signal about answer correctness.
        self.skip_trailing_eos = bool(skip_trailing_eos)

    def _think_prefix(self) -> str:
        """Assistant-turn prefix for the reward forward. The Qwen3.5-VL
        `<think>\\n\\n</think>\\n\\n` (enable_thinking=False) template is emitted
        ONLY for the Qwen3.5 family and ONLY when ``disable_thinking_prefix`` is
        set; for q25 / q3-VL it is forced empty (the prefix is OOD there)."""
        return ("<think>\n\n</think>\n\n"
                if (self.disable_thinking_prefix and self._is_qwen35) else "")

    # ----------------------------------------------------------------- public

    @torch.no_grad()
    def compute_logprobs(
        self,
        masked_images: Sequence,
        question: str,
        gold_answer: str,
        device: Optional[torch.device] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute log p(gold_answer | masked_image_k, q) for k in [K].

        Args:
            masked_images: list of K PIL images, all paired with the
                same question. Will be passed to the processor.
            question: user prompt text.
            gold_answer: target answer whose log-prob is the reward.
            device: device for the forward pass; defaults to
                ``next(self.model.parameters()).device``.
            reduction: ``"sum"`` for total log-prob (matches the
                policy-gradient theorem); ``"mean"`` for average per
                token (length-normalized; the RL recipe).

        Returns:
            Tensor of shape ``(K,)``: per-image log-prob of the gold
            answer tokens.
        """
        if reduction not in {"sum", "mean"}:
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")

        K = len(masked_images)
        if K == 0:
            target_device = device or next(self.model.parameters()).device
            return torch.empty((0,), device=target_device)

        if device is None:
            device = next(self.model.parameters()).device

        # 0. The policy model has SD-RPN's 2-stage ROI augmentation enabled
        #    (``roi_enable2stage=True`` and ``enable_twig=True``). For the
        #    reward forward we explicitly want a plain LLM forward on the
        #    *already-masked* image -- no further ROI cropping. Temporarily
        #    disable both flags and restore them after the forward so the
        #    policy path is unaffected.
        #
        #    Additionally, ``SKIP_POST_BRANCH=1`` (default) makes the
        #    model skip both post-twig main-path layers AND the lm_head
        #    matmul during ``self.training=True``, returning a [B, 1, V]
        #    placeholder logits tensor. We need the FULL lm_head output
        #    here to score the gold answer, so flip the model to eval()
        #    mode for the reward forward.
        lm_node = getattr(self.model, "model", self.model)
        lm_node = getattr(lm_node, "language_model", lm_node)
        saved_roi = getattr(lm_node, "roi_enable2stage", None)
        saved_twig = getattr(lm_node, "enable_twig", None)
        if saved_roi is not None:
            lm_node.roi_enable2stage = False
        if saved_twig is not None:
            lm_node.enable_twig = False
        was_training = self.model.training
        self.model.eval()

        try:
            return self._compute_logprobs_inner(
                masked_images, question, gold_answer, device, reduction,
            )
        finally:
            if saved_roi is not None:
                lm_node.roi_enable2stage = saved_roi
            if saved_twig is not None:
                lm_node.enable_twig = saved_twig
            if was_training:
                self.model.train()

    @torch.no_grad()
    def _compute_logprobs_inner(
        self,
        masked_images: Sequence,
        question: str,
        gold_answer: str,
        device: torch.device,
        reduction: str,
    ) -> torch.Tensor:
        """Internal: same logic as :meth:`compute_logprobs` but without
        the ``roi_enable2stage`` / ``enable_twig`` toggling (the public
        wrapper handles that).
        """
        K = len(masked_images)
        # 1. Build the chat-formatted text strings.
        text_full, text_prompt = self._format_chat_strings(question, gold_answer)

        # 2. Locate the answer-token span via text-only tokenization
        #    (no image expansion). The same number of trailing tokens
        #    appears at the end of the multimodal input_ids.
        full_ids_text = self.tokenizer(
            text_full, add_special_tokens=False
        ).input_ids
        prompt_ids_text = self.tokenizer(
            text_prompt, add_special_tokens=False
        ).input_ids
        n_answer_tokens_full = len(full_ids_text) - len(prompt_ids_text)
        if n_answer_tokens_full <= 0:
            raise ValueError(
                f"Could not locate answer span: full={len(full_ids_text)}, "
                f"prompt={len(prompt_ids_text)}, gold_answer={gold_answer!r}"
            )
        # Drop the trailing `<|im_end|>` token from the answer mask
        # (always the last token in text_full).
        n_skip_trailing = 1 if self.skip_trailing_eos else 0
        n_answer_tokens = n_answer_tokens_full - n_skip_trailing
        if n_answer_tokens <= 0:
            raise ValueError(
                f"After skipping {n_skip_trailing} trailing token(s), "
                f"answer span is empty; gold_answer={gold_answer!r}"
            )

        # 3. Process K rollouts (one image each).
        images_flat = list(masked_images)
        # Guard: Qwen smart_resize raises if a raw image side < patch*merge
        # factor (qwen2.5 = 28; a few pool images are tiny, e.g. 26px). Upscale
        # such images (preserving aspect) so the processor's resize-to-min_pixels
        # can proceed -- matches the resolution the policy/eval see. No-op for
        # normal images (min side >= 64).
        def _ensure_min_side(_im, _min_side=64):
            try:
                _w, _h = _im.size
            except Exception:
                return _im
            if min(_w, _h) >= _min_side:
                return _im
            from PIL import Image as _PILImage
            _sc = float(_min_side) / float(max(1, min(_w, _h)))
            return _im.resize(
                (max(_min_side, int(round(_w * _sc))),
                 max(_min_side, int(round(_h * _sc)))),
                _PILImage.BICUBIC,
            )
        images_flat = [_ensure_min_side(_im) for _im in images_flat]
        if getattr(self, "_is_gemma", False):
            from qwenvl.train.region_level_grpo.gemma_support import (
                gemma_processor_call,
            )
            inputs = gemma_processor_call(
                self.processor, [text_full] * K, images_flat)
        else:
            inputs = self.processor(
                text=[text_full] * K,
                images=images_flat,
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        # 4. Forward through the (frozen) model.
        # Memory:
        #  - ``use_cache=False``: no need to keep KV across layers (we
        #    do a single teacher-forced forward, not autoregressive
        #    generation).
        #  - Replace explicit ``log_softmax + gather`` with fused
        #    ``F.cross_entropy(reduction='none')``: the kernel computes
        #    log_softmax(logits[i])[label[i]] row-by-row without
        #    materializing the full ``(B, S-1, V)`` fp32 tensor.
        outputs = self.model(**inputs, use_cache=False)
        # logits: (K, S, V); shift by 1 for next-token prediction.
        logits = outputs.logits[:, :-1, :]   # bf16, (K, S-1, V)
        labels = inputs["input_ids"][:, 1:]  # (K, S-1)

        B, T, V = logits.shape
        # Chunk the cross_entropy by rollout so peak memory only depends on a
        # few rows, not K of them (REWARD_CE_CHUNK rows per chunk).
        ce_chunk = int(qz_getenv("REWARD_CE_CHUNK", "1") or "1")
        ce_chunk = max(1, min(ce_chunk, B))
        gathered = torch.empty(
            (B, T), dtype=torch.float32, device=logits.device,
        )
        for s in range(0, B, ce_chunk):
            e = min(s + ce_chunk, B)
            chunk_logits = logits[s:e].reshape(-1, V)   # (chunk*T, V) bf16
            chunk_labels = labels[s:e].reshape(-1)
            chunk_neg_log_p = F.cross_entropy(
                chunk_logits, chunk_labels, reduction="none",
            )  # (chunk*T,) fp32
            gathered[s:e] = (-chunk_neg_log_p).reshape(e - s, T)
            del chunk_logits, chunk_labels, chunk_neg_log_p
        # Free the (K, T, V) logits ASAP so the answer-mask + sum has room.
        del logits, outputs

        # 5. Build per-sample answer-mask: last n_answer_tokens of valid
        #    positions (right-padded sequences need attention_mask).
        attn = inputs.get("attention_mask")
        answer_mask = self._build_answer_mask(
            shape=gathered.shape,
            attention_mask=attn[:, 1:] if attn is not None else None,
            n_answer_tokens=n_answer_tokens,
            device=device,
            n_skip_trailing=n_skip_trailing,
        )

        if reduction == "sum":
            return (gathered * answer_mask.float()).sum(dim=-1)
        # mean: divide by per-sample answer-token count.
        denom = answer_mask.float().sum(dim=-1).clamp(min=1.0)
        return (gathered * answer_mask.float()).sum(dim=-1) / denom

    # --------------------------------------------------------------- internal

    def _format_chat_strings(
        self,
        question: str,
        gold_answer: str,
    ) -> tuple[str, str]:
        """Build ``(text_full, text_prompt)`` using the Qwen chat template.

        We construct strings manually to avoid version-dependent
        behavior of ``apply_chat_template`` and to ensure the
        ``<|image_pad|>`` placeholder is at a known position for
        processor expansion.
        """
        if getattr(self, "_is_gemma", False):
            from qwenvl.train.region_level_grpo.gemma_support import (
                gemma_chat_strings,
            )
            return gemma_chat_strings(self.processor, question, gold_answer)
        sys = self.system_message
        user_with_image = self.image_token_template + question
        # Official Qwen3.5-VL ``enable_thinking=False`` template --
        # ``<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}``.
        # Family-gated (q35 only).
        assistant_prefix = self._think_prefix()
        text_full = (
            f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\n{user_with_image}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}{gold_answer}<|im_end|>"
        )
        text_prompt = (
            f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\n{user_with_image}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}"
        )
        return text_full, text_prompt

    @staticmethod
    def _build_answer_mask(
        shape: tuple[int, int],
        attention_mask: Optional[torch.Tensor],
        n_answer_tokens: int,
        device: torch.device,
        n_skip_trailing: int = 0,
    ) -> torch.Tensor:
        """For each sample, mark the ``n_answer_tokens`` positions
        ending ``n_skip_trailing`` positions before the last valid
        position. With ``n_skip_trailing=0`` this is the last
        ``n_answer_tokens`` valid positions. With ``n_skip_trailing=1``
        we exclude the trailing ``<|im_end|>`` from the answer span so
        the reduction operates only on the gold-answer content tokens.

        The "answer" portion sits at the tail of input_ids, but right-
        padding can push the actual end inward; we use attention_mask
        to find the last valid position per sample.
        """
        K, S_minus_one = shape
        mask = torch.zeros((K, S_minus_one), dtype=torch.bool, device=device)
        if attention_mask is None:
            if n_skip_trailing > 0:
                end = S_minus_one - int(n_skip_trailing)
                start = end - int(n_answer_tokens)
                if start < 0:
                    start = 0
                if end > start:
                    mask[:, start:end] = True
            else:
                mask[:, -n_answer_tokens:] = True
            return mask

        # For each sample, valid positions are where attn==1; the
        # answer span is the last n_answer_tokens of those, optionally
        # shifted inward by n_skip_trailing.
        for k in range(K):
            valid_pos = attention_mask[k].nonzero(as_tuple=True)[0]
            if int(n_skip_trailing) > 0 and len(valid_pos) > int(n_skip_trailing):
                valid_pos = valid_pos[: -int(n_skip_trailing)]
            if len(valid_pos) >= n_answer_tokens:
                mask[k, valid_pos[-n_answer_tokens:]] = True
            elif len(valid_pos) > 0:
                # Truncation case -- mark all valid positions.
                mask[k, valid_pos] = True
        return mask


# ------------------------------------------------------------------ tests --

def _self_test_api_only() -> None:
    """API smoke test that doesn't require a real model.

    Verifies the chat-string formatting and the answer-span locator
    using a stub tokenizer. The full integration test (with a real
    LLM forward) should be run separately on a server.
    """

    class _StubTokenizer:
        """Minimal byte-level tokenizer for shape-checking."""

        def __call__(self, text, add_special_tokens=False):
            class _Out:
                pass

            o = _Out()
            o.input_ids = list(text.encode("utf-8"))
            return o

    class _StubProcessor:
        def __init__(self):
            self.tokenizer = _StubTokenizer()

        def __call__(self, text, images, return_tensors, padding):
            raise NotImplementedError("not exercised in API-only test")

    rm = RewardModel(
        model=None,  # type: ignore[arg-type]
        processor=_StubProcessor(),
    )

    text_full, text_prompt = rm._format_chat_strings(
        question="What color is the car?",
        gold_answer="red",
    )
    assert "system" in text_full
    assert "<|vision_start|><|image_pad|><|vision_end|>" in text_full
    assert text_prompt.endswith("<|im_start|>assistant\n")
    # Full has the answer + closing tag appended after prompt.
    assert text_full.startswith(text_prompt)
    suffix = text_full[len(text_prompt):]
    assert suffix == "red<|im_end|>"

    # Answer span = bytes("red<|im_end|>") = 13 bytes under stub tokenizer.
    n_answer_bytes = len("red<|im_end|>")
    full_bytes = len(text_full.encode("utf-8"))
    prompt_bytes = len(text_prompt.encode("utf-8"))
    assert full_bytes - prompt_bytes == n_answer_bytes, (
        f"answer-byte delta mismatch: {full_bytes - prompt_bytes} vs {n_answer_bytes}"
    )

    # Answer-mask builder logic (no model needed).
    K, S = 3, 10
    attn = torch.ones(K, S, dtype=torch.long)
    attn[1, -2:] = 0  # sample 1 has 2 trailing pad tokens
    n_answer = 4
    mask = RewardModel._build_answer_mask(
        shape=(K, S),
        attention_mask=attn,
        n_answer_tokens=n_answer,
        device=torch.device("cpu"),
    )
    # Sample 0: last 4 of 10.
    assert mask[0, -n_answer:].all() and not mask[0, :-n_answer].any()
    # Sample 1: last 4 of 8 (positions 4..7).
    assert mask[1, 4:8].all() and not mask[1, :4].any() and not mask[1, 8:].any()
    # Sample 2: same as sample 0.
    assert mask[2, -n_answer:].all() and not mask[2, :-n_answer].any()

    # skip_trailing_eos=1: span shifts inward by one position.
    mask_skip = RewardModel._build_answer_mask(
        shape=(K, S), attention_mask=attn, n_answer_tokens=n_answer,
        device=torch.device("cpu"), n_skip_trailing=1,
    )
    assert mask_skip[0, 5:9].all() and not mask_skip[0, 9] and not mask_skip[0, :5].any()

    # Mean-vs-sum reduction (numerical check on a fake gathered tensor).
    gathered = torch.full((K, S), -1.0)
    full_mask = torch.zeros(K, S, dtype=torch.bool)
    full_mask[:, -n_answer:] = True
    sum_lp = (gathered * full_mask.float()).sum(dim=-1)
    mean_lp = sum_lp / full_mask.float().sum(dim=-1).clamp(min=1.0)
    assert torch.equal(sum_lp, torch.full((K,), -float(n_answer)))
    assert torch.equal(mean_lp, torch.full((K,), -1.0))

    print("OK: API-only smoke test passed.")


if __name__ == "__main__":
    _self_test_api_only()
