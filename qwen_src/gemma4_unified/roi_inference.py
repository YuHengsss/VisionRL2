"""Two-stage SD-RPN RoI inference for Gemma-4-12B-IT (stage1-trained twig).

Pipeline (PORT_PLAN inference half, pragmatic first implementation):
  1. source pass: native-aspect image @560 tier, chat-template prompt,
     ONE prefill forward (logits_to_keep=1) — the twig fork fires at prefill
     and returns twig_hidden_states + pre_twig_ctx;
  2. RPN heatmap: last twig layer's q·k (query = LAST prefill position, the
     SD-RPN inference convention) over image cols → 23x23 sigmoid map;
  3. box extraction: recipe "q25" or "q35" (ported conventions, see
     extract_box) with confidence roi_conf_thresh;
  4. crop from the ORIGINAL image (grid box → pixel box on the square canvas
     → intersect content region → original coords);
  5. EXPLICIT scaling (Gemma has no min_pixels knob): if the crop's native
     area < min_tier*48*48, bilinear-upscale by sqrt(target/area); the 560
     branch budget then caps it in the processor;
  6. answer pass: two-image prompt [source, crop, question] via the native
     multi-image path (crop gets its own BOI/EOI span, bidirectional block
     on sliding layers, crop-local coord embeddings — PORT_PLAN option (a)).
  Fallback: no box → plain single-image generate (baseline behavior).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from qwen_src.gemma4_unified.modeling_gemma4_unified_batch import (
    STAGE_ACC,
    _stage_resolve,
    apply_rotary_pos_emb,
    create_masks_for_vision_model,
    get_block_sequence_ids_for_mask,
    repeat_kv,
)


def _grab_stage_acc(timer, suffix: str):
    """Drain the in-forward stage accumulator (twig/gate/vision segments,
    GEMMA_STAGE_TIMING=1) into the current timer row, keyed per pass."""
    if not timer.enabled:
        return
    # The in-forward marks are CUDA events; read them here, once, rather than
    # synchronising at each mark inside the layer loop.
    try:
        _stage_resolve()
    except NameError:
        pass
    for k, v in list(STAGE_ACC.items()):
        kk = k[:-3] + suffix + "_ms" if k.endswith("_ms") else k + suffix
        timer.row[kk] = timer.row.get(kk, 0.0) + v
    STAGE_ACC.clear()


class StageTimer:
    """cuda-synced per-sample stage timer (env GEMMA_STAGE_TIMING=1).

    Rows are appended as json lines to GEMMA_TIMING_OUT; a .runinfo.json
    sidecar records the run configuration once."""

    def __init__(self, enabled: bool, out_path: Optional[str], runinfo: dict):
        self.enabled = enabled and out_path is not None
        self.out_path = out_path
        self.row = {}
        self._t = None
        if self.enabled:
            info_path = out_path + ".runinfo.json"
            if not os.path.exists(info_path):
                base = {
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "torch": torch.__version__,
                    "bs": 1,
                    "sync": "torch.cuda.synchronize",
                    "warmup_samples": 5,
                    "vllm": False,
                }
                base.update(runinfo or {})
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(info_path, "w") as f:
                    json.dump(base, f, indent=2)

    def start(self, **row_init):
        if not self.enabled:
            return
        STAGE_ACC.clear()
        self.row = dict(row_init)
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        self._t = self._t0

    def mark(self, key: str):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.row[key] = self.row.get(key, 0.0) + (now - self._t) * 1000.0
        self._t = now

    def finish(self, **extra):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.row["t_total_ms"] = (now - self._t0) * 1000.0
        self.row.update(extra)
        # Kept so the eval wrapper can copy it into the samples jsonl, where
        # it sits next to the doc_id instead of relying on write order.
        self.last_row = dict(self.row)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(self.row) + "\n")


def expand2square(img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


@torch.no_grad()
def compute_rpn_grid(model, out, input_ids, image_position_ids):
    """Last-position q·k RoI heatmap → (grid_logits [gh, gw], grid shape)."""
    twig_hidden = out.twig_hidden_states
    if twig_hidden is None:
        return None
    tm = model.model.language_model
    last_twig = tm.twig_layers[-1]
    attn = last_twig.self_attn
    head_dim = attn.head_dim
    B, S, _ = twig_hidden.shape

    normed = last_twig.input_layernorm(twig_hidden)
    q = attn.q_proj(normed).view(B, S, -1, head_dim)
    q = attn.q_norm(q)
    k = attn.k_proj(normed).view(B, S, -1, head_dim)
    k = attn.k_norm(k)
    ctx = out.pre_twig_ctx or {}
    if ctx.get("position_embeddings") is not None:
        cos, sin = ctx["position_embeddings"]
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    image_token_id = model.config.image_token_id
    img_mask = (input_ids[0] == image_token_id)
    idxs = img_mask.nonzero(as_tuple=False).squeeze(-1)
    if idxs.numel() == 0:
        return None
    img_slice = slice(int(idxs[0]), int(idxs[-1]) + 1)

    ipos = image_position_ids[0]
    valid = ipos[(ipos != -1).all(dim=-1)]
    gw = int(valid[:, 0].max()) + 1
    gh = int(valid[:, 1].max()) + 1
    flat_idx = (valid[:, 1] * gw + valid[:, 0]).long()

    kb = k
    if attn.num_key_value_groups > 1:
        kb = repeat_kv(kb, attn.num_key_value_groups)
    q_last = q[:, :, -1:, :]                       # last prefill position
    kb = kb[:, :, img_slice, :]
    score = torch.matmul(q_last, kb.transpose(-1, -2)) * attn.scaling
    score = score.mean(dim=1).squeeze(0).squeeze(0)  # [N_img]

    grid = torch.zeros(gh * gw, dtype=torch.float32, device=score.device)
    grid[flat_idx.to(score.device)] = score.float()
    return grid.view(gh, gw).cpu()


def extract_box(grid_logits: torch.Tensor, recipe: str, conf: float,
                ) -> Optional[Tuple[int, int, int, int]]:
    """Heatmap → grid box (r0, c0, r1, c1) inclusive, or None (skip stage 2).

    Faithful port of the qwen two-stage box extraction (recon 2026-08-14,
    qzoom-revision mm_utils.get_batched_sub_images_v2 / qwen_src/roi/
    crop_budget.py): RAW q·k logit map → sigmoid (exactly once) → sink
    suppression → gaussian blur(kernel=3, σ=1.0) → threshold → bbox of the
    UNION of above-threshold cells (no connected components), empty mask ⇒
    no augmentation. One crop per sample.

    Sink suppression: qwen zeroes the top-left attention-sink band
    ([:H//4, 0] and [0, :W//4], grids >256 cells only). Gemma's sink band
    is the OUTERMOST RING (head_config.py SINK_RULE, same convention the
    stage1 labels were built with) — we zero the ring instead.

    Recipes:
      q35      — fixed threshold, production default conf 0.10;
      q25      — same fixed-threshold skeleton, task-published conf values
                 (chartqa 0.125, docvqa/textvqa-class 0.15);
      q35_peak — qwen3.5 dynamic peak_ratio mode (min_gate 0.03,
                 ratio_thresh 3.0, thr = peak*0.3), conf arg ignored.
    """
    gh, gw = grid_logits.shape
    prob = torch.sigmoid(grid_logits.float())
    if gh * gw > 256:
        sink = torch.ones_like(prob)
        sink[0, :] = 0
        sink[-1, :] = 0
        sink[:, 0] = 0
        sink[:, -1] = 0
        prob = prob * sink
    blurred = TF.gaussian_blur(prob[None, None], kernel_size=3, sigma=1.0)[0, 0]

    if recipe in ("q35", "q25"):
        mask = blurred > conf
    elif recipe == "q35_peak":
        peak = blurred.max()
        mean = blurred.mean()
        if float(peak) < 0.03:                       # min_gate
            return None
        if float(peak / (mean + 1e-8)) < 3.0:        # ratio_thresh
            return None
        mask = blurred > (peak * 0.3)                # peak_fraction
    else:
        raise ValueError(f"unknown recipe {recipe!r}")
    if not mask.any():
        return None
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


class GemmaRoIPipeline:
    def __init__(self, model, processor, device, *, conf_thresh=0.10,
                 min_tier=280, recipe="q35", max_soft_tokens=560,
                 model_patch=48, debug_dir=None, kv_reuse=False,
                 gate_thresh=None,
                 crop_target_tok=256, crop_max_upscale_edge=3.0,
                 sparse_k_max=3.0, sparse_dilation=1, smooth_sigma="auto2"):
        self.model = model
        self.processor = processor
        self.tok = processor.tokenizer
        self.device = device
        self.conf = float(conf_thresh)
        self.min_tier = int(min_tier)
        # roi_sparse (paper-protocol crop rule + sparse crop encoding)
        self.crop_target_tok = int(crop_target_tok)
        self.crop_max_upscale_edge = float(crop_max_upscale_edge)
        self.sparse_k_max = float(sparse_k_max)
        self.sparse_dilation = int(sparse_dilation)
        self.smooth_sigma = str(smooth_sigma)
        self.recipe = recipe
        self.tier = int(max_soft_tokens)
        self.model_patch = model_patch
        self.debug_dir = debug_dir
        self.kv_reuse = bool(kv_reuse)
        # stage3 Need-Refine threshold for the gated_reuse mode
        self.gate_thresh = None if gate_thresh is None \
            else float(gate_thresh)
        # Gate early exit (2026-08-25): stop pass 1 at the twig/gate fork
        # depth (layer twig_K-1 = 26 of 48) instead of running the whole
        # stack before reading the gate. See answer_v2. GEMMA_GATE_EARLY_EXIT=0
        # restores the un-split pass-1 forward (the A/B kill switch).
        self.gate_early_exit = os.environ.get(
            "GEMMA_GATE_EARLY_EXIT", "1") == "1"
        try:
            self.fill = tuple(int(x * 255)
                              for x in processor.image_processor.image_mean)
        except Exception:
            self.fill = (0, 0, 0)
        self.n_aug = 0
        self.n_total = 0
        self.n_reuse_fallback = 0
        # Optional system turn (qwen roi-arm mining convention: the RoI
        # dump runs WITH 'You are a helpful assistant.'; eval paths leave
        # this None).
        self.system_text: Optional[str] = None
        self.timer = StageTimer(
            os.environ.get("GEMMA_STAGE_TIMING", "0") == "1",
            os.environ.get("GEMMA_TIMING_OUT"),
            runinfo={
                "recipe": recipe, "conf": self.conf, "min_tier": self.min_tier,
                "src_tier": self.tier, "kv_reuse": self.kv_reuse,
                "gate_early_exit": self.gate_early_exit,
                "attn": getattr(model.config, "_attn_implementation", "?"),
            },
        )

    # ------------------------------------------------------------------
    # v2 (timing/deployment) path: Q-Zoom INSERTION layout with optional
    # partial-prefill KV-cache reuse.
    #
    # Both RoI arms use the SAME token sequence: the squared-source pass-1
    # prompt with the RoI span inserted directly after the source image
    # span (identical to the two-image chat-template rendering, verified
    # by a prefix-identity assert). Arms differ only in how the sequence
    # is prefetched:
    #   roi_full  — full re-prefill of the inserted sequence;
    #   roi_reuse — pass-1 KV cache cropped to the prefix (…source EOI],
    #               then a PARTIAL prefill of [RoI span + question tail]
    #               with past_key_values. Prefix outputs are position-
    #               and content-identical in both arms (causal), sliding
    #               windows and the vision-block bidirectional overlay
    #               apply equally, so reuse is exact up to kernel-order
    #               numerics (acceptance: bit-identical answers on ~50
    #               samples, mirroring the Qwen 97.9% result).
    # ------------------------------------------------------------------

    def _eos_set(self):
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tok.eos_token_id
        if not isinstance(eos, (list, tuple)):
            eos = [eos]
        return {int(e) for e in eos if e is not None}

    @torch.no_grad()
    def _decode_greedy(self, cache, last_logits, max_new_tokens: int):
        eos = self._eos_set()
        ids = []
        tok = int(last_logits.argmax(-1))
        for _ in range(int(max_new_tokens)):
            ids.append(tok)
            if tok in eos:
                break
            out = self.model(
                input_ids=torch.tensor([[tok]], device=self.device),
                past_key_values=cache, use_cache=True, logits_to_keep=1,
            )
            cache = out.past_key_values
            tok = int(out.logits[0, -1].argmax(-1))
        return ids

    @torch.no_grad()
    def answer_v2(self, image: Image.Image, question: str, *,
                  max_new_tokens=64, mode: Optional[str] = None) -> str:
        """mode in {baseline, roi_full, roi_reuse, gated_reuse};
        default per kv_reuse. gated_reuse = stage3 deployment path:
        gate decides after pass-1; bypass decodes off the pass-1
        cache, fired continues as roi_reuse."""
        if mode is None:
            mode = "roi_reuse" if self.kv_reuse else "roi_full"
        if mode in ("roi_sparse", "roi_dense"):
            from qwen_src.gemma4_unified.roi_sparse import answer_sparse
            return answer_sparse(self, image, question, max_new_tokens=max_new_tokens,
                                 dense=(mode == "roi_dense"))
        self.n_total += 1
        image = image.convert("RGB")
        tm = self.model.model.language_model
        timer = self.timer
        timer.start(mode=mode, sample_idx=self.n_total,
                    warmup=self.n_total <= 5)
        try:
            if mode == "baseline":
                tm.enable_twig = False
                templ = self._prompt(question, 1)
                inputs = self.processor(
                    text=[templ], images=[[image]],
                    images_kwargs={"max_soft_tokens": self.tier},
                    return_tensors="pt").to(self.device)
                out = self.model(**inputs, use_cache=True, logits_to_keep=1)
                timer.mark("t_src_prefill_ms")
                _grab_stage_acc(timer, "_p1")
                ids = self._decode_greedy(out.past_key_values,
                                          out.logits[:, -1], max_new_tokens)
                timer.mark("t_decode_ms")
                ans = self.tok.decode(ids, skip_special_tokens=True).strip()
                timer.mark("t_postprocess_ms")
                timer.finish(gen_len=len(ids),
                             n_src_tokens=int(inputs["input_ids"].shape[1]))
                return ans

            # ---- pass 1: squared source, twig + gate on ----
            tm.enable_twig = True
            # pass 2 switches this off; without re-arming it here the
            # gate would stay dead for every sample after the first.
            tm.enable_high_res = True
            # The twig is skipped inside this forward once the
            # gate scores at or below the threshold, so the
            # threshold has to be set before the forward runs.
            # Only the deployed gated path has one to give.
            tm.gate_skip_twig_thresh = (
                self.gate_thresh if mode == "gated_reuse" else None)
            # ---- gate early exit ----
            # Everything the gate decision needs (Need-Refine score, twig
            # heatmap) exists after base layer twig_K-1 = 26; layers 27..47 of
            # pass 1 only build the pass-1 KV cache and the pass-1 logits.
            # Those matter on the BYPASS branch alone, so pass 1 stops at the
            # fork and the branch decides:
            #   bypass -> model.resume_pass1() runs 27..47 on the stashed
            #             state (same tensors/masks/cache => bit-identical);
            #   fired  -> model.discard_pass1(); the two-image arm re-prefills
            #             from scratch and 21 of 48 layers are never paid for.
            # Exception: when the roi_reuse arm turns out to be able to reuse
            # the pass-1 prefix cache (short-sequence samples that clear the
            # sliding-window guard), the tail IS needed after all and is
            # resumed further down before cache.crop.
            _ee_on = (mode == "gated_reuse") and self.gate_early_exit
            tm.gate_early_exit = _ee_on
            # native pass 1: the processor picks an aspect-fit grid, the
            # twig localises on the full-resolution content, and no
            # max(W,H)^2 canvas is ever built (Qwen deployment convention).
            templ1 = self._prompt(question, 1)
            inputs1 = self.processor(
                text=[templ1], images=[[image]],
                images_kwargs={"max_soft_tokens": self.tier},
                return_tensors="pt").to(self.device)
            out1 = self.model(**inputs1, use_cache=True, logits_to_keep=1)
            timer.mark("t_src_prefill_ms")
            _grab_stage_acc(timer, "_p1")
            # Pass 2 and every decode step must run the full stack.
            tm.gate_early_exit = False
            _p1_pending = bool(getattr(out1, "early_exited", False))
            tm.enable_twig = False  # later forwards don't need the fork
            # The gate has already decided and its score is read from
            # tm.high_res_pred below; scoring again in pass 2 is pure
            # overhead.  The branch is stateless, so switching it off
            # here cannot change what the model generates.
            tm.enable_high_res = False

            _gate_score = None
            if mode == "gated_reuse":
                # Need-Refine gate decides BEFORE any RoI work. Gate
                # compute ran inside the pass-1 forward (stage3 ckpt,
                # enable_high_res) — its cost sits in t_src_prefill_ms;
                # overhead = gated.src_prefill - roi_full.src_prefill.
                _hp = getattr(tm, "high_res_pred", None)
                _gate_score = float(_hp.reshape(-1)[0]) \
                    if _hp is not None else -1.0
                _thr = self.gate_thresh \
                    if self.gate_thresh is not None else 0.5
                if _gate_score <= _thr:
                    # bypass: the pass-1 cache and logits ARE the answer
                    # context, so finish the stack that the early exit
                    # suspended (layers 27..47) before decoding.
                    if _p1_pending:
                        out1 = self.model.resume_pass1(logits_to_keep=1)
                        _p1_pending = False
                        timer.mark("t_resume_ms")
                    ids = self._decode_greedy(out1.past_key_values,
                                              out1.logits[:, -1],
                                              max_new_tokens)
                    timer.mark("t_decode_ms")
                    ans = self.tok.decode(
                        ids, skip_special_tokens=True).strip()
                    timer.mark("t_postprocess_ms")
                    timer.finish(
                        gen_len=len(ids), fired=0,
                        gate_score=_gate_score,
                        early_exit=int(_ee_on),
                        n_src_tokens=int(inputs1["input_ids"].shape[1]))
                    return ans
                mode = "roi_reuse"  # fired: continue as the reuse arm

            grid = compute_rpn_grid(self.model, out1, inputs1["input_ids"],
                                    inputs1["image_position_ids"])
            box = extract_box(grid, self.recipe, self.conf) \
                if grid is not None else None
            crop = self._grid_box_to_crop(image, box, *grid.shape) \
                if box is not None else None
            if crop is not None:
                crop = self._scale_crop(crop)
            timer.mark("t_rpn_ms")

            if crop is None:
                # bypass: continue decoding straight off the pass-1 cache
                # (gate fired but the heatmap produced no box) — same as the
                # gate-bypass branch, the suspended tail is needed.
                if _p1_pending:
                    out1 = self.model.resume_pass1(logits_to_keep=1)
                    _p1_pending = False
                    timer.mark("t_resume_ms")
                ids = self._decode_greedy(out1.past_key_values,
                                          out1.logits[:, -1], max_new_tokens)
                timer.mark("t_decode_ms")
                ans = self.tok.decode(ids, skip_special_tokens=True).strip()
                timer.mark("t_postprocess_ms")
                timer.finish(gen_len=len(ids), fired=0,
                             early_exit=int(_ee_on),
                             n_src_tokens=int(inputs1["input_ids"].shape[1]))
                return ans

            self.n_aug += 1
            # ---- inserted sequence: native source + crop ----
            # prepare_inputs answers from [native_src, crop] and produced the
            # accuracy numbers; the squared canvas belongs to pass 1 alone,
            # where the twig's training convention requires it.  Answering
            # from it re-encodes a max(W,H)^2 buffer and spends most of the
            # soft-token budget on padding.
            templ2 = self._prompt(question, 2)
            inputs2 = self.processor(
                text=[templ2], images=[[image, crop]],
                images_kwargs={"max_soft_tokens": self.tier},
                return_tensors="pt").to(self.device)
            timer.mark("t_roi_encode_ms")

            ids1 = inputs1["input_ids"][0]
            ids2 = inputs2["input_ids"][0]
            eoi_id = int(getattr(self.model.config, "eoi_token_id", 258882))
            reuse_ok = False
            if mode == "roi_reuse":
                eoi_pos = (ids1 == eoi_id).nonzero(as_tuple=False)
                if eoi_pos.numel() > 0:
                    P = int(eoi_pos[0]) + 1
                    if P < ids2.shape[0] and torch.equal(ids2[:P], ids1[:P]):
                        reuse_ok = True
                # Sliding-window guard (GEMMA_KV_REUSE_GUARD=window|off,
                # default window): beyond sliding_window (1024) two things
                # break — (a) sliding layers stop tracking past states, so
                # cache.crop RAISES once L1 >= window (observed on the
                # vstar 1120-tier arm), and (b) the chunked-append sliding
                # masks diverge from full-prefill (the source of the
                # 72%-answer-identity drift; all diverging samples had
                # L2 > 1024). Reuse only when BOTH sequences fit the
                # window; otherwise clean fallback to full re-prefill.
                if reuse_ok and os.environ.get(
                        "GEMMA_KV_REUSE_GUARD", "window") == "window":
                    sw = int(getattr(
                        self.model.config.get_text_config(),
                        "sliding_window", 1024) or 1024)
                    L1_ids = int(ids1.shape[0])
                    L2_ids = int(ids2.shape[0])
                    if L1_ids > sw or L2_ids > sw:
                        reuse_ok = False
                if not reuse_ok:
                    self.n_reuse_fallback += 1

            if mode == "roi_reuse" and reuse_ok and _p1_pending:
                # The fired arm turned out to want the pass-1 prefix cache
                # after all (both sequences fit the sliding window), and that
                # cache is only half-built. Finish it before cropping: the
                # resumed layers write exactly the KV the un-split pass 1
                # wrote, so reuse stays exact. Costs this sample the LLM tail
                # it would have paid anyway — the early exit simply saves
                # nothing here, it never changes the result.
                out1 = self.model.resume_pass1(logits_to_keep=1)
                _p1_pending = False
                timer.mark("t_resume_ms")
            if _p1_pending:
                # fired + no cache reuse: the pass-1 tail is dead weight.
                self.model.discard_pass1()
                _p1_pending = False

            if mode == "roi_reuse" and reuse_ok:
                cache = out1.past_key_values
                L1 = int(cache.get_seq_length())
                # transformers 5.15: NEGATIVE arg = "remove this many tokens
                # from the end" (a positive arg is the deprecated legacy
                # "final absolute size" — passing L1-P there silently kept
                # only ~30 tokens and broke reuse exactness).
                try:
                    if L1 > P:
                        cache.crop(-(L1 - P))
                except RuntimeError as e:
                    # sliding layer past-state tracking exhausted (>window
                    # prefill with guard off) — fall back to full re-prefill.
                    print(f"[roi_reuse] cache.crop failed ({e}); "
                          f"falling back to full prefill", flush=True)
                    reuse_ok = False
                    self.n_reuse_fallback += 1
            if mode == "roi_reuse" and reuse_ok:
                # crop-only tensors for the partial prefill — built ONLY once
                # reuse is confirmed (2026-08-17). Building this eagerly cost
                # every fired sample a second processor pass over the crop
                # (+129..203 ms measured) for tensors the window guard then
                # discarded. Cost now lands in t_secondary_prefill_ms on the
                # (rare) reused path.
                inputs_c = self.processor(
                    text=[self.tok.image_token], images=[[crop]],
                    images_kwargs={"max_soft_tokens": self.tier},
                    return_tensors="pt").to(self.device)
                append_ids = ids2[P:].unsqueeze(0)
                kw = {}
                append_mm = None
                if "mm_token_type_ids" in inputs2:
                    append_mm = inputs2["mm_token_type_ids"][:, P:]
                    kw["mm_token_type_ids"] = append_mm
                # EXPLICIT mask-dict build (PORT_PLAN insertion point 5):
                # the non-generate forward builds block ids from the append
                # segment WITHOUT padding them for the cached prefix — the
                # blockwise bidirectional overlay on sliding layers then
                # misindexes kv positions (observed as 1.5-9 logit drift).
                # create_masks_for_vision_model pads block ids to kv length
                # (the generate-path machinery), so we pre-build the dict
                # and hand it to forward as attention_mask.
                dummy_embeds = self.model.model.get_input_embeddings()(
                    torch.where(append_ids >= 0, append_ids,
                                torch.zeros_like(append_ids)))
                pos_ids = torch.arange(
                    P, int(ids2.shape[0]), device=self.device).unsqueeze(0)
                blk = (get_block_sequence_ids_for_mask(append_mm, self.device)
                       if append_mm is not None else
                       torch.full(append_ids.shape, -1, device=self.device))
                mask_dict = create_masks_for_vision_model(
                    config=self.model.config.get_text_config(),
                    inputs_embeds=dummy_embeds,
                    attention_mask=None,
                    past_key_values=cache,
                    position_ids=pos_ids,
                    block_sequence_ids=blk,
                )
                out2 = self.model(
                    input_ids=append_ids,
                    pixel_values=inputs_c["pixel_values"],
                    image_position_ids=inputs_c["image_position_ids"],
                    attention_mask=mask_dict,
                    position_ids=pos_ids,
                    past_key_values=cache,
                    use_cache=True, logits_to_keep=1, **kw)
                timer.mark("t_secondary_prefill_ms")
                _grab_stage_acc(timer, "_p2")
                prefix_len = P
            else:
                out2 = self.model(**inputs2, use_cache=True, logits_to_keep=1)
                timer.mark("t_secondary_prefill_ms")
                _grab_stage_acc(timer, "_p2")
                prefix_len = 0

            ids = self._decode_greedy(out2.past_key_values,
                                      out2.logits[:, -1], max_new_tokens)
            timer.mark("t_decode_ms")
            ans = self.tok.decode(ids, skip_special_tokens=True).strip()
            timer.mark("t_postprocess_ms")
            timer.finish(
                gen_len=len(ids), fired=1,
                gate_score=_gate_score,
                early_exit=int(_ee_on),
                reused=int(mode == "roi_reuse" and reuse_ok),
                prefix_len=int(prefix_len),
                n_src_tokens=int(ids1.shape[0]),
                n_full_tokens=int(ids2.shape[0]),
            )
            self._last_debug = (image, grid, box, crop, question)
            return ans
        finally:
            tm.enable_twig = True
            # Never leave the switch armed or a stash holding a pass-1 cache:
            # the next sample's pass 1 arms it again itself, and any other
            # caller of this model must see the un-split forward.
            tm.gate_early_exit = False
            tm._resume_state = None

    def _prompt(self, question, n_images):
        content = [{"type": "image"}] * n_images + [
            {"type": "text", "text": question}]
        messages = []
        if self.system_text:
            messages.append({"role": "system", "content": [
                {"type": "text", "text": self.system_text}]})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def _grid_box_to_crop(self, img: Image.Image, box, gh, gw,
                          min_size: int = 48):
        """Grid box on the native pass-1 grid → crop of the ORIGINAL image.

        Pass 1 runs on the native image, so the gh×gw grid tiles it directly:
        per-axis cells, no centring offsets.  The rest mirrors qwen
        get_bbox4src: clamp the box to a minimum side (one model-patch cell,
        48 px for Gemma vs qwen's 28) by expanding symmetrically from the
        center and SLIDING the box back inside the frame (translate, don't
        shrink), final clamp to bounds.
        """
        w, h = img.size
        cell_w = w / gw
        cell_h = h / gh
        r0, c0, r1, c1 = box  # inclusive grid coords
        X0, X1 = c0 * cell_w, (c1 + 1) * cell_w
        Y0, Y1 = r0 * cell_h, (r1 + 1) * cell_h
        X0, Y0 = max(0.0, X0), max(0.0, Y0)
        X1, Y1 = min(float(w), X1), min(float(h), Y1)
        if X1 <= X0 or Y1 <= Y0:
            return None
        # min-size clamp: center expansion
        dw = max(0.0, min_size - (X1 - X0))
        dh = max(0.0, min_size - (Y1 - Y0))
        X0, X1 = X0 - dw / 2, X1 + dw / 2
        Y0, Y1 = Y0 - dh / 2, Y1 + dh / 2
        # slide inside the frame (translate, don't shrink)
        if X0 < 0:
            X1 -= X0
            X0 = 0.0
        if Y0 < 0:
            Y1 -= Y0
            Y0 = 0.0
        if X1 > w:
            X0 = max(0.0, X0 - (X1 - w))
            X1 = float(w)
        if Y1 > h:
            Y0 = max(0.0, Y0 - (Y1 - h))
            Y1 = float(h)
        X0, Y0 = int(math.floor(X0)), int(math.floor(Y0))
        X1, Y1 = int(math.ceil(X1)), int(math.ceil(Y1))
        if X1 - X0 < 2 or Y1 - Y0 < 2:
            return None
        return img.crop((X0, Y0, X1, Y1))

    def _scale_crop(self, crop: Image.Image) -> Image.Image:
        """Explicit min-tier upscaling (Gemma has no min_pixels knob)."""
        w, h = crop.size
        area = w * h
        target = self.min_tier * self.model_patch * self.model_patch
        if area < target:
            s = math.sqrt(target / area)
            crop = crop.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.BILINEAR)
        return crop

    @torch.no_grad()
    def prepare_inputs(self, image: Image.Image, question: str):
        """RoI stage: heatmap → box → crop. Returns the ANSWER-pass
        processor inputs: [native_src, crop] when a box fires, else
        [native_src] alone — the no-box path is EXACTLY the baseline
        single-image prompt (native aspect, no expand2square)."""
        self.n_total += 1
        image = image.convert("RGB")

        # ---- native pass 1: prefill only (Qwen deployment convention) ----
        templ = self._prompt(question, 1)
        inputs = self.processor(
            text=[templ], images=[[image]],
            images_kwargs={"max_soft_tokens": self.tier},
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs, use_cache=False, logits_to_keep=1)
        grid = compute_rpn_grid(self.model, out, inputs["input_ids"],
                                inputs["image_position_ids"])
        box = None
        if grid is not None:
            box = extract_box(grid, self.recipe, self.conf)
        crop = None
        if box is not None:
            crop = self._grid_box_to_crop(image, box, *grid.shape)
        if crop is not None:
            crop = self._scale_crop(crop)
            self.n_aug += 1

        # ---- answer-pass inputs (native-aspect source, baseline-style) ----
        imgs = [image] if crop is None else [image, crop]
        templ2 = self._prompt(question, len(imgs))
        inputs2 = self.processor(
            text=[templ2], images=[imgs],
            images_kwargs={"max_soft_tokens": self.tier},
            return_tensors="pt",
        ).to(self.device)
        self._last_debug = (image, grid, box, crop, question)
        return inputs2

    @torch.no_grad()
    def answer(self, image: Image.Image, question: str, *,
               max_new_tokens=64, **gen_kwargs) -> str:
        inputs2 = self.prepare_inputs(image, question)
        gen = self.model.generate(
            **inputs2, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, **gen_kwargs,
        )
        ans = self.tok.decode(
            gen[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
        if self.debug_dir and self.n_total <= 16:
            img, grid, box, crop, q = self._last_debug
            self._dump_debug(img, grid, box, crop, q, ans)
        return ans

    def _dump_debug(self, image, grid, box, crop, question, ans):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(self.debug_dir, exist_ok=True)
            fig, axes = plt.subplots(1, 3, figsize=(11, 4))
            axes[0].imshow(image); axes[0].set_title("src", fontsize=8)
            if grid is not None:
                axes[1].imshow(torch.sigmoid(grid.float()), cmap="viridis",
                               vmin=0, vmax=1)
            axes[1].set_title(f"rpn box={box}", fontsize=8)
            if crop is not None:
                axes[2].imshow(crop)
            axes[2].set_title(f"crop", fontsize=8)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"Q: {question[:70]} | A: {ans[:40]}", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(
                self.debug_dir, f"roi_{self.n_total:03d}.png"), dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"[roi debug] {e}", flush=True)
