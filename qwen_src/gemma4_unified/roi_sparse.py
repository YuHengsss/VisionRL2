"""Gemma-4 two-stage RoI answer pass with SPARSE crop encoding (paper protocol).

Port of the Qwen3.5 "ours" evaluation convention to the encoder-free Gemma-4
tiers (70 / 140 / 280 / 560 / 1120 soft tokens; the processor always fills the
requested tier, so a crop's token count is exactly its tier):

  1. pass 1: native-aspect source at the SOURCE tier (1120 = the largest);
     SD-RPN heatmap from the last prompt token (``compute_rpn_grid``).
  2. region mask (paper convention): sigmoid -> outer-ring sink zeroed ->
     gaussian blur with the ``auto2`` sigma schedule (sigma from the source
     token count) -> peak-ratio gate (min_gate 0.03, peak/mean >= 3, thr =
     0.3 * peak) -> box = bbox of the mask (union of cells).
  3. crop tokens: the constant-target rule of ``qwen_src/roi/crop_budget.py``
     (PROBE_CROP_TARGET_TOK = T, PROBE_CROP_MAX_UPSCALE_EDGE = 3,
     cap C = min(3T, src/2)):  kept = min(max(native, min(9*native, T)), C),
     floor 64; then Mode B (``window_sparse_mode=token_budget``): encode the
     crop at kept * k^2 tokens, k = min(sqrt(1/fg_ratio), k_max), and keep
     only the foreground cells (mask dilated by 1 cell) so the kept-token
     count lands near T. Gemma quantisation: the crop tier is the smallest
     tier >= kept * k^2 (capped at 1120).
  4. sparse drop at the processor-output level: the crop's pooled-token rows
     (``pixel_values`` / ``image_position_ids``) outside the dilated mask and
     the same number of ``<|image|>`` placeholders are removed, so the model's
     ordered masked_scatter of image features stays exact (positions keep
     their true 2-D ids).
  5. pass 2: full prefill of [source @1120, sparse crop, question], greedy
     decode. No box -> decode off the pass-1 cache (baseline behaviour).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from qwen_src.gemma4_unified.roi_inference import _grab_stage_acc, compute_rpn_grid

GEMMA_TIERS = (70, 140, 280, 560, 1120)
CELL_PX = 48 * 48          # one pooled token = 3x3 patches of 16 px


def auto2_sigma(src_tok: int) -> Tuple[float, int]:
    """ROI_EVAL_SMOOTH_SIGMA=auto2 schedule (crop_budget.py)."""
    s = float(src_tok)
    if s <= 2048.0:
        sig = min(max(1.0 + (s - 512.0) / (2048.0 - 512.0), 1.0), 2.0)
    else:
        sig = min(2.0 * (s / 2048.0) ** 0.5, 4.0)
    ker = max(3, int(round(3.0 * sig)))
    if ker % 2 == 0:
        ker += 1
    return sig, ker


def extract_mask_peak_ratio(grid_logits: torch.Tensor, sigma_mode: str = "auto2",
                            min_gate: float = 0.03, ratio_thresh: float = 3.0,
                            peak_fraction: float = 0.3):
    """Paper-protocol region mask on the heatmap grid. Returns (mask, box, sigma)
    with box = (r0, c0, r1, c1) inclusive, or (None, None, sigma)."""
    gh, gw = grid_logits.shape
    prob = torch.sigmoid(grid_logits.float())
    if gh * gw > 256:                       # Gemma sink band = outer ring
        prob[0, :] = 0; prob[-1, :] = 0; prob[:, 0] = 0; prob[:, -1] = 0
    if str(sigma_mode).lower() == "auto2":
        sig, ker = auto2_sigma(gh * gw)
    else:
        sig = float(sigma_mode); ker = max(3, int(round(3.0 * sig)))
        if ker % 2 == 0:
            ker += 1
    blurred = TF.gaussian_blur(prob[None, None], kernel_size=ker, sigma=sig)[0, 0]
    peak = float(blurred.max()); mean = float(blurred.mean())
    if peak < min_gate or peak / (mean + 1e-8) < ratio_thresh:
        return None, None, sig
    mask = blurred > (peak * peak_fraction)
    if not bool(mask.any()):
        return None, None, sig
    ys, xs = torch.nonzero(mask, as_tuple=True)
    box = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    return mask, box, sig


def crop_tier_rule(crop_w: int, crop_h: int, fg_ratio: float, *, target_tok: int,
                   src_tier: int, max_upscale_edge: float = 3.0, k_max: float = 3.0,
                   max_tier: int = 1120):
    """Constant-target crop rule + Mode-B k^2 compensation -> Gemma tier."""
    native = max(1.0, crop_w * crop_h / CELL_PX)
    cap = min(3 * target_tok, src_tier // 2)
    kept = min(max(native, min(max_upscale_edge ** 2 * native, float(target_tok))), float(cap))
    kept = max(kept, 64.0)
    k = min(math.sqrt(1.0 / max(fg_ratio, 1e-3)), float(k_max))
    want = kept * k * k
    tier = next((t for t in GEMMA_TIERS if t >= want), max_tier)
    return min(tier, max_tier), kept, k


def _dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    m = mask.float()[None, None]
    return (F.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r)[0, 0] > 0.5)


@torch.no_grad()
def answer_sparse(pipe, image: Image.Image, question: str, *, max_new_tokens: int = 64,
                  dense: bool = False) -> str:
    """``GemmaRoIPipeline`` method body for mode == 'roi_sparse' (or 'roi_dense'
    with ``dense=True``: the bbox crop is encoded in full at the constant-target
    tier, no Mode-B k^2 compensation and no token dropping -- the SD-RPN arm)."""
    model, proc, tok, dev = pipe.model, pipe.processor, pipe.tok, pipe.device
    tm = model.model.language_model
    timer = pipe.timer
    pipe.n_total += 1
    image = image.convert("RGB")
    T = int(getattr(pipe, "crop_target_tok", 256))
    r_edge = float(getattr(pipe, "crop_max_upscale_edge", 3.0))
    k_max = 1.0 if dense else float(getattr(pipe, "sparse_k_max", 3.0))
    dil = int(getattr(pipe, "sparse_dilation", 1))
    sigma_mode = str(getattr(pipe, "smooth_sigma", "auto2"))
    img_id = int(model.config.image_token_id)
    timer.start(mode=("roi_dense" if dense else "roi_sparse"), sample_idx=pipe.n_total,
                warmup=pipe.n_total <= 5)
    try:
        # ---- pass 1: native source at the source tier, twig on ----
        tm.enable_twig = True
        tm.enable_high_res = False
        tm.gate_early_exit = False
        templ1 = pipe._prompt(question, 1)
        inputs1 = proc(text=[templ1], images=[[image]],
                       images_kwargs={"max_soft_tokens": pipe.tier},
                       return_tensors="pt").to(dev)
        out1 = model(**inputs1, use_cache=True, logits_to_keep=1)
        timer.mark("t_src_prefill_ms")
        _grab_stage_acc(timer, "_p1")
        tm.enable_twig = False
        grid = compute_rpn_grid(model, out1, inputs1["input_ids"], inputs1["image_position_ids"])
        mask = box = None
        sig = None
        if grid is not None:
            mask, box, sig = extract_mask_peak_ratio(grid, sigma_mode)
        crop = pipe._grid_box_to_crop(image, box, *grid.shape) if box is not None else None
        timer.mark("t_rpn_ms")
        n_src = int(inputs1["input_ids"].shape[1])
        if crop is None:
            ids = pipe._decode_greedy(out1.past_key_values, out1.logits[:, -1], max_new_tokens)
            timer.mark("t_decode_ms")
            ans = tok.decode(ids, skip_special_tokens=True).strip()
            timer.mark("t_postprocess_ms")
            timer.finish(gen_len=len(ids), fired=0, n_src_tokens=n_src, sigma=sig)
            return ans
        pipe.n_aug += 1
        del out1

        # ---- crop tier (constant-target rule + Mode B) ----
        r0, c0, r1, c1 = box
        mask_box = mask[r0:r1 + 1, c0:c1 + 1]
        fg_ratio = float(mask_box.float().mean())
        tier_c, kept_target, k = crop_tier_rule(
            crop.size[0], crop.size[1], fg_ratio, target_tok=T, src_tier=pipe.tier,
            max_upscale_edge=r_edge, k_max=k_max)
        inputs_c = proc(text=[tok.image_token], images=[[crop]],
                        images_kwargs={"max_soft_tokens": tier_c}, return_tensors="pt")
        pv_c = inputs_c["pixel_values"][0]                      # (tier_c, D)
        pos_c = inputs_c["image_position_ids"][0]               # (tier_c, 2) (x, y)
        valid_c = (pos_c != -1).all(dim=-1)
        gw_c = int(pos_c[valid_c][:, 0].max()) + 1
        gh_c = int(pos_c[valid_c][:, 1].max()) + 1
        # foreground cells of the crop grid: the heatmap mask inside the box,
        # resized (nearest) to the crop grid and dilated by ``dil`` cells
        keep_grid = F.interpolate(mask_box.float()[None, None], size=(gh_c, gw_c),
                                  mode="nearest")[0, 0] > 0.5
        keep_grid = _dilate(keep_grid, dil)
        keep_rows = valid_c.clone()
        if not dense:
            xs = pos_c[:, 0].clamp(min=0); ys = pos_c[:, 1].clamp(min=0)
            keep_rows &= keep_grid[ys, xs].to(keep_rows.device)
        n_valid = int(valid_c.sum())
        n_keep = int(keep_rows.sum())
        if n_keep < 16:                                        # degenerate mask: keep all
            keep_rows = valid_c.clone(); n_keep = n_valid
        timer.mark("t_roi_encode_ms")

        # ---- pass 2 sequence: two-image template at the source tier, crop span trimmed ----
        templ2 = pipe._prompt(question, 2)
        inputs2 = proc(text=[templ2], images=[[image, crop]],
                       images_kwargs={"max_soft_tokens": pipe.tier}, return_tensors="pt")
        ids2 = inputs2["input_ids"][0]
        img_pos = (ids2 == img_id).nonzero(as_tuple=False).squeeze(-1)
        # split the placeholder positions into the two contiguous runs
        breaks = (img_pos[1:] - img_pos[:-1] > 1).nonzero(as_tuple=False).squeeze(-1)
        assert breaks.numel() == 1, f"expected 2 image runs, got {breaks.numel() + 1}"
        s2 = int(img_pos[int(breaks[0]) + 1]); e2 = int(img_pos[-1]) + 1
        keep_tok = torch.ones(ids2.shape[0], dtype=torch.bool)
        keep_tok[s2 + n_keep:e2] = False                        # trim the crop run to n_keep
        new_ids = ids2[keep_tok].unsqueeze(0)
        kw = {}
        if "mm_token_type_ids" in inputs2:
            kw["mm_token_type_ids"] = inputs2["mm_token_type_ids"][0][keep_tok].unsqueeze(0)
        R = inputs2["pixel_values"].shape[1]                    # rows per image (= source tier)
        pv = inputs2["pixel_values"].clone()
        pos = inputs2["image_position_ids"].clone()
        pv[1].zero_(); pos[1].fill_(-1)
        pv[1, :n_keep] = pv_c[keep_rows][:R]
        pos[1, :n_keep] = pos_c[keep_rows][:R]
        out2 = model(input_ids=new_ids.to(dev), attention_mask=torch.ones_like(new_ids).to(dev),
                     pixel_values=pv.to(dev), image_position_ids=pos.to(dev),
                     use_cache=True, logits_to_keep=1,
                     **{k_: v.to(dev) for k_, v in kw.items()})
        timer.mark("t_secondary_prefill_ms")
        _grab_stage_acc(timer, "_p2")
        ids = pipe._decode_greedy(out2.past_key_values, out2.logits[:, -1], max_new_tokens)
        timer.mark("t_decode_ms")
        ans = tok.decode(ids, skip_special_tokens=True).strip()
        timer.mark("t_postprocess_ms")
        timer.finish(gen_len=len(ids), fired=1, n_src_tokens=n_src,
                     n_full_tokens=int(new_ids.shape[1]), crop_tier=int(tier_c),
                     crop_tokens_valid=n_valid, crop_tokens_kept=n_keep,
                     crop_native_tok=round(crop.size[0] * crop.size[1] / CELL_PX, 1),
                     kept_target=round(kept_target, 1), fg_ratio=round(fg_ratio, 4),
                     k=round(k, 3), sigma=sig, crop_wh=list(crop.size))
        pipe._last_debug = (image, grid, box, crop, question)
        return ans
    finally:
        tm.enable_twig = True
        tm.gate_early_exit = False
        tm._resume_state = None
