"""Online ROI **single-region** supervision labels.

Build ROI supervision targets ON THE FLY from the model's own response→image
attention, using the GPU-batched single-region (peak-ratio + 1-CC) extraction
(same pipeline as
``qwenvl/train/region_level_grpo/components.extract_single_region_batched``).

Per sample, over the response tokens (AFTER skipping the leading
``[Visual Evidence]\\n`` prefix), we keep only tokens whose RAW answer→image
foreground is a SINGLE connected region (peak/mean ratio gate + threshold +
8-connectivity CC). The union of those raw foregrounds is the positive; the
Gaussian-smoothing halo around the union is the ignore band; everything else is
background.

Label convention — matches ``qwen_src/roi/heatmap.create_pseudo_labels`` so the
existing ROI/twig BCE loss (``valid = tgt != -100``) consumes it unchanged:

    1     foreground  (🟢 raw union, no smoothing)
    0     background
  -100    ignore      (🟠 Gaussian-smoothing halo around the union)

Validation runs on the RAW map (no smoothing) by default — set
``extract_smooth_sigma=0.25`` for light denoising if the raw map fragments.

Connected components run on the GPU via kornia (``num_iterations`` label
propagation), with a scipy CPU fallback if kornia is unavailable.

Entry point: :func:`single_region_roi_label` (one ``[h, w]`` label from a
sample's per-token response→image attention); called from
``online_pseudo_label.online_single_region_label_for_sample``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

# Defaults of the single-region foreground pipeline.
DEFAULT_RATIO_THRESH = 3.0          # peak/mean gate ("peak ratio of 3")
DEFAULT_PEAK_FRACTION = 0.3         # threshold = peak_fraction * peak
DEFAULT_EXTRACT_SMOOTH_SIGMA = 0.0  # validate on the RAW map (no smoothing)
DEFAULT_HALO_SMOOTH_SIGMA = 1.0     # Gaussian for the ignore halo
DEFAULT_HALO_THRESH_FRAC = 0.3      # blurred-union > frac*peak → dilated region
DEFAULT_CC_NUM_ITERATIONS = 64      # kornia label-propagation iters
DEFAULT_SKIP_LEADING_TOKENS = 5     # "[Visual Evidence]\n" response prefix
IGNORE_INDEX = -100
FG_LABEL = 1
BG_LABEL = 0


# --------------------------------------------------------------------------- #
# Batched GPU primitives (shared with the vis / training extraction)
# --------------------------------------------------------------------------- #

def _gaussian_blur_batched(x: torch.Tensor, sigma: float,
                           kernel_size: int = 5) -> torch.Tensor:
    """Separable Gaussian over a batch of 2D maps ``[N, h, w]`` (one conv2d,
    on ``x``'s device). No-op if ``sigma <= 0``."""
    if not sigma or sigma <= 0:
        return x
    k = int(kernel_size)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(ax ** 2) / (2.0 * float(sigma) ** 2))
    g = g / g.sum()
    ker = (g[:, None] * g[None, :])[None, None]
    return F.conv2d(x[:, None], ker, padding=k // 2)[:, 0]


def _cc_count_batched(masks: torch.Tensor,
                      num_iterations: int = DEFAULT_CC_NUM_ITERATIONS) -> torch.Tensor:
    """Per-map connected-component COUNT for ``[N, h, w]`` bool masks. kornia
    GPU label-propagation (count distinct nonzero labels per map via one sorted
    pass), with a scipy CPU fallback. Returns a long tensor ``[N]``."""
    n = int(masks.shape[0])
    try:
        import kornia
        lab = kornia.contrib.connected_components(
            masks[:, None].float(), num_iterations=int(num_iterations))
        s, _ = lab.reshape(n, -1).sort(dim=1)
        new = torch.ones_like(s, dtype=torch.bool)
        new[:, 1:] = s[:, 1:] != s[:, :-1]
        return (new & (s != 0)).sum(dim=1).long()
    except Exception:  # noqa: BLE001 — kornia missing / failure → CPU scipy
        import numpy as np
        from scipy.ndimage import label as _lbl
        st = np.ones((3, 3), np.int32)
        m = masks.detach().cpu().numpy()
        counts = [int(_lbl(x, structure=st)[1]) for x in m]
        return torch.as_tensor(counts, device=masks.device, dtype=torch.long)


def _single_region_masks(
    maps: torch.Tensor, ratio_thresh: float, peak_fraction: float,
    smooth_sigma: float, cc_num_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``maps`` ``[N, h, w]`` raw → ``(masks bool [N,h,w], single bool [N])``.

    Per map: normalize-by-max → optional Gaussian → peak/mean ratio gate →
    threshold at ``peak_fraction*peak`` → CC count; ``single = (n_cc == 1)``."""
    m = maps.float()
    mx = m.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    sm = _gaussian_blur_batched(m / mx, smooth_sigma)
    peak = sm.amax(dim=(1, 2))
    mean = sm.mean(dim=(1, 2))
    gate = (peak / (mean + 1e-8)) >= float(ratio_thresh)
    masks = (sm > (peak * float(peak_fraction))[:, None, None]) & gate[:, None, None]
    n_cc = _cc_count_batched(masks, cc_num_iterations)
    return masks, (n_cc == 1)


def _union_to_label(
    raw_union: torch.Tensor, *, halo_smooth_sigma: float,
    halo_thresh_frac: float, ignore_index: int,
) -> torch.Tensor:
    """Turn a ``[h, w]`` bool raw-union mask into a long label map:
    raw_union → ``FG_LABEL``; Gaussian-smoothing halo → ``ignore_index``;
    rest → ``BG_LABEL``."""
    h, w = raw_union.shape
    dev = raw_union.device
    if halo_smooth_sigma > 0 and bool(raw_union.any()):
        blurred = _gaussian_blur_batched(
            raw_union[None].float(), halo_smooth_sigma)[0]
        dilated = blurred > halo_thresh_frac * float(blurred.max())
        halo = dilated & (~raw_union)
    else:
        halo = torch.zeros((h, w), dtype=torch.bool, device=dev)
    lbl = torch.full((h, w), BG_LABEL, dtype=torch.long, device=dev)
    lbl[halo] = int(ignore_index)
    lbl[raw_union] = int(FG_LABEL)
    return lbl


def _reduce_to_token_maps(
    attn: torch.Tensor,
    layers: Optional[Sequence[int]],
    heads: Optional[Sequence[int]],
) -> torch.Tensor:
    """Accept per-token response→image attention as ``[T, h, w]`` (already
    reduced) or ``[T, L, H, h, w]`` (reduce over the given layer/head subsets,
    else all) → ``[T, h, w]``."""
    if attn.ndim == 3:
        return attn.float()
    if attn.ndim == 5:
        a = attn
        if layers:
            a = a[:, list(layers)]
        if heads:
            a = a[:, :, list(heads)]
        return a.float().mean(dim=(1, 2))
    raise ValueError(
        f"attn must be [T,h,w] or [T,L,H,h,w]; got {tuple(attn.shape)!r}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def single_region_roi_label(
    per_token_attn: Union[torch.Tensor, "object"],
    *,
    layers: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
    skip_leading_tokens: int = DEFAULT_SKIP_LEADING_TOKENS,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
    peak_fraction: float = DEFAULT_PEAK_FRACTION,
    extract_smooth_sigma: float = DEFAULT_EXTRACT_SMOOTH_SIGMA,
    halo_smooth_sigma: float = DEFAULT_HALO_SMOOTH_SIGMA,
    halo_thresh_frac: float = DEFAULT_HALO_THRESH_FRAC,
    cc_num_iterations: int = DEFAULT_CC_NUM_ITERATIONS,
    ignore_index: int = IGNORE_INDEX,
    empty_sample_ignore: bool = True,
    device=None,
    return_stats: bool = False,
):
    """Build one ``[h, w]`` long ROI label for a sample from its per-token
    response→image attention.

    Args:
        per_token_attn: ``[T, h, w]`` (already averaged over the grounding
            layers/heads) or ``[T, L, H, h, w]`` (reduced internally via
            ``layers``/``heads``). ``T`` indexes RESPONSE tokens; index 0 must
            be the first response token.
        skip_leading_tokens: drop the leading ``[Visual Evidence]\\n`` prefix
            tokens (default 5: ``'['  'Visual'  ' Evidence'  ']'  '\\n'``).
        empty_sample_ignore: if no token yields a single-region foreground,
            return an ALL-IGNORE label (skip the sample in the loss). Set False
            to instead return all-background (treat as a hard negative).

    Returns:
        ``labels`` ``[h, w]`` long with values ``{1, 0, ignore_index}`` (or
        ``(labels, stats)`` if ``return_stats``).
    """
    if not torch.is_tensor(per_token_attn):
        per_token_attn = torch.as_tensor(per_token_attn)
    if device is not None:
        per_token_attn = per_token_attn.to(device)
    A = _reduce_to_token_maps(per_token_attn, layers, heads)   # [T, h, w]
    T, h, w = A.shape
    dev = A.device

    A = A[int(skip_leading_tokens):]                           # drop prefix
    if A.shape[0] == 0:
        raw_union = torch.zeros((h, w), dtype=torch.bool, device=dev)
        single = torch.zeros((0,), dtype=torch.bool, device=dev)
    else:
        masks, single = _single_region_masks(
            A, ratio_thresh, peak_fraction,
            extract_smooth_sigma, cc_num_iterations)
        raw_union = (masks[single].any(dim=0) if bool(single.any())
                     else torch.zeros((h, w), dtype=torch.bool, device=dev))

    if not bool(raw_union.any()) and empty_sample_ignore:
        lbl = torch.full((h, w), int(ignore_index), dtype=torch.long,
                         device=dev)
    else:
        lbl = _union_to_label(
            raw_union, halo_smooth_sigma=halo_smooth_sigma,
            halo_thresh_frac=halo_thresh_frac, ignore_index=ignore_index)

    if return_stats:
        stats = {
            "n_response_tokens": int(max(0, T - int(skip_leading_tokens))),
            "n_survivors": int(single.sum()) if single.numel() else 0,
            "n_fg": int((lbl == FG_LABEL).sum()),
            "n_ignore": int((lbl == ignore_index).sum()),
            "n_bg": int((lbl == BG_LABEL).sum()),
        }
        return lbl, stats
    return lbl
