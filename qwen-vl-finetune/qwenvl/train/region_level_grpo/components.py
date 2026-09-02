"""Connected-component extraction + scoring for region-level GR-REINFORCE.

Pipeline at one prompt:

    Z_theta  (mean-head logits, shape (Hg, Wg))
        |  sigmoid
        v
    P  (probability map, in [0, 1])
        |  Gaussian blur (kernel 3, sigma 1.0)  -- prevent over-fragmentation
        v
    P_smooth
        |  peak-ratio threshold (rho=3, peak_fraction=0.3 in the RL recipe)
        v
    binary mask
        |  scipy connected-component labeling
        v
    list[Component]   <-- each scored via top-p mean of Z_theta within
                          the component, minus a size penalty
        |  sort by score desc, keep top-R
        v
    list[Component] (top-R)

The action space for the policy is built on top of these components in
``actions.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from torchvision.transforms.functional import gaussian_blur


# ---------------------------------------------------------------- Component --

@dataclass(frozen=True)
class Component:
    """A single connected foreground component on the feature grid.

    Attributes:
        mask: bool numpy array of shape ``(Hg, Wg)``, True for grid cells
            belonging to this component.
        score: scalar score from :func:`score_component`. Higher = more
            "valuable" candidate ROI under the current policy.
        area: number of foreground grid cells (= ``mask.sum()``).
        bbox: ``(r1, c1, r2, c2)`` axis-aligned bounding box in grid
            coordinates with exclusive end (``r2 = max_row + 1``).
    """

    mask: np.ndarray
    score: float
    area: int
    bbox: Tuple[int, int, int, int]


# -------------------------------------------------------------- pipeline --

def smooth_prob_map(
    p_map: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Gaussian-blur a 2D probability map at feature-grid resolution.

    Prevents pixel-level noise from over-fragmenting the foreground when
    the threshold + connected-component step is applied. Operates in
    place geometrically: the output shape is preserved.
    """
    if p_map.ndim != 2:
        raise ValueError(
            f"p_map must be 2D (Hg, Wg); got shape {tuple(p_map.shape)!r}"
        )
    # No-op short-circuit: kernel_size<=1 or sigma<=0 means "disable
    # smoothing" (connected components on the raw heatmap).
    if kernel_size <= 1 or sigma <= 0:
        return p_map
    blurred = gaussian_blur(
        p_map.unsqueeze(0).unsqueeze(0),  # (1, 1, Hg, Wg)
        kernel_size=kernel_size,
        sigma=sigma,
    )
    return blurred.squeeze(0).squeeze(0)


def threshold_fixed(
    p_map: torch.Tensor,
    threshold: float = 0.04,
) -> torch.Tensor:
    """Static threshold: ``p_map > threshold``. Returns a bool mask."""
    return p_map > float(threshold)


def threshold_peak_ratio(
    p_map: torch.Tensor,
    ratio_thresh: float = 3.0,
    peak_fraction: float = 0.15,
    min_gate: float = 0.03,
) -> torch.Tensor:
    """Apply the dynamic peak-ratio threshold; return a bool mask.

    Returns an all-False mask if either:
      - peak < min_gate (signal too weak), or
      - peak / mean < ratio_thresh (no localized hotspot).

    Otherwise the threshold is ``peak * peak_fraction`` and the mask is
    ``p_map > threshold``.
    """
    peak = p_map.max()
    if peak.item() < min_gate:
        return torch.zeros_like(p_map, dtype=torch.bool)
    mean = p_map.mean()
    if (peak / (mean + 1e-8)).item() < ratio_thresh:
        return torch.zeros_like(p_map, dtype=torch.bool)
    return p_map > peak * peak_fraction


def _connectivity_structure(connectivity: int = 1) -> np.ndarray:
    if connectivity == 1:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if connectivity == 2:
        return np.ones((3, 3), dtype=bool)
    raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")


def label_connected_components(
    binary_mask: torch.Tensor,
    connectivity: int = 1,
) -> Tuple[np.ndarray, int]:
    """scipy connected-component labeling. Returns ``(label_array, n_cc)``."""
    np_mask = binary_mask.detach().cpu().numpy().astype(bool)
    structure = _connectivity_structure(connectivity)
    labeled, n_cc = cc_label(np_mask, structure=structure)
    return labeled, int(n_cc)


def score_component(
    z_logits: torch.Tensor,
    component_mask: np.ndarray,
    p: float = 0.5,
    beta: float = 1.0,
    gamma: float = 0.6,
) -> float:
    """Score a connected component via top-p mean of pre-sigmoid logits.

    .. math::

        s_\\theta(C) = \\operatorname{TopPMean}_p\\!\\left(
            \\{Z_\\theta(u) \\mid u \\in C\\}
        \\right)
        - \\beta \\cdot \\max(0, \\, \\text{area}(C) / |I| - \\gamma)

    Args:
        z_logits: ``(Hg, Wg)`` tensor of pre-sigmoid logits.
        component_mask: ``(Hg, Wg)`` bool numpy array.
        p: top-p fraction. ``p=1.0`` → mean over all in-component logits
            (the RL recipe); ``p=0.5`` → mean over the top 50% (by value)
            within the component.
        beta: size-penalty weight.
        gamma: size-penalty kicks in above ``area / |I| > gamma``
            (default 0.6 = 60% of the image).

    Returns:
        Scalar Python float. ``-inf`` if the component is empty.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1]; got {p}")
    if z_logits.ndim != 2:
        raise ValueError(
            f"z_logits must be 2D (Hg, Wg); got shape {tuple(z_logits.shape)!r}"
        )

    z_np = z_logits.detach().cpu().numpy().astype(np.float32)
    n_grid = int(component_mask.size)  # Hg * Wg
    n_in = int(component_mask.sum())
    if n_in == 0:
        return float("-inf")

    vals = z_np[component_mask]
    if p < 1.0:
        k = max(1, int(round(p * n_in)))
        vals = np.sort(vals)[-k:]
    pool = float(vals.mean())

    area_frac = n_in / float(n_grid)
    size_penalty = beta * max(0.0, area_frac - gamma)

    return pool - size_penalty


def score_component_torch(
    z_logits: torch.Tensor,
    component_mask: torch.Tensor,
    p: float = 0.5,
    beta: float = 1.0,
    gamma: float = 0.6,
) -> torch.Tensor:
    """Differentiable variant of :func:`score_component`.

    Uses ``torch.topk`` so gradients flow from the returned score back
    through ``z_logits``. The component mask itself is treated as a
    constant (no gradient).

    Args:
        z_logits: ``(Hg, Wg)`` logits, requires grad.
        component_mask: ``(Hg, Wg)`` bool tensor (or 0/1 numeric),
            detached. Same device as ``z_logits`` recommended.
        p: top-p fraction, in (0, 1].
        beta: size-penalty weight.
        gamma: size-penalty area threshold.

    Returns:
        0-dim scalar tensor with grad through ``z_logits``.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1]; got {p}")
    mask = component_mask.bool()
    n_in = int(mask.sum().item())
    if n_in == 0:
        return z_logits.new_full((), float("-inf"))

    in_comp = z_logits[mask]  # (n_in,) with grad
    if p < 1.0:
        k = max(1, int(round(p * n_in)))
        top_vals, _ = torch.topk(in_comp, k=k, sorted=False)
        pool = top_vals.mean()
    else:
        pool = in_comp.mean()

    n_grid = float(int(component_mask.numel()))
    area_frac = float(n_in) / n_grid
    size_penalty = float(beta) * max(0.0, area_frac - float(gamma))

    return pool - size_penalty


def extract_components(
    p_map: torch.Tensor,
    z_logits: torch.Tensor,
    *,
    smooth_kernel: int = 3,
    smooth_sigma: float = 1.0,
    threshold_mode: str = "peak_ratio",
    fixed_threshold: float = 0.04,
    ratio_thresh: float = 3.0,
    peak_fraction: float = 0.15,
    min_gate: float = 0.03,
    score_p: float = 0.5,
    score_beta: float = 1.0,
    score_gamma: float = 0.6,
    connectivity: int = 1,
) -> List[Component]:
    """End-to-end: smooth → threshold → CC → score → sort.

    ``threshold_mode``:
      - ``"peak_ratio"`` (default): per-sample dynamic threshold,
        ``mask = p > peak * peak_fraction`` with rejection on weak peaks.
      - ``"fixed"``: static absolute threshold ``mask = p > fixed_threshold``.

    Returns components sorted by score descending so callers can take
    ``components[:R]`` for top-R selection.
    """
    smoothed = smooth_prob_map(p_map, kernel_size=smooth_kernel, sigma=smooth_sigma)
    if threshold_mode == "peak_ratio":
        binary = threshold_peak_ratio(
            smoothed,
            ratio_thresh=ratio_thresh,
            peak_fraction=peak_fraction,
            min_gate=min_gate,
        )
    elif threshold_mode == "fixed":
        binary = threshold_fixed(smoothed, threshold=fixed_threshold)
    else:
        raise ValueError(
            "threshold_mode must be 'peak_ratio' or 'fixed'; "
            f"got {threshold_mode!r}"
        )
    if not binary.any():
        return []

    labeled, n_cc = label_connected_components(binary, connectivity)
    if n_cc == 0:
        return []

    out: List[Component] = []
    for label_id in range(1, n_cc + 1):
        mask = (labeled == label_id)
        if not mask.any():
            continue
        score = score_component(
            z_logits, mask, p=score_p, beta=score_beta, gamma=score_gamma,
        )
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r1 = int(np.where(rows)[0][0])
        r2 = int(np.where(rows)[0][-1]) + 1
        c1 = int(np.where(cols)[0][0])
        c2 = int(np.where(cols)[0][-1]) + 1
        out.append(
            Component(
                mask=mask,
                score=float(score),
                area=int(mask.sum()),
                bbox=(r1, c1, r2, c2),
            )
        )

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def rank_top_r(components: List[Component], R: int) -> List[Component]:
    """Top-R components by score. ``components`` is already sorted desc."""
    if R < 0:
        raise ValueError(f"R must be non-negative; got {R}")
    return list(components[:R])


# ----------------------------------------------------------------- tests --

def _self_test() -> None:
    """Synthetic-heatmap self-test. Run as ``python -m ...components``."""
    torch.manual_seed(0)

    # 1. Two-blob heatmap → expect 2 components, distinct scores.
    Hg, Wg = 16, 16
    yy, xx = np.mgrid[0:Hg, 0:Wg]
    blob_a = np.exp(-((xx - 4) ** 2 + (yy - 4) ** 2) / 4.0)   # high peak
    blob_b = 0.4 * np.exp(-((xx - 12) ** 2 + (yy - 12) ** 2) / 4.0)  # weaker
    p = torch.from_numpy((blob_a + blob_b).astype(np.float32))
    p = p / p.max()
    z = torch.logit(p.clamp(1e-3, 1 - 1e-3))

    comps = extract_components(
        p, z, peak_fraction=0.15, ratio_thresh=2.0,
    )
    print(f"[two-blob] n_components={len(comps)}, scores={[round(c.score, 3) for c in comps]}")
    assert len(comps) >= 2, f"expected >= 2 components, got {len(comps)}"
    # Sorted descending by score.
    assert comps[0].score >= comps[1].score

    # 2. Single-blob → 1 component.
    p_one = torch.from_numpy(blob_a.astype(np.float32))
    p_one = p_one / p_one.max()
    z_one = torch.logit(p_one.clamp(1e-3, 1 - 1e-3))
    comps_one = extract_components(p_one, z_one, peak_fraction=0.15, ratio_thresh=2.0)
    print(f"[one-blob] n_components={len(comps_one)}")
    assert len(comps_one) == 1

    # 3. Below-threshold heatmap (no peak strong enough) → empty list.
    p_flat = torch.full((Hg, Wg), 0.02, dtype=torch.float32)
    z_flat = torch.logit(p_flat.clamp(1e-3, 1 - 1e-3))
    comps_flat = extract_components(p_flat, z_flat, peak_fraction=0.15)
    print(f"[flat-below-min-gate] n_components={len(comps_flat)}")
    assert comps_flat == []

    # 4. score_component edge cases.
    mask_full = np.ones((Hg, Wg), dtype=bool)
    s_full = score_component(z, mask_full, p=1.0, beta=1.0, gamma=0.6)
    # Whole image -> area_frac=1, size penalty = beta*(1-gamma) = 0.4.
    s_no_penalty = score_component(z, mask_full, p=1.0, beta=0.0, gamma=0.6)
    assert abs(s_full - (s_no_penalty - 0.4)) < 1e-4, (
        f"size penalty mismatch: {s_full} vs {s_no_penalty - 0.4}"
    )

    # 5. top-p mean: verify p=0.5 picks the top half.
    z_sorted = torch.tensor(np.linspace(-1, 1, Hg * Wg).reshape(Hg, Wg), dtype=torch.float32)
    s_p1 = score_component(z_sorted, mask_full, p=1.0, beta=0.0, gamma=1.0)
    s_p5 = score_component(z_sorted, mask_full, p=0.5, beta=0.0, gamma=1.0)
    assert s_p5 > s_p1, f"top-p=0.5 should be > p=1.0 on monotone data; {s_p5} vs {s_p1}"

    # 6. rank_top_r preserves order.
    if len(comps) >= 2:
        top = rank_top_r(comps, R=1)
        assert len(top) == 1
        assert top[0] is comps[0]

    print("OK: components.py self-test passed.")


if __name__ == "__main__":
    _self_test()
