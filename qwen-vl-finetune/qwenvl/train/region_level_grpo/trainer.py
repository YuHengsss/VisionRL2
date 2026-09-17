"""Phase-B.1 trainer for region-level GRPO on SD-RPN (connected-component v1).

Pieces:

1. :func:`compute_phase_b1_loss_v2` -- pure function that takes one
   prompt's data (per-head logits, PIL image, question, gold answer,
   Phase-A reference heatmap) and returns the region-drop GR-REINFORCE
   loss. The logic dispatches on K (component count) into three paths:

       K == 0  -> only the KL anchor (no foreground; no learning)
       K == 1  -> BCE-anchor on heatmap + KL anchor (no policy term)
       K >= 2  -> policy gradient + KL anchor (full GR-REINFORCE)

2. :func:`compute_source_map_group_loss` -- the additive "supplement"
   group: regions the cached answer->image evidence maps attend to but the
   policy foreground misses are rewarded additively and pushed up.

3. :class:`RegionLevelGRPOTrainer` -- HF ``Trainer`` subclass. Pops
   the Phase-B.1 extras (PIL images, gold answers, refs, evidence maps)
   from each batch, runs the frozen Phase-A reference twig for the online
   KL anchor, and averages the two groups per sample (1/2 each).
"""

from __future__ import annotations

import logging
import math

# --- Q-Zoom centralized env-knob accessor (Phase A) ---
try:
    from qwen_src.qzoom_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from qzoom_config import getenv as qz_getenv
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import Trainer
except ImportError:  # transformers optional for unit tests
    Trainer = object  # type: ignore[assignment, misc]

from qwenvl.train.region_level_grpo.actions import (
    enumerate_removal_actions,
)
from qwenvl.train.region_level_grpo.components import (
    extract_components,
    rank_top_r,
    score_component_torch,
)
from qwenvl.train.region_level_grpo.losses import (
    k1_anchor_bce_loss_logit,
    kl_anchor_loss_logit,
    policy_gradient_loss,
    policy_gradient_loss_empty_baseline,
)
from qwenvl.train.region_level_grpo.policy import (
    compute_log_pi,
    select_actions_for_rewards,
)
from qwenvl.train.region_level_grpo.reward_model import RewardModel


log = logging.getLogger(__name__)


# Reserved keys the dataset/collator must populate per batch.
KEY_PIL_IMAGES = "_pil_images"           # list[PIL.Image]      length B
KEY_QUESTIONS = "_questions"             # list[str]            length B
KEY_GOLD_ANSWERS = "_gold_answers"       # list[str]            length B
KEY_P_REFS = "_p_refs"                   # list[Tensor (Hg,Wg)] length B
KEY_P_REF_BINARIES = "_p_ref_binaries"   # list[Tensor (Hg,Wg)] length B (or None)
# Optional per-sample cached multi-layer response→image single-region maps
# (evidence maps) for the source_map_group + attention-fg gradient mask:
# list[Tensor (n_layers, Hg, Wg) uint8] | None.
KEY_EV_MAPS = "_ev_maps"                 # list[Tensor (L,Hg,Wg)] length B (or None)


# ============================================================================
# Config
# ============================================================================

@dataclass
class PhaseB1Config:
    """Hyperparameters for one Phase-B.1 GR-REINFORCE step.

    Defaults are the released recipe (Qwen3.5-VL-4B/9B and Qwen2.5-VL-7B
    share every value except ``placebo_kappa``).
    """

    # ---- heatmap → components ----
    threshold_mode: str = "peak_ratio"   # "peak_ratio" | "fixed"
    # Absolute threshold for threshold_mode="fixed"; ALSO the binarization
    # threshold of the reference heatmap for the K=1 BCE anchor and the
    # ``heatmap_pos_frac`` diagnostic.
    fixed_threshold: float = 0.02
    smooth_kernel: int = 3
    smooth_sigma: float = 1.0
    # Peak-ratio params (used when threshold_mode='peak_ratio').
    ratio_thresh: float = 3.0
    peak_fraction: float = 0.3
    # Peak-ratio rejection gate: an all-background mask when peak < min_gate.
    min_gate: float = 0.03
    score_p: float = 1.0
    score_beta: float = 1.0
    score_gamma: float = 0.6
    connectivity: int = 1            # 4-conn

    # --- Source-map supplement group ----------------------------------------
    # When True, add a second loss group driven by the cached answer→image
    # evidence maps (``ev_maps``): SUPP = evidence fg − policy fg (cells the
    # answer attends to but the policy misses), split into regions, reward
    # each ADDITIVELY by c_j = R(cores∪SUPP) − R(cores∪SUPP\supp_j)
    # (Δheight − β·size), advantage = c_j/(std+eps) (raw, NO mean-centering),
    # PG pushes the heatmap UP on supp_j. Averaged 1/2-1/2 with the region-drop
    # group; KL applied once at lambda_kl.
    source_map_group: bool = True
    source_map_policy_sigma: float = 1.0   # σ for the policy's own foreground
    # Additive supplement mechanism (the only supported source-map mode).
    source_map_supp_additive: bool = True
    source_map_supp_k_max: int = 4         # max supp regions per sample
    source_map_supp_min_cells: int = 1     # drop supp regions smaller than this
    # multilayer: pool SUPP from ALL ev_maps layers, IoU>thr merge (majority
    # vote). The only supported pooling mode.
    source_map_supp_multilayer: bool = True
    source_map_supp_merge_iou: float = 0.5
    # supp group height: clipped-logit (same g0-referenced transform as the
    # region-drop group) instead of raw log_p.
    source_map_supp_logit_clip: bool = True
    # True: every merged supp region has weight 1 (independent region-level
    # terms). False: weight = (#layers voting for the region) / n_layers.
    source_map_supp_uniform_weight: bool = True

    # ---- attention-fg gradient mask (source-map vote) ----
    # For each POSITIVE region (c_k = R(∅)−R(drop k) > 0), keep the per-cell
    # policy gradient only where the cached source-map vote (#layers whose
    # single-region foreground includes the cell, from ev_maps) exceeds
    # attention_fg_vote_threshold; stop-gradient (detach) the below-threshold
    # cells. Negative/zero-c_k regions + cells outside all components are
    # unaffected. Needs ev_maps; zero extra forward cost.
    attention_fg_gradient_mask: bool = True
    attention_fg_vote_threshold: int = 3
    # If a POSITIVE region has EVERY cell below the vote threshold, zeroing
    # its whole gate would detach the region entirely (a foreground-collapse
    # channel). With this flag ON such fully-below regions keep the FULL
    # region gradient instead. Partially-covered regions keep the per-cell mask.
    attention_fg_fully_below_full_grad: bool = True
    # k×k max-pool dilation of the binarized vote mask before it is used as
    # the gradient-keep mask (a cell is foreground if ANY cell in its k×k
    # window passed the vote). 0 or 1 = off (raw vote mask). Odd k.
    attention_fg_vote_dilation_k: int = 3

    # ---- top-R + action set ----
    R_max: int = 6                    # |Ω| caps at 1+6 singleton drops
    enumerate_threshold: int = 12     # if |Ω| <= this, enumerate all actions
    rollout_K: int = 4                # else Gumbel-top-K WOR with this size

    # ---- policy ----
    softmax_T: float = 1.0
    removal_beta_keep: float = 1.0
    removal_gamma_keep: float = 0.6

    # ---- reward shaping ----
    # Linear size penalty: R = h_a − β·keep_frac. 0.0 in the released recipe.
    reward_size_beta: float = 0.0
    # Linear N_cc reward shaping: R_extra = −α · N_cc(Y_keep) where
    # N_cc = K − |D| (top-R components are disjoint). Constant marginal bonus
    # per dropped component. 0.0 in the released recipe.
    reward_linear_ncc_alpha: float = 0.0
    # Reward HEIGHT = clipped log-odds contribution about the ∅ operating
    # point: h_a = g0 + clip(logit(p_a) − g0, ±δ), g0 = logit(p_∅). Raw log_p
    # saturates near p≈1; the logit Δ is difficulty-linear and the clip bounds it.
    reward_logit_clip: bool = True
    reward_logit_clip_delta: float = 5.0

    # ---- loss weights ----
    lambda_kl: float = 0.5
    lambda_anchor_k1: float = 1.0

    # ---- online Phase-A reference for the KL anchor ----
    # When True, the trainer uses a frozen Phase-A twig (RefTwigModule) to
    # compute the KL-anchor reference at the SAME (Hg, Wg) as the live policy
    # heatmap, instead of a cached p_ref captured at the filter's resolution.
    online_p_ref: bool = True

    # ---- winnability-weighted PG ----
    # Weight each sample's PG loss by w = max_a p(a), EMA-normalized to
    # mean≈1 so the overall gradient scale (and LR) stays put while hopeless
    # samples (max_p≈0) contribute ~0 instead of z-scored noise.
    winnability_weight: str = "max_p"        # "off" | "max_p"
    winnability_ema_decay: float = 0.99
    winnability_floor: float = 0.05          # floor on the EMA denominator
    winnability_max: float = 3.0             # clip on w_used
    winnability_ema_state: float = -1.0      # runtime EMA (mutated in place)

    # ---- action set + advantage ----
    # singleton_only_actions=True: enumerate_removal_actions returns only
    # [∅, drop_0, drop_1, ..., drop_{R-1}] — no pair drops (pair rewards from
    # the LM are noisy because spurious context dependencies compound
    # non-additively).
    singleton_only_actions: bool = True
    # advantage_std_eps: additional constant added to reward_std in the
    # advantage denominator, damping noise amplification when the reward
    # variance is small: advantage = (R − baseline) / (std + advantage_std_eps).
    advantage_std_eps: float = 1.0

    # ---- placebo-bar subtractor ----
    # "placebo": score 2 additive null probes (keep = ∅-union ∪ a connected
    # low-evidence blob of the median dropped-region area) with the frozen
    # reward; bar = kappa · max|h(∅) − h(probe)| (capped at placebo_bar_max)
    # is the per-sample reward-noise magnitude. The region-drop advantage is
    # then ∅-baselined against R(∅) − bar (policy_gradient_loss_empty_baseline)
    # so a region is kept only if its contribution beats the bar. EMA fallback
    # when the low-evidence zone cannot host the blobs. "none" = plain z-score.
    subtractor_mode: str = "placebo"         # "placebo" | "none"
    placebo_kappa: float = 1.25              # 1.25 (Qwen3.5-4B) / 1.0 (9B, Qwen2.5-7B)
    placebo_p_thresh: float = 0.02           # low-evidence zone: p < this
    placebo_bar_max: float = 1.0             # hard cap on the bar

    # ---- SD-RPN scoring convention (policy + online ref) ----
    # Apply RoPE to the SD-RPN training-time scoring (modeling-side) and to
    # the RefTwigModule QK^T — closes the train/eval scoring-convention gap.
    roi_score_with_rope: bool = True
    # Query token set for the SD-RPN Q*K^T scoring: "last_prompt" = the single
    # position just before the first response token (matches the inference
    # path's proxy query); "response_tokens" = mean over the gold-answer rows.
    roi_score_query_mode: str = "last_prompt"
    # Independent control over the online ref_twig's QK^T scoring.
    ref_score_with_rope: bool = True
    ref_score_query_mode: str = "last_prompt"
    # Reward LM teacher-forcing format: prepend the official Qwen3.5-VL
    # ``enable_thinking=False`` prefix (family-gated inside RewardModel) ...
    reward_disable_thinking_prefix: bool = True
    # ... and exclude the closing `<|im_end|>` from the answer mask.
    reward_skip_trailing_eos: bool = True


# ============================================================================
# Helpers
# ============================================================================

def _aggregate_groups(subs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine a list of per-group loss dicts into one (equal weight).

    ``loss = mean(sub losses)`` (differentiable). The KL anchor is group-
    independent, so averaging N copies returns the full ``lambda_kl * KL``;
    PG and the K=1 anchor average per-group. The caller may override
    ``loss`` afterwards (the trainer uses the 1/2-1/2 split explicitly).

    Scalar diagnostics are averaged; ``rewards`` / ``advantage`` tensors are
    concatenated across groups for logging. ``K`` reports the finest split
    (max), ``n_actions`` the sum across groups.
    """
    losses = [s["loss"] for s in subs]
    out: Dict[str, Any] = {}
    out["loss"] = torch.stack(losses).mean() if len(losses) > 1 else losses[0]
    out["K"] = max(int(s.get("K", 0)) for s in subs)
    out["n_actions"] = sum(int(s.get("n_actions", 0)) for s in subs)
    out["branch"] = "multigroup"
    out["n_groups"] = len(subs)
    _skip = {"loss", "K", "n_actions", "branch", "rewards", "advantage"}

    def _scalar(v):
        # Accept python numbers AND 0-dim tensors (loss_kl / loss_policy /
        # loss_anchor_k1 come back as detached 0-dim tensors). Returning None
        # for anything else excludes it from the averaged diagnostics.
        if isinstance(v, (int, float)):
            return float(v)
        if torch.is_tensor(v) and v.numel() == 1:
            return float(v.detach())
        return None

    _float_keys = set()
    for s in subs:
        for k, v in s.items():
            if k not in _skip and _scalar(v) is not None:
                _float_keys.add(k)
    for k in _float_keys:
        vals = [_scalar(s[k]) for s in subs if k in s and _scalar(s[k]) is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    for k in ("rewards", "advantage"):
        ts = [s[k] for s in subs
              if s.get(k) is not None and hasattr(s[k], "numel") and s[k].numel() > 0]
        if ts:
            out[k] = torch.cat([t.reshape(-1) for t in ts])
    return out


_PLACEBO_EMA = {"v": None}


def _build_placebo_blobs(p_np, exclude_mask, area, n_blobs, p_thresh, seed):
    """Placebo-bar subtractor: grow ``n_blobs`` connected evidence-free blobs
    of ``area`` cells inside the low-evidence zone (p < p_thresh, outside
    exclude_mask + 1-cell dilation). Scored as additive null probes;
    deterministic via ``seed``. Returns [] when the zone cannot host the
    blobs (caller falls back to the EMA bar)."""
    import random as _rnd
    from scipy.ndimage import binary_dilation as _bdil
    Hg, Wg = p_np.shape
    _ex = (_bdil(exclude_mask > 0, iterations=1)
           if exclude_mask is not None else np.zeros_like(p_np, dtype=bool))
    zone = (p_np < float(p_thresh)) & (~_ex)
    cells = np.argwhere(zone)
    if len(cells) < area * n_blobs:
        return []
    rng = _rnd.Random(int(seed))
    blobs = []
    used = np.zeros_like(zone, dtype=bool)
    for _ in range(n_blobs):
        order = list(range(len(cells)))
        rng.shuffle(order)
        blob = None
        for oi in order:
            r0, c0 = int(cells[oi][0]), int(cells[oi][1])
            if used[r0, c0]:
                continue
            m = np.zeros_like(zone, dtype=bool)
            q = [(r0, c0)]
            m[r0, c0] = True
            cnt = 1
            while q and cnt < area:
                r, c = q.pop(0)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < Hg and 0 <= cc < Wg and zone[rr, cc]
                            and not m[rr, cc] and not used[rr, cc]):
                        m[rr, cc] = True
                        cnt += 1
                        q.append((rr, cc))
                        if cnt >= area:
                            break
            if cnt >= area:
                blob = m
                break
        if blob is None:
            break
        blobs.append(blob.astype(np.uint8))
        used |= blob
    return blobs


# ============================================================================
# Source-map supplement group
# ============================================================================

def compute_source_map_group_loss(
    *,
    per_head_scores: torch.Tensor,
    feat_hw: Tuple[int, int],
    ev_maps: Optional[torch.Tensor],
    pil_image,
    question: str,
    gold_answer: str,
    p_ref: torch.Tensor,
    ref_per_head_score: Optional[torch.Tensor],
    reward_model: RewardModel,
    config: PhaseB1Config,
    winn_w: float = 1.0,
) -> Dict[str, Any]:
    """Answer→image SUPP additive group (the "discover" channel).

    SUPP = union over evidence-map layers of (layer fg − policy fg), split
    into connected regions and IoU-merged across layers. Reward each supp
    region additively by c_j = R(cores∪SUPP) − R(cores∪SUPP\\supp_j)
    (Δheight − β·size); advantage = c_j/(std+eps) (raw, no mean-centering);
    the PG pushes the heatmap UP on supp_j. ``lambda_kl * KL`` is included
    (so the averaged multi-group loss keeps the full KL).

    Returns the same dict shape as :func:`compute_phase_b1_loss_v2` so it
    feeds :func:`_aggregate_groups`.
    """
    device = per_head_scores.device
    Hg, Wg = int(feat_hw[0]), int(feat_hw[1])
    n_grid = Hg * Wg

    z_flat = per_head_scores.float().mean(dim=0)
    Z_theta = z_flat.view(Hg, Wg)
    P = torch.sigmoid(Z_theta)

    # --- reference (z_ref) for the KL anchor: mirror compute_phase_b1_loss_v2 ---
    if ref_per_head_score is not None:
        ref_flat = ref_per_head_score.float().mean(dim=0).to(device=device)
        z_ref = ref_flat.view(Hg, Wg).to(dtype=P.dtype).detach()
    else:
        p_ref_t = p_ref.to(device=device, dtype=P.dtype).detach()
        if p_ref_t.shape != (Hg, Wg):
            p_ref_t = F.interpolate(
                p_ref_t.unsqueeze(0).unsqueeze(0), size=(Hg, Wg),
                mode="nearest").squeeze(0).squeeze(0)
        z_ref = torch.logit(p_ref_t.clamp(min=1e-5, max=1.0 - 1e-5))

    if config.lambda_kl > 0:
        _kl = kl_anchor_loss_logit(Z_theta, z_ref)
        _kl_loss, _kl_det = config.lambda_kl * _kl, _kl.detach()
    else:
        _kl_loss = P.sum() * 0.0
        _kl_det = _kl_loss.new_zeros(())
    _zero = _kl_loss.new_zeros(())

    def _kl_anchor_only(tag):
        return {"loss": _kl_loss, "loss_kl": _kl_det, "loss_policy": _zero,
                "loss_anchor_k1": _zero, "K": 0, "n_actions": 0, "branch": tag,
                "reward_mean": _zero, "reward_std": _zero,
                "supp_n": 0.0, "supp_adv_abs_mean": _zero, "supp_c_mean": _zero}

    if (not config.source_map_supp_additive
            or ev_maps is None or ev_maps.numel() == 0):
        # No evidence maps for this sample -> KL anchor only.
        return _kl_anchor_only("source_map_kl_anchor")

    from scipy.ndimage import label as _cc_label

    _em = ev_maps
    if _em.dim() == 2:
        _em = _em.unsqueeze(0)
    _em = _em.to(torch.float32)
    if _em.shape[-2:] != (Hg, Wg):
        _em = F.interpolate(_em.unsqueeze(1), size=(Hg, Wg),
                            mode="nearest").squeeze(1)
    _em = (_em > 0.5).cpu().numpy().astype(np.uint8)
    # policy foreground (cores) at source_map_policy_sigma
    _ps2 = float(config.source_map_policy_sigma)
    _pk2 = max(1, int(round(config.smooth_kernel * _ps2 / max(1e-6, config.smooth_sigma))))
    if _pk2 % 2 == 0:
        _pk2 += 1
    try:
        _ptop = rank_top_r(extract_components(
            P.detach(), Z_theta.detach(), smooth_kernel=_pk2, smooth_sigma=_ps2,
            threshold_mode=config.threshold_mode,
            fixed_threshold=config.fixed_threshold,
            ratio_thresh=config.ratio_thresh, peak_fraction=config.peak_fraction,
            min_gate=config.min_gate, score_p=config.score_p,
            score_beta=config.score_beta, score_gamma=config.score_gamma,
            connectivity=config.connectivity), R=config.R_max)
    except Exception:
        _ptop = []
    cores = np.zeros((Hg, Wg), dtype=np.uint8)
    for _c in _ptop:
        cores = np.maximum(cores, _c.mask.astype(np.uint8))
    if int(cores.sum()) == 0:
        return _kl_anchor_only("supp_no_cores")
    # ---- build SUPP regions (+ per-region vote weight) ----------------------
    _struct = (np.ones((3, 3), dtype=np.int32)
               if int(config.connectivity) == 2 else None)
    _minc = max(1, int(config.source_map_supp_min_cells))
    _kmax = int(config.source_map_supp_k_max)
    _nlayers = int(_em.shape[0])
    # SUPP from EVERY layer (layer_fg − cores), pooled, then IoU>thr merged.
    # merged mask = majority vote (ave membership ≥ 0.5); vote = #distinct
    # layers in the cluster (cross-layer agreement).
    _miou = float(config.source_map_supp_merge_iou)
    _cand = []   # (mask uint8, layer_idx)
    for _li in range(_nlayers):
        _sl = ((_em[_li] > 0) & (cores == 0)).astype(np.uint8)
        if int(_sl.sum()) == 0:
            continue
        _l2, _n2 = _cc_label(_sl > 0, structure=_struct)
        for _ri in range(1, _n2 + 1):
            _rm = (_l2 == _ri).astype(np.uint8)
            if int(_rm.sum()) >= _minc:
                _cand.append((_rm, _li))
    if not _cand:
        return _kl_anchor_only("supp_covered")
    _cand.sort(key=lambda t: -int(t[0].sum()))
    _used = [False] * len(_cand)
    supp_regions, supp_votes = [], []
    for _i in range(len(_cand)):
        if _used[_i]:
            continue
        _seed, _sl0 = _cand[_i]
        _used[_i] = True
        _members, _layers = [_seed], {_sl0}
        for _j in range(_i + 1, len(_cand)):
            if _used[_j]:
                continue
            _mj, _lj = _cand[_j]
            _inter = int(((_seed > 0) & (_mj > 0)).sum())
            _uni = int(((_seed > 0) | (_mj > 0)).sum())
            if _uni > 0 and (_inter / _uni) > _miou:
                _used[_j] = True
                _members.append(_mj)
                _layers.add(_lj)
        _stack = np.stack(_members, axis=0).astype(np.float32)
        _merged = (_stack.mean(axis=0) >= 0.5).astype(np.uint8)
        if int(_merged.sum()) == 0:                 # fallback: union
            _merged = (_stack.sum(axis=0) > 0).astype(np.uint8)
        supp_regions.append(_merged)
        supp_votes.append(len(_layers))
    _ord = sorted(range(len(supp_regions)),
                  key=lambda i: -int(supp_regions[i].sum()))[:_kmax]
    supp_regions = [supp_regions[i] for i in _ord]
    supp_weights = [supp_votes[i] / float(_nlayers) for i in _ord]
    if config.source_map_supp_uniform_weight:
        # every merged supp region contributes as an independent
        # region-level term with weight 1 (no vote-altitude divider).
        supp_weights = [1.0] * len(supp_weights)
    Ks = len(supp_regions)
    if Ks == 0:
        return _kl_anchor_only("supp_no_regions")
    supp_union = np.zeros((Hg, Wg), dtype=np.uint8)
    for _m in supp_regions:
        supp_union = np.maximum(supp_union, _m)
    # ---- action set: leave-one-out from the full set. baseline = cores∪SUPP;
    # action_j DROPS supp_j (keep = full\supp_j) and pushes supp_j up;
    # c_j = R(full) − R(drop_j).
    push_regions = [supp_regions[j] for j in range(Ks)]
    act_w = [float(supp_weights[j]) for j in range(Ks)]
    full = np.maximum(cores, supp_union)
    keep_list = [full.astype(np.uint8)] + [
        ((full > 0) & (_pr == 0)).astype(np.uint8) for _pr in push_regions]
    keep_np = np.stack(keep_list, axis=0).astype(np.uint8)   # (1+A, Hg, Wg)
    _nA = keep_np.shape[0] - 1               # number of push-actions
    _pils = _build_masked_pils(pil_image.convert("RGB"), keep_np)
    _lp = reward_model.compute_logprobs(
        masked_images=_pils, question=question, gold_answer=gold_answer,
        device=device, reduction="mean").to(dtype=P.dtype)
    _kt = torch.from_numpy(keep_np).to(device=device, dtype=P.dtype)
    _kf = _kt.sum(dim=(-1, -2)) / float(n_grid)
    _beta = float(config.reward_size_beta)
    # shaped reward = height − β·keep_frac. height = raw log_p OR — under
    # source_map_supp_logit_clip — the SAME g0-referenced clipped logit the
    # region-drop group uses, with the FULL mask (no-drop, index 0) as the g0
    # operating point, so both PG groups live in the same reward space.
    if config.source_map_supp_logit_clip:
        _spe = 1e-6
        _sp = torch.exp(_lp).clamp(max=1.0 - _spe)
        _slogit = _lp - torch.log1p(-_sp)               # logit(p) = log p − log(1−p)
        _sg0 = _slogit[0].detach()                       # g0 = logit(p_full)
        _scd = float(config.reward_logit_clip_delta)
        _hgt = _sg0 + torch.clamp(_slogit - _sg0, min=-_scd, max=_scd)
    else:
        _hgt = _lp
    _R = _hgt - _beta * _kf
    _eps = float(config.advantage_std_eps) if config.advantage_std_eps > 0 else 1.0
    # contribution c_j = R(full) − R(full\r_j) (drop). raw c / std, NO mean-centering.
    c = (_R[0] - _R[1:]).detach()
    _std = c.std() if _nA > 1 else c.new_zeros(())
    adv = (c / (_std + _eps)).detach()
    # per-region weight: 1.0 (uniform) or v/n_layers (cross-layer agreement).
    _w = torch.tensor(act_w, device=device, dtype=P.dtype)
    _logsig = F.logsigmoid(Z_theta)
    _logpi = torch.stack([
        _logsig[torch.from_numpy(_m.astype(np.bool_)).to(device=device)].mean()
        for _m in push_regions])
    _cpos = (c.detach() > 0)
    _n_pos = float(_cpos.sum().item())
    pg_loss = -(_w * adv * _logpi).sum()     # ascent on weighted-adv inclusion
    loss = _kl_loss + pg_loss * float(winn_w)
    _vmean = float(np.mean(supp_weights) * _nlayers)
    return {
        "loss": loss, "loss_kl": _kl_det, "loss_policy": pg_loss.detach(),
        "loss_anchor_k1": _zero, "K": Ks, "n_actions": int(1 + _nA),
        "branch": f"source_map_supp_ml_v{_vmean:.2f}",
        "reward_mean": _R.mean().detach(), "reward_std": _R.std().detach(),
        "supp_n": float(Ks), "supp_n_pos": _n_pos,
        "supp_adv_abs_mean": adv.abs().mean().detach(),
        "supp_c_mean": c.mean().detach(),
        "supp_vote_mean": _vmean,
        "rewards": _R.detach(), "advantage": adv,
    }


# ============================================================================
# Region-drop group (pure loss function)
# ============================================================================

def compute_phase_b1_loss_v2(
    per_head_scores: torch.Tensor,
    feat_hw: Tuple[int, int],
    pil_image,
    question: str,
    gold_answer: str,
    p_ref: torch.Tensor,
    p_ref_binary: Optional[torch.Tensor],
    reward_model: RewardModel,
    config: PhaseB1Config,
    rng: Optional[torch.Generator] = None,
    ref_per_head_score: Optional[torch.Tensor] = None,
    # attention-fg gradient mask: cached source-map vote (L,Hg,Wg)
    ev_maps: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compute the connected-component GR-REINFORCE loss for one prompt.

    Args:
        per_head_scores: ``(num_heads, k_img)`` raw logits with grad,
            from ``model.forward(..., return_per_head_score=True)``.
            ``k_img == Hg * Wg``.
        feat_hw: ``(Hg, Wg)`` so we can reshape ``per_head_scores``.
        pil_image: original (un-masked) PIL image.
        question: user prompt text.
        gold_answer: target answer text whose log-prob is the reward.
        p_ref: ``(Hg, Wg)`` cached Phase-A ``sigma(Z_PhaseA)``, no grad.
            Ignored when ``ref_per_head_score`` is given (online ref).
        p_ref_binary: ``(Hg, Wg)`` Phase-A binarized foreground for the
            K=1 anchor. If ``None``, computed on the fly via
            ``p_ref > config.fixed_threshold``.
        reward_model: configured :class:`RewardModel` wrapper.
        config: hyperparameters.
        rng: optional torch RNG for deterministic Gumbel sampling.
        ref_per_head_score: ``(num_heads, k_img)`` frozen Phase-A twig
            scores on the same grid (online_p_ref).
        ev_maps: ``(L, Hg, Wg)`` cached evidence maps (attention-fg mask).

    Returns:
        Dict with at least ``'loss'`` (scalar tensor with grad). Other
        keys (``'loss_policy'``, ``'loss_kl'``, ``'loss_anchor_k1'``,
        ``'K'``, ``'n_actions'``, ``'rewards'``, ``'advantage'``) are
        present where applicable for diagnostics.
    """
    device = per_head_scores.device
    Hg, Wg = int(feat_hw[0]), int(feat_hw[1])
    n_grid = Hg * Wg

    # --- Mean over heads -> single Z_theta(Hg, Wg) with grad ---
    z_flat = per_head_scores.float().mean(dim=0)
    if z_flat.shape[0] != n_grid:
        raise ValueError(
            f"k_img mismatch: per_head_scores last-dim {z_flat.shape[0]} "
            f"!= Hg*Wg {n_grid} (Hg={Hg}, Wg={Wg})"
        )
    Z_theta = z_flat.view(Hg, Wg)
    P = torch.sigmoid(Z_theta)

    # Diagnostic stats (detached, scalar, used for tensorboard).
    _heatmap_max = float(P.detach().max())
    _heatmap_pos_frac = float(
        (P.detach() > config.fixed_threshold).to(P.dtype).mean()
    )

    # online_p_ref: when the frozen Phase-A ref twig supplied a per-head
    # score for this sample, derive p_ref / p_ref_binary from it inline
    # (resolution matches Z_theta exactly by construction — no
    # nearest-upsampling needed). Cached p_ref is ignored in this path.
    # Otherwise fall through to the cached-ref behavior below.
    if ref_per_head_score is not None:
        ref_flat = ref_per_head_score.float().mean(dim=0).to(device=device)
        if ref_flat.numel() != n_grid:
            raise ValueError(
                f"ref_per_head_score numel {ref_flat.numel()} != Hg*Wg {n_grid}; "
                "ref-twig grid must match policy grid by construction."
            )
        Z_ref_t = ref_flat.view(Hg, Wg).to(dtype=P.dtype).detach()
        p_ref = torch.sigmoid(Z_ref_t).detach()
        p_ref_binary = (p_ref > config.fixed_threshold).to(dtype=P.dtype)
    else:
        p_ref = p_ref.to(device=device, dtype=P.dtype).detach()
        # Align p_ref to the policy heatmap shape. Filter-time and train-time
        # image processing can produce off-by-one image_grid_thw values for
        # the same image, so we resize p_ref via nearest-neighbor (preserves
        # binary-ish foreground regions) instead of crashing on shape
        # mismatch.
        if p_ref.shape != (Hg, Wg):
            p_ref = torch.nn.functional.interpolate(
                p_ref.unsqueeze(0).unsqueeze(0),
                size=(Hg, Wg),
                mode="nearest",
            ).squeeze(0).squeeze(0)
        if p_ref_binary is None:
            p_ref_binary = (p_ref > config.fixed_threshold).to(dtype=P.dtype)
        else:
            p_ref_binary = p_ref_binary.to(device=device, dtype=P.dtype).detach()
            if p_ref_binary.shape != (Hg, Wg):
                p_ref_binary = torch.nn.functional.interpolate(
                    p_ref_binary.unsqueeze(0).unsqueeze(0),
                    size=(Hg, Wg),
                    mode="nearest",
                ).squeeze(0).squeeze(0)

    # Recover z_ref = logit(p_ref) once for the logit-space KL. eps=1e-5
    # is comfortably above fp32 spacing at 1.0 (~5.96e-8) so the clamp
    # actually clips, mapping cached p=0/1 to z_ref=∓11.51 (finite).
    z_ref = torch.logit(p_ref.clamp(min=1e-5, max=1.0 - 1e-5))

    # Smoothing (kernel, sigma). gaussian_blur needs an odd kernel.
    _sk, _ss = int(config.smooth_kernel), float(config.smooth_sigma)
    if _sk % 2 == 0:
        _sk += 1

    # --- Connected components on detached values (CC labeling is non-diff) ---
    components = extract_components(
        P.detach(),
        Z_theta.detach(),
        smooth_kernel=_sk,
        smooth_sigma=_ss,
        threshold_mode=config.threshold_mode,
        fixed_threshold=config.fixed_threshold,
        ratio_thresh=config.ratio_thresh,
        peak_fraction=config.peak_fraction,
        min_gate=config.min_gate,
        score_p=config.score_p,
        score_beta=config.score_beta,
        score_gamma=config.score_gamma,
        connectivity=config.connectivity,
    )
    top_R = rank_top_r(components, R=config.R_max)
    K = len(top_R)

    # ----------- K = 0: empty foreground -> only KL anchor -----------------
    # If lambda_kl == 0 we MUST skip the KL compute outright. Route the zero
    # through P/twig so backward has a grad_fn.
    if K == 0:
        if config.lambda_kl > 0:
            kl = kl_anchor_loss_logit(Z_theta, z_ref)
            loss = config.lambda_kl * kl
            kl_det = kl.detach()
        else:
            loss = P.sum() * 0.0
            kl_det = loss.new_zeros(())
        return {
            "loss": loss,
            "loss_kl": kl_det,
            "loss_policy": loss.new_zeros(()),
            "loss_anchor_k1": loss.new_zeros(()),
            "K": 0,
            "n_actions": 0,
            "branch": "K=0",
            "heatmap_max": _heatmap_max,
            "heatmap_pos_frac": _heatmap_pos_frac,
        }

    # ----------- K = 1: BCE-anchor on heatmap + KL anchor ------------------
    if K == 1:
        if config.lambda_kl > 0:
            kl = kl_anchor_loss_logit(Z_theta, z_ref)
            kl_term = config.lambda_kl * kl
            kl_det = kl.detach()
        else:
            kl_term = P.sum() * 0.0
            kl_det = kl_term.new_zeros(())
        if config.lambda_anchor_k1 > 0:
            anchor = k1_anchor_bce_loss_logit(Z_theta, p_ref_binary)
            anchor_term = config.lambda_anchor_k1 * anchor
            anchor_det = anchor.detach()
        else:
            anchor_term = P.sum() * 0.0
            anchor_det = anchor_term.new_zeros(())
        loss = anchor_term + kl_term
        return {
            "loss": loss,
            "loss_kl": kl_det,
            "loss_anchor_k1": anchor_det,
            "loss_policy": loss.new_zeros(()),
            "K": 1,
            "n_actions": 1,
            "branch": "K=1",
            "heatmap_max": _heatmap_max,
            "heatmap_pos_frac": _heatmap_pos_frac,
        }

    # ----------- K >= 2: standard policy gradient --------------------------
    # Score the top-R components with grad through Z_theta.
    component_scores = torch.stack([
        score_component_torch(
            Z_theta,
            torch.from_numpy(c.mask).to(device=device),
            p=config.score_p,
            beta=config.score_beta,
            gamma=config.score_gamma,
        )
        for c in top_R
    ])  # (K,) with grad

    # Build action space: enumerate ∅ + singleton drops (+ pair drops unless
    # singleton_only_actions). Gumbel-top-K WOR picks which to reward below.
    actions = enumerate_removal_actions(
        top_R, singleton_only=config.singleton_only_actions,
    )

    if not actions:
        # Defensive: shouldn't happen for K >= 2, but if it does just KL.
        kl = kl_anchor_loss_logit(Z_theta, z_ref)
        loss = config.lambda_kl * kl
        return {
            "loss": loss,
            "loss_kl": kl.detach(),
            "loss_policy": loss.new_zeros(()),
            "loss_anchor_k1": loss.new_zeros(()),
            "K": K, "n_actions": 0, "branch": "no_actions",
        }

    # log pi(a) over the action set; gradient flows via component_scores.
    log_pi = compute_log_pi(
        component_scores,
        actions,
        n_grid,
        T=config.softmax_T,
        beta_keep=config.removal_beta_keep,
        gamma_keep=config.removal_gamma_keep,
    )

    # Pick which actions to compute rewards for via Gumbel-top-K WOR.
    selected_idx = select_actions_for_rewards(
        log_pi,
        enumerate_threshold=config.enumerate_threshold,
        rollout_K=config.rollout_K,
        generator=rng,
    )
    selected_idx = selected_idx.to(device=device)

    selected_actions = [actions[int(i)] for i in selected_idx.cpu().tolist()]

    # Build masked images (crop-to-bbox + mean-fill) and call the frozen
    # reward LLM.
    keep_masks_np = np.stack([a.keep_mask for a in selected_actions], axis=0)
    keep_masks_t = torch.from_numpy(keep_masks_np)
    src_pil = pil_image.convert("RGB")

    masked_pils = _build_masked_pils(src_pil, keep_masks_np)
    log_probs = reward_model.compute_logprobs(
        masked_images=masked_pils,
        question=question,
        gold_answer=gold_answer,
        device=device,
        reduction="mean",
    ).to(dtype=P.dtype)  # (n_sel,)

    keep_areas = keep_masks_t.float().sum(dim=(-1, -2))  # (n_sel,)
    keep_frac = (keep_areas / float(n_grid)).to(device=device, dtype=P.dtype)

    # ---- placebo-bar subtractor: score 2 additive null probes (keep =
    # ∅-union ∪ low-evidence blob, area = median dropped-region area) with
    # the same frozen reward; the bar (computed below in height units) is
    # the per-sample reward-noise magnitude. ----
    _pl_mode = str(config.subtractor_mode) == "placebo"
    _pl_lp = None
    _pl_fallback = 0
    if _pl_mode:
        _union_np = None
        for _a in selected_actions:
            if not _a.discard:
                _union_np = _a.keep_mask.astype(np.uint8)
                break
        _drop_areas = [int(_union_np.sum() - _a.keep_mask.sum())
                       for _a in selected_actions if len(_a.discard) == 1] \
            if _union_np is not None else []
        if _union_np is not None and _drop_areas:
            import zlib as _zlib
            _pl_seed = _zlib.crc32(
                (str(question) + "|" + str(gold_answer)).encode()) & 0x7FFFFFFF
            _blobs = _build_placebo_blobs(
                P.detach().float().cpu().numpy(), _union_np,
                max(1, int(np.median(_drop_areas))), 2,
                float(config.placebo_p_thresh), _pl_seed)
            if _blobs:
                _pl_keeps = np.stack(
                    [np.maximum(_union_np, _bm) for _bm in _blobs], axis=0)
                _pl_lp = reward_model.compute_logprobs(
                    masked_images=_build_masked_pils(src_pil, _pl_keeps),
                    question=question, gold_answer=gold_answer,
                    device=device, reduction="mean").to(dtype=P.dtype)

    # ∅ (keep-all) index = the per-sample operating point p̄ = p(∅).
    _empty_idx = next(
        (i for i, a in enumerate(selected_actions) if not a.discard), 0)
    _beta = float(config.reward_size_beta)
    _alpha = float(config.reward_linear_ncc_alpha)

    # ---- logit-clip reward HEIGHT (saturation-free) ----
    # Replace the raw log_p height with a clipped log-odds contribution about the
    # ∅ operating point: h_a = g0 + clip(logit(p_a) − g0, ±δ), g0 = logit(p_∅).
    # Raw log_p saturates near p≈1 (a helpful region yields a tiny Δlog_p the
    # size penalty can dominate); logit Δ is difficulty-linear, the clip bounds it.
    if config.reward_logit_clip:
        _eps = 1e-6
        _p = torch.exp(log_probs).clamp(max=1.0 - _eps)
        _logit = log_probs - torch.log1p(-_p)        # logit = log p − log(1−p)
        _g0 = _logit[_empty_idx].detach()
        _cd = float(config.reward_logit_clip_delta)
        _heights = _g0 + torch.clamp(_logit - _g0, min=-_cd, max=_cd)
    else:
        _heights = log_probs

    # placebo bar in HEIGHT units (max |h(∅) − h(probe)| over the 2 probes);
    # EMA fallback when the low-evidence zone could not host the blobs.
    _pl_bar = None
    if _pl_mode:
        if _pl_lp is not None:
            if config.reward_logit_clip:
                _plp = torch.exp(_pl_lp).clamp(max=1.0 - 1e-6)
                _pl_h = _g0 + torch.clamp(
                    (_pl_lp - torch.log1p(-_plp)) - _g0, min=-_cd, max=_cd)
            else:
                _pl_h = _pl_lp
            # bar = kappa * measured noise magnitude, hard-capped at
            # placebo_bar_max.
            _pl_kappa = float(config.placebo_kappa)
            _pl_bar = min(
                _pl_kappa * float(
                    (_heights[_empty_idx].detach() - _pl_h.detach()).abs().max().item()),
                float(config.placebo_bar_max))
            _ema = _PLACEBO_EMA["v"]
            _PLACEBO_EMA["v"] = (_pl_bar if _ema is None
                                 else 0.9 * _ema + 0.1 * _pl_bar)
        else:
            _pl_bar = _PLACEBO_EMA["v"]
            _pl_fallback = 1

    # comparison diagnostic: the group-mean contribution c̄ (height units) —
    # the implicit adaptive subtractor a z-score/mean-center form would use.
    _drop_ix_diag = [i for i, _a in enumerate(selected_actions) if _a.discard]
    _g1_c_mean = (float((_heights[_empty_idx].detach()
                         - _heights[_drop_ix_diag].detach()).mean().item())
                  if _drop_ix_diag else float("nan"))

    size_penalty = _beta * keep_frac
    rewards = _heights - size_penalty

    # ---- linear N_cc reward shaping ----
    # Subtract α·N_cc(Y_keep) where N_cc = K - |D| (top-R components are
    # disjoint by construction). Constant marginal bonus per dropped component.
    if _alpha > 0.0:
        n_kept = torch.tensor(
            [max(1, K - len(a.discard)) for a in selected_actions],
            device=rewards.device, dtype=rewards.dtype,
        )
        rewards = rewards - _alpha * n_kept

    # ---- h_ave / Ncc0 / ∅-reward diagnostic ----
    # h0 = ∅ height (g0); h_ave_drop = mean height of the singleton-drop actions;
    # Ncc0 = components ∅ keeps (=K). R_empty, R_ave_drop = the corresponding
    # rewards.
    _dd = rewards.detach(); _hh = _heights.detach()
    _drop_ix = [i for i, a in enumerate(selected_actions) if a.discard]
    diag_h_empty = float(_hh[_empty_idx].item())
    diag_ncc0 = float(K)
    diag_R_empty = float(_dd[_empty_idx].item())
    if _drop_ix:
        _ix = torch.tensor(_drop_ix, device=_hh.device)
        diag_h_ave_drop = float(_hh.index_select(0, _ix).mean().item())
        diag_R_ave_drop = float(_dd.index_select(0, _ix).mean().item())
    else:
        diag_h_ave_drop = diag_h_empty
        diag_R_ave_drop = diag_R_empty

    # ---- attention-fg gradient mask (vote>thr) on POSITIVE regions ----
    # For each top-R component with positive contribution c_k = R(∅)−R(drop k),
    # keep the per-cell policy gradient only where the cached source-map vote
    # (#layers whose foreground includes the cell) exceeds the threshold; detach
    # the below-threshold cells. Recompute log_pi from the gated Z (forward
    # values unchanged → action selection / rewards above are unaffected; only
    # the gradient to Z is masked).
    attn_fg_active = 0
    attn_fg_n_pos = 0
    attn_fg_n_pos_fully_below = 0   # positive regions with EVERY cell vote<=thr
    attn_fg_grad_keep_frac = 1.0
    if (config.attention_fg_gradient_mask
            and ev_maps is not None and ev_maps.numel() > 0):
        _vthr = int(config.attention_fg_vote_threshold)
        _em = ev_maps.to(device=device).float()
        if tuple(_em.shape[-2:]) != (Hg, Wg):
            _em = torch.nn.functional.interpolate(
                _em.unsqueeze(1), size=(Hg, Wg), mode="nearest").squeeze(1)
        vote_keep = (_em.sum(dim=0) > float(_vthr))          # bool (Hg, Wg)
        # k×k MAX-pool dilation of the binary vote mask: a cell is foreground
        # if ANY (≥1) cell in its k×k window passed the vote.
        _dilk = int(config.attention_fg_vote_dilation_k)
        if _dilk > 1:
            _vk = vote_keep.to(Z_theta.dtype)[None, None]
            _vk = torch.nn.functional.max_pool2d(
                _vk, kernel_size=_dilk, stride=1, padding=_dilk // 2)
            vote_keep = (_vk[0, 0] > 0)
        R_zero = float(rewards[_empty_idx].detach().item())
        ck_by_comp = {}
        for _si, _a in enumerate(selected_actions):
            if len(_a.discard) == 1:
                ck_by_comp[int(_a.discard[0])] = (
                    R_zero - float(rewards[_si].detach().item()))
        grad_gate = torch.ones((Hg, Wg), device=device, dtype=Z_theta.dtype)
        for _ci, _comp in enumerate(top_R):
            _ck = ck_by_comp.get(_ci, None)
            if _ck is None or _ck <= 0.0:
                continue                       # only POSITIVE regions are masked
            attn_fg_n_pos += 1
            _cm = torch.from_numpy(_comp.mask.astype(np.bool_)).to(device=device)
            _vk_region = vote_keep[_cm]
            # fully-below: a positive region whose EVERY cell is <=thr would get
            # its entire gate zeroed -> wholly detached (no PG grow/shrink).
            _fully_below = (_vk_region.numel() > 0 and not bool(_vk_region.any()))
            if _fully_below:
                attn_fg_n_pos_fully_below += 1
            if _fully_below and config.attention_fg_fully_below_full_grad:
                # keep the FULL region gradient instead of silencing it.
                grad_gate[_cm] = 1.0
            else:
                grad_gate[_cm] = _vk_region.to(Z_theta.dtype)
        if attn_fg_n_pos > 0:
            Z_gated = grad_gate * Z_theta + (1.0 - grad_gate) * Z_theta.detach()
            component_scores = torch.stack([
                score_component_torch(
                    Z_gated, torch.from_numpy(c.mask).to(device=device),
                    p=config.score_p, beta=config.score_beta,
                    gamma=config.score_gamma)
                for c in top_R
            ])
            log_pi = compute_log_pi(
                component_scores, actions, n_grid, T=config.softmax_T,
                beta_keep=config.removal_beta_keep,
                gamma_keep=config.removal_gamma_keep,
            )
            attn_fg_active = 1
            attn_fg_grad_keep_frac = float(grad_gate.mean().item())

    # Policy gradient.
    log_pi_selected = log_pi.index_select(0, selected_idx)
    if _pl_mode and _pl_bar is not None:
        # ---- ∅-baselined, no-mean-center advantage with the placebo bar ----
        # R'_k = R(drop k) − (R(∅) − bar); advantage = R'_k / (std(R') + eps).
        # The ∅ action is the baseline (and excluded from the PG group). Falls
        # back to the standard z-score if no ∅ action is present.
        empty_idx = None
        for i, a in enumerate(selected_actions):
            if not a.discard:
                empty_idx = i
                break
        if empty_idx is None:
            pol = policy_gradient_loss(
                log_pi_selected, rewards.detach(),
                std_regularizer=config.advantage_std_eps,
            )
        else:
            keep_mask = torch.ones(
                rewards.shape[0], dtype=torch.bool, device=rewards.device,
            )
            keep_mask[empty_idx] = False
            # reference = R(∅) − noise bar, so a region is kept only if its
            # contribution beats the bar.
            r_empty = rewards[empty_idx].detach() - float(_pl_bar)
            pg_rewards = rewards[keep_mask]
            pg_log_pi = log_pi_selected[keep_mask]
            pol = policy_gradient_loss_empty_baseline(
                pg_log_pi, pg_rewards.detach(), r_empty,
                std_regularizer=config.advantage_std_eps,
            )
    else:
        pol = policy_gradient_loss(
            log_pi_selected, rewards.detach(),
            std_regularizer=config.advantage_std_eps,
        )

    if config.lambda_kl > 0:
        kl = kl_anchor_loss_logit(Z_theta, z_ref)
        kl_term = config.lambda_kl * kl
        kl_det = kl.detach()
    else:
        kl_term = P.sum() * 0.0
        kl_det = kl_term.new_zeros(())

    # ---- EMA-normalized winnability weight on the PG term ----
    # w = max_a p(a); divide by an EMA of w so the mean weight ≈ 1 (overall
    # gradient scale / LR unchanged) while hopeless samples (max_p ≈ 0)
    # contribute ~0 gradient instead of z-scored noise. The weight is a
    # detached scalar; KL + K=1 anchor are NOT weighted.
    winn_w = 1.0
    if config.winnability_weight != "off" and log_probs.numel() > 0:
        _w_raw = float(np.exp(float(log_probs.detach().max().item())))
        _ema = config.winnability_ema_state
        _ema = _w_raw if _ema < 0.0 else (
            config.winnability_ema_decay * _ema
            + (1.0 - config.winnability_ema_decay) * _w_raw)
        config.winnability_ema_state = _ema
        winn_w = _w_raw / max(_ema, config.winnability_floor)
        winn_w = float(min(max(winn_w, 0.0), config.winnability_max))

    pol_term = pol["loss"] * winn_w
    loss = pol_term + kl_term

    return {
        "loss": loss,
        "loss_policy": pol["loss"].detach(),
        "loss_kl": kl_det,
        "loss_anchor_k1": loss.new_zeros(()),
        "winnability_w": float(winn_w),
        "ncc_alpha_eff": float(_alpha),
        "size_beta_eff": float(_beta),
        "reward_p_empty": float(np.exp(float(log_probs[_empty_idx].detach().item()))),
        **({"placebo_bar": float(_pl_bar)}
           if (_pl_mode and _pl_bar is not None) else {}),
        **({"placebo_fallback": float(_pl_fallback)} if _pl_mode else {}),
        **({"g1_c_mean": _g1_c_mean} if _g1_c_mean == _g1_c_mean else {}),
        "advantage": pol["advantage"].detach(),
        "rewards": rewards.detach(),
        "log_probs": log_probs.detach(),
        "K": K,
        "n_actions": len(actions),
        "n_sampled": len(selected_actions),
        "branch": "K>=2",
        "heatmap_max": _heatmap_max,
        "heatmap_pos_frac": _heatmap_pos_frac,
        "keep_frac_mean": float(keep_frac.mean()),
        # attention-fg gradient mask diagnostics.
        "attn_fg_active": float(attn_fg_active),
        "attn_fg_n_pos": float(attn_fg_n_pos),
        "attn_fg_n_pos_fully_below": float(attn_fg_n_pos_fully_below),
        "attn_fg_grad_keep_frac": float(attn_fg_grad_keep_frac),
        # h_ave / Ncc0 / ∅-reward diagnostic.
        "diag_h_empty": diag_h_empty,
        "diag_h_ave_drop": diag_h_ave_drop,
        "diag_ncc0": diag_ncc0,
        "diag_R_empty": diag_R_empty,
        "diag_R_ave_drop": diag_R_ave_drop,
    }


# ============================================================================
# Mask -> PIL helper
# ============================================================================

# Module-level thread pool, lazy-initialized once and reused across calls so we
# don't pay startup cost for every sample. Keyed on (n_workers).
_MASKED_PIL_POOLS: Dict[int, "object"] = {}


def _get_masked_pil_pool(n_workers: int):
    """Return a process-wide ThreadPoolExecutor sized to ``n_workers``."""
    from concurrent.futures import ThreadPoolExecutor

    pool = _MASKED_PIL_POOLS.get(n_workers)
    if pool is None:
        pool = ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="masked_pil"
        )
        _MASKED_PIL_POOLS[n_workers] = pool
    return pool


# Reward masked-PIL perf defaults. The source image is capped to the reward
# processor's max_pixels (= max visual tokens × patch²) before the per-mask
# np.repeat expansion — lossless, since the masked crop is resized to that
# exact budget by the processor anyway. RewardModel registers its
# image_processor (set_reward_mask_image_processor); until then we fall back
# to a 2048-token cap. Mask building uses REWARD_MASK_PIL_WORKERS threads.
_REWARD_MASK_IP = None                          # reward image_processor (registered)
_REWARD_MASK_SRC_MAX_PIXELS_FALLBACK = 2097152  # 2048 tokens × 1024 px
REWARD_MASK_PIL_WORKERS_DEFAULT = int(
    qz_getenv("REWARD_MASK_PIL_WORKERS", "16") or "16")


def set_reward_mask_image_processor(ip):
    """RewardModel calls this so _build_masked_pils caps source images at the
    reward processor's live max_pixels (max visual tokens × patch²)."""
    global _REWARD_MASK_IP
    _REWARD_MASK_IP = ip


def _build_masked_pils(
    src_pil,
    keep_masks: np.ndarray,
    n_workers: int = 1,
) -> list:
    """Apply each keep_mask (Hg, Wg) to ``src_pil``, return list of PIL.

    Crop-then-mask: bbox of the keep_mask, then mean-fill INSIDE the crop.
    Background fill = dataset mean RGB to match Qwen normalization
    behavior at the image level (the processor will re-normalize). Aligns
    the reward image format with deployment, which crops to the foreground
    union bbox before re-encoding under the min/max token budget.

    When ``n_workers > 1``, the K-action loop is dispatched onto a
    persistent ``ThreadPoolExecutor``. The inner work
    (``np.copy``/boolean-mask assignment / ``Image.fromarray``) releases
    the GIL, so threads parallelize cleanly.
    """
    from PIL import Image

    # Cap the source resolution before the per-mask full-res np.repeat
    # expansion. The masked crop is resized to the reward token budget by the
    # processor ANYWAY, so building masks at the raw image resolution (pool
    # infographics reach ~67 Mpx) is wasted single-core numpy and the dominant
    # cause of the large-K DDP straggler stall.
    _cap = int(getattr(_REWARD_MASK_IP, "max_pixels", 0) or 0) \
        or _REWARD_MASK_SRC_MAX_PIXELS_FALLBACK
    _w0, _h0 = src_pil.size
    if _w0 * _h0 > _cap:
        _s = math.sqrt(_cap / float(_w0 * _h0))
        src_pil = src_pil.resize(
            (max(1, int(_w0 * _s)), max(1, int(_h0 * _s))), Image.BILINEAR)
    if n_workers <= 1:
        n_workers = REWARD_MASK_PIL_WORKERS_DEFAULT

    K, Hg, Wg = keep_masks.shape
    arr = np.asarray(src_pil).astype(np.float32) / 255.0  # (H, W, 3)
    H, W, _ = arr.shape
    # Fill = the processor's image_mean (what normalizes to zero): CLIP mean
    # for the Qwen processors; black for Gemma-4 (no normalization, and the
    # same fill its Phase-A expand2square padding used).
    mean_rgb = np.array(
        [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
    )
    if _REWARD_MASK_IP is not None:
        _im = getattr(_REWARD_MASK_IP, "image_mean", None)
        if not getattr(_REWARD_MASK_IP, "do_normalize", True):
            mean_rgb = np.zeros(3, dtype=np.float32)
        elif _im is not None and len(_im) == 3:
            mean_rgb = np.asarray(_im, dtype=np.float32)

    def _one(k: int):
        m = keep_masks[k].astype(bool)
        sh = max(1, H // Hg)
        sw = max(1, W // Wg)
        m_full = np.repeat(np.repeat(m, sh, axis=0), sw, axis=1)
        # Pad/crop to (H, W) if not exact multiples.
        if m_full.shape[0] < H:
            m_full = np.concatenate(
                [m_full, np.zeros((H - m_full.shape[0], m_full.shape[1]), dtype=bool)],
                axis=0,
            )
        elif m_full.shape[0] > H:
            m_full = m_full[:H]
        if m_full.shape[1] < W:
            m_full = np.concatenate(
                [m_full, np.zeros((m_full.shape[0], W - m_full.shape[1]), dtype=bool)],
                axis=1,
            )
        elif m_full.shape[1] > W:
            m_full = m_full[:, :W]
        ys, xs = np.where(m_full)
        if ys.size == 0:
            # Defensive: empty mask shouldn't happen (action enumeration
            # excludes the all-discard action), but mean-fill the whole
            # image as a safe fallback.
            masked = np.broadcast_to(mean_rgb, arr.shape).copy()
            return Image.fromarray((masked * 255).astype(np.uint8))
        y1, y2 = int(ys.min()), int(ys.max() + 1)
        x1, x2 = int(xs.min()), int(xs.max() + 1)
        img_crop = arr[y1:y2, x1:x2].copy()
        m_crop = m_full[y1:y2, x1:x2]
        img_crop[~m_crop] = mean_rgb
        return Image.fromarray((img_crop * 255).astype(np.uint8))

    if n_workers <= 1 or K <= 1:
        return [_one(k) for k in range(K)]
    pool = _get_masked_pil_pool(n_workers)
    return list(pool.map(_one, range(K)))


# ============================================================================
# HF Trainer subclass
# ============================================================================

class RegionLevelGRPOTrainer(Trainer):
    """HF ``Trainer`` subclass for Phase-B.1 GR-REINFORCE.

    Expected batch schema (the dataset/collator must populate these in
    addition to the standard model-forward inputs):

      - ``KEY_PIL_IMAGES``    -> ``list[PIL.Image]``   (length ``B``)
      - ``KEY_QUESTIONS``     -> ``list[str]``         (length ``B``)
      - ``KEY_GOLD_ANSWERS``  -> ``list[str]``         (length ``B``)
      - ``KEY_P_REFS``        -> ``list[Tensor(Hg,Wg)]`` (length ``B``)
      - ``KEY_P_REF_BINARIES``-> ``list[Tensor(Hg,Wg) | None]`` (length ``B``)
      - ``KEY_EV_MAPS``       -> ``list[Tensor(L,Hg,Wg) | None]`` (length ``B``)

    The standard inputs (``input_ids``, ``attention_mask``,
    ``pixel_values``, ``image_grid_thw``, ``labels``) flow normally to
    ``model.forward`` after the extras are popped.
    """

    def __init__(
        self,
        *args,
        phase_b1_config: Optional[PhaseB1Config] = None,
        reward_system_message: str = "You are a helpful assistant.",
        ref_twig_module=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.phase_b1_config = phase_b1_config or PhaseB1Config()
        self._reward_system_message = reward_system_message
        self._reward_model_cache: Optional[RewardModel] = None
        # Frozen Phase-A reference twig for the online KL anchor. When set,
        # compute_loss runs it on the saved pre_twig_ctx after the main
        # forward to get a resolution-matched Z_ref.
        self._ref_twig_module = ref_twig_module
        # Per-microbatch diagnostic buffers, drained on each ``self.log()``
        # call (i.e. on every ``logging_steps`` boundary).
        self._pending_diag: Dict[str, List[float]] = defaultdict(list)
        self._pending_K_counts: Counter = Counter()

    def _get_reward_model(self) -> RewardModel:
        """Lazily build the reward-model wrapper bound to ``self.model``.

        The Phase-B.1 LLM backbone is frozen, so the reward forward
        shares weights with the policy. We unwrap DDP / FSDP if present.
        """
        if self._reward_model_cache is None:
            model = getattr(self.model, "module", self.model)
            self._reward_model_cache = RewardModel(
                model=model,
                processor=self.processing_class,
                system_message=self._reward_system_message,
                disable_thinking_prefix=bool(
                    self.phase_b1_config.reward_disable_thinking_prefix
                ),
                skip_trailing_eos=bool(
                    self.phase_b1_config.reward_skip_trailing_eos
                ),
            )
        return self._reward_model_cache

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        # 1. Pop Phase-B.1 extras.
        pil_images = inputs.pop(KEY_PIL_IMAGES, None)
        questions = inputs.pop(KEY_QUESTIONS, None)
        gold_answers = inputs.pop(KEY_GOLD_ANSWERS, None)
        p_refs = inputs.pop(KEY_P_REFS, None)
        p_ref_binaries = inputs.pop(KEY_P_REF_BINARIES, None)
        ev_maps_list = inputs.pop(KEY_EV_MAPS, None)
        if (pil_images is None or questions is None
                or gold_answers is None or p_refs is None):
            raise RuntimeError(
                "Batch is missing Phase-B.1 extras. "
                f"Required keys: {KEY_PIL_IMAGES}, {KEY_QUESTIONS}, "
                f"{KEY_GOLD_ANSWERS}, {KEY_P_REFS}. Did you wire up the "
                "RegionLevelGRPO collator?"
            )

        # 2. Force per-head-score capture and disable 2-stage ROI
        #    augmentation during the policy forward. We only need the
        #    first-stage heatmap; the 2-stage path would re-forward the
        #    LM on a cropped sub-image (slow + unnecessary for RL).
        cfg = getattr(model, "config", None)
        # Older deepspeed (conda qwen / tf4.51, used for Qwen2.5-VL) exposes the
        # DeepSpeed config *dict* as ``model.config`` while proxying ``.model``
        # to the wrapped module; the HF config lives on ``model.module``. Unwrap
        # so ``return_per_head_score`` lands on the config the forward reads.
        # No-op for qwen3.x (newer deepspeed already proxies .config -> HF config).
        if not hasattr(cfg, "return_per_head_score"):
            cfg = getattr(getattr(model, "module", model), "config", cfg)
        original_phs = bool(getattr(cfg, "return_per_head_score", False)) if cfg is not None else False
        if cfg is not None:
            cfg.return_per_head_score = True

        lm_node = getattr(model, "model", model)
        lm_node = getattr(lm_node, "language_model", lm_node)
        saved_roi2 = getattr(lm_node, "roi_enable2stage", None)
        if saved_roi2 is not None:
            lm_node.roi_enable2stage = False
        try:
            outputs = model(**inputs)
        finally:
            if cfg is not None:
                cfg.return_per_head_score = original_phs
            if saved_roi2 is not None:
                lm_node.roi_enable2stage = saved_roi2

        per_head_scores_list = getattr(outputs, "per_head_scores", None) or []
        feat_hw_list = getattr(outputs, "feat_hw", None) or []

        # online_p_ref: run the frozen Phase-A ref twig on the saved
        # pre-twig context, in lockstep with the policy forward, to get a
        # resolution-matched Z_ref. ref_phs_list[b] is the ref per-head
        # score tensor for sample b (shape (num_heads, k_img_b)), or None
        # if the sample had no visual / response tokens.
        ref_phs_list: List[Optional[torch.Tensor]] = []
        if self._ref_twig_module is not None:
            pre_twig_ctx = getattr(outputs, "pre_twig_ctx", None)
            # When the twig branch didn't fire this step (e.g. labels
            # all-truncated at very high MAX_PIXELS, or pre-RL filter
            # edge case), pre_twig_ctx is None and so is the per_head
            # capture — the trainer's empty-per_head_scores check below
            # then returns a zero loss for this microbatch.
            if pre_twig_ctx is not None:
                # The ref twig forward needs the same dtype / device as
                # the pre-twig hidden states. Move the ref module
                # accordingly.
                policy_dev = pre_twig_ctx["hidden_states"].device
                policy_dtype = pre_twig_ctx["hidden_states"].dtype
                if (next(self._ref_twig_module.parameters()).device != policy_dev
                        or next(self._ref_twig_module.parameters()).dtype != policy_dtype):
                    self._ref_twig_module = self._ref_twig_module.to(
                        device=policy_dev, dtype=policy_dtype,
                    )
                _cfg_src = getattr(model, "module", model)
                image_token_id = getattr(getattr(_cfg_src, "config", None), "image_token_id", None)
                ref_phs_list = self._ref_twig_module.compute_per_head_scores(
                    pre_twig_ctx=pre_twig_ctx,
                    input_ids=inputs.get("input_ids"),
                    labels=inputs.get("labels"),
                    image_token_id=image_token_id,
                    image_grid_thw=inputs.get("image_grid_thw"),
                    apply_rope=bool(self.phase_b1_config.ref_score_with_rope),
                    query_mode=str(self.phase_b1_config.ref_score_query_mode),
                )
                # One-time proof that the ONLINE ref twig (not cached p_ref)
                # is driving the KL anchor for this run. Prints the live ref
                # grid resolved from the first sample that produced a score.
                if not getattr(self, "_online_p_ref_proof_done", False):
                    self._online_p_ref_proof_done = True
                    _gthw = inputs.get("image_grid_thw")
                    _hw = None
                    if _gthw is not None and len(_gthw) > 0:
                        try:
                            _hw = (int(_gthw[0][1]) // 2, int(_gthw[0][2]) // 2)
                        except Exception:  # noqa: BLE001
                            _hw = None
                    _n_ref = sum(1 for x in ref_phs_list if x is not None)
                    _gated = getattr(self._ref_twig_module, "_q_gated", None)
                    import sys as _sys
                    print(
                        f"[phase_b1] online_p_ref ACTIVE: online RefTwigModule "
                        f"q_gated={_gated} ref grid≈(Hg,Wg)={_hw} "
                        f"n_samples_with_ref={_n_ref}/{len(ref_phs_list)}",
                        file=_sys.stderr, flush=True,
                    )

        if not per_head_scores_list:
            zero = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
            if not isinstance(zero, torch.Tensor):
                zero = torch.zeros((), device=next(model.parameters()).device)
            return (zero, outputs) if return_outputs else zero

        # 3. Per-sample loss.
        #
        # When PHASE_B1_DEBUG=1, dump per-sample loss components and flag
        # any NaN/Inf -> stdout. This is the primary diagnostic for the
        # "step 1 produces NaN gradients" failure mode.
        debug = bool(int(qz_getenv("PHASE_B1_DEBUG", "0")))
        reward_model = self._get_reward_model()
        per_sample_losses: List[torch.Tensor] = []
        K_counts: Dict[int, int] = {}
        # diag_acc is a per-call buffer of single-sample diagnostics; we
        # aggregate it into self._pending_diag at the end of compute_loss
        # so it can be flushed to tensorboard on the next log() boundary.
        diag_acc: Dict[str, List[float]] = defaultdict(list)
        for b, phs in enumerate(per_head_scores_list):
            if phs is None:
                continue
            _b1cfg = self.phase_b1_config

            # cached source-map vote (L,Hg,Wg) for this sample — used by both
            # the attn-fg gradient mask (region-drop group) and the supp group.
            _ev = (ev_maps_list[b]
                   if ev_maps_list is not None and b < len(ev_maps_list)
                   else None)

            def _call_b1():
                return compute_phase_b1_loss_v2(
                    per_head_scores=phs,
                    feat_hw=tuple(feat_hw_list[b]),
                    pil_image=pil_images[b],
                    question=questions[b],
                    gold_answer=gold_answers[b],
                    p_ref=p_refs[b],
                    p_ref_binary=(p_ref_binaries[b] if p_ref_binaries is not None else None),
                    reward_model=reward_model,
                    config=_b1cfg,
                    ref_per_head_score=(ref_phs_list[b] if b < len(ref_phs_list) else None),
                    ev_maps=_ev,
                )

            if _b1cfg.source_map_group:
                # region-drop group + source-map supplement group, equal 1/2
                # each. KL is identical in both subs, so it is preserved once
                # at λ_kl regardless of the PG split.
                _sub_g1 = _call_b1()
                _winn_raw = _sub_g1.get("winnability_w", 1.0)
                _winn = 1.0 if _winn_raw is None else float(_winn_raw)
                # K<=1 sub-dicts carry no winnability_w; the 1.0 fallback must
                # still respect the configured clamp (winnability_max=0 zeroes
                # ALL PG groups, incl. smap on K<=1 samples).
                _winn = min(_winn, float(_b1cfg.winnability_max))
                _sub_smap = compute_source_map_group_loss(
                    per_head_scores=phs,
                    feat_hw=tuple(feat_hw_list[b]),
                    ev_maps=_ev,
                    pil_image=pil_images[b],
                    question=questions[b],
                    gold_answer=gold_answers[b],
                    p_ref=p_refs[b],
                    ref_per_head_score=(ref_phs_list[b]
                                        if b < len(ref_phs_list) else None),
                    reward_model=reward_model,
                    config=_b1cfg,
                    # SAME per-sample winnability weight as the region-drop
                    # group (its compute_phase_b1_loss_v2 already updated the EMA).
                    winn_w=_winn,
                )
                if not getattr(self, "_smap_proof_done", False):
                    self._smap_proof_done = True
                    import sys as _sys
                    print(f"[phase_b1] SOURCE_MAP_GROUP active: region-drop σ"
                          f"{float(_b1cfg.smooth_sigma)}+supp (½/½) "
                          f"(policy_σ={_b1cfg.source_map_policy_sigma}, "
                          f"ev_maps={'yes' if _ev is not None else 'NONE'}, "
                          f"branch={_sub_smap.get('branch')}, "
                          f"n_cand={_sub_smap.get('K')})",
                          file=_sys.stderr, flush=True)
                # Diagnostics via the standard aggregate (equal-weight means);
                # the LOSS is the 1/2-1/2 PG split. KL preserved.
                res = _aggregate_groups([_sub_g1, _sub_smap])
                res["loss"] = 0.5 * _sub_g1["loss"] + 0.5 * _sub_smap["loss"]
                res["branch"] = "multigroup"
            else:
                res = _call_b1()
            per_sample_losses.append(res["loss"])
            K_counts[res["K"]] = K_counts.get(res["K"], 0) + 1
            diag_acc["loss_policy"].append(float(res.get("loss_policy", 0.0)))
            diag_acc["loss_kl"].append(float(res.get("loss_kl", 0.0)))
            diag_acc["loss_anchor_k1"].append(float(res.get("loss_anchor_k1", 0.0)))
            diag_acc["n_actions"].append(int(res.get("n_actions", 0)))
            diag_acc["K"].append(int(res.get("K", 0)))
            if "n_sampled" in res:
                diag_acc["n_sampled"].append(int(res["n_sampled"]))
            if "heatmap_max" in res:
                diag_acc["heatmap_max"].append(float(res["heatmap_max"]))
            if "heatmap_pos_frac" in res:
                diag_acc["heatmap_pos_frac"].append(float(res["heatmap_pos_frac"]))
            if "keep_frac_mean" in res:
                diag_acc["keep_frac_mean"].append(float(res["keep_frac_mean"]))
            for _key in (
                # winnability diagnostics.
                "winnability_w", "ncc_alpha_eff", "size_beta_eff",
                "reward_p_empty",
                # attention-fg gradient mask diagnostics.
                "attn_fg_active", "attn_fg_n_pos",
                "attn_fg_n_pos_fully_below", "attn_fg_grad_keep_frac",
                "diag_h_empty", "diag_h_ave_drop", "diag_ncc0",
                "diag_R_empty", "diag_R_ave_drop",
                # answer→image SUPP additive source-map group diagnostics.
                "supp_n", "supp_n_pos", "supp_vote_mean", "supp_adv_abs_mean", "supp_c_mean",
                # placebo-bar subtractor diagnostics (batch-mean → tensorboard).
                "placebo_bar", "placebo_fallback", "g1_c_mean",
            ):
                if _key in res:
                    try:
                        diag_acc[_key].append(float(res[_key]))
                    except (TypeError, ValueError):
                        pass
            rewards = res.get("rewards")
            if rewards is not None and rewards.numel() > 0:
                diag_acc["reward_mean"].append(float(rewards.mean()))
                diag_acc["reward_std"].append(float(rewards.std(unbiased=False)))
            advantage = res.get("advantage")
            if advantage is not None and advantage.numel() > 0:
                diag_acc["advantage_abs_mean"].append(
                    float(advantage.abs().mean())
                )
            if debug:
                _l = float(res["loss"].detach())
                _lp = float(res.get("loss_policy", 0.0))
                _lk = float(res.get("loss_kl", 0.0))
                _la = float(res.get("loss_anchor_k1", 0.0))
                bad = []
                for nm, v in (("L", _l), ("L_pol", _lp),
                              ("L_kl", _lk), ("L_anchor", _la)):
                    if math.isnan(v) or math.isinf(v):
                        bad.append(nm)
                # Reward stats (K>=2 only).
                rew = res.get("rewards")
                rew_str = "n/a"
                if rew is not None and rew.numel() > 0:
                    _rmin = float(rew.min())
                    _rmax = float(rew.max())
                    _rmean = float(rew.mean())
                    _rstd = float(rew.std(unbiased=False))
                    rew_str = f"min={_rmin:.3f} max={_rmax:.3f} mean={_rmean:.3f} std={_rstd:.3f}"
                    if any(map(math.isnan, (_rmin, _rmax, _rmean, _rstd))) \
                            or any(map(math.isinf, (_rmin, _rmax, _rmean, _rstd))):
                        bad.append("rewards")
                # Heatmap stats (NaN/Inf check on the policy heatmap).
                phs_min = float(phs.detach().float().min())
                phs_max = float(phs.detach().float().max())
                if math.isnan(phs_min) or math.isinf(phs_min) \
                        or math.isnan(phs_max) or math.isinf(phs_max):
                    bad.append("per_head_scores")
                tag = f"!!NaN/Inf:{','.join(bad)}!!" if bad else "ok"
                ds = (questions[b][:40] + "...") if questions and len(questions[b]) > 40 \
                    else (questions[b] if questions else "?")
                print(
                    f"[phase_b1_debug] b={b} K={res['K']} "
                    f"branch={res.get('branch')} L={_l:.4f} "
                    f"L_pol={_lp:.4f} L_kl={_lk:.4f} L_anchor={_la:.4f} "
                    f"phs=[{phs_min:.3f},{phs_max:.3f}] rew={rew_str} "
                    f"q='{ds}' [{tag}]",
                    flush=True,
                )

        if not per_sample_losses:
            zero = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
            if not isinstance(zero, torch.Tensor):
                zero = torch.zeros((), device=next(model.parameters()).device)
            return (zero, outputs) if return_outputs else zero

        loss = torch.stack(per_sample_losses).mean()

        # Push per-microbatch diagnostics onto the trainer-level buffers.
        # The ``log()`` override below averages and emits these on each
        # ``logging_steps`` boundary.
        for k, vals in diag_acc.items():
            if vals:
                self._pending_diag[k].extend(vals)
        for k, v in K_counts.items():
            self._pending_K_counts[k] += v

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Merge accumulated Phase-B.1 diagnostics into the HF log dict.

        HF Trainer calls ``self.log({...})`` on every ``logging_steps``
        boundary (and on epoch end). We splice in buffer means here so the
        TensorBoard / WandB callbacks pick them up without any extra hooks.
        Buffers are cleared after each merge.
        """
        if self._pending_diag or self._pending_K_counts:
            extra: Dict[str, float] = {}
            for k, vals in self._pending_diag.items():
                if vals:
                    extra[k] = float(sum(vals) / len(vals))
            total_K = sum(self._pending_K_counts.values())
            if total_K > 0:
                k0 = self._pending_K_counts.get(0, 0)
                k1 = self._pending_K_counts.get(1, 0)
                kge2 = sum(v for k, v in self._pending_K_counts.items() if k >= 2)
                extra["K_frac_0"] = k0 / total_K
                extra["K_frac_1"] = k1 / total_K
                extra["K_frac_ge2"] = kge2 / total_K
                extra["K_mean"] = sum(
                    k * v for k, v in self._pending_K_counts.items()
                ) / total_K
            logs.update(extra)
            self._pending_diag.clear()
            self._pending_K_counts.clear()
        super().log(logs, *args, **kwargs)


# ============================================================================
# Self-test (mocked model + reward)
# ============================================================================

def _self_test() -> None:
    """End-to-end test of compute_phase_b1_loss_v2 with mock reward."""
    from PIL import Image

    torch.manual_seed(0)
    Hg, Wg = 16, 16
    H_img, W_img = 64, 64
    n_heads = 4

    # Synthetic per-head logits with a clear two-blob structure.
    yy, xx = np.mgrid[0:Hg, 0:Wg]
    blob_a = np.exp(-((xx - 4) ** 2 + (yy - 4) ** 2) / 4.0)
    blob_b = np.exp(-((xx - 12) ** 2 + (yy - 12) ** 2) / 4.0)
    p_template = (0.7 * blob_a + 0.3 * blob_b)
    p_template = p_template / p_template.max()
    z_template = np.log(np.clip(p_template, 1e-3, 1 - 1e-3) /
                        (1 - np.clip(p_template, 1e-3, 1 - 1e-3)))
    per_head = torch.from_numpy(
        np.stack([z_template] * n_heads, axis=0).astype(np.float32)
    ).reshape(n_heads, Hg * Wg).requires_grad_(True)

    p_ref = torch.sigmoid(torch.from_numpy(z_template.astype(np.float32)))
    p_ref_binary = (p_ref > 0.04).float()

    pil = Image.fromarray(
        (np.random.rand(H_img, W_img, 3) * 255).astype(np.uint8)
    )

    # Fake reward model.
    class _FakeRewardModel:
        image_token_template = "<|image_pad|>"
        system_message = "you"

        def compute_logprobs(self, masked_images, question, gold_answer,
                             device=None, reduction="sum"):
            # Reward correlates with how much foreground each mask retains.
            K = len(masked_images)
            scores = []
            for im in masked_images:
                arr = np.asarray(im).astype(bool).any(axis=-1)
                scores.append(float(arr.mean()))
            return torch.tensor(scores, dtype=torch.float32) * 4.0 - 2.0

    cfg = PhaseB1Config(threshold_mode="fixed", fixed_threshold=0.04, R_max=4)
    out = compute_phase_b1_loss_v2(
        per_head_scores=per_head,
        feat_hw=(Hg, Wg),
        pil_image=pil,
        question="what is in the image?",
        gold_answer="a blob",
        p_ref=p_ref,
        p_ref_binary=p_ref_binary,
        reward_model=_FakeRewardModel(),
        config=cfg,
    )

    # Loss is a scalar with grad.
    assert out["loss"].dim() == 0
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert per_head.grad is not None
    assert torch.isfinite(per_head.grad).all()

    print(f"[mocked] branch={out['branch']}, K={out['K']}, "
          f"n_actions={out['n_actions']}, loss={float(out['loss']):.4f}")
    assert out["K"] >= 1, f"expected at least 1 component, got {out['K']}"

    # Supp group with synthetic evidence maps (2 layers, one extra region).
    ev = torch.zeros(2, Hg, Wg, dtype=torch.uint8)
    ev[:, 2:6, 10:14] = 1
    ev[0, 12:14, 2:4] = 1
    out_smap = compute_source_map_group_loss(
        per_head_scores=per_head, feat_hw=(Hg, Wg), ev_maps=ev, pil_image=pil,
        question="q", gold_answer="a", p_ref=p_ref, ref_per_head_score=None,
        reward_model=_FakeRewardModel(), config=cfg,
    )
    print(f"[mocked supp] branch={out_smap['branch']}, K={out_smap['K']}, "
          f"loss={float(out_smap['loss']):.4f}")
    assert out_smap["loss"].requires_grad
    out_smap["loss"].backward()

    # K=0 path: zero out the heatmap.
    zero_per_head = torch.zeros_like(per_head, requires_grad=True) - 5.0
    out_k0 = compute_phase_b1_loss_v2(
        per_head_scores=zero_per_head,
        feat_hw=(Hg, Wg),
        pil_image=pil,
        question="q", gold_answer="a",
        p_ref=p_ref, p_ref_binary=p_ref_binary,
        reward_model=_FakeRewardModel(),
        config=cfg,
    )
    print(f"[mocked K=0] branch={out_k0['branch']}, K={out_k0['K']}, "
          f"loss={float(out_k0['loss']):.4f}")
    out_k0["loss"].backward()  # backprop sanity

    # K=1 path: single very-narrow blob.
    z_single = np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 1.0)
    z_single = z_single / z_single.max() * 4.0 - 2.0
    z_single_t = torch.from_numpy(z_single.astype(np.float32))
    per_head_k1 = z_single_t.reshape(1, Hg * Wg).repeat(n_heads, 1).requires_grad_(True)
    out_k1 = compute_phase_b1_loss_v2(
        per_head_scores=per_head_k1,
        feat_hw=(Hg, Wg),
        pil_image=pil,
        question="q", gold_answer="a",
        p_ref=p_ref, p_ref_binary=p_ref_binary,
        reward_model=_FakeRewardModel(),
        config=cfg,
    )
    print(f"[mocked K=1] branch={out_k1['branch']}, K={out_k1['K']}, "
          f"loss_anchor_k1={float(out_k1['loss_anchor_k1']):.4f}")
    out_k1["loss"].backward()
    assert out_k1["K"] == 1

    print("OK: trainer.py self-test passed.")


if __name__ == "__main__":
    _self_test()
