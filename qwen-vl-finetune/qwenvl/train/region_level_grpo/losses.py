"""Loss components for region-level GR-REINFORCE (connected-component v1).

Three terms, combined per-sample inside the trainer:

1. **Policy gradient** (REINFORCE with a per-prompt group baseline)
   ::

       L_policy = -mean_n A_n * log pi(a_n)

   where ``A_n`` is the standardized advantage over the action subset for
   which rewards were computed. Two baselines are provided: the group mean
   (:func:`policy_gradient_loss`) and the keep-all action ``R(∅)``
   (:func:`policy_gradient_loss_empty_baseline`, used by the placebo-bar
   subtractor). Used for samples with ``len(actions) >= 2``.

2. **KL anchor** on the (mean-head) heatmap, applied to **all**
   samples (including K=1) so the policy doesn't drift on prompts the
   policy gradient can't touch::

       L_kl = mean_pixel KL_Bernoulli( P || P_ref )

   computed in logit space (:func:`kl_anchor_loss_logit`).

3. **K=1 anchor-SFT** for samples whose action space collapses to
   ``{∅}`` (single component) -- pulls the heatmap toward Phase-A's
   binarized foreground::

       L_anchor_k1 = BCE( P, binarize(P_ref) )

   This term is *only* active for K=1 samples and replaces the
   policy-loss zero in that regime (:func:`k1_anchor_bce_loss_logit`).

Total::

    L = L_policy [if K>=2] + lambda_anchor * L_anchor_k1 [if K=1]
        + lambda_kl * L_kl
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# ============================================================================
# Policy gradient
# ============================================================================

def policy_gradient_loss(
    log_pi: torch.Tensor,
    rewards: torch.Tensor,
    eps: float = 1e-8,
    std_regularizer: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Group-relative REINFORCE: ``-mean_n A_n * log pi(a_n)``.

    Args:
        log_pi: ``(N,)`` log-probabilities of the actions for which
            rewards were computed. Requires grad through the policy.
        rewards: ``(N,)`` raw scalar rewards. Detached from the graph.
        eps: numerical-stability epsilon for std denominator.
        std_regularizer: additional constant added to the advantage
            denominator. When >0, dampens noise amplification on
            low-reward-variance groups. advantage = (R - mean(R)) /
            (std(R) + std_regularizer + eps).

    Returns:
        Dict with ``loss`` (scalar w/ grad), ``advantage`` (detached),
        ``reward_mean``, ``reward_std`` for logging.
    """
    if log_pi.shape != rewards.shape:
        raise ValueError(
            f"log_pi shape {tuple(log_pi.shape)} != "
            f"rewards shape {tuple(rewards.shape)}"
        )
    R = rewards.detach().to(log_pi.dtype)

    if R.numel() < 2:
        # Degenerate: zero advantage, zero gradient, but emit a 0-loss
        # tensor that has a grad_fn so the outer compute_loss doesn't
        # crash autograd.
        zero = log_pi.sum() * 0.0
        return {
            "loss": zero,
            "advantage": torch.zeros_like(R),
            "reward_mean": R.mean(),
            "reward_std": R.new_zeros(()),
        }

    R_mean = R.mean()
    R_std = R.std(unbiased=False)
    advantage = (R - R_mean) / (R_std + std_regularizer + eps)

    loss = -(advantage * log_pi).mean()
    return {
        "loss": loss,
        "advantage": advantage,
        "reward_mean": R_mean,
        "reward_std": R_std,
    }


def policy_gradient_loss_empty_baseline(
    log_pi: torch.Tensor,
    rewards: torch.Tensor,
    r_empty: torch.Tensor,
    eps: float = 1e-8,
    std_regularizer: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """∅-baselined, NO-mean-center advantage.

    ``advantage = (R − R(∅)) / (std(R − R(∅)) + std_regularizer + eps)``.

    The baseline is the empty / keep-all action ``R(∅)`` instead of the group
    mean. This preserves the natural zero of the per-region marginal value:
    ``R'_k = R(drop k) − R(∅) = −c_k + penalty_reduction``, so a region that
    neither changes log_p (``c_k≈0``) nor shrinks the ROI gets ~0 advantage, an
    important region (``c_k`` high) gets negative advantage (drop discouraged →
    kept), and an irrelevant/large region gets positive (drop encouraged). No
    group-mean subtraction, so it does NOT force the group mean to zero.

    The placebo-bar subtractor (``subtractor_mode="placebo"``) calls this with
    ``r_empty = R(∅) − bar`` so a region is only kept when its contribution
    beats the per-sample reward-noise bar.

    Args:
        log_pi: ``(N,)`` log-probs of the N drop actions (∅ already excluded).
        rewards: ``(N,)`` shaped rewards ``R(drop k)``. Detached from the graph.
        r_empty: scalar ``R(∅)`` baseline (detached).
        std_regularizer: constant added to the std denominator.
    """
    if log_pi.shape != rewards.shape:
        raise ValueError(
            f"log_pi shape {tuple(log_pi.shape)} != "
            f"rewards shape {tuple(rewards.shape)}"
        )
    R = rewards.detach().to(log_pi.dtype)
    Rp = R - r_empty.detach().to(log_pi.dtype)   # R'_k = R(drop k) − R(∅)
    if Rp.numel() < 2:
        zero = log_pi.sum() * 0.0
        return {
            "loss": zero,
            "advantage": torch.zeros_like(Rp),
            "reward_mean": R.mean(),
            "reward_std": R.new_zeros(()),
        }
    Rp_std = Rp.std(unbiased=False)
    advantage = Rp / (Rp_std + std_regularizer + eps)   # NO mean subtraction
    loss = -(advantage * log_pi).mean()
    return {
        "loss": loss,
        "advantage": advantage,
        "reward_mean": R.mean(),
        "reward_std": Rp_std,
    }


# ============================================================================
# KL anchor — logit-space (numerically robust)
# ============================================================================

def kl_anchor_loss_logit(
    z_current: torch.Tensor,
    z_ref: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel KL(σ(z_current) || σ(z_ref)), computed in logit space.

    Numerically robust formulation that never invokes ``log(0)`` or
    ``log(1)`` regardless of saturation. Uses the identity::

        log σ(z)    = -softplus(-z)
        log(1-σ(z)) = -softplus( z)

    so that::

        KL(σ(z_p) || σ(z_q)) =
            σ(z_p) · (softplus(-z_q) - softplus(-z_p))
          + (1-σ(z_p)) · (softplus( z_q) - softplus( z_p))

    Mean-pooled across pixels.

    Args:
        z_current: ``(Hg, Wg)`` pre-sigmoid logits ``Z_theta``,
            requires grad.
        z_ref: ``(Hg, Wg)`` reference (Phase-A) logits, no grad. When the
            reference comes from a cached probability map, callers recover
            it via ``torch.logit(p_ref, eps=1e-5)`` (eps must be > fp32
            spacing at 1.0 ≈ 5.96e-8; 1e-5 gives safety margin).

    Returns:
        Scalar tensor with grad through ``z_current``.
    """
    if z_current.shape != z_ref.shape:
        raise ValueError(
            f"shape mismatch: z_current {tuple(z_current.shape)} vs "
            f"z_ref {tuple(z_ref.shape)}"
        )
    # MUST run fp32: under HF Trainer's autocast(bf16) the softplus terms
    # lose the precision the KL difference depends on.
    device_type = z_current.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        z_p = z_current.float()
        z_q = z_ref.detach().float()
        p = torch.sigmoid(z_p)
        sp_neg_p = F.softplus(-z_p)
        sp_pos_p = F.softplus(z_p)
        sp_neg_q = F.softplus(-z_q)
        sp_pos_q = F.softplus(z_q)
        kl = p * (sp_neg_q - sp_neg_p) + (1.0 - p) * (sp_pos_q - sp_pos_p)
        return kl.mean()


# ============================================================================
# K=1 anchor-SFT — logit-space (numerically robust)
# ============================================================================

def k1_anchor_bce_loss_logit(
    z_current: torch.Tensor,
    target_binary: torch.Tensor,
) -> torch.Tensor:
    """BCE anchor for K=1 samples, using LOGITS (not probabilities).

    For K=1 prompts the action space collapses to ``{∅}`` and the policy
    gradient is identically zero (one action -> ``log pi = 0``). This
    loss replaces the policy term with direct supervision against the
    Phase-A binarized foreground (a 0/1 mask), giving these samples
    real gradient signal while preventing the heatmap from drifting on
    them. ``F.binary_cross_entropy_with_logits`` uses log-sum-exp
    internally, so it's safe for any finite ``z_current``.

    Args:
        z_current: ``(Hg, Wg)`` pre-sigmoid logits ``Z_theta``,
            requires grad.
        target_binary: ``(Hg, Wg)`` 0/1 mask of the Phase-A foreground.
            Detached.

    Returns:
        Scalar tensor with grad through ``z_current``.
    """
    if z_current.shape != target_binary.shape:
        raise ValueError(
            f"shape mismatch: z_current {tuple(z_current.shape)} vs "
            f"target {tuple(target_binary.shape)}"
        )
    device_type = z_current.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        target = target_binary.detach().float()
        return F.binary_cross_entropy_with_logits(
            z_current.float(), target, reduction="mean",
        )


# ============================================================================
# Self-test
# ============================================================================

def _self_test() -> None:
    torch.manual_seed(0)

    # ---- 1. Policy gradient direction --------------------------------------
    # Three actions with rewards (1, 0, -1). Standardized A = (1, 0, -1)/sigma.
    # Loss = -mean(A * log_pi). Increasing log_pi[0] (high reward) -> loss
    # decreases -> grad on log_pi[0] is negative under autograd.
    log_pi = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, -1.0])
    out = policy_gradient_loss(log_pi, rewards)
    out["loss"].backward()
    g = log_pi.grad
    assert g is not None
    # g[0] (high reward) should be negative; g[2] (low reward) positive.
    assert g[0] < g[2], (
        f"expected grad[0] < grad[2] (high reward should pull log_pi up), "
        f"got {g.tolist()}"
    )

    # ---- 2. Single-action degenerate path returns 0-grad -------------------
    log_pi_one = torch.tensor([-0.5], requires_grad=True)
    rewards_one = torch.tensor([1.0])
    out_one = policy_gradient_loss(log_pi_one, rewards_one)
    out_one["loss"].backward()
    assert log_pi_one.grad is not None
    assert torch.allclose(log_pi_one.grad, torch.zeros_like(log_pi_one), atol=1e-7), (
        f"single-action loss must have zero grad; got {log_pi_one.grad}"
    )

    # ---- 3. ∅-baselined advantage: sign follows R(drop k) − R(∅) -----------
    log_pi_eb = torch.tensor([-1.0, -1.0], requires_grad=True)
    rewards_eb = torch.tensor([0.5, -0.5])
    out_eb = policy_gradient_loss_empty_baseline(
        log_pi_eb, rewards_eb, torch.tensor(0.0))
    assert out_eb["advantage"][0] > 0 > out_eb["advantage"][1]
    out_eb["loss"].backward()
    assert torch.isfinite(log_pi_eb.grad).all()

    # ---- 4b. Logit-space KL: zero when z == z_ref --------------------------
    z = torch.randn(8, 8, requires_grad=True) * 5.0
    kl_logit = kl_anchor_loss_logit(z, z.detach().clone())
    assert kl_logit.item() < 1e-4, f"KL(z||z) should be ~0; got {kl_logit.item()}"

    # ---- 4c. Logit-space KL is finite even with saturated targets ----------
    z_p = torch.full((8, 8), -3.0, requires_grad=True)
    # p_ref with exact 0 / 1 pixels. Recovered via torch.logit(p, eps=1e-5):
    # p=1 → z≈+11.51, p=0 → z≈-11.51. Both finite, KL stays finite.
    p_ref_saturated = torch.zeros(8, 8)
    p_ref_saturated[0, :] = 1.0   # one row of saturated foreground
    p_ref_saturated[-1, :] = 0.0  # one row of exact background
    p_ref_saturated[3:5, :] = 0.5
    z_q = torch.logit(p_ref_saturated.clamp(1e-5, 1.0 - 1e-5))
    kl_sat = kl_anchor_loss_logit(z_p, z_q)
    assert torch.isfinite(kl_sat).item(), \
        f"KL must be finite for saturated p_ref; got {kl_sat.item()}"
    kl_sat.backward()
    assert torch.isfinite(z_p.grad).all().item()

    # ---- 4d. KL anchor: gradient pulls z toward z_ref ----------------------
    z_far = torch.full((8, 8), 2.0, requires_grad=True)
    z_ref_low = torch.full((8, 8), -2.0)
    kl_far = kl_anchor_loss_logit(z_far, z_ref_low)
    kl_far.backward()
    assert kl_far.item() > 0
    assert z_far.grad.mean().item() > 0  # push z down toward -2

    # ---- 5. K=1 anchor BCE (logits): near zero when logits match target ----
    target = torch.tensor([[1, 1, 0, 0]] * 4, dtype=torch.bool)
    z_match = (target.float() * 20.0 - 10.0).clone().detach().requires_grad_(True)
    bce = k1_anchor_bce_loss_logit(z_match, target)
    assert bce.item() < 1e-3, f"BCE on matching target should be near zero, got {bce.item()}"
    bce.backward()
    assert torch.isfinite(z_match.grad).all()

    # ---- 8. Shape-mismatch error path --------------------------------------
    try:
        kl_anchor_loss_logit(torch.rand(8, 8), torch.rand(7, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")

    print("OK: losses.py self-test passed.")


if __name__ == "__main__":
    _self_test()
