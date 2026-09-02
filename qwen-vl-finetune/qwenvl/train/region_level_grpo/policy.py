"""Categorical policy over removal actions for region-level GR-REINFORCE.

For one prompt with top-R components and the enumerated action set
``A = [∅, {0}, {1}, ..., {R-1}, {0,1}, ..., {R-2,R-1}]`` from
:mod:`actions`, the policy is::

    pi_theta(D | x) = softmax_D in A( s^rm_theta(D) / T )

with the per-action score::

    s^rm_theta(D) = - sum_{C in D} s_theta(C)
                    - beta_keep * max(0, |Y_keep(D)| / |I| - gamma_keep)

Components NOT in ``D`` (i.e. components that are *kept*) contribute
nothing to the score beyond their per-component ``s_theta(C)``: we want
the component score to drive both inclusion in top-R (via ranking) and
the policy's preference for keeping it (via the discard-penalty
appearing only in the actions that drop it).

Gradients flow through ``component_scores`` -> log pi(a) -> the
reinforce loss.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from qwenvl.train.region_level_grpo.actions import RemovalAction


# ---------------------------------------------------------- score function --

def removal_action_score(
    component_scores: torch.Tensor,
    action: RemovalAction,
    grid_size: int,
    beta_keep: float = 1.0,
    gamma_keep: float = 0.6,
) -> torch.Tensor:
    """Compute ``s^rm_theta(D)`` for one removal action.

    Args:
        component_scores: ``(R,)`` tensor of per-component scores. Must
            require_grad to backprop.
        action: a :class:`RemovalAction` with discard-set indices.
        grid_size: ``Hg * Wg`` for the size penalty denominator.
        beta_keep: weight of the size penalty on Y_keep.
        gamma_keep: penalty kicks in above ``|Y_keep| / |I| > gamma_keep``.

    Returns:
        Scalar tensor (0-dim) carrying gradient through
        ``component_scores``.
    """
    if action.discard:
        idxs = torch.tensor(
            action.discard, dtype=torch.long, device=component_scores.device,
        )
        discard_sum = component_scores.index_select(0, idxs).sum()
    else:
        discard_sum = component_scores.new_zeros(())

    keep_area = float(int(action.keep_mask.sum()))  # numpy bool -> int -> float
    keep_frac = keep_area / float(max(grid_size, 1))
    size_penalty = float(beta_keep) * max(0.0, keep_frac - float(gamma_keep))

    return -discard_sum - size_penalty


def compute_action_scores(
    component_scores: torch.Tensor,
    actions: List[RemovalAction],
    grid_size: int,
    beta_keep: float = 1.0,
    gamma_keep: float = 0.6,
) -> torch.Tensor:
    """Stack ``s^rm_theta(D)`` across all actions; shape ``(N,)``."""
    if not actions:
        return component_scores.new_zeros(0)
    return torch.stack([
        removal_action_score(
            component_scores, a, grid_size,
            beta_keep=beta_keep, gamma_keep=gamma_keep,
        )
        for a in actions
    ])


# -------------------------------------------------------------- log policy --

def compute_log_pi(
    component_scores: torch.Tensor,
    actions: List[RemovalAction],
    grid_size: int,
    T: float = 1.0,
    beta_keep: float = 1.0,
    gamma_keep: float = 0.6,
) -> torch.Tensor:
    """Compute ``log pi(a)`` for every action.

    Returns:
        Tensor of shape ``(N_actions,)``. ``logsumexp(log_pi)`` is
        numerically zero (within fp eps); gradients flow through
        ``component_scores``.
    """
    raw = compute_action_scores(
        component_scores, actions, grid_size,
        beta_keep=beta_keep, gamma_keep=gamma_keep,
    )
    if raw.numel() == 0:
        return raw
    return torch.log_softmax(raw / float(T), dim=0)


# ------------------------------------------------------------ sampling-WOR --

def sample_without_replacement(
    log_pi: torch.Tensor,
    K: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``K`` unique action indices via the Gumbel-top-K trick.

    The Gumbel-top-K identity says that adding i.i.d. Gumbel(0, 1) noise
    to log probabilities and taking the top-K gives ``K`` unique samples
    drawn proportionally to ``pi`` without replacement.

    Args:
        log_pi: ``(N,)`` log-probabilities. ``log_pi.detach()`` is used
            for sampling -- gradients of the sampling step are zero.
        K: number of unique samples.
        generator: optional ``torch.Generator`` for deterministic seed.

    Returns:
        Long tensor of shape ``(min(K, N),)`` with unique action indices.
    """
    N = int(log_pi.shape[0])
    if K >= N:
        return torch.arange(N, device=log_pi.device, dtype=torch.long)
    if generator is None:
        u = torch.rand(N, dtype=log_pi.dtype, device=log_pi.device)
    else:
        u = torch.rand(
            N, generator=generator, dtype=log_pi.dtype, device=log_pi.device,
        )
    gumbel = -torch.log(-torch.log(u.clamp(min=1e-20)) + 1e-20)
    perturbed = log_pi.detach() + gumbel
    return torch.topk(perturbed, k=K, sorted=False).indices


def select_actions_for_rewards(
    log_pi: torch.Tensor,
    enumerate_threshold: int = 12,
    rollout_K: int = 8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pick action indices for which to compute rewards.

    - If ``N <= enumerate_threshold``: return all indices ``[0..N)``.
    - Else: sample ``rollout_K`` unique indices via Gumbel-top-K.
    """
    N = int(log_pi.shape[0])
    if N <= enumerate_threshold:
        return torch.arange(N, device=log_pi.device, dtype=torch.long)
    return sample_without_replacement(log_pi, rollout_K, generator=generator)


# ----------------------------------------------------------------- tests --

def _self_test() -> None:
    """Self-test for the policy module. Run via ``python -m ...policy``."""
    import numpy as np
    from qwenvl.train.region_level_grpo.components import Component

    Hg, Wg = 8, 8
    grid_size = Hg * Wg

    # Build R=4 dummy components with hand-set scores.
    comps: List[Component] = []
    for i, score in enumerate([3.0, 2.0, 1.0, 0.5]):
        mask = np.zeros((Hg, Wg), dtype=bool)
        mask[i * 2:(i + 1) * 2, :] = True  # disjoint horizontal strips
        comps.append(
            Component(mask=mask, score=score, area=2 * Wg, bbox=(i*2, 0, (i+1)*2, Wg))
        )

    # Build actions.
    from qwenvl.train.region_level_grpo.actions import enumerate_removal_actions
    actions = enumerate_removal_actions(comps)
    assert len(actions) == 11  # R=4 -> 1+4+6

    component_scores = torch.tensor(
        [c.score for c in comps], dtype=torch.float32, requires_grad=True,
    )

    # 1. log_pi shape + normalization.
    log_pi = compute_log_pi(component_scores, actions, grid_size, T=1.0)
    assert log_pi.shape == (11,)
    lse = torch.logsumexp(log_pi, dim=0).item()
    assert abs(lse - 0.0) < 1e-5, f"log-softmax should sum to 0; got {lse}"

    # 2. Highest score should be the empty discard action (∅): no discard
    #    sum, and keep all has the largest area. With our scores all > 0,
    #    discarding any component subtracts from the score, so ∅ is best.
    best_idx = int(log_pi.argmax().item())
    assert actions[best_idx].discard == (), (
        f"expected ∅ to win, got {actions[best_idx].discard}"
    )

    # 3. Gradient flow.
    loss = -log_pi[5]  # arbitrary action
    loss.backward()
    assert component_scores.grad is not None
    assert torch.isfinite(component_scores.grad).all()

    # 4. Score of singleton-discarding-comp-0 = -comp[0].score - size penalty.
    score_disc_0 = removal_action_score(
        component_scores.detach(), actions[1], grid_size,
        beta_keep=1.0, gamma_keep=0.6,
    )
    # actions[1] discards comp 0; keep_mask is rows 2-7 = 6*8=48 cells. 48/64=0.75.
    # size_penalty = 1.0 * max(0, 0.75 - 0.6) = 0.15.
    expected = -3.0 - 0.15
    assert abs(score_disc_0.item() - expected) < 1e-4, (
        f"score_disc_0 expected {expected}, got {score_disc_0.item()}"
    )

    # 5. Higher temperature flattens distribution.
    log_pi_T1 = compute_log_pi(component_scores, actions, grid_size, T=1.0)
    log_pi_T10 = compute_log_pi(component_scores, actions, grid_size, T=10.0)
    pi_T1 = log_pi_T1.exp()
    pi_T10 = log_pi_T10.exp()
    # T=10 should be more uniform: max prob lower, min prob higher.
    assert pi_T10.max() < pi_T1.max()
    assert pi_T10.min() > pi_T1.min()

    # 6. Sampling without replacement: K=3 from N=11 returns 3 unique.
    g = torch.Generator().manual_seed(0)
    sampled = sample_without_replacement(log_pi.detach(), K=3, generator=g)
    assert sampled.shape == (3,)
    assert len(set(sampled.tolist())) == 3, "WOR sampling must return unique"

    # 7. select_actions: small N enumerates, large N samples.
    idx_small = select_actions_for_rewards(
        log_pi, enumerate_threshold=12, rollout_K=8,
    )
    assert idx_small.shape == (11,) and torch.equal(
        idx_small.sort().values, torch.arange(11)
    )
    # Force a "large" case by lowering the threshold.
    idx_sample = select_actions_for_rewards(
        log_pi, enumerate_threshold=5, rollout_K=8,
    )
    assert idx_sample.shape == (8,)
    assert len(set(idx_sample.tolist())) == 8

    print("OK: policy.py self-test passed.")


if __name__ == "__main__":
    _self_test()
