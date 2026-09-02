"""Removal-action enumeration for region-level GR-REINFORCE.

An action is a subset ``D`` of the top-R retained components to *discard*
from the high-recall foreground proposal. The kept ROI is::

    Y_keep(D) = union over k not in D of C_k.mask

The action space enumerated here::

    A = { ∅ }                              # no removal (keep all)
        ∪ { {C_i}     : 0 <= i < R }       # discard one
        ∪ { {C_i,C_j} : 0 <= i < j < R }   # discard two

i.e. ``1 + R + R(R-1)/2`` actions in total.

The all-discard action (``Y_keep = ∅``) is **excluded** because the
empty mask gives a degenerate teacher-forced log-prob (the reward LLM
sees a blank image). We accordingly skip:

* the singleton action when ``R == 1`` (it would discard everything);
* the pair action when ``R == 2``      (it would discard everything).

For ``R >= 3`` no such truncation is needed.

Sampling-vs-enumeration is orthogonal to this module — see the trainer.
This file just produces the full ``A`` list deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from qwenvl.train.region_level_grpo.components import Component


# ------------------------------------------------------------ data model --

@dataclass(frozen=True)
class RemovalAction:
    """A removal-action specifying which top-R components to discard.

    Attributes:
        discard: tuple of component indices (sorted ascending) to remove.
            The empty tuple ``()`` represents the no-removal action.
        keep_mask: ``(Hg, Wg)`` bool numpy array, the union of
            ``C_k.mask`` over indices ``k`` NOT in ``discard``.
    """

    discard: Tuple[int, ...]
    keep_mask: np.ndarray


# ------------------------------------------------------------ enumeration --

def enumerate_removal_actions(
    components: List[Component],
    singleton_only: bool = False,
) -> List[RemovalAction]:
    """Enumerate ∅ + singletons + pairs over the given (top-R) components.

    Order: ``[∅, {0}, {1}, ..., {R-1}, {0,1}, {0,2}, ..., {R-2,R-1}]``.

    Excludes any action whose discard set equals all components
    (``Y_keep = ∅``), because the reward LLM cannot meaningfully
    evaluate an empty mask.

    Iter 33: ``singleton_only=True`` skips the pair-enumeration block.
    Rationale: pair rewards from the LM are noisy due to non-additive
    spurious-context dependencies (see analysis/case_studies/
    infovqa_val_doc55). Singleton-only gives a clean per-component
    contribution signal: c_k = R(∅) - R(drop_k).

    Returns:
        List of :class:`RemovalAction`. Length:

          - ``R == 0``: ``[]`` (no foreground)
          - ``R == 1``: ``[∅]``                                 (1 action)
          - ``R == 2``: ``[∅, {0}, {1}]``                       (3 actions)
          - ``R == 3``: ``[∅, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}]``  (7 actions)
          - ``R == 4``: 11 actions
          - ``R == 8``: 37 actions
    """
    R = len(components)
    if R == 0:
        return []

    union_all = np.zeros_like(components[0].mask)
    for c in components:
        union_all = union_all | c.mask

    actions: List[RemovalAction] = []

    # 1. The empty discard set — keep everything.
    actions.append(RemovalAction(discard=tuple(), keep_mask=union_all.copy()))

    # 2. Singletons. Skip if R == 1 (would discard everything).
    if R > 1:
        for i in range(R):
            keep = union_all & (~components[i].mask)
            actions.append(RemovalAction(discard=(i,), keep_mask=keep))

    # 3. Pairs. Skip if R == 2 (would discard everything) OR if
    # singleton_only mode is on (iter 33).
    if R > 2 and not singleton_only:
        for i in range(R):
            for j in range(i + 1, R):
                keep = union_all & (~components[i].mask) & (~components[j].mask)
                actions.append(RemovalAction(discard=(i, j), keep_mask=keep))

    return actions


def expected_action_count(R: int) -> int:
    """Number of actions :func:`enumerate_removal_actions` returns for top-R."""
    if R <= 0:
        return 0
    if R == 1:
        return 1                 # only ∅
    if R == 2:
        return 1 + R             # ∅ + singletons
    return 1 + R + R * (R - 1) // 2


# ----------------------------------------------------------------- tests --

def _self_test() -> None:
    """Self-test for action enumeration. Run via ``python -m ...actions``."""
    Hg, Wg = 8, 8

    def make_components(n: int) -> List[Component]:
        comps: List[Component] = []
        for i in range(n):
            mask = np.zeros((Hg, Wg), dtype=bool)
            row = i % Hg
            mask[row, :] = True  # one full row each, all disjoint
            comps.append(
                Component(
                    mask=mask, score=float(n - i), area=Wg,
                    bbox=(row, 0, row + 1, Wg),
                )
            )
        return comps

    # R = 0: empty input -> empty output.
    assert enumerate_removal_actions([]) == []
    assert expected_action_count(0) == 0

    # R = 1: just the empty action.
    a1 = enumerate_removal_actions(make_components(1))
    assert len(a1) == 1 == expected_action_count(1)
    assert a1[0].discard == ()
    assert a1[0].keep_mask.sum() == Wg  # one row

    # R = 2: ∅ + 2 singletons = 3.
    comps2 = make_components(2)
    a2 = enumerate_removal_actions(comps2)
    assert len(a2) == 3 == expected_action_count(2)
    assert a2[0].discard == ()
    assert a2[1].discard == (0,)
    assert a2[2].discard == (1,)
    # Singleton i drops component i from union.
    union2 = comps2[0].mask | comps2[1].mask
    assert np.array_equal(a2[1].keep_mask, union2 & ~comps2[0].mask)
    assert np.array_equal(a2[2].keep_mask, union2 & ~comps2[1].mask)

    # R = 3: ∅ + 3 + 3 = 7.
    comps3 = make_components(3)
    a3 = enumerate_removal_actions(comps3)
    assert len(a3) == 7 == expected_action_count(3)
    expected_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_actions = [a for a in a3 if len(a.discard) == 2]
    assert [a.discard for a in pair_actions] == expected_pairs

    # R = 4: 1 + 4 + 6 = 11.
    a4 = enumerate_removal_actions(make_components(4))
    assert len(a4) == 11 == expected_action_count(4)

    # R = 8: 1 + 8 + 28 = 37.
    a8 = enumerate_removal_actions(make_components(8))
    assert len(a8) == 37 == expected_action_count(8)

    # No keep_mask should be all-False (Y_keep != ∅).
    for a in a4 + a8:
        assert a.keep_mask.any(), f"degenerate action {a.discard} -> empty keep_mask"

    print("OK: actions.py self-test passed.")


if __name__ == "__main__":
    _self_test()
