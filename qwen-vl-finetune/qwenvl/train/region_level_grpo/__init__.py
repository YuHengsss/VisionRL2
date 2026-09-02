"""Region-level Group-Relative REINFORCE for SD-RPN (Phase B.1).

Connected-component policy with REINFORCE + group baseline on the frozen
Phase-A (SD-RPN) twig. See ``experiments/exploration/head_level_grpo/README.md``
for the recipe and ``connected_region_sampling_problem.tex`` for the formal
motivation.

Modules:
    components.py    - smooth + threshold + CC + score (numpy & torch)
    actions.py       - enumerate removal actions over top-R components
    policy.py        - softmax over component scores; Gumbel-top-K WOR
    losses.py        - policy gradient + KL anchor + K=1 BCE-anchor (logit space)
    reward_model.py  - frozen-LLM teacher-forced log-prob of gold answer
    ref_twig.py      - frozen Phase-A twig for the online KL reference
    dataset.py       - filtered-pool dataset + collator
    trainer.py       - HF Trainer subclass orchestrating compute_loss
"""

from qwenvl.train.region_level_grpo.components import (
    Component,
    extract_components,
    rank_top_r,
    score_component,
    score_component_torch,
)
from qwenvl.train.region_level_grpo.actions import (
    RemovalAction,
    enumerate_removal_actions,
    expected_action_count,
)
from qwenvl.train.region_level_grpo.policy import (
    compute_log_pi,
    removal_action_score,
    sample_without_replacement,
    select_actions_for_rewards,
)
from qwenvl.train.region_level_grpo.losses import (
    k1_anchor_bce_loss_logit,
    kl_anchor_loss_logit,
    policy_gradient_loss,
    policy_gradient_loss_empty_baseline,
)
from qwenvl.train.region_level_grpo.trainer import (
    PhaseB1Config,
    RegionLevelGRPOTrainer,
    compute_phase_b1_loss_v2,
)

__all__ = [
    # components
    "Component",
    "extract_components",
    "rank_top_r",
    "score_component",
    "score_component_torch",
    # actions
    "RemovalAction",
    "enumerate_removal_actions",
    "expected_action_count",
    # policy
    "compute_log_pi",
    "removal_action_score",
    "sample_without_replacement",
    "select_actions_for_rewards",
    # losses
    "k1_anchor_bce_loss_logit",
    "kl_anchor_loss_logit",
    "policy_gradient_loss",
    "policy_gradient_loss_empty_baseline",
    # trainer
    "PhaseB1Config",
    "RegionLevelGRPOTrainer",
    "compute_phase_b1_loss_v2",
]
