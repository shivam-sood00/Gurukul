from __future__ import annotations

import copy

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, handle_deprecated_rsl_rl_cfg


_SHARED_POLICY_CLASS_NAMES = {
    "ContactTrailActorCritic",
    "CTSActorCritic",
    "MultiCriticActorCritic",
    "RealTeacherActorCritic",
    "StartActorCritic",
    "StudentTeacherDepthBackbone",
    "StudentTeacherDepthBackboneRecurrent",
    "StudentTeacherDepthBackboneRecurrentSTART",
}


def migrate_custom_policy_cfg(agent_cfg, installed_version):
    """Run Isaac Lab's migration without discarding custom shared-policy configs."""
    policy = getattr(agent_cfg, "policy", None)
    policy_class_name = getattr(policy, "class_name", "")
    policy_backup = copy.deepcopy(policy) if policy_class_name in _SHARED_POLICY_CLASS_NAMES else None
    use_beta_action_distribution = bool(getattr(agent_cfg, "use_beta_action_distribution", False))

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if policy_backup is not None:
        agent_cfg.policy = policy_backup
    if use_beta_action_distribution:
        agent_cfg.actor.distribution_cfg = RslRlMLPModelCfg.DistributionCfg(class_name="BetaDistribution")
    return agent_cfg
