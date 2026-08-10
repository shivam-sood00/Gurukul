"""RSL-RL 5 PPO adapter for the shared REAL teacher policy."""

from __future__ import annotations

from .contact_trail_actor_critic import ContactTrailPPO
from .real_teacher import RealTeacherActorCritic


class RealTeacherPPO(ContactTrailPPO):
    """Train the monolithic REAL actor-critic without optimizing shared parameters twice."""

    def __init__(self, policy, storage=None, **kwargs):
        super().__init__(policy, storage=storage, contact_quality_loss_coef=0.0, **kwargs)

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
        from rsl_rl.storage import RolloutStorage
        from rsl_rl.utils import resolve_callable, resolve_obs_groups

        alg_class = RealTeacherPPO
        algorithm_class_name = cfg["algorithm"].pop("class_name", None)
        if algorithm_class_name and algorithm_class_name not in {"RealTeacherPPO", "PPO"}:
            if ":" in algorithm_class_name or "." in algorithm_class_name:
                alg_class = resolve_callable(algorithm_class_name)
            else:
                raise ValueError(f"Unsupported REAL teacher algorithm class: {algorithm_class_name}")

        policy_cfg = dict(cfg["policy"])
        policy_class = RealTeacherActorCritic
        policy_class_name = policy_cfg.pop("class_name", None)
        if policy_class_name and policy_class_name != "RealTeacherActorCritic":
            if ":" in policy_class_name or "." in policy_class_name:
                policy_class = resolve_callable(policy_class_name)
            else:
                raise ValueError(f"Unsupported REAL teacher policy class: {policy_class_name}")

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        policy = policy_class(obs, cfg["obs_groups"], env.num_actions, **policy_cfg).to(device)
        print(f"REAL Teacher Actor-Critic Model: {policy}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(
            policy,
            storage=storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
