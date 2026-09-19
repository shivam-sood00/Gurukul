# Copyright (c) 2025- Shenzhen Zhongqing Robot Technology Co., Ltd. ("EngineAI")
# Copyright (c) 2026, Gurukul contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL configuration for PM01 velocity following with EngineAI AMP."""

from pathlib import Path

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from Gurukul.tasks.manager_based.locomotion.velocity.pm01_amp_data import (
    PM01_AMP_FRAME_DIM,
    PM01_AMP_REFERENCE_SHA256,
)

from .rsl_rl_ppo_cfg import EngineAiPm01FlatPPORunnerCfg

_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "motion" / "locomotion.npz"


@configclass
class EngineAiPm01AmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "Gurukul.tasks.manager_based.locomotion.velocity.pm01_amp_ppo:Pm01AmpPPO"
    amp_dataset_path: str = str(_REFERENCE_PATH)
    amp_dataset_sha256: str = PM01_AMP_REFERENCE_SHA256
    amp_history_length: int = 5
    amp_frame_dim: int = PM01_AMP_FRAME_DIM

    # Upstream adds ``0.01 * 2.0 * style`` per step at a 100 Hz policy, i.e. 2.0 * style per
    # second of simulated time. Gurukul runs the actor at 50 Hz and scales by ``policy_dt``,
    # so the same per-second prior strength is ``amp_style_reward_weight = 2.0``. Against this
    # task's ~5 reward/s the prior is then worth up to ~30%, in line with published AMP splits.
    amp_style_reward_weight: float = 2.0
    # No schedule: upstream applies the prior from the first iteration. A warm-up long enough
    # for PPO to converge first (the previous 1500 + 1500) only perturbs a settled policy.
    amp_style_reward_warmup_iterations: int = 0
    amp_style_reward_ramp_iterations: int = 0

    # Drop the ~8.5 s of standing that closes EngineAI's clip so the prior describes walking.
    amp_reference_min_speed: float = 0.15

    amp_discriminator_hidden_dims: list[int] = [256, 128]
    amp_feature_normalization: bool = True
    amp_discriminator_learning_rate: float = 1.0e-4
    amp_discriminator_weight_decay: float = 1.0e-4
    amp_discriminator_update_interval: int = 4
    amp_batch_size: int = 16_384
    amp_gradient_penalty_scale: float = 10.0


@configclass
class EngineAiPm01AmpFlatPPORunnerCfg(EngineAiPm01FlatPPORunnerCfg):
    """EngineAI AMP hyperparameters on the maintained 24-action PM01 policy."""

    algorithm = EngineAiPm01AmpAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 10_000
        self.save_interval = 500
        self.experiment_name = "engineai_pm01_official_amp_flat"
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
