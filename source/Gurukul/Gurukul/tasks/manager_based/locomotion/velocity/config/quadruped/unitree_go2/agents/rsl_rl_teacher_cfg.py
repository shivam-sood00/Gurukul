# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaaclab/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class RealTeacherActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the REAL-style privileged teacher actor-critic."""

    class_name: str = "RealTeacherActorCritic"
    proprio_obs_group: str = "real_teacher_proprio"
    terrain_obs_group: str = "real_teacher_terrain"
    privileged_obs_group: str = "real_teacher_privileged"
    terrain_scan_shape: tuple[int, int] | None = (16, 10)
    attention_embed_dim: int = 128
    attention_num_heads: int = 4
    terrain_encoder_hidden_dim: int = 128
    privileged_latent_dim: int = 128
    privileged_hidden_dims: list[int] = [128]
    use_scan_positional_encoding: bool = True
    pretrained_attention_checkpoint: str | None = None
    freeze_pretrained_attention: bool = False
    load_pretrained_attention_into_actor: bool = True
    load_pretrained_attention_into_critic: bool = True


from .rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg, UnitreeGo2RoughPPORunnerCfg


@configclass
class UnitreeGo2RoughTeacherPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    # Teacher uses privileged observations for both actor and critic.
    obs_groups = {"actor": ["critic"], "critic": ["critic"]}
    experiment_name = "unitree_go2_rough_teacher"


@configclass
class UnitreeGo2RoughFullTeacherPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Full-state rough-terrain teacher with temporal observation history."""

    def __post_init__(self):
        super().__post_init__()
        self.obs_groups = {"actor": ["teacher_full"], "critic": ["teacher_full"]}
        self.experiment_name = "unitree_go2_rough_teacher_full"
        # Increase model capacity for stacked privileged observations.
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[1024, 512, 256],
            critic_hidden_dims=[1024, 512, 256],
            activation="elu",
        )


@configclass
class UnitreeGo2RoughRealTeacherPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """REAL-style rough-terrain privileged teacher runner."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "RealTeacherPPO"
        self.obs_groups = {
            "actor": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
            "critic": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
        }
        self.experiment_name = "unitree_go2_rough_real_teacher"
        self.policy = RealTeacherActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            attention_embed_dim=128,
            attention_num_heads=4,
            terrain_encoder_hidden_dim=128,
            privileged_latent_dim=128,
            privileged_hidden_dims=[128],
            terrain_scan_shape=(16, 10),
            use_scan_positional_encoding=True,
        )


@configclass
class UnitreeGo2RoughRealTeacherPretrainedPPORunnerCfg(UnitreeGo2RoughRealTeacherPPORunnerCfg):
    """REAL runner that initializes the offline terrain-token/key prior and fine-tunes it."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_rough_real_teacher_pretrained"
        self.policy.freeze_pretrained_attention = False


@configclass
class UnitreeGo2RoughRealTeacherFrozenPPORunnerCfg(UnitreeGo2RoughRealTeacherPPORunnerCfg):
    """REAL runner that freezes the offline encoder while fine-tuning its transferred key projection."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_rough_real_teacher_frozen"
        self.policy.freeze_pretrained_attention = True


@configclass
class UnitreeGo2RoughRealBeamPPORunnerCfg(UnitreeGo2RoughRealTeacherPPORunnerCfg):
    """REAL-style beam-only privileged teacher runner."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_rough_real_teacher_beam"


@configclass
class UnitreeGo2RoughRealBeamPretrainedPPORunnerCfg(UnitreeGo2RoughRealBeamPPORunnerCfg):
    """Beam REAL runner that initializes the offline terrain-token/key prior and fine-tunes it."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_rough_real_teacher_beam_pretrained"
        self.policy.freeze_pretrained_attention = False


@configclass
class UnitreeGo2RoughRealBeamFrozenPPORunnerCfg(UnitreeGo2RoughRealBeamPPORunnerCfg):
    """Beam REAL runner that initializes the terrain-attention blocks from an offline checkpoint and freezes the terrain-side prior."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_rough_real_teacher_beam_frozen"
        self.policy.freeze_pretrained_attention = True


@configclass
class UnitreeGo2FlatTeacherPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    # Teacher uses privileged observations for both actor and critic.
    obs_groups = {"actor": ["critic"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_flat_teacher"
