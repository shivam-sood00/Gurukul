# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os

import toml

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)


def _register_custom_distillation_policies():
    """Register custom policy/algorithm classes into RSL-RL eval namespaces."""
    try:
        import rsl_rl.algorithms as rsl_algorithms
        import rsl_rl.models as rsl_models
        import rsl_rl.modules as rsl_modules
        import rsl_rl.runners.distillation_runner as distillation_runner
        import rsl_rl.runners.on_policy_runner as on_policy_runner

        from Gurukul.tasks.manager_based.locomotion.velocity.depth_student_teacher import (
            CombinedDistillation,
            DepthBackboneActorModel,
            StartDistillation,
            StudentTeacherDepthBackbone,
            StudentTeacherDepthBackboneRecurrent,
            StudentTeacherDepthBackboneRecurrentSTART,
        )
        from Gurukul.tasks.manager_based.concurrent_teacher_student import CTSActorCritic, CTSPPO
        from Gurukul.tasks.manager_based.locomotion.velocity.real_teacher import RealTeacherActorCritic
        from Gurukul.tasks.manager_based.locomotion.velocity.real_teacher_ppo import RealTeacherPPO
        from Gurukul.tasks.manager_based.locomotion.velocity.start_actor_critic import (
            StartActorCritic,
            StartPPO,
        )
        from Gurukul.tasks.manager_based.locomotion.velocity.contact_trail_actor_critic import (
            ContactTrailActorCritic,
            ContactTrailPPO,
        )

        # Make the class discoverable via eval("StudentTeacherDepthBackbone") in DistillationRunner.
        rsl_modules.StudentTeacherDepthBackbone = StudentTeacherDepthBackbone
        rsl_modules.StudentTeacherDepthBackboneRecurrent = StudentTeacherDepthBackboneRecurrent
        rsl_modules.StudentTeacherDepthBackboneRecurrentSTART = StudentTeacherDepthBackboneRecurrentSTART
        rsl_modules.DepthBackboneActorModel = DepthBackboneActorModel
        rsl_models.DepthBackboneActorModel = DepthBackboneActorModel
        distillation_runner.StudentTeacherDepthBackbone = StudentTeacherDepthBackbone
        distillation_runner.StudentTeacherDepthBackboneRecurrent = StudentTeacherDepthBackboneRecurrent
        distillation_runner.StudentTeacherDepthBackboneRecurrentSTART = StudentTeacherDepthBackboneRecurrentSTART
        # Make the algorithm discoverable via eval("StartDistillation") in DistillationRunner.
        rsl_algorithms.StartDistillation = StartDistillation
        distillation_runner.StartDistillation = StartDistillation
        rsl_algorithms.CombinedDistillation = CombinedDistillation
        distillation_runner.CombinedDistillation = CombinedDistillation

        # Make REAL teacher and START single-stage classes discoverable via eval(...) in OnPolicyRunner.
        rsl_modules.RealTeacherActorCritic = RealTeacherActorCritic
        on_policy_runner.RealTeacherActorCritic = RealTeacherActorCritic
        rsl_algorithms.RealTeacherPPO = RealTeacherPPO
        on_policy_runner.RealTeacherPPO = RealTeacherPPO
        rsl_modules.StartActorCritic = StartActorCritic
        on_policy_runner.StartActorCritic = StartActorCritic
        on_policy_runner.DepthBackboneActorModel = DepthBackboneActorModel
        rsl_algorithms.StartPPO = StartPPO
        on_policy_runner.StartPPO = StartPPO
        rsl_modules.CTSActorCritic = CTSActorCritic
        on_policy_runner.CTSActorCritic = CTSActorCritic
        rsl_algorithms.CTSPPO = CTSPPO
        on_policy_runner.CTSPPO = CTSPPO
        rsl_modules.ContactTrailActorCritic = ContactTrailActorCritic
        on_policy_runner.ContactTrailActorCritic = ContactTrailActorCritic
        rsl_algorithms.ContactTrailPPO = ContactTrailPPO
        on_policy_runner.ContactTrailPPO = ContactTrailPPO
    except Exception as exc:
        # Keep task registration robust even if rsl_rl is unavailable in non-training contexts.
        print(f"[WARN] Failed to register custom RSL-RL extensions: {exc}")


_register_custom_distillation_policies()
