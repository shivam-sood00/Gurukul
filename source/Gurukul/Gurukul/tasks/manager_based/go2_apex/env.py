"""APEX manager environment with opt-in terminal observations and curriculum checkpoints."""

from __future__ import annotations

import copy

import torch
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv


class ApexManagerBasedRLEnv(ManagerBasedRLEnv):
    """Keep the standard step order; capture critic inputs immediately before automatic reset."""

    def enable_terminal_observations(self, groups):
        self._terminal_groups = groups

    def step(self, action):
        self.extras.pop("terminal_observations", None)
        self.extras.pop("terminal_time_outs", None)
        self._in_policy_step = True
        try:
            return super().step(action)
        finally:
            self._in_policy_step = False

    def _reset_idx(self, env_ids):
        if getattr(self, "_in_policy_step", False) and getattr(self, "_terminal_groups", None):
            # A true failure takes precedence over an episode time limit.
            timeouts = (self.reset_time_outs & ~self.reset_terminated).clone()
            self.extras["terminal_time_outs"] = timeouts
            if torch.any(timeouts):
                self.extras["terminal_observations"] = self._terminal_critic_observations()
        super()._reset_idx(env_ids)

    def _terminal_critic_observations(self):
        motion = self.command_manager.get_term("motion")
        original_steps = motion.time_steps
        # The normal next observation follows command advancement. Preview that
        # frame without resampling a clip, changing the robot or updating the sampler.
        if hasattr(motion, "_motion_frame_accumulator"):
            fps = motion.motion.fps_values[motion.current_motion_ids]
            increment = torch.floor(motion._motion_frame_accumulator + fps * self.step_dt).long()
            motion.time_steps = torch.minimum(original_steps + increment, motion.time_step_ends)
        else:
            # BeyondMimic's G1 APEX command advances exactly one frame per step.
            motion.time_steps = (original_steps + 1).clamp(max=motion.motion.time_step_total - 1)
        history = self.observation_manager._group_obs_term_history_buffer
        saved_history = {name: history[name] for name in self._terminal_groups}
        try:
            motion._refresh_relative_motion_state()
            # Preview the latest sample in any critic observation history without
            # appending a second sample to surviving environments' real histories.
            for name in self._terminal_groups:
                history[name] = copy.deepcopy(saved_history[name])
            observations = {
                name: self.observation_manager.compute_group(name, update_history=True).clone()
                for name in self._terminal_groups
            }
            return TensorDict(observations, batch_size=[self.num_envs])
        finally:
            history.update(saved_history)
            motion.time_steps = original_steps
            motion._refresh_relative_motion_state()

    def training_state_dict(self):
        motion = self.command_manager.get_term("motion")
        action = self.action_manager.get_term("joint_pos")
        return {
            "common_step_counter": self.common_step_counter,
            "decap_step": action._current_decap_step() if hasattr(action, "_current_decap_step") else None,
            "bin_failed_count": motion.bin_failed_count.clone(),
            "motion_files": self._motion_files(motion),
            "termination_resume_offsets": {
                name: self.termination_manager.get_term_cfg(name).params["resume_iteration"]
                for name in self.termination_manager.active_terms
                if "resume_iteration" in self.termination_manager.get_term_cfg(name).params
            }
            if hasattr(self, "termination_manager")
            else {},
        }

    def load_training_state_dict(self, state):
        motion = self.command_manager.get_term("motion")
        if self._motion_files(motion) != state["motion_files"]:
            raise ValueError("Resume requires the same ordered motion files")
        bins = state["bin_failed_count"].to(self.device)
        if bins.shape != motion.bin_failed_count.shape:
            raise ValueError("Resume requires the same adaptive motion sampling bins")
        self.common_step_counter = int(state["common_step_counter"])
        action = self.action_manager.get_term("joint_pos")
        # The CLI may already have seeded the schedule from the checkpoint name.
        # Replace that offset using saved counters; never add it twice.
        if state["decap_step"] is not None:
            if not hasattr(action, "_current_decap_step"):
                raise ValueError("Checkpoint contains DecAP state but the current action term does not use DecAP")
            action._decap_resume_step_offset = int(state["decap_step"]) - self.common_step_counter
        # Restoring common_step_counter already restores elapsed training time.
        # Replace the CLI's checkpoint-derived offset to avoid counting it twice.
        for name, offset in state.get("termination_resume_offsets", {}).items():
            self.termination_manager.get_term_cfg(name).params["resume_iteration"] = offset
        motion.bin_failed_count.copy_(bins)
        self.reset()

    @staticmethod
    def _motion_files(motion):
        if hasattr(motion.motion, "motion_files"):
            return list(motion.motion.motion_files)
        return [motion.cfg.motion_file]
