"""RSL-RL wrapper that streams one env to a Viser server on every step."""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from Gurukul.utils.live_env_viewer import LiveEnvViewer


class LiveViserRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL vec-env wrapper with a guaranteed per-step Viser publish hook."""

    def __init__(self, env, clip_actions: float | None = None, viewer: LiveEnvViewer | None = None):
        self._live_viewer = viewer
        self._live_step = 0
        super().__init__(env, clip_actions=clip_actions)
        if self._live_viewer is not None:
            self._live_viewer.set_num_envs(self.num_envs)
            self._publish_live_frame(reward=None, done=None)

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        if self._live_viewer is not None:
            env_id = min(self._live_viewer.env_id, self.num_envs - 1)
            reward = float(rew[env_id].item()) if rew.numel() > env_id else None
            done = bool(dones[env_id].item()) if dones.numel() > env_id else None
            self._publish_live_frame(reward=reward, done=done)
        return obs, rew, dones, extras

    def close(self):
        if self._live_viewer is not None:
            self._live_viewer.close()
        return super().close()

    def _publish_live_frame(self, *, reward: float | None, done: bool | None) -> None:
        assert self._live_viewer is not None
        self._live_viewer.update(self.unwrapped, self._live_step, reward=reward, done=done)
        self._live_step += 1
