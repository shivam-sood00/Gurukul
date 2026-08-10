from __future__ import annotations

import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan


class ContactTrailOnPolicyRunner(OnPolicyRunner):
    """On-policy runner with contact-trail TensorBoard metrics and optional map dumps."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self._log_contact_trail_metrics(it)
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(f"{self.logger.log_dir}/model_{it}.pt")

        if self.logger.writer is not None:
            self.save(f"{self.logger.log_dir}/model_{self.current_learning_iteration}.pt")
            self.logger.stop_logging_writer()

    def _log_contact_trail_metrics(self, iteration: int) -> None:
        policy = getattr(self.alg, "policy", None)
        if policy is None:
            return

        if hasattr(policy, "set_debug_context"):
            policy.set_debug_context(log_dir=self.logger.log_dir, iteration=iteration)

        writer = getattr(self.logger, "writer", None)
        if hasattr(policy, "pop_logging_stats") and writer is not None:
            metrics = policy.pop_logging_stats()
            for tag, value in metrics.items():
                writer.add_scalar(tag, value, iteration)

        if hasattr(policy, "maybe_save_debug_maps"):
            policy.maybe_save_debug_maps()
