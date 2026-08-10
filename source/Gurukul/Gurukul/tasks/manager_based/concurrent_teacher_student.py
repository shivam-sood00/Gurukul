from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_nn_activation, resolve_optimizer
from torch.distributions import Normal


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    """Build one of the ELU MLPs listed in CTS Table I."""
    act = resolve_nn_activation(activation)
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(copy.deepcopy(act))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class CTSActorCritic(nn.Module):
    """Concurrent teacher-student actor-critic from CTS §II-B.

    Teacher actions use privileged observations to build a latent. Student actions use a flattened
    deployable observation history to predict the same latent space. The shared actor consumes
    deployable policy observations plus the selected latent.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        privileged_encoder_hidden_dims: list[int] = [512, 256],
        history_encoder_hidden_dims: list[int] = [512, 256],
        latent_dim: int = 32,
        latent_norm: str | None = "l2",
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        student_history_group: str = "student_history",
        teacher_obs_group: str = "teacher",
        role_obs_group: str = "cts_role",
        **kwargs,
    ):
        if kwargs:
            print("CTSActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs)))
        super().__init__()

        if noise_std_type != "scalar":
            raise ValueError("CTSActorCritic currently supports only scalar action noise.")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError(
                "Use runner-level empirical normalization; CTSActorCritic internal normalization is disabled."
            )

        self.obs_groups = obs_groups
        self.student_history_group = student_history_group
        self.teacher_obs_group = teacher_obs_group
        self.role_obs_group = role_obs_group

        self.actor_obs_dim = sum(int(obs[group].shape[-1]) for group in obs_groups["policy"])
        self.critic_obs_dim = sum(int(obs[group].shape[-1]) for group in obs_groups["critic"])
        self.teacher_obs_dim = sum(int(obs[group].shape[-1]) for group in obs_groups[teacher_obs_group])
        self.history_obs_dim = sum(int(obs[group].shape[-1]) for group in obs_groups[student_history_group])

        latent_dim = int(latent_dim)
        self.latent_dim = latent_dim
        if latent_norm not in (None, "none", "l2"):
            raise ValueError(f"Unsupported CTS latent_norm '{latent_norm}'. Supported values: None, 'none', 'l2'.")
        self.latent_norm = latent_norm
        self.privileged_encoder = _build_mlp(
            self.teacher_obs_dim, privileged_encoder_hidden_dims, latent_dim, activation
        )
        self.history_encoder = _build_mlp(self.history_obs_dim, history_encoder_hidden_dims, latent_dim, activation)
        self.actor = _build_mlp(self.actor_obs_dim + latent_dim, actor_hidden_dims, num_actions, activation)
        # CTS §II-B and Table I: V_phi receives the full state and the
        # group-appropriate latent representation.
        self.critic = _build_mlp(self.critic_obs_dim + latent_dim, critic_hidden_dims, 1, activation)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _concat_group(self, obs: dict[str, torch.Tensor], group_name: str) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups[group_name]], dim=-1)

    def get_actor_obs(self, obs):
        return self._concat_group(obs, "policy")

    def get_critic_obs(self, obs):
        return self._concat_group(obs, "critic")

    def get_teacher_obs(self, obs):
        return self._concat_group(obs, self.teacher_obs_group)

    def get_history_obs(self, obs):
        return self._concat_group(obs, self.student_history_group)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def output_std(self):
        return self.std.clamp_min(1.0e-6)

    @property
    def output_distribution_params(self):
        return (self.action_mean, self.action_std)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def output_entropy(self):
        return self.entropy

    def reset(self, dones=None, hidden_state=None):
        pass

    def _update_distribution_from_mean(self, mean: torch.Tensor) -> None:
        # [UNSPECIFIED] CTS does not specify action-noise parameterization.
        # Using: RSL-RL's state-independent diagonal Gaussian with a positive minimum scale.
        # Alternatives: log-standard-deviation parameterization or fixed exploration noise.
        self.distribution = Normal(mean, mean * 0.0 + self.output_std)

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Map z to the unit hypersphere as specified in CTS §II-B."""
        if self.latent_norm == "l2":
            return F.normalize(latent, p=2.0, dim=-1)
        return latent

    def _role_mask(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.role_obs_group not in obs:
            raise ValueError(f"CTS role observation group '{self.role_obs_group}' is missing.")
        role = obs[self.role_obs_group]
        while role.ndim > 1:
            role = role[..., 0]
        return role.reshape(-1) > 0.5

    def _mixed_latent(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build z_t from E_teacher or E_student for the shared critic (CTS §II-B)."""
        teacher_mask = self._role_mask(obs)
        latent = self.get_actor_obs(obs).new_empty((teacher_mask.shape[0], self.latent_dim))
        if torch.any(teacher_mask):
            teacher_obs = {name: value[teacher_mask] for name, value in obs.items()}
            latent[teacher_mask] = self._normalize_latent(self.privileged_encoder(self.get_teacher_obs(teacher_obs)))
        if torch.any(~teacher_mask):
            student_obs = {name: value[~teacher_mask] for name, value in obs.items()}
            latent[~teacher_mask] = self._normalize_latent(self.history_encoder(self.get_history_obs(student_obs)))
        # Algorithm 1 updates phi with the value loss, theta_teacher with policy
        # gradient, and theta_student with reconstruction. The critic therefore
        # consumes z without sending value-loss gradients into either encoder.
        return latent.detach()

    def forward_teacher(self, obs):
        actor_obs = self.get_actor_obs(obs)
        latent = self._normalize_latent(self.privileged_encoder(self.get_teacher_obs(obs)))
        return self.actor(torch.cat((actor_obs, latent), dim=-1))

    def forward_student(self, obs):
        actor_obs = self.get_actor_obs(obs)
        # CTS Algorithm 1 updates the shared policy with student trajectories but
        # reserves E_student updates for the reconstruction phase (line 11).
        latent = self._normalize_latent(self.history_encoder(self.get_history_obs(obs))).detach()
        return self.actor(torch.cat((actor_obs, latent), dim=-1))

    def forward(self, obs):
        return self.act_inference(obs)

    def act_teacher(self, obs):
        mean = self.forward_teacher(obs)
        self._update_distribution_from_mean(mean)
        return self.distribution.sample()

    def act_student(self, obs):
        mean = self.forward_student(obs)
        self._update_distribution_from_mean(mean)
        return self.distribution.sample()

    def act(self, obs, masks=None, hidden_states=None):
        return self.act_student(obs)

    def act_inference(self, obs):
        return self.forward_student(obs)

    def evaluate(self, obs, masks=None, hidden_states=None):
        return self.critic(torch.cat((self.get_critic_obs(obs), self._mixed_latent(obs)), dim=-1))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_output_log_prob(self, actions):
        return self.get_actions_log_prob(actions)

    def get_hidden_states(self):
        return None, None

    def get_hidden_state(self):
        return None

    def update_normalization(self, obs):
        pass

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)


class CTSPPO(PPO):
    """CTS Algorithm 1 with Equations 3--8 and an RSL-RL rollout buffer."""

    policy: CTSActorCritic

    def __init__(
        self,
        policy,
        storage=None,
        role_obs_group: str = "cts_role",
        encoder_learning_rate: float = 1.0e-3,
        num_encoder_epochs: int = 5,
        teacher_loss_coef: float = 1.0,
        student_loss_coef: float = 1.0,
        encoder_loss_coef: float = 1.0,
        multi_gpu_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ):
        if rnd_cfg:
            raise ValueError("CTSPPO does not support RND.")
        if symmetry_cfg and (symmetry_cfg.get("use_data_augmentation") or symmetry_cfg.get("use_mirror_loss")):
            raise ValueError("CTSPPO does not support symmetry augmentation or mirror losses.")
        if storage is None:
            raise ValueError("CTSPPO requires an initialized RolloutStorage instance.")
        self.device = kwargs.pop("device", "cpu")
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = symmetry_cfg
        self.policy = policy.to(self.device)
        self.actor = self.policy
        self.critic = self.policy
        self._raw_actor = self.policy
        self._raw_critic = self.policy
        self.storage = storage
        from rsl_rl.storage import RolloutStorage

        self.transition = RolloutStorage.Transition()
        self.clip_param = kwargs.pop("clip_param", 0.2)
        self.num_learning_epochs = kwargs.pop("num_learning_epochs", 5)
        self.num_mini_batches = kwargs.pop("num_mini_batches", 4)
        self.value_loss_coef = kwargs.pop("value_loss_coef", 1.0)
        self.entropy_coef = kwargs.pop("entropy_coef", 0.01)
        self.gamma = kwargs.pop("gamma", 0.99)
        self.lam = kwargs.pop("lam", 0.95)
        self.max_grad_norm = kwargs.pop("max_grad_norm", 1.0)
        self.use_clipped_value_loss = kwargs.pop("use_clipped_value_loss", True)
        self.desired_kl = kwargs.pop("desired_kl", None)
        self.schedule = kwargs.pop("schedule", "fixed")
        self.learning_rate = kwargs.pop("learning_rate", 1.0e-3)
        self.normalize_advantage_per_mini_batch = kwargs.pop("normalize_advantage_per_mini_batch", False)
        kwargs.pop("share_cnn_encoders", None)
        optimizer_name = kwargs.pop("optimizer", "adam")
        if kwargs:
            raise TypeError(f"Unexpected CTSPPO arguments: {sorted(kwargs)}")
        self.role_obs_group = role_obs_group
        self.num_encoder_epochs = int(num_encoder_epochs)
        self.teacher_loss_coef = float(teacher_loss_coef)
        self.student_loss_coef = float(student_loss_coef)
        self.encoder_loss_coef = float(encoder_loss_coef)
        self.rl_params = (
            list(self.policy.actor.parameters())
            + list(self.policy.critic.parameters())
            + list(self.policy.privileged_encoder.parameters())
            + [self.policy.std]
        )
        self.optimizer = resolve_optimizer(optimizer_name)(self.rl_params, lr=self.learning_rate)
        self.history_encoder_optimizer = optim.Adam(self.policy.history_encoder.parameters(), lr=encoder_learning_rate)

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        """Construct CTS directly from a single shared actor-critic config."""
        from rsl_rl.storage import RolloutStorage
        from rsl_rl.utils import resolve_obs_groups

        alg_class = CTSPPO
        policy_class = CTSActorCritic
        algorithm_class_name = cfg["algorithm"].pop("class_name", None)
        if algorithm_class_name and (":" in algorithm_class_name or "." in algorithm_class_name):
            from rsl_rl.utils import resolve_callable

            alg_class = resolve_callable(algorithm_class_name)

        policy_class_name = cfg["policy"].pop("class_name", None)
        if policy_class_name and (":" in policy_class_name or "." in policy_class_name):
            from rsl_rl.utils import resolve_callable

            policy_class = resolve_callable(policy_class_name)

        default_sets = ["policy", "teacher", "student_history", "critic", "cts_role"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        policy_cfg = dict(cfg["policy"])
        actor_cfg = cfg.get("actor", {}) or {}
        critic_cfg = cfg.get("critic", {}) or {}
        if actor_cfg:
            policy_cfg.setdefault("actor_hidden_dims", actor_cfg.get("hidden_dims"))
            policy_cfg.setdefault("activation", actor_cfg.get("activation"))
            policy_cfg.setdefault("actor_obs_normalization", actor_cfg.get("obs_normalization", False))
            policy_cfg.setdefault("init_noise_std", actor_cfg.get("init_noise_std", 1.0))
            policy_cfg.setdefault("noise_std_type", actor_cfg.get("noise_std_type", "scalar"))
        if critic_cfg:
            policy_cfg.setdefault("critic_hidden_dims", critic_cfg.get("hidden_dims"))
            policy_cfg.setdefault("critic_obs_normalization", critic_cfg.get("obs_normalization", False))

        cts_policy_keys = (
            "actor_hidden_dims",
            "critic_hidden_dims",
            "privileged_encoder_hidden_dims",
            "history_encoder_hidden_dims",
            "latent_dim",
            "latent_norm",
            "activation",
            "init_noise_std",
            "noise_std_type",
            "actor_obs_normalization",
            "critic_obs_normalization",
            "student_history_group",
            "teacher_obs_group",
        )
        for key in cts_policy_keys:
            if key in cfg["algorithm"]:
                policy_cfg[key] = cfg["algorithm"].pop(key)
        policy_cfg = {key: value for key, value in policy_cfg.items() if value is not None}
        policy = policy_class(obs, cfg["obs_groups"], env.num_actions, **policy_cfg).to(device)
        print(f"CTS Actor-Critic Model: {policy}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(policy, storage=storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

    def _role_mask(self, obs_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.role_obs_group not in obs_batch:
            raise ValueError(f"CTS role observation group '{self.role_obs_group}' is missing.")
        role = obs_batch[self.role_obs_group]
        while role.ndim > 1:
            role = role[..., 0]
        return role.reshape(-1) > 0.5

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        teacher_mask = self._role_mask(obs)
        actions = torch.empty((teacher_mask.shape[0], self.policy.std.shape[0]), device=self.device)
        action_log_prob = torch.empty((teacher_mask.shape[0],), device=self.device)
        action_mean = torch.empty_like(actions)
        action_sigma = torch.empty_like(actions)

        if torch.any(teacher_mask):
            teacher_obs = {name: value[teacher_mask] for name, value in obs.items()}
            teacher_actions = self.policy.act_teacher(teacher_obs).detach()
            actions[teacher_mask] = teacher_actions
            action_log_prob[teacher_mask] = self.policy.get_actions_log_prob(teacher_actions).detach()
            action_mean[teacher_mask] = self.policy.action_mean.detach()
            action_sigma[teacher_mask] = self.policy.action_std.detach()
        if torch.any(~teacher_mask):
            student_obs = {name: value[~teacher_mask] for name, value in obs.items()}
            student_actions = self.policy.act_student(student_obs).detach()
            actions[~teacher_mask] = student_actions
            action_log_prob[~teacher_mask] = self.policy.get_actions_log_prob(student_actions).detach()
            action_mean[~teacher_mask] = self.policy.action_mean.detach()
            action_sigma[~teacher_mask] = self.policy.action_std.detach()

        self.transition.actions = actions
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = action_log_prob.detach()
        self.transition.action_mean = action_mean.detach()
        self.transition.action_sigma = action_sigma.detach()
        self.transition.distribution_params = (action_mean.detach(), action_sigma.detach())
        self.transition.observations = obs
        return self.transition.actions

    def _ppo_surrogate(self, actions_log_prob, old_actions_log_prob, advantages):
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _encoder_loss(self, obs_batch):
        """Compute Equation 8 over student trajectories only."""
        student_mask = ~self._role_mask(obs_batch)
        if not torch.any(student_mask):
            return None
        student_obs = {name: value[student_mask] for name, value in obs_batch.items()}
        prediction = self.policy._normalize_latent(
            self.policy.history_encoder(self.policy.get_history_obs(student_obs))
        )
        with torch.no_grad():
            target = self.policy._normalize_latent(
                self.policy.privileged_encoder(self.policy.get_teacher_obs(student_obs))
            )
        # CTS §II-C, Eq. 8 averages squared L2 norms over student samples/time; elementwise
        # MSE would silently scale the objective by 1 / latent_dim.
        return torch.sum(torch.square(prediction - target), dim=-1).mean()

    def _student_encoder_batches(self):
        """Yield the student-only minibatches specified by CTS §III-B, Table III."""
        observations = self.storage.observations.flatten(0, 1)
        student_indices = torch.nonzero(~self._role_mask(observations), as_tuple=False).flatten()
        mini_batch_size = student_indices.numel() // self.num_mini_batches
        if mini_batch_size == 0:
            return
        permutation = torch.randperm(student_indices.numel(), device=student_indices.device)
        shuffled_indices = student_indices[permutation]
        for _ in range(self.num_encoder_epochs):
            for mini_batch in range(self.num_mini_batches):
                start = mini_batch * mini_batch_size
                stop = start + mini_batch_size
                yield observations[shuffled_indices[start:stop]]

    def _adapt_learning_rate(
        self,
        old_distribution_params: tuple[torch.Tensor, torch.Tensor] | None,
        new_distribution_params: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Apply the desired-KL schedule from CTS §III-B, Table III using RSL-RL's PPO rule."""
        if old_distribution_params is None or self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            old_mean, old_std = old_distribution_params
            new_mean, new_std = new_distribution_params
            kl_mean = (
                torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std))
                .sum(dim=-1)
                .mean()
            )
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                learning_rate = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(learning_rate, src=0)
                self.learning_rate = learning_rate.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def compute_returns(self, obs):
        """Bootstrap with the critic while keeping policy ``forward`` actor-facing for playback."""
        storage = self.storage
        last_values = self.policy.evaluate(obs).detach()
        advantage = 0
        for step in reversed(range(storage.num_transitions_per_env)):
            next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
            next_is_not_terminal = 1.0 - storage.dones[step].float()
            delta = storage.rewards[step] + next_is_not_terminal * self.gamma * next_values - storage.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            storage.returns[step] = advantage + storage.values[step]
        storage.advantages = storage.returns - storage.values
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages = (storage.advantages - storage.advantages.mean()) / (storage.advantages.std() + 1.0e-8)

    def update(self):  # noqa: C901
        if self.rnd or (self.symmetry and (self.symmetry["use_data_augmentation"] or self.symmetry["use_mirror_loss"])):
            raise NotImplementedError("CTSPPO does not support RND or symmetry augmentation yet.")

        mean_value_loss = 0.0
        mean_teacher_surrogate_loss = 0.0
        mean_student_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_encoder_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if hasattr(batch, "observations"):
                obs_batch = batch.observations
                actions_batch = batch.actions
                target_values_batch = batch.values
                advantages_batch = batch.advantages
                returns_batch = batch.returns
                old_actions_log_prob_batch = batch.old_actions_log_prob
                old_distribution_params_batch = batch.old_distribution_params
            else:
                (
                    obs_batch,
                    _critic_obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    *_,
                ) = batch
                old_distribution_params_batch = (old_mu_batch, old_sigma_batch)
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            teacher_mask = self._role_mask(obs_batch)
            teacher_surrogate_loss = actions_batch.new_tensor(0.0)
            student_surrogate_loss = actions_batch.new_tensor(0.0)
            entropy_terms = []
            action_mean = torch.empty_like(actions_batch)
            action_sigma = torch.empty_like(actions_batch)

            if torch.any(teacher_mask):
                teacher_obs = {name: value[teacher_mask] for name, value in obs_batch.items()}
                self.policy.act_teacher(teacher_obs)
                teacher_surrogate_loss = self._ppo_surrogate(
                    self.policy.get_actions_log_prob(actions_batch[teacher_mask]),
                    old_actions_log_prob_batch[teacher_mask],
                    advantages_batch[teacher_mask],
                )
                entropy_terms.append(self.policy.entropy)
                action_mean[teacher_mask] = self.policy.action_mean
                action_sigma[teacher_mask] = self.policy.action_std
            if torch.any(~teacher_mask):
                student_obs = {name: value[~teacher_mask] for name, value in obs_batch.items()}
                self.policy.act_student(student_obs)
                student_surrogate_loss = self._ppo_surrogate(
                    self.policy.get_actions_log_prob(actions_batch[~teacher_mask]),
                    old_actions_log_prob_batch[~teacher_mask],
                    advantages_batch[~teacher_mask],
                )
                entropy_terms.append(self.policy.entropy)
                action_mean[~teacher_mask] = self.policy.action_mean
                action_sigma[~teacher_mask] = self.policy.action_std

            self._adapt_learning_rate(old_distribution_params_batch, (action_mean, action_sigma))

            entropy_batch = torch.cat(entropy_terms).mean() if entropy_terms else actions_batch.new_tensor(0.0)
            value_batch = self.policy.evaluate(obs_batch)
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                self.teacher_loss_coef * teacher_surrogate_loss
                + self.student_loss_coef * student_surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch
            )
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._reduce_parameter_gradients(self.rl_params)
            nn.utils.clip_grad_norm_(self.rl_params, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += float(value_loss.detach().item())
            mean_teacher_surrogate_loss += float(teacher_surrogate_loss.detach().item())
            mean_student_surrogate_loss += float(student_surrogate_loss.detach().item())
            mean_entropy += float(entropy_batch.detach().item())

        # CTS Algorithm 1 lines 10--12: run a distinct student-encoder phase
        # after the PPO epochs. Filtering each mixed batch produces the Table III
        # student minibatch size at the paper's 3:1 rollout split.
        num_encoder_updates = 0
        for encoder_obs in self._student_encoder_batches():
            encoder_loss = self._encoder_loss(encoder_obs)
            if encoder_loss is None:
                continue
            encoder_loss = self.encoder_loss_coef * encoder_loss
            self.history_encoder_optimizer.zero_grad()
            encoder_loss.backward()
            if self.is_multi_gpu:
                self._reduce_parameter_gradients(self.policy.history_encoder.parameters())
            nn.utils.clip_grad_norm_(self.policy.history_encoder.parameters(), self.max_grad_norm)
            self.history_encoder_optimizer.step()
            mean_encoder_loss += float(encoder_loss.detach().item())
            num_encoder_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "teacher_surrogate": mean_teacher_surrogate_loss / num_updates,
            "student_surrogate": mean_student_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "history_encoder": mean_encoder_loss / max(1, num_encoder_updates),
        }

    def save(self) -> dict:
        """Save both optimizers so resumed CTS training preserves encoder momentum."""
        saved_dict = super().save()
        saved_dict["history_encoder_optimizer_state_dict"] = self.history_encoder_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load CTS state, accepting older checkpoints that lack the encoder optimizer."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        should_load_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        if should_load_optimizer and "history_encoder_optimizer_state_dict" in loaded_dict:
            self.history_encoder_optimizer.load_state_dict(loaded_dict["history_encoder_optimizer_state_dict"])
        return load_iteration

    def _reduce_parameter_gradients(self, parameters) -> None:
        """Average one unique parameter set across distributed workers."""
        parameters = [parameter for parameter in parameters if parameter.grad is not None]
        if not parameters:
            return
        flattened = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
        flattened /= self.gpu_world_size
        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            parameter.grad.copy_(flattened[offset : offset + numel].view_as(parameter.grad))
            offset += numel
