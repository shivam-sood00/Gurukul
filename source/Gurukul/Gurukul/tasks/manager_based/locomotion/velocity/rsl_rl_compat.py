"""Compatibility helpers for local RSL-RL policy extensions."""

from __future__ import annotations

import torch

from rsl_rl.modules import EmpiricalNormalization, MLP, RNN


class Memory(RNN):
    """RSL-RL 5 recurrent wrapper with the old Gurukul memory API."""

    def __init__(
        self,
        input_size: int,
        type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int | None = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_dim=hidden_dim if hidden_size is None else int(hidden_size),
            num_layers=num_layers,
            type=type,
        )

    @property
    def hidden_states(self):
        return self.hidden_state

    @hidden_states.setter
    def hidden_states(self, value):
        self.hidden_state = value

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_states=None,
        hidden_state=None,
    ) -> None:
        if hidden_states is not None and hidden_state is not None:
            raise ValueError("Pass only one of hidden_states or hidden_state.")
        super().reset(dones=dones, hidden_state=hidden_states if hidden_states is not None else hidden_state)

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        self.detach_hidden_state(dones=dones)
