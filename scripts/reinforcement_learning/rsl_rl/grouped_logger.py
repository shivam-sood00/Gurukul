from __future__ import annotations

import pathlib
import statistics
import time
from collections.abc import Iterable

import torch
from rsl_rl.utils.logger import Logger

_EXTRA_GROUP_ORDER = (
    ("Episode rewards", ("Episode_Reward/",)),
    ("Episode terminations", ("Episode_Termination/",)),
    ("Motion metrics", ("Metrics/motion/",)),
    ("DecAP metrics", ("Metrics/decap/",)),
    ("Episode metrics", ("Episode/",)),
    ("Other metrics", ()),
)


class GroupedConsoleLogger(Logger):
    """RSL-RL logger with deterministic grouped console output for episode extras."""

    @staticmethod
    def _sorted_extra_keys(ep_extras: Iterable[dict]) -> list[tuple[str, str]]:
        keys = {key for ep_info in ep_extras for key in ep_info}
        grouped_keys: list[tuple[str, str]] = []
        used_keys: set[str] = set()

        for group_name, prefixes in _EXTRA_GROUP_ORDER[:-1]:
            group_keys = sorted(key for key in keys if any(key.startswith(prefix) for prefix in prefixes))
            grouped_keys.extend((group_name, key) for key in group_keys)
            used_keys.update(group_keys)

        other_keys = sorted(keys - used_keys)
        grouped_keys.extend((_EXTRA_GROUP_ORDER[-1][0], key) for key in other_keys)
        return grouped_keys

    def _mean_extra_value(self, key: str) -> torch.Tensor:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in self.ep_extras:
            if key not in ep_info:
                continue
            value = ep_info[key]
            if not isinstance(value, torch.Tensor):
                value = torch.Tensor([value])
                ep_info[key] = value
            if len(value.shape) == 0:
                value = value.unsqueeze(0)
                ep_info[key] = value
            infotensor = torch.cat((infotensor, value.to(self.device)))
        return torch.mean(infotensor)

    @staticmethod
    def _display_extra_key(key: str) -> str:
        return key if "/" in key else f"Mean episode {key}"

    def _log_episode_extras(self, it: int, pad: int) -> str:
        extras_string = ""
        previous_group_name = None
        for group_name, key in self._sorted_extra_keys(self.ep_extras):
            value = self._mean_extra_value(key)
            if "/" in key:
                self.writer.add_scalar(key, value, it)  # type: ignore
            else:
                self.writer.add_scalar("Episode/" + key, value, it)  # type: ignore

            if group_name != previous_group_name:
                extras_string += f"""{f"-- {group_name} --":>{pad}}\n"""
                previous_group_name = group_name
            extras_string += f"""{f"{self._display_extra_key(key)}:":>{pad}} {value:.4f}\n"""
        return extras_string

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log training metrics and print grouped episode extras to the console."""
        if self.writer is None:
            return

        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        extras_string = self._log_episode_extras(it, pad) if self.ep_extras else ""

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)

        self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(self.rewbuffer) > 0:
            if self.cfg["algorithm"]["rnd_cfg"]:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                self.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time))
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                )

        log_string = f"""{"#" * width}\n"""
        log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

        run_name = self.cfg.get("run_name")
        log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

        log_string += (
            f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
            f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
            f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
            f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
        )

        for key, value in loss_dict.items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

        if len(self.rewbuffer) > 0:
            if self.cfg["algorithm"]["rnd_cfg"]:
                log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

        log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""

        if not print_minimal:
            log_string += extras_string

        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / done_it * remaining_it
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
        )
        print(log_string)

        if self.logger_type == "wandb":
            for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore
                self.writer.save_video(video, it)  # type: ignore

        self.ep_extras.clear()
