import torch

import isaaclab.utils.math as math_utils


def camera_follow(
    env,
    mode: str = "follow",
    window_size: int = 50,
    env_index: int = 0,
    follow_offset: tuple[float, float, float] | None = None,
):
    if not hasattr(camera_follow, "history"):
        camera_follow.history = {}

    window_size = max(1, int(window_size))
    history_key = (int(env_index), str(mode))
    smooth_camera_positions = camera_follow.history.setdefault(history_key, [])

    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]

    if mode == "follow":
        offset = follow_offset if follow_offset is not None else (-3.0, 0.0, 0.5)
        camera_offset = torch.tensor(offset, dtype=torch.float32, device=env.device)
        camera_pos = math_utils.transform_points(
            camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
        ).squeeze(0)
    elif mode == "isometric":
        # Keep camera in world frame for a stable diagonal overview.
        camera_offset = torch.tensor([-3.5, 3.5, 2.8], dtype=torch.float32, device=env.device)
        camera_pos = robot_pos + camera_offset
    elif mode == "topdown":
        camera_offset = torch.tensor([0.0, 0.0, 6.0], dtype=torch.float32, device=env.device)
        camera_pos = robot_pos + camera_offset
    else:
        raise ValueError(f"Unsupported camera follow mode: {mode}")

    smooth_camera_positions.append(camera_pos)
    if len(smooth_camera_positions) > window_size:
        smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(smooth_camera_positions), dim=0)

    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=env_index)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )
