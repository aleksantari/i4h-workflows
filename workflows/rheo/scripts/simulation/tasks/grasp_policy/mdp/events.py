# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Event (reset) functions for the grasp-policy pick-and-place task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_task_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset the task stage counter and all reward caches for the given envs."""
    if hasattr(env, "_task_stage") and env._task_stage is not None:
        env._task_stage[env_ids] = 0

    for attr in ("_prev_stage_grasp", "_prev_stage_transport", "_prev_stage_place"):
        if hasattr(env, attr):
            getattr(env, attr)[env_ids] = 0


def reset_block_random_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
) -> None:
    """Randomise the block position within a small range around its default spawn.

    The randomisation is relative to the block's default initial position.
    """
    block = env.scene[block_cfg.name]
    num_reset = len(env_ids)
    device = env.device

    # Get default root state (pos + quat + lin_vel + ang_vel = 13)
    default_state = block.data.default_root_state[env_ids].clone()  # (num_reset, 13)

    # Add random offsets to x and y (relative to env origins)
    dx = torch.empty(num_reset, device=device).uniform_(*x_range)
    dy = torch.empty(num_reset, device=device).uniform_(*y_range)
    default_state[:, 0] += dx
    default_state[:, 1] += dy

    # Write to sim
    block.write_root_state_to_sim(default_state, env_ids)


def reset_block_to_tray_slot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    slot_pos: tuple[float, float, float] = (-1.55, 1.86, 0.875),
    slot_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    xy_noise: tuple[float, float] = (-0.005, 0.005),
    yaw_noise_deg: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset the block (grasp target) to its tray slot position.

    Teleports the block to the slot centre with optional XY jitter and
    random yaw rotation for training robustness. Base orientation is
    *slot_rot* (w, x, y, z), with yaw noise composed on top.
    Velocities are zeroed.
    """
    block = env.scene[block_cfg.name]
    num_reset = len(env_ids)
    device = env.device

    # Build root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
    default_state = block.data.default_root_state[env_ids].clone()

    # Override position to slot centre
    default_state[:, 0] = slot_pos[0]
    default_state[:, 1] = slot_pos[1]
    default_state[:, 2] = slot_pos[2]

    # Add XY noise for training robustness
    if xy_noise[1] > xy_noise[0]:
        dx = torch.empty(num_reset, device=device).uniform_(*xy_noise)
        dy = torch.empty(num_reset, device=device).uniform_(*xy_noise)
        default_state[:, 0] += dx
        default_state[:, 1] += dy

    # Base orientation (w, x, y, z)
    w0, x0, y0, z0 = slot_rot

    if yaw_noise_deg[1] > yaw_noise_deg[0]:
        # Random yaw around Z, composed with slot_rot: q_final = q_yaw * q_slot
        yaw = torch.empty(num_reset, device=device).uniform_(
            math.radians(yaw_noise_deg[0]),
            math.radians(yaw_noise_deg[1]),
        )
        half = yaw * 0.5
        cw = torch.cos(half)  # q_yaw.w
        sz = torch.sin(half)  # q_yaw.z (x,y = 0 for Z-axis rotation)
        # Hamilton product: q_yaw (cw,0,0,sz) * q_slot (w0,x0,y0,z0)
        default_state[:, 3] = cw * w0 - sz * z0
        default_state[:, 4] = cw * x0 - sz * y0
        default_state[:, 5] = cw * y0 + sz * x0
        default_state[:, 6] = cw * z0 + sz * w0
    else:
        default_state[:, 3] = w0
        default_state[:, 4] = x0
        default_state[:, 5] = y0
        default_state[:, 6] = z0

    # Zero velocities
    default_state[:, 7:] = 0.0

    block.write_root_state_to_sim(default_state, env_ids)
