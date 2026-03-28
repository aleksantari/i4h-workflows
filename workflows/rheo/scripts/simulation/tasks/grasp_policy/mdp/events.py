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
