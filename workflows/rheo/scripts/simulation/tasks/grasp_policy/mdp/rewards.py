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

"""Reward functions for the grasp-policy pick-and-place task.

Three-stage sparse rewards:
  Stage 0 -> 1: Block lifted above table
  Stage 1 -> 2: Block positioned over bin (x, y within bin bounds)
  Stage 2 -> 3: Block placed inside bin (z below rim, within bounds)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_task_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current task stage tensor (num_envs,), initialising on first call."""
    if not hasattr(env, "_task_stage") or env._task_stage is None:
        env._task_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._task_stage


def update_task_stage(
    env: ManagerBasedRLEnv,
    table_height: float = 0.855,
    lift_threshold: float = 0.05,
    bin_x_min: float = -1.90,
    bin_x_max: float = -1.60,
    bin_y_min: float = 1.55,
    bin_y_max: float = 1.85,
    bin_rim_z: float = 0.935,
    print_log: bool = False,
) -> torch.Tensor:
    """Check all stage transitions for the grasp-policy task.

    Stages only advance forward, never backward.

    Args:
        table_height: Height of the table surface (m).
        lift_threshold: How far above the table the block must be lifted (m).
        bin_x_min/max, bin_y_min/max: Horizontal bounds of the bin interior.
        bin_rim_z: Z height of the bin rim (top of walls).
        print_log: Print debug info.
    """
    stage = get_task_stage(env)
    block: torch.Tensor = env.scene["block"].data.root_pos_w  # (num_envs, 3)

    # Subtract env origins for multi-env support
    env_origins = env.scene.env_origins  # (num_envs, 3)
    block_local = block - env_origins

    bx, by, bz = block_local[:, 0], block_local[:, 1], block_local[:, 2]

    # Stage 0 -> 1: Block lifted above table
    lifted = bz > (table_height + lift_threshold)
    can_advance_0 = (stage == 0) & lifted
    stage = torch.where(can_advance_0, torch.ones_like(stage), stage)

    # Stage 1 -> 2: Block horizontally over bin
    over_bin = (bx > bin_x_min) & (bx < bin_x_max) & (by > bin_y_min) & (by < bin_y_max)
    can_advance_1 = (stage == 1) & over_bin
    stage = torch.where(can_advance_1, torch.full_like(stage, 2), stage)

    # Stage 2 -> 3: Block placed inside bin (below rim and within bounds)
    in_bin = over_bin & (bz < bin_rim_z) & (bz > table_height)
    can_advance_2 = (stage == 2) & in_bin
    stage = torch.where(can_advance_2, torch.full_like(stage, 3), stage)

    env._task_stage = stage

    if print_log and env.episode_length_buf[0].item() % 50 == 0:
        print(f"[GraspPolicy] step={env.episode_length_buf[0].item()} stages={stage.tolist()}")

    return stage


def grasp_reward(
    env: ManagerBasedRLEnv,
    table_height: float = 0.855,
    lift_threshold: float = 0.05,
    bin_x_min: float = -1.90,
    bin_x_max: float = -1.60,
    bin_y_min: float = 1.55,
    bin_y_max: float = 1.85,
    bin_rim_z: float = 0.935,
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for stage 0 -> 1 transition (grasping and lifting the block)."""
    update_task_stage(
        env,
        table_height=table_height,
        lift_threshold=lift_threshold,
        bin_x_min=bin_x_min,
        bin_x_max=bin_x_max,
        bin_y_min=bin_y_min,
        bin_y_max=bin_y_max,
        bin_rim_z=bin_rim_z,
        print_log=print_log,
    )
    stage = env._task_stage

    if not hasattr(env, "_prev_stage_grasp"):
        env._prev_stage_grasp = torch.zeros_like(stage)

    reward = ((stage >= 1) & (env._prev_stage_grasp < 1)).float()
    env._prev_stage_grasp = stage.clone()

    if use_sparse_reward:
        reward = reward / env.step_dt
    return reward


def transport_reward(
    env: ManagerBasedRLEnv,
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for stage 1 -> 2 transition (moving block over the bin)."""
    stage = get_task_stage(env)

    if not hasattr(env, "_prev_stage_transport"):
        env._prev_stage_transport = torch.zeros_like(stage)

    reward = ((stage >= 2) & (env._prev_stage_transport < 2)).float()
    env._prev_stage_transport = stage.clone()

    if use_sparse_reward:
        reward = reward / env.step_dt
    return reward


def place_reward(
    env: ManagerBasedRLEnv,
    use_sparse_reward: bool = True,
    print_log: bool = False,
) -> torch.Tensor:
    """Reward for stage 2 -> 3 transition (placing block inside the bin)."""
    stage = get_task_stage(env)

    if not hasattr(env, "_prev_stage_place"):
        env._prev_stage_place = torch.zeros_like(stage)

    reward = ((stage >= 3) & (env._prev_stage_place < 3)).float()
    env._prev_stage_place = stage.clone()

    if use_sparse_reward:
        reward = reward / env.step_dt
    return reward
