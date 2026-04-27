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

"""
Base closed-loop policy with action chunking and joint remapping.

Provides the common infrastructure shared by GR00T, ACT, and future VLA
policy wrappers. Subclasses only need to implement model loading and
observation/action format conversion.
"""

from abc import abstractmethod
from typing import Any

import gymnasium as gym
import torch
from isaaclab_arena.policy.policy_base import PolicyBase


class BaseClosedloopPolicy(PolicyBase):
    """Abstract base class for closed-loop policies with action chunking.

    Manages the action chunk lifecycle: when the current chunk is exhausted,
    requests a new one from the subclass. Handles per-environment state
    tracking for parallel evaluation.

    Subclasses must implement:
        - _load_model(): Load the model and return it.
        - _get_action_chunk(observation): Run model forward, return (num_envs, chunk_len, sim_action_dim).

    Args:
        num_envs: Number of parallel environments.
        action_chunk_length: Actions predicted per forward pass.
        sim_action_dim: Simulator action dimension (43 for G1 full robot).
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int = 1,
        action_chunk_length: int = 16,
        sim_action_dim: int = 43,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.action_chunk_length = action_chunk_length
        self.sim_action_dim = sim_action_dim
        self.device = device

        # Action chunking state
        self.current_action_chunk = torch.zeros(
            (num_envs, action_chunk_length, sim_action_dim),
            dtype=torch.float32,
            device=device,
        )
        self.env_requires_new_action_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the underlying model. Called during __init__."""
        ...

    @abstractmethod
    def _get_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        """Run model forward pass and return actions in simulator joint order.

        Args:
            observation: Raw IsaacLab observation dict.

        Returns:
            Action chunk tensor of shape (num_envs, action_chunk_length, sim_action_dim).
        """
        ...

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Get the next action from the current action chunk.

        If the chunk is exhausted, requests a new one via _get_action_chunk().

        Returns:
            action: Shape (num_envs, sim_action_dim).
        """
        if any(self.env_requires_new_action_chunk):
            new_chunk = self._get_action_chunk(observation)
            self.current_action_chunk[self.env_requires_new_action_chunk] = new_chunk[
                self.env_requires_new_action_chunk
            ]
            self.current_action_index[self.env_requires_new_action_chunk] = 0
            self.env_requires_new_action_chunk[self.env_requires_new_action_chunk] = False

        # Select action at current index per env
        action = self.current_action_chunk[torch.arange(self.num_envs), self.current_action_index]
        self.current_action_index += 1

        # Reset envs that exhausted their chunk
        reset_mask = self.current_action_index >= self.action_chunk_length
        self.current_action_chunk[reset_mask] = 0.0
        self.env_requires_new_action_chunk[reset_mask] = True
        self.current_action_index[reset_mask] = -1

        return action

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset action chunking state for specified environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True

    @staticmethod
    def pad_28d_to_43d(actions_28d: torch.Tensor) -> torch.Tensor:
        """Pad 28D policy actions to 43D sim actions (prepend 15 zeros for legs/waist).

        Args:
            actions_28d: Shape (..., 28).

        Returns:
            Shape (..., 43) with 15 leading zeros.
        """
        pad_shape = actions_28d.shape[:-1] + (15,)
        pad = torch.zeros(pad_shape, dtype=actions_28d.dtype, device=actions_28d.device)
        return torch.cat([pad, actions_28d], dim=-1)
