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
ACT (Action Chunking Transformer) closed-loop policy wrapper for IsaacLab evaluation.

Loads a LeRobot-trained ACT checkpoint and wraps it in the PolicyBase interface
used by the rheo evaluation pipeline.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_arena.policy.policy_base import PolicyBase


class ACTClosedloopPolicy(PolicyBase):
    """Closed-loop policy wrapper for LeRobot ACT models.

    Implements the same action chunking pattern as CustomGr00tClosedloopPolicy
    but loads an ACT checkpoint from LeRobot's training pipeline.

    Args:
        policy_config_yaml_path: Path to YAML config with model_path, action settings, etc.
        num_envs: Number of parallel environments.
        device: Device to run inference on.
    """

    # Indices into the 87D robot_joint_state for shoulder/arm joints (same as rlinf_ext)
    _ARM_JOINT_SLICE = slice(15, 29)  # 14 joints: left_arm(7) + right_arm(7)

    def __init__(self, policy_config_yaml_path: Path, num_envs: int = 1, device: str = "cuda"):
        with open(policy_config_yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.num_envs = num_envs
        self.device = device
        self.model_path = Path(self.config["model_path"])
        self.action_chunk_length = self.config.get("action_chunk_length", 100)
        self.language_instruction = self.config.get("language_instruction", "pick up block and place in bin")

        # Image target size (H, W, C)
        self.target_image_size = tuple(self.config.get("target_image_size", [480, 640, 3]))

        # Action dim: 28D policy output padded to 43D for sim (15 zeros for legs/waist)
        self.policy_action_dim = self.config.get("policy_action_dim", 28)
        self.sim_action_dim = self.config.get("sim_action_dim", 43)
        self.leg_waist_pad = self.sim_action_dim - self.policy_action_dim  # 15

        # Load ACT policy
        self.policy = self._load_policy()

        # Action chunking state
        self.current_action_chunk = torch.zeros(
            (num_envs, self.action_chunk_length, self.sim_action_dim),
            dtype=torch.float32,
            device=device,
        )
        self.env_requires_new_action_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def _load_policy(self):
        """Load ACT policy from LeRobot checkpoint."""
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        pretrained_path = str(self.model_path)
        policy = ACTPolicy.from_pretrained(pretrained_path)
        policy.eval()
        policy.to(self.device)
        print(f"[ACT] Loaded policy from {pretrained_path}")
        return policy

    def _extract_observations(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert IsaacLab observation dict to ACT input format.

        ACT expects a flat dict with keys like:
          - observation.state: (B, 28) joint positions
          - observation.images.cam_left_wrist: (B, C, H, W)
          - observation.images.cam_right_wrist: (B, C, H, W)
          - observation.images.cam_room: (B, C, H, W)
        """
        # Extract joint states (28D: arms + hands)
        body_state = observation["policy"]["robot_joint_state"]  # (B, 87)
        dex3_state = observation["policy"]["robot_dex3_joint_state"]  # (B, 14)
        arm_state = body_state[:, self._ARM_JOINT_SLICE]  # (B, 14): left_arm(7) + right_arm(7)
        state_28d = torch.cat([arm_state, dex3_state], dim=-1)  # (B, 28)

        # Extract camera images and convert to (B, C, H, W) float32 [0, 1]
        camera_obs = observation["camera_images"]
        cam_keys_map = {
            "left_wrist_camera": "observation.images.cam_left_wrist",
            "right_wrist_camera": "observation.images.cam_right_wrist",
            "front_camera": "observation.images.cam_room",
        }

        act_obs = {"observation.state": state_28d.to(self.device)}

        for sim_key, act_key in cam_keys_map.items():
            img = camera_obs[sim_key]  # (B, H, W, C) uint8 on GPU
            img = img.float() / 255.0  # normalize to [0, 1]
            img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
            act_obs[act_key] = img.to(self.device)

        return act_obs

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Get the next action from the current action chunk.

        Returns:
            action: Shape (num_envs, sim_action_dim=43)
        """
        if any(self.env_requires_new_action_chunk):
            new_chunk = self._get_action_chunk(observation)
            self.current_action_chunk[self.env_requires_new_action_chunk] = new_chunk[
                self.env_requires_new_action_chunk
            ]
            self.current_action_index[self.env_requires_new_action_chunk] = 0
            self.env_requires_new_action_chunk[self.env_requires_new_action_chunk] = False

        # Select action at current index for each env
        action = self.current_action_chunk[torch.arange(self.num_envs), self.current_action_index]
        self.current_action_index += 1

        # Reset envs that exhausted their chunk
        reset_mask = self.current_action_index >= self.action_chunk_length
        self.current_action_chunk[reset_mask] = 0.0
        self.env_requires_new_action_chunk[reset_mask] = True
        self.current_action_index[reset_mask] = -1

        return action

    @torch.no_grad()
    def _get_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        """Run ACT forward pass to get a chunk of actions.

        Returns:
            action_chunk: Shape (num_envs, action_chunk_length, sim_action_dim=43)
        """
        act_obs = self._extract_observations(observation)

        # ACT forward: returns (B, chunk_size, action_dim) or processes per-env
        # LeRobot ACT select_action returns (chunk_size, action_dim) for single env
        chunks = []
        for i in range(self.num_envs):
            single_obs = {k: v[i : i + 1] for k, v in act_obs.items()}
            action = self.policy.select_action(single_obs)  # (chunk_size, action_dim)
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            chunks.append(action)

        # Stack: (num_envs, chunk_size, 28)
        policy_actions = torch.stack(chunks, dim=0).to(self.device)

        # Pad 28D → 43D (prepend 15 zeros for legs/waist)
        pad = torch.zeros(
            self.num_envs, policy_actions.shape[1], self.leg_waist_pad,
            dtype=policy_actions.dtype, device=self.device,
        )
        sim_actions = torch.cat([pad, policy_actions], dim=-1)  # (B, chunk, 43)

        # Truncate or pad to action_chunk_length
        chunk_len = sim_actions.shape[1]
        if chunk_len >= self.action_chunk_length:
            sim_actions = sim_actions[:, : self.action_chunk_length, :]
        else:
            # Pad with last action repeated
            repeat_count = self.action_chunk_length - chunk_len
            last_action = sim_actions[:, -1:, :].expand(-1, repeat_count, -1)
            sim_actions = torch.cat([sim_actions, last_action], dim=1)

        return sim_actions.float()

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset action chunking state for specified envs."""
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True
        # Reset ACT's internal action queue
        if hasattr(self.policy, "reset"):
            self.policy.reset()
