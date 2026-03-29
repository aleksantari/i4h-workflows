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
Model-agnostic observation processor for IsaacLab environments.

Extracts camera images and joint states from the raw IsaacLab observation dict
into a standardized format that can be consumed by any policy wrapper.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ProcessedObservation:
    """Standardized observation format produced by ObsProcessor.

    Attributes:
        cameras: Dict mapping camera name to (B, H, W, C) uint8 tensors.
        arm_joints: (B, 14) tensor — left_arm(7) + right_arm(7) joint positions.
        hand_joints: (B, 14) tensor — left_hand(7) + right_hand(7) joint positions.
        state_28d: (B, 28) tensor — concatenation of arm_joints + hand_joints.
        language: Task description string.
    """

    cameras: dict[str, torch.Tensor] = field(default_factory=dict)
    arm_joints: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    hand_joints: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    state_28d: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    language: str = ""


class ObsProcessor:
    """Extracts observations from IsaacLab env output in a model-agnostic way.

    This processor handles the common pattern of extracting:
    - 3 camera views (front, left_wrist, right_wrist) from camera_images
    - 28D joint state (arms + hands) from policy observations
    - Language instruction for the task

    Usage:
        processor = ObsProcessor(language="pick up block and place in bin")
        processed = processor.process(raw_obs)
        # processed.cameras["front_camera"] -> (B, H, W, C) tensor
        # processed.state_28d -> (B, 28) tensor
    """

    # Indices into the 87D robot_joint_state for arm joints
    ARM_JOINT_SLICE = slice(15, 29)  # 14 joints: left_arm(7) + right_arm(7)

    # Default camera key mapping (IsaacLab key -> standard name)
    DEFAULT_CAMERA_KEYS = ["front_camera", "left_wrist_camera", "right_wrist_camera"]

    def __init__(self, language: str = "", camera_keys: list[str] | None = None):
        self.language = language
        self.camera_keys = camera_keys or self.DEFAULT_CAMERA_KEYS

    def process(self, observation: dict[str, Any]) -> ProcessedObservation:
        """Extract standardized observations from IsaacLab env output.

        Args:
            observation: Raw observation dict from env.step() with keys:
                - "policy": {"robot_joint_state": (B, 87), "robot_dex3_joint_state": (B, 14)}
                - "camera_images": {"front_camera": (B,H,W,C), ...}

        Returns:
            ProcessedObservation with extracted tensors.
        """
        result = ProcessedObservation(language=self.language)

        # Extract camera images
        if "camera_images" in observation:
            for key in self.camera_keys:
                if key in observation["camera_images"]:
                    result.cameras[key] = observation["camera_images"][key]

        # Extract joint states
        if "policy" in observation:
            policy = observation["policy"]

            if "robot_joint_state" in policy:
                body_state = policy["robot_joint_state"]  # (B, 87)
                result.arm_joints = body_state[:, self.ARM_JOINT_SLICE]  # (B, 14)

            if "robot_dex3_joint_state" in policy:
                result.hand_joints = policy["robot_dex3_joint_state"]  # (B, 14)

            # Concatenate into 28D state
            if result.arm_joints.numel() > 0 and result.hand_joints.numel() > 0:
                result.state_28d = torch.cat([result.arm_joints, result.hand_joints], dim=-1)

        return result
