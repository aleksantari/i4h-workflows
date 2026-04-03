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

"""Binary gripper retargeter for Inspire FTP teleop.

Replaces per-finger dex-retargeting with a simple binary open/close gripper,
analogous to how the Dex3 teleop works. All actuated finger joints close to a
uniform angle when the user pinches, and mimic joints follow via URDF multipliers.

This keeps IL (imitation learning) simple — the policy learns binary grip patterns.
RL post-training on the full Inspire env can then unlock per-finger control.

Output: 38D [left_wrist(7), right_wrist(7), hand_joints(24)] — same format
as UnitreeG1Retargeter, so PinkIK config stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import (
    UnitreeG1Retargeter,
)

# ---------------------------------------------------------------------------
# Mimic rules (from mimic_action.py — URDF-style joint names)
# ---------------------------------------------------------------------------
# Each tuple: (mimic_joint, parent_joint, multiplier)
# Order matters: thumb_3 depends on thumb_2, thumb_4 depends on thumb_3.
_MIMIC_RULES_PER_SIDE: list[tuple[str, str, float]] = [
    ("{side}_index_2_joint", "{side}_index_1_joint", 1.0843),
    ("{side}_middle_2_joint", "{side}_middle_1_joint", 1.0843),
    ("{side}_ring_2_joint", "{side}_ring_1_joint", 1.0843),
    ("{side}_little_2_joint", "{side}_little_1_joint", 1.0843),
    ("{side}_thumb_3_joint", "{side}_thumb_2_joint", 0.8024),
    ("{side}_thumb_4_joint", "{side}_thumb_3_joint", 0.9487),
]

MIMIC_RULES: list[tuple[str, str, float]] = []
for _side in ("left", "right"):
    for _mimic_tmpl, _parent_tmpl, _mult in _MIMIC_RULES_PER_SIDE:
        MIMIC_RULES.append((_mimic_tmpl.format(side=_side), _parent_tmpl.format(side=_side), _mult))

# Actuated joints per hand (URDF-style names, same order as in joint_names).
# 6 per hand: 4 finger proximal + thumb_1 (yaw) + thumb_2 (pitch).
_LEFT_ACTUATED = [
    "left_index_1_joint",
    "left_little_1_joint",
    "left_middle_1_joint",
    "left_ring_1_joint",
    "left_thumb_1_joint",
    "left_thumb_2_joint",
]
_RIGHT_ACTUATED = [
    "right_index_1_joint",
    "right_little_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
]


def _compute_closed_joints(hand_joint_names: list[str], closed_angle: float) -> np.ndarray:
    """Pre-compute the 24D closed hand joint array.

    Sets all actuated joints to ``closed_angle``, then applies mimic rules
    sequentially (order matters for the thumb chain).
    """
    name_to_idx = {n: i for i, n in enumerate(hand_joint_names)}
    joints = np.zeros(len(hand_joint_names), dtype=np.float64)

    # Set actuated joints
    for name in _LEFT_ACTUATED + _RIGHT_ACTUATED:
        if name in name_to_idx:
            joints[name_to_idx[name]] = closed_angle

    # Apply mimic rules (sequential — thumb_4 depends on thumb_3 which depends on thumb_2)
    for mimic_name, parent_name, mult in MIMIC_RULES:
        mimic_idx = name_to_idx.get(mimic_name)
        parent_idx = name_to_idx.get(parent_name)
        if mimic_idx is not None and parent_idx is not None:
            joints[mimic_idx] = joints[parent_idx] * mult

    return joints


class InspireGripperRetargeter(UnitreeG1Retargeter):
    """Binary gripper retargeter for Inspire FTP hands.

    Inherits wrist retargeting (``_retarget_abs``) from ``UnitreeG1Retargeter``
    but replaces per-finger dex-retargeting with a binary pinch-based gripper.

    Pinch detection uses hysteresis (same thresholds as Dex3):
    - Close when thumb-index distance < ``pinch_close_distance``
    - Open when distance > ``pinch_open_distance``
    """

    def __init__(self, cfg: InspireGripperRetargeterCfg):
        # Skip UnitreeG1Retargeter.__init__ to avoid UnitreeG1DexRetargeting import.
        # We only need wrist retargeting (_retarget_abs, inherited) + visualization.
        RetargeterBase.__init__(self, cfg)

        self._hand_joint_names = cfg.hand_joint_names
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device

        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

        # Gripper state (hysteresis)
        self._prev_left_grip = 0.0
        self._prev_right_grip = 0.0
        self._pinch_close_distance = cfg.pinch_close_distance
        self._pinch_open_distance = cfg.pinch_open_distance

        # Pre-compute open/closed 24D joint arrays
        self._open_joints = np.zeros(len(self._hand_joint_names), dtype=np.float64)
        self._closed_joints = _compute_closed_joints(self._hand_joint_names, cfg.gripper_closed_angle)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand tracking data to 38D robot command with binary gripper.

        Returns:
            38D tensor: [left_wrist(7), right_wrist(7), hand_joints(24)]
        """
        left_hand_poses = data[DeviceBase.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[DeviceBase.TrackingTarget.HAND_RIGHT]

        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        # Visualization (same as parent)
        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        # Wrist retargeting (inherited from UnitreeG1Retargeter)
        left_wrist_tensor = torch.tensor(
            self._retarget_abs(left_wrist, True), dtype=torch.float32, device=self._sim_device
        )
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, False), dtype=torch.float32, device=self._sim_device
        )

        # Binary gripper from pinch distance
        left_grip = self._pinch_gripper(left_hand_poses, self._prev_left_grip)
        self._prev_left_grip = left_grip

        right_grip = self._pinch_gripper(right_hand_poses, self._prev_right_grip)
        self._prev_right_grip = right_grip

        # Expand gripper state → 24D hand joints
        hand_joints = self._open_joints.copy()
        if left_grip > 0.5:
            # Only overwrite left-hand indices (first 12 joints in USD order: indices 0-4 actuated, 10-14 mimic, etc.)
            for i, name in enumerate(self._hand_joint_names):
                if name.startswith("left_"):
                    hand_joints[i] = self._closed_joints[i]
        if right_grip > 0.5:
            for i, name in enumerate(self._hand_joint_names):
                if name.startswith("right_"):
                    hand_joints[i] = self._closed_joints[i]

        hand_joints_tensor = torch.tensor(hand_joints, dtype=torch.float32, device=self._sim_device)

        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def _pinch_gripper(self, hand_poses: dict, prev_state: float) -> float:
        """Binary gripper from thumb-index pinch distance with hysteresis.

        Same logic as the Dex3 gripper in handtracking.py:
        - If currently open (prev < 0.5): close when distance < close_threshold
        - If currently closed (prev >= 0.5): open when distance > open_threshold
        """
        thumb_tip = hand_poses.get("thumb_tip")
        index_tip = hand_poses.get("index_tip")
        if thumb_tip is None or index_tip is None:
            return prev_state

        distance = np.linalg.norm(np.array(thumb_tip[:3]) - np.array(index_tip[:3]))

        if prev_state < 0.5:
            return 1.0 if distance < self._pinch_close_distance else 0.0
        else:
            return 0.0 if distance > self._pinch_open_distance else 1.0


@dataclass
class InspireGripperRetargeterCfg(RetargeterCfg):
    """Configuration for the Inspire FTP binary gripper retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 2 * 26
    hand_joint_names: list[str] | None = None
    pinch_close_distance: float = 0.03  # meters — same as Dex3
    pinch_open_distance: float = 0.05  # meters — same as Dex3
    gripper_closed_angle: float = 1.0  # radians — uniform for all actuated joints
    retargeter_type: type[RetargeterBase] = InspireGripperRetargeter
