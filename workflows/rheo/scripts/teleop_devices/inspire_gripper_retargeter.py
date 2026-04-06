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

"""Hybrid gripper retargeter for Inspire FTP teleop.

Uses **dex-retargeting** (DexPilot IK) for the thumb and a **binary gripper**
for the other four fingers. This gives the thumb natural opposition from the
operator's hand pose while keeping finger control simple for IL.

When ``use_dex_thumb=False``, falls back to fixed per-joint thumb angles
(Option A) — useful if the dex-retargeting URDF assets are unavailable.

Output: 38D [left_wrist(7), right_wrist(7), hand_joints(24)] — same format
as UnitreeG1Retargeter, so PinkIK config stays unchanged.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URDF → Nucleus joint name mapping (needed for dex-retargeting)
# ---------------------------------------------------------------------------
# The dex-retargeting library loads hand-only Nucleus URDFs whose joints use
# L_/R_ prefixes and anatomical suffixes. Our env uses URDF-style names.
# This mapping converts positionally so the retargeter output aligns with
# our 24D hand joint array.
_URDF_TO_NUCLEUS: dict[str, str] = {
    # Left hand
    "left_index_1_joint": "L_index_proximal_joint",
    "left_index_2_joint": "L_index_intermediate_joint",
    "left_little_1_joint": "L_pinky_proximal_joint",
    "left_little_2_joint": "L_pinky_intermediate_joint",
    "left_middle_1_joint": "L_middle_proximal_joint",
    "left_middle_2_joint": "L_middle_intermediate_joint",
    "left_ring_1_joint": "L_ring_proximal_joint",
    "left_ring_2_joint": "L_ring_intermediate_joint",
    "left_thumb_1_joint": "L_thumb_proximal_yaw_joint",
    "left_thumb_2_joint": "L_thumb_proximal_pitch_joint",
    "left_thumb_3_joint": "L_thumb_intermediate_joint",
    "left_thumb_4_joint": "L_thumb_distal_joint",
    # Right hand
    "right_index_1_joint": "R_index_proximal_joint",
    "right_index_2_joint": "R_index_intermediate_joint",
    "right_little_1_joint": "R_pinky_proximal_joint",
    "right_little_2_joint": "R_pinky_intermediate_joint",
    "right_middle_1_joint": "R_middle_proximal_joint",
    "right_middle_2_joint": "R_middle_intermediate_joint",
    "right_ring_1_joint": "R_ring_proximal_joint",
    "right_ring_2_joint": "R_ring_intermediate_joint",
    "right_thumb_1_joint": "R_thumb_proximal_yaw_joint",
    "right_thumb_2_joint": "R_thumb_proximal_pitch_joint",
    "right_thumb_3_joint": "R_thumb_intermediate_joint",
    "right_thumb_4_joint": "R_thumb_distal_joint",
}

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

# Thumb joints (URDF-style) for identifying which joints get dex-retargeted
_THUMB_YAW_JOINTS = {"left_thumb_1_joint", "right_thumb_1_joint"}
_THUMB_PITCH_JOINTS = {"left_thumb_2_joint", "right_thumb_2_joint"}
_THUMB_ACTUATED_JOINTS = _THUMB_YAW_JOINTS | _THUMB_PITCH_JOINTS


def _compute_closed_joints(
    hand_joint_names: list[str],
    closed_angle: float,
    thumb_yaw_closed_angle: float | None = None,
    thumb_pitch_closed_angle: float | None = None,
) -> np.ndarray:
    """Pre-compute the 24D closed hand joint array.

    Sets finger proximal joints to ``closed_angle``. Thumb joints use separate
    angles to prevent the thumb from tucking under the other fingers:
    - ``thumb_1`` (yaw): controls thumb rotation toward/away from palm.
      A low value (~0.2 rad) keeps the thumb in opposition.
    - ``thumb_2`` (pitch): controls thumb curl. URDF limit is 0.5864 rad.

    Mimic rules are applied sequentially (order matters for thumb chain).
    """
    name_to_idx = {n: i for i, n in enumerate(hand_joint_names)}
    joints = np.zeros(len(hand_joint_names), dtype=np.float64)

    # Set actuated joints with per-joint thumb overrides
    for name in _LEFT_ACTUATED + _RIGHT_ACTUATED:
        if name not in name_to_idx:
            continue
        if name in _THUMB_YAW_JOINTS and thumb_yaw_closed_angle is not None:
            joints[name_to_idx[name]] = thumb_yaw_closed_angle
        elif name in _THUMB_PITCH_JOINTS and thumb_pitch_closed_angle is not None:
            joints[name_to_idx[name]] = thumb_pitch_closed_angle
        else:
            joints[name_to_idx[name]] = closed_angle

    # Apply mimic rules (sequential — thumb_4 depends on thumb_3 which depends on thumb_2)
    for mimic_name, parent_name, mult in MIMIC_RULES:
        mimic_idx = name_to_idx.get(mimic_name)
        parent_idx = name_to_idx.get(parent_name)
        if mimic_idx is not None and parent_idx is not None:
            joints[mimic_idx] = joints[parent_idx] * mult

    return joints


def _apply_mimic_rules(joints: np.ndarray, name_to_idx: dict[str, int]) -> None:
    """Apply mimic rules in-place to a 24D joint array."""
    for mimic_name, parent_name, mult in MIMIC_RULES:
        mimic_idx = name_to_idx.get(mimic_name)
        parent_idx = name_to_idx.get(parent_name)
        if mimic_idx is not None and parent_idx is not None:
            joints[mimic_idx] = joints[parent_idx] * mult


class InspireGripperRetargeter(UnitreeG1Retargeter):
    """Hybrid gripper retargeter for Inspire FTP hands.

    Inherits wrist retargeting (``_retarget_abs``) from ``UnitreeG1Retargeter``.

    **Thumb:** Uses DexPilot IK retargeting from the operator's hand pose,
    giving natural opposition and curl that follows the human thumb.

    **Other 4 fingers:** Binary pinch-based gripper (same as before).
    Close when thumb-index distance < ``pinch_close_distance``,
    open when distance > ``pinch_open_distance``.

    When ``use_dex_thumb=False``, falls back to fixed per-joint thumb angles
    (no dex-retargeting dependency).
    """

    def __init__(self, cfg: InspireGripperRetargeterCfg):
        # Skip UnitreeG1Retargeter.__init__ to avoid its full dex-retargeting init.
        # We only need wrist retargeting (_retarget_abs, inherited) + visualization.
        RetargeterBase.__init__(self, cfg)

        self._hand_joint_names = cfg.hand_joint_names
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device
        self._use_dex_thumb = cfg.use_dex_thumb

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

        # Build name→index map for the 24D hand joint array
        self._name_to_idx = {n: i for i, n in enumerate(self._hand_joint_names)}

        # Gripper state (hysteresis) — used for the 4 non-thumb fingers
        self._prev_left_grip = 0.0
        self._prev_right_grip = 0.0
        self._pinch_close_distance = cfg.pinch_close_distance
        self._pinch_open_distance = cfg.pinch_open_distance

        # Pre-compute open/closed 24D joint arrays (used for 4-finger binary + fallback thumb)
        self._open_joints = np.zeros(len(self._hand_joint_names), dtype=np.float64)
        self._closed_joints = _compute_closed_joints(
            self._hand_joint_names,
            cfg.gripper_closed_angle,
            thumb_yaw_closed_angle=cfg.thumb_yaw_closed_angle,
            thumb_pitch_closed_angle=cfg.thumb_pitch_closed_angle,
        )

        # Initialize dex-retargeting for thumb IK
        if self._use_dex_thumb:
            from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_dex_retargeting_utils import (
                UnitreeG1DexRetargeting,
            )

            _nucleus_names = [_URDF_TO_NUCLEUS[n] for n in self._hand_joint_names]
            self._dex_retarget = UnitreeG1DexRetargeting(_nucleus_names)
            logger.info("[InspireGripperRetargeter] Dex-retargeting enabled for thumb")

            # Build index mapping: dex output (6D Nucleus-style) → 24D URDF-style array
            # Dex output order per hand: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
            # We only use indices 0 (yaw) and 1 (pitch) for the thumb.
            self._left_thumb_dex_indices = [0, 1]  # in dex output
            self._right_thumb_dex_indices = [0, 1]

            # URDF-style thumb joint indices in the 24D hand array
            self._left_thumb_yaw_idx = self._name_to_idx["left_thumb_1_joint"]
            self._left_thumb_pitch_idx = self._name_to_idx["left_thumb_2_joint"]
            self._right_thumb_yaw_idx = self._name_to_idx["right_thumb_1_joint"]
            self._right_thumb_pitch_idx = self._name_to_idx["right_thumb_2_joint"]

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand tracking data to 38D robot command.

        Thumb: dex-retargeted from hand pose (or fixed angles if use_dex_thumb=False).
        Other fingers: binary gripper from pinch distance.

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

        # Binary gripper for 4 non-thumb fingers
        left_grip = self._pinch_gripper(left_hand_poses, self._prev_left_grip)
        self._prev_left_grip = left_grip

        right_grip = self._pinch_gripper(right_hand_poses, self._prev_right_grip)
        self._prev_right_grip = right_grip

        # Build 24D hand joint array
        hand_joints = self._open_joints.copy()

        # --- Set 4-finger binary gripper (exclude thumb joints) ---
        if left_grip > 0.5:
            for i, name in enumerate(self._hand_joint_names):
                if name.startswith("left_") and name not in _THUMB_ACTUATED_JOINTS:
                    hand_joints[i] = self._closed_joints[i]
        if right_grip > 0.5:
            for i, name in enumerate(self._hand_joint_names):
                if name.startswith("right_") and name not in _THUMB_ACTUATED_JOINTS:
                    hand_joints[i] = self._closed_joints[i]

        # --- Set thumb joints ---
        if self._use_dex_thumb:
            # Dex-retarget: compute per-finger angles from hand pose, extract thumb only
            left_dex = self._dex_retarget.compute_left(left_hand_poses)   # 6D
            right_dex = self._dex_retarget.compute_right(right_hand_poses)  # 6D

            hand_joints[self._left_thumb_yaw_idx] = left_dex[0]    # thumb yaw
            hand_joints[self._left_thumb_pitch_idx] = left_dex[1]  # thumb pitch
            hand_joints[self._right_thumb_yaw_idx] = right_dex[0]
            hand_joints[self._right_thumb_pitch_idx] = right_dex[1]
        else:
            # Fallback: use fixed per-joint thumb angles (Option A)
            if left_grip > 0.5:
                for name in ("left_thumb_1_joint", "left_thumb_2_joint"):
                    hand_joints[self._name_to_idx[name]] = self._closed_joints[self._name_to_idx[name]]
            if right_grip > 0.5:
                for name in ("right_thumb_1_joint", "right_thumb_2_joint"):
                    hand_joints[self._name_to_idx[name]] = self._closed_joints[self._name_to_idx[name]]

        # Apply mimic rules for all joints (thumb mimic chain + finger _2 joints)
        _apply_mimic_rules(hand_joints, self._name_to_idx)

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
    """Configuration for the Inspire FTP hybrid gripper retargeter."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 2 * 26
    hand_joint_names: list[str] | None = None
    pinch_close_distance: float = 0.03  # meters — same as Dex3
    pinch_open_distance: float = 0.05  # meters — same as Dex3
    gripper_closed_angle: float = 1.0  # radians — for index/middle/ring/little proximal joints
    thumb_yaw_closed_angle: float = 0.2  # radians — thumb_1 fallback (used when use_dex_thumb=False)
    thumb_pitch_closed_angle: float = 0.55  # radians — thumb_2 fallback (URDF limit: 0.5864)
    use_dex_thumb: bool = True  # Use DexPilot IK for thumb (False = fixed angles)
    retargeter_type: type[RetargeterBase] = InspireGripperRetargeter
