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

from dataclasses import dataclass

import numpy as np
import torch
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import G1LowerBodyStandingMotionControllerRetargeterCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode, XrCfg
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

import isaaclab.utils.math as PoseUtils

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


class G1HandtrackingGripperRetargeter(RetargeterBase):
    """Retargeter for G1 that converts AVP hand tracking to binary gripper + wrist poses.

    Gripper: Uses thumb-index pinch distance with hysteresis (same output as motion controller trigger).
    Wrist: Retargets absolute pose from OpenXR hand tracking to robot frame.

    Output format matches G1TriHandUpperBodyMotionControllerGripperRetargeter:
        [left_gripper(1), right_gripper(1), left_wrist(7), right_wrist(7)] = 16D
    """

    def __init__(self, cfg: "G1HandtrackingGripperRetargeterCfg"):
        super().__init__(cfg)
        self._cfg = cfg
        self._prev_left_state: float = 0.0
        self._prev_right_state: float = 0.0

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand tracking data to gripper state + wrist poses.

        Args:
            data: Dictionary with TrackingTarget.HAND_LEFT/RIGHT keys.
                  Each value is a dict mapping joint name -> 7D pose [x,y,z,qw,qx,qy,qz].

        Returns:
            Tensor: [left_gripper(1), right_gripper(1), left_wrist(7), right_wrist(7)] = 16D
        """
        left_hand_poses = data.get(DeviceBase.TrackingTarget.HAND_LEFT, {})
        right_hand_poses = data.get(DeviceBase.TrackingTarget.HAND_RIGHT, {})

        # --- Gripper: pinch distance with hysteresis ---
        left_gripper = self._compute_pinch_gripper(left_hand_poses, self._prev_left_state)
        right_gripper = self._compute_pinch_gripper(right_hand_poses, self._prev_right_state)
        self._prev_left_state = left_gripper
        self._prev_right_state = right_gripper

        gripper_tensor = torch.tensor([left_gripper, right_gripper], dtype=torch.float32, device=self._sim_device)

        # --- Wrist: retarget absolute pose ---
        default_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        left_wrist = left_hand_poses.get("wrist", default_wrist)
        right_wrist = right_hand_poses.get("wrist", default_wrist)

        left_wrist_tensor = torch.tensor(
            self._retarget_abs(left_wrist, is_left=True), dtype=torch.float32, device=self._sim_device
        )
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, is_left=False), dtype=torch.float32, device=self._sim_device
        )

        return torch.cat([gripper_tensor, left_wrist_tensor, right_wrist_tensor])

    def _compute_pinch_gripper(self, hand_poses: dict, prev_state: float) -> float:
        """Compute binary gripper state from thumb-index pinch distance with hysteresis.

        Args:
            hand_poses: Dict mapping joint name -> 7D pose.
            prev_state: Previous gripper state (0.0=open, 1.0=closed).

        Returns:
            Gripper state: 0.0 (open) or 1.0 (closed).
        """
        thumb_tip = hand_poses.get("thumb_tip")
        index_tip = hand_poses.get("index_tip")

        if thumb_tip is None or index_tip is None:
            return 0.0

        distance = np.linalg.norm(thumb_tip[:3] - index_tip[:3])

        if prev_state < 0.5:  # Currently open
            return 1.0 if distance < self._cfg.pinch_close_distance else 0.0
        else:  # Currently closed
            return 0.0 if distance > self._cfg.pinch_open_distance else 1.0

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        """Retarget absolute wrist pose from OpenXR hand tracking to robot frame.

        Uses hand-tracking-specific rotation offsets (different from motion controller offsets).
        Left hand:  (0, 90, 90) euler -> [0.7071, 0, 0.7071, 0]
        Right hand: (0, -90, -90) euler -> [0, -0.7071, 0, 0.7071]
        """
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        if is_left:
            combined_quat = torch.tensor([0.7071, 0, 0.7071, 0], dtype=torch.float32)
        else:
            combined_quat = torch.tensor([0, -0.7071, 0, 0.7071], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

    def reset(self):
        self._prev_left_state = 0.0
        self._prev_right_state = 0.0


@dataclass
class G1HandtrackingGripperRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 hand tracking gripper retargeter."""

    pinch_close_distance: float = 0.03  # meters — pinch closer than this = close gripper
    pinch_open_distance: float = 0.05  # meters — pinch wider than this = open gripper
    retargeter_type: type[RetargeterBase] = G1HandtrackingGripperRetargeter


class TrocarG1HandtrackingGripperRetargeter(G1HandtrackingGripperRetargeter):
    """Trocar-specific variant that applies robot-origin offset to wrist positions."""

    _ROBOT_ORIGIN_W = np.array([-1.84919, 1.94, 0.81168], dtype=np.float32)

    def retarget(self, data: dict) -> torch.Tensor:
        action = super().retarget(data).clone()
        action[2:5] = action.new_tensor(action[2:5].detach().cpu().numpy() - self._ROBOT_ORIGIN_W)
        action[9:12] = action.new_tensor(action[9:12].detach().cpu().numpy() - self._ROBOT_ORIGIN_W)
        return action

    def reset(self):
        self._prev_left_state = 0.0
        self._prev_right_state = 0.0


@dataclass
class TrocarG1HandtrackingGripperRetargeterCfg(G1HandtrackingGripperRetargeterCfg):
    retargeter_type: type[RetargeterBase] = TrocarG1HandtrackingGripperRetargeter


@register_device
class HandtrackingTeleopDevice(TeleopDeviceBase):
    """Teleop device for AVP hand tracking (OpenXR hand joints → binary gripper + wrist poses)."""

    name = "handtracking"

    def __init__(self, sim_device: str | None = None):
        super().__init__(sim_device=sim_device)

    def get_teleop_device_cfg(
        self,
        embodiment: object | None = None,
        xr_cfg: XrCfg | None = None,
        use_trocar_retargeter: bool = False,
    ) -> DevicesCfg:
        """Build the teleop device configuration (hand tracking gripper + lower body standing)."""

        if xr_cfg is None:
            if embodiment is not None and hasattr(embodiment, "get_xr_cfg"):
                xr_cfg = embodiment.get_xr_cfg()
            else:
                xr_cfg = XrCfg()

        xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        if use_trocar_retargeter:
            retargeters = [
                TrocarG1HandtrackingGripperRetargeterCfg(
                    sim_device=self.sim_device,
                ),
                G1LowerBodyStandingMotionControllerRetargeterCfg(
                    sim_device=self.sim_device,
                ),
            ]
        else:
            retargeters = [
                G1HandtrackingGripperRetargeterCfg(
                    sim_device=self.sim_device,
                ),
                G1LowerBodyStandingMotionControllerRetargeterCfg(
                    sim_device=self.sim_device,
                ),
            ]

        return DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=retargeters,
                    sim_device=self.sim_device,
                    xr_cfg=xr_cfg,
                ),
            }
        )
