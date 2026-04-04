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

"""Teleop environment configuration for the Inspire FTP grasp-policy task.

Inherits the RL env config and overrides actions to use PinkIK (38D),
extends episode length for human teleoperation, and registers the
InspireGripperRetargeter for AVP binary gripper control.

Key differences from the Dex3 teleop variant:
- PinkIK (38D) instead of WBC+PINK (23D)
- Binary gripper (pinch-based open/close, uniform angle for all fingers)
- No WBC state reset needed (PinkIK is stateless per-step)

Action format (38D):
  [left_wrist_pos(3) | left_wrist_quat(4) | right_wrist_pos(3) | right_wrist_quat(4) | hand_joints(24)]
"""

import tempfile

import torch
from pink.tasks import FrameTask

import carb

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from teleop_devices.inspire_gripper_retargeter import InspireGripperRetargeterCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode, XrCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from simulation.tasks.grasp_policy_inspire.g1_grasp_policy_inspire_env_cfg import (
    G1GraspPolicyInspireEnvCfg,
    joint_names,  # Full 53-joint list (NOT actuated_joint_names) — PinkIK needs all 24 hand joints.
)

# All 24 hand joints in USD articulation order (indices 29-52 of the full 53-joint list).
# This order MUST match robot.joint_names[-24:] for the retargeter output
# to be correctly interpreted by the PinkIK action.
HAND_JOINT_NAMES: list[str] = joint_names[29:]


@configclass
class TeleopActionsCfg:
    """38D PinkIK action: arm IK (14D wrist poses) + direct hand joint targets (24D)."""

    pink_ik_cfg: ActionTermCfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        hand_joint_names=HAND_JOINT_NAMES,
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=24,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_rev_1_0_with_inspire_hand_FTP_left_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_rev_1_0_with_inspire_hand_FTP_right_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_rev_1_0_with_inspire_hand_FTP_left_wrist_yaw_link",
                        "g1_29dof_rev_1_0_with_inspire_hand_FTP_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
        enable_gravity_compensation=False,
    )


@configclass
class G1GraspPolicyInspireTeleopEnvCfg(G1GraspPolicyInspireEnvCfg):
    """Teleop variant of the Inspire FTP grasp-policy task.

    Overrides:
    - Actions: PinkIK (38D) instead of direct joint control (53D)
    - Episode length: 300s (humans need time)
    - Render interval: 2 (smoother XR visuals)
    - XR config: anchor on robot pelvis
    - Teleop devices: AVP hand tracking with dex-retargeting
    """

    actions: TeleopActionsCfg = TeleopActionsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
    )

    temp_urdf_dir: str = tempfile.gettempdir()

    # Idle action to hold robot in default pose (38D).
    # Format: [left_wrist_pos(3), left_wrist_quat(4),
    #          right_wrist_pos(3), right_wrist_quat(4),
    #          hand_joints(24)]
    # NOTE: Wrist positions are approximate for our robot at
    # (-1.84919, 1.94, 0.81168) with rot (1,0,0,0). May need
    # empirical tuning at first run via FK at default joint pose.
    idle_action: torch.Tensor = torch.tensor(
        [
            # Left wrist pose (7D) — approximate world-frame FK
            -1.9979, 2.1438, 1.0952,  # position (x, y, z)
            0.707, 0.0, 0.0, 0.707,   # quaternion (w, x, y, z)
            # Right wrist pose (7D) — approximate world-frame FK
            -1.7005, 2.1438, 1.0952,  # position (x, y, z)
            0.707, 0.0, 0.0, 0.707,   # quaternion (w, x, y, z)
            # Hand joints (24D) — all zero (open)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = 2
        self.episode_length_s = 300.0

        # Convert USD to URDF for PinkIK solver
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        # XR anchor follows robot pelvis
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        # Register AVP hand tracking with binary gripper retargeting.
        # Uses HAND_JOINT_NAMES directly (URDF-style names matching joint_names[29:]).
        # No need for RETARGETER_HAND_JOINT_NAMES — the Nucleus-to-URDF bridge was
        # only required by UnitreeG1DexRetargeting, which we no longer use.
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        InspireGripperRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=HAND_JOINT_NAMES,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            },
        )
