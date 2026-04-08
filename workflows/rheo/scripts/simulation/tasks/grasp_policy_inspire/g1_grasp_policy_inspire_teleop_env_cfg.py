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
UnitreeG1Retargeter for full 5-finger DexPilot IK retargeting via AVP.

Key differences from the Dex3 teleop variant:
- PinkIK (38D) instead of WBC+PINK (23D)
- Full dex-retargeting for all 5 fingers (DexPilot IK)
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
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import (
    UnitreeG1RetargeterCfg,
)
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

# The UnitreeG1DexRetargeting retargeter loads Nucleus hand-only URDFs whose joints use
# L_/R_ prefixes and anatomical suffixes ("pinky" instead of "little", etc.).
# We must pass Nucleus-style names in the same positional order as HAND_JOINT_NAMES
# so the retargeted values end up in the correct slots for PinkIK.
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

RETARGETER_HAND_JOINT_NAMES: list[str] = [_URDF_TO_NUCLEUS[n] for n in HAND_JOINT_NAMES]


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
                    gain=0.8,
                ),
                FrameTask(
                    "g1_29dof_rev_1_0_with_inspire_hand_FTP_right_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10,
                    gain=0.8,
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

        # Register AVP hand tracking with full 5-finger DexPilot IK retargeting.
        # Uses Nucleus-style joint names (RETARGETER_HAND_JOINT_NAMES) so the
        # dex-retargeting output aligns positionally with PinkIK's hand joint order.
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=RETARGETER_HAND_JOINT_NAMES,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            },
        )
