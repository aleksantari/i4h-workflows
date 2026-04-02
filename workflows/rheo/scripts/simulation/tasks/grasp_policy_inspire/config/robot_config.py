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

"""Robot configuration for the G1 + Inspire FTP grasp_policy_inspire task.

Uses the pre-tested nucleus USD with L_/R_ hand joint naming.
Actuator parameters for the hands follow the IsaacLab reference config.
"""

from typing import Dict, Optional, Tuple

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

# Local USD converted from our URDF (includes d435_link camera mount).
# Generate with: python scripts/utils/convert_inspire_ftp_urdf_to_usd.py
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "assets")
UNITREE_G1_29DOF_INSPIRE_FTP_USD = os.path.abspath(
    os.path.join(_ASSETS_DIR, "robots", "g1-29dof-inspire-ftp-usd", "g1_29dof_inspire_ftp.usd")
)

# Default joint positions (body same as Dex3 task, hands at zero).
DEFAULT_JOINT_POS: Dict[str, float] = {
    # legs
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.3,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.3,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    # Inspire FTP hands — all zero
    ".*_index_.*": 0.0,
    ".*_middle_.*": 0.0,
    ".*_ring_.*": 0.0,
    ".*_little_.*": 0.0,
    ".*_thumb_.*": 0.0,
}


G129_CFG_WITH_INSPIRE_FTP_BASE_FIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UNITREE_G1_29DOF_INSPIRE_FTP_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            effort_limit=50.0,
            velocity_limit=37.0,
            armature=0.03,
            friction=0.03,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=10000.0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_.*_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 15.0,
                ".*_shoulder_roll_joint": 15.0,
                ".*_shoulder_yaw_joint": 8.0,
                ".*_elbow_joint": 8.0,
                ".*_wrist_.*_joint": 4.0,
            },
            armature={".*_shoulder_.*": 0.03, ".*_elbow_.*": 0.03, ".*_wrist_.*_joint": 0.03},
            friction=0.03,
        ),
        # Inspire FTP hands — all 24 joints (actuated + mimic) use implicit actuator.
        # Mimic enforcement happens in the custom action class, not the actuator.
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_.*",
                ".*_middle_.*",
                ".*_thumb_.*",
                ".*_ring_.*",
                ".*_little_.*",
            ],
            effort_limit=30.0,
            velocity_limit=10.0,
            stiffness=10.0,
            damping=0.2,
            armature=0.001,
        ),
    },
)


def make_g1_29dof_inspire_ftp_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.744),
    init_rot: Tuple[float, float, float, float] = (0.7071, 0, 0, 0.7071),
    custom_joint_pos: Optional[Dict[str, float]] = None,
    base_config: ArticulationCfg = G129_CFG_WITH_INSPIRE_FTP_BASE_FIX,
) -> ArticulationCfg:
    """Create the Inspire FTP robot articulation cfg."""
    joint_pos = DEFAULT_JOINT_POS.copy()
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return base_config.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class G1InspireRobotPresets:
    """G1 + Inspire FTP robot preset configuration collection."""

    @classmethod
    def g1_29dof_inspire_ftp_base_fix(
        cls,
        init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.76),
        init_rot: Tuple[float, float, float, float] = (0.7071, 0, 0, 0.7071),
    ) -> ArticulationCfg:
        """Tabletop manipulation configuration — fixed base, Inspire FTP hands."""
        return make_g1_29dof_inspire_ftp_cfg(init_pos=init_pos, init_rot=init_rot)
