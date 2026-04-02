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

"""Environment configuration for the Inspire FTP grasp-policy pick-and-place task.

Same block pick-and-place task as the Dex3 variant, but using the G1 robot
with Inspire FTP 5-finger hands.

Key differences from the Dex3 variant:
- 53D action space (29 body + 24 hand) with mimic enforcement
- 12D hand observation (6 actuated per hand)
- Front camera only (no wrist cameras)
- Policy dim: 26D (14 arm + 12 actuated hand)
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from simulation.assets.assets import TROCAR_ASSEMBLY_SCENE_USD
from simulation.tasks.grasp_policy_inspire import mdp

from simulation.tasks.grasp_policy_inspire.config import CameraPresets, G1InspireRobotPresets  # isort: skip

# ---------------------------------------------------------------------------
# 53-joint name list: 29 body + 24 Inspire FTP hand joints.
# NOTE: The exact ordering MUST match the USD articulation ordering.
# Run scripts/utils/inspect_inspire_ftp_joints.py to verify/update.
# The body joints (indices 0-28) are the same as the Dex3 task.
# Hand joints (indices 29-52) follow the USD's tree traversal order.
# ---------------------------------------------------------------------------
joint_names = [
    # --- Body (29): USD articulation tree traversal order ---
    # Legs, waist, and arms are interleaved left/right in the USD.
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8
    "left_knee_joint",            # 9
    "right_knee_joint",           # 10
    "left_shoulder_pitch_joint",  # 11
    "right_shoulder_pitch_joint", # 12
    "left_ankle_pitch_joint",     # 13
    "right_ankle_pitch_joint",    # 14
    "left_shoulder_roll_joint",   # 15
    "right_shoulder_roll_joint",  # 16
    "left_ankle_roll_joint",      # 17
    "right_ankle_roll_joint",     # 18
    "left_shoulder_yaw_joint",    # 19
    "right_shoulder_yaw_joint",   # 20
    "left_elbow_joint",           # 21
    "right_elbow_joint",          # 22
    "left_wrist_roll_joint",      # 23
    "right_wrist_roll_joint",     # 24
    "left_wrist_pitch_joint",     # 25
    "right_wrist_pitch_joint",    # 26
    "left_wrist_yaw_joint",       # 27
    "right_wrist_yaw_joint",      # 28
    # --- Hands (24): left/right interleaved, actuated then mimic ---
    "left_index_1_joint",         # 29 [actuated]
    "left_little_1_joint",        # 30 [actuated]
    "left_middle_1_joint",        # 31 [actuated]
    "left_ring_1_joint",          # 32 [actuated]
    "left_thumb_1_joint",         # 33 [actuated]
    "right_index_1_joint",        # 34 [actuated]
    "right_little_1_joint",       # 35 [actuated]
    "right_middle_1_joint",       # 36 [actuated]
    "right_ring_1_joint",         # 37 [actuated]
    "right_thumb_1_joint",        # 38 [actuated]
    "left_index_2_joint",         # 39 [mimic]
    "left_little_2_joint",        # 40 [mimic]
    "left_middle_2_joint",        # 41 [mimic]
    "left_ring_2_joint",          # 42 [mimic]
    "left_thumb_2_joint",         # 43 [actuated]
    "right_index_2_joint",        # 44 [mimic]
    "right_little_2_joint",       # 45 [mimic]
    "right_middle_2_joint",       # 46 [mimic]
    "right_ring_2_joint",         # 47 [mimic]
    "right_thumb_2_joint",        # 48 [actuated]
    "left_thumb_3_joint",         # 49 [mimic]
    "right_thumb_3_joint",        # 50 [mimic]
    "left_thumb_4_joint",         # 51 [mimic]
    "right_thumb_4_joint",        # 52 [mimic]
]

offset_dict = {
    "left_elbow_joint": -0.3,
    "right_elbow_joint": -0.3,
}

# ---------------------------------------------------------------------------
# Target pad geometry (same as Dex3 variant)
# ---------------------------------------------------------------------------
TARGET_W = 0.15
TARGET_D = 0.15
TARGET_T = 0.005
BIN_CX = -1.55
BIN_CY = 1.61
TABLE_Z = 0.855
TARGET_Z = 0.835


@configclass
class GraspPolicyInspireSceneCfg(InteractiveSceneCfg):
    """Scene: G1 + Inspire FTP robot + block + bin."""

    robot = G1InspireRobotPresets.g1_29dof_inspire_ftp_base_fix(
        init_pos=(-1.84919, 1.94, 0.81168), init_rot=(1.0, 0, 0, 0.0)
    )

    # Front camera only (Inspire FTP has no wrist camera mount links)
    front_camera = CameraPresets.g1_front_camera(focal_length=10.5)

    # Background scene (surgical room with table)
    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(usd_path=TROCAR_ASSEMBLY_SCENE_USD),
    )

    # Block to grasp
    block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/block",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.55, 1.90, 0.885)),
    )

    # Target pad
    target_pad = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_pad",
        spawn=sim_utils.CuboidCfg(
            size=(TARGET_W, TARGET_D, TARGET_T),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(BIN_CX, BIN_CY, TARGET_Z + TARGET_T / 2)),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """53D joint control with Inspire FTP mimic enforcement."""

    joint_pos = mdp.InspireFTPJointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=1.0,
        use_default_offset=False,
        offset=offset_dict,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups: body state + hand state + front camera."""

    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_body_joint_states)
        robot_inspire_joint_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraImagesCfg(ObsGroup):
        front_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.task_success_termination,
        time_out=False,
        params={"success_stage": 3},
    )

    object_drop = DoneTerm(
        func=mdp.object_drop_termination,
        time_out=True,
        params={
            "drop_height_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("block"),
        },
    )


@configclass
class RewardsCfg:
    """3-stage sparse rewards. Total reward for full task = 3.0."""

    grasp = RewTerm(
        func=mdp.grasp_reward,
        weight=1.0,
        params={
            "table_height": TABLE_Z,
            "lift_threshold": 0.05,
            "bin_x_min": BIN_CX - TARGET_W / 2,
            "bin_x_max": BIN_CX + TARGET_W / 2,
            "bin_y_min": BIN_CY - TARGET_D / 2,
            "bin_y_max": BIN_CY + TARGET_D / 2,
            "bin_rim_z": TARGET_Z + TARGET_T + 0.05,
            "bin_floor_z": TARGET_Z - 0.05,
            "use_sparse_reward": True,
        },
    )

    transport = RewTerm(
        func=mdp.transport_reward,
        weight=1.0,
        params={"use_sparse_reward": True},
    )

    place = RewTerm(
        func=mdp.place_reward,
        weight=1.0,
        params={"use_sparse_reward": True},
    )


@configclass
class EventCfg:
    """Event configuration for scene reset."""

    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_task_stage = EventTermCfg(func=mdp.reset_task_stage, mode="reset")
    reset_block_position = EventTermCfg(
        func=mdp.reset_block_random_position,
        mode="reset",
        params={
            "block_cfg": SceneEntityCfg("block"),
            "x_range": (-0.03, 0.03),
            "y_range": (-0.03, 0.03),
        },
    )


@configclass
class G1GraspPolicyInspireEnvCfg(ManagerBasedRLEnvCfg):
    """Inspire FTP grasp-policy environment configuration (RL mode, 53D joint control)."""

    scene: GraspPolicyInspireSceneCfg = GraspPolicyInspireSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=True,
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(-0.5, 2.4, 1.6),
        lookat=(-5.4, 0.2, -1.2),
        cam_prim_path="/OmniverseKit_Persp",
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.render.enable_translucency = True
        self.sim.render.carb_settings = {
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
        self.sim.render.rendering_mode = "quality"
        self.sim.render.antialiasing_mode = "DLAA"


@configclass
class G1GraspPolicyInspireEvalEnvCfg(G1GraspPolicyInspireEnvCfg):
    """Eval variant — deterministic block placement per env index."""

    pass
