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
- 41D action space (29 body + 12 actuated hand) with mimic enforcement
- 12D hand observation (6 actuated per hand)
- 3 cameras (front + left wrist + right wrist), matching Dex3
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
from simulation.assets.assets import SINUS_TOOL_USD_PATHS, SURGICAL_TRAY_USD, TROCAR_ASSEMBLY_SCENE_USD
from simulation.tasks.grasp_policy_inspire import mdp

from simulation.tasks.grasp_policy_inspire.config import CameraPresets, G1InspireRobotPresets  # isort: skip

# Joint identity constants live in a single Kit-free source of truth so they
# can be shared with the HDF5→LeRobot converter (non-Kit context). Aliased
# here to preserve the public names this module already exports.
# See utils/inspire/joint_constants.py and docs/inspire/joint_spaces.md.
from inspire_joint_constants import (  # noqa: E402
    JOINT_NAMES as joint_names,
    MIMIC_JOINT_NAMES as _MIMIC_JOINT_NAMES,
    ACTUATED_JOINT_NAMES as actuated_joint_names,
)

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

# ---------------------------------------------------------------------------
# Surgical tray configuration
# Tray origin sits on the table surface. Slot offsets are from Xform markers
# embedded in the USD by the asset creator (6 prims: /root/tool_0 .. tool_5).
# Tray bounding box: 63cm x 23cm x 8cm, centered at origin.
# ---------------------------------------------------------------------------
TRAY_POS = (-1.49919, 2.03365, 0.84554)  # from trocar task, +5cm in x
TRAY_ROT = (0.70711, 0.0, 0.0, -0.70711)  # 90° CW around Z (from trocar task)
TOOL_ROT = (0.70711, 0.0, 0.0, -0.70711)  # match tray rotation

# Local slot offsets from USD Xform markers, rotated 90° CW to match tray.
# Original local coords (x, y) rotated by (y, -x).
# fmt: off
_SLOT_LOCAL = [
    ( 0.055,   -0.12097, 0.03),  # slot 0 — /root/tool_0
    ( 0.000,   -0.12097, 0.03),  # slot 1 — /root/tool_1
    (-0.055,   -0.12097, 0.03),  # slot 2 — /root/tool_2
    ( 0.055,    0.11557, 0.03),  # slot 3 — /root/tool_3
    ( 0.000,    0.11557, 0.03),  # slot 4 — /root/tool_4
    (-0.055,    0.11557, 0.03),  # slot 5 — /root/tool_5
]
# fmt: on

# World-space slot positions (tray pos + local offset)
TRAY_SLOT_POSITIONS = [
    (TRAY_POS[0] + dx, TRAY_POS[1] + dy, TRAY_POS[2] + dz) for dx, dy, dz in _SLOT_LOCAL
]
ACTIVE_SLOT_IDX = 4  # slot 4 is the physics-enabled grasp target


@configclass
class GraspPolicyInspireSceneCfg(InteractiveSceneCfg):
    """Scene: G1 + Inspire FTP robot + surgical tray + single tool + bin."""

    robot = G1InspireRobotPresets.g1_29dof_inspire_base_fix(
        init_pos=(-1.84919, 1.94, 0.81168), init_rot=(1.0, 0, 0, 0.0)
    )

    # Cameras
    front_camera = CameraPresets.g1_front_camera(focal_length=10.5)
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera(focal_length=12.0)
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera(focal_length=12.0)

    # Background scene (surgical room with table)
    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(usd_path=TROCAR_ASSEMBLY_SCENE_USD),
    )

    # Surgical tray — static prop centered on table (no physics)
    surgical_tray = AssetBaseCfg(
        prim_path="/World/envs/env_.*/surgical_tray",
        spawn=UsdFileCfg(
            usd_path=SURGICAL_TRAY_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TRAY_POS, rot=TRAY_ROT),
    )

    # Active grasp target — single tool (default: tool_0), overridable via CLI.
    # Named "block" so shared MDP code (rewards, terminations) works unmodified.
    # Positioned at tray slot 4; reset event teleports it back with XY/yaw noise.
    block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/block",
        spawn=UsdFileCfg(
            usd_path=SINUS_TOOL_USD_PATHS["tool_0"],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TRAY_SLOT_POSITIONS[ACTIVE_SLOT_IDX], rot=TOOL_ROT
        ),
    )

    # Target pad (bin)
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
    """41D joint control (29 body + 12 actuated hand) with mimic enforcement."""

    joint_pos = mdp.InspireJointPositionActionCfg(
        asset_name="robot",
        joint_names=actuated_joint_names,
        scale=1.0,
        use_default_offset=False,
        offset=offset_dict,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups: body state + hand state + front camera + wrist cameras."""

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
        left_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
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
        func=mdp.reset_block_to_tray_slot,
        mode="reset",
        params={
            "block_cfg": SceneEntityCfg("block"),
            "slot_pos": TRAY_SLOT_POSITIONS[ACTIVE_SLOT_IDX],
            "slot_rot": TOOL_ROT,
            "xy_noise": (-0.02, 0.02),
            "yaw_noise_deg": (-15.0, 15.0),
        },
    )


@configclass
class G1GraspPolicyInspireEnvCfg(ManagerBasedRLEnvCfg):
    """Inspire FTP grasp-policy environment configuration (RL mode, 41D joint control)."""

    scene: GraspPolicyInspireSceneCfg = GraspPolicyInspireSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=False,
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
        self.episode_length_s = 200.0
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
    """Eval variant — deterministic block placement (no XY / yaw noise).

    Inherits everything from the RL config but zeros the reset noise so
    checkpoint comparisons see the exact same initial block pose on every
    episode. Used by RLinf eval rollouts and (optionally) the IL eval script.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_block_position.params["xy_noise"] = (0.0, 0.0)
        self.events.reset_block_position.params["yaw_noise_deg"] = (0.0, 0.0)
