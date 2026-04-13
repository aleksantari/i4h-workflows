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

"""Teleop environment configuration for the grasp-policy task.

Inherits the RL env config and overrides actions to use WBC+PINK (23D),
extends episode length for human teleoperation, and registers XR devices.
"""

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode, XrCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_arena_g1.g1_env.mdp import g1_events as g1_events_mdp
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg
from simulation.tasks.grasp_policy.g1_grasp_policy_dex3_env_cfg import G1GraspPolicyEnvCfg
from teleop_devices.handtracking import HandtrackingTeleopDevice
from teleop_devices.motion_controllers import MotionControllersTeleopDevice


class G1GraspPolicyFixedLegsWBCPinkAction(G1DecoupledWBCPinkAction):
    """WBC+PINK action with fixed lower body for tabletop manipulation.

    23-D input: gripper(2) + wrist poses(14) + nav(3) + height(1) + torso(3).
    Lower body joints are frozen — only arms and hands move.
    """

    # Indices of lower-body joints to freeze (15 DOF: 12 leg + 3 waist)
    _FIXED_LEG_JOINT_INDICES = list(range(15))
    _FIXED_BASE_HEIGHT_CMD = 0.75

    def process_actions(self, actions):
        """Override to zero out lower body and navigation commands."""
        # Zero the lower-body portion of the action before WBC processing
        actions = actions.clone()
        # Navigation commands (indices 16-18 in the 23D action) -> zero
        actions[:, 16:19] = 0.0
        # Base height command (index 19) -> fixed standing height
        actions[:, 19] = self._FIXED_BASE_HEIGHT_CMD
        # Torso orientation (indices 20-22) -> zero
        actions[:, 20:23] = 0.0
        return super().process_actions(actions)


@configclass
class G1GraspPolicyFixedLegsWBCPinkActionCfg(G1DecoupledWBCPinkActionCfg):
    class_type = G1GraspPolicyFixedLegsWBCPinkAction


@configclass
class TeleopActionsCfg:
    g1_action: ActionTermCfg = G1GraspPolicyFixedLegsWBCPinkActionCfg(
        asset_name="robot",
        joint_names=[".*"],
    )


@configclass
class G1GraspPolicyTeleopEnvCfg(G1GraspPolicyEnvCfg):
    """Teleop variant of the grasp-policy task.

    Overrides:
    - Actions: WBC+PINK (23D) instead of direct joint control (43D)
    - Episode length: 300s (humans need time)
    - Render interval: 2 (smoother XR visuals)
    - XR config: anchor on robot pelvis
    - Teleop devices: motion controllers + hand tracking
    """

    actions: TeleopActionsCfg = TeleopActionsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = 2
        self.episode_length_s = 300.0

        # Reset WBC policy state on env reset
        self.events.reset_wbc_policy = EventTermCfg(
            func=g1_events_mdp.reset_decoupled_wbc_pink_policy,
            mode="reset",
        )

        # XR anchor follows robot pelvis
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        # Register teleop devices (motion controllers + AVP hand tracking)
        mc = MotionControllersTeleopDevice(sim_device=self.sim.device)
        ht = HandtrackingTeleopDevice(sim_device=self.sim.device)
        self.teleop_devices = DevicesCfg(
            devices={
                **mc.get_teleop_device_cfg(xr_cfg=self.xr, use_trocar_retargeter=True).devices,
                **ht.get_teleop_device_cfg(xr_cfg=self.xr, use_trocar_retargeter=True).devices,
            }
        )
