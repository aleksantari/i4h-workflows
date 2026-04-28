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

"""Inspect the joint ordering of the active Inspire FTP USD.

Loads the local URDF-converted USD that ``robot_config.py`` points at
(``UNITREE_G1_29DOF_INSPIRE_FTP_USD``) and prints the full joint list with
indices plus body / hand / actuated / mimic classification.

This is a one-off introspection tool. Primary use cases:

  - **Bootstrapping a new embodiment.** When ``env_cfg.joint_names`` does not
    yet exist for a USD (e.g. a hardware revision or different hand variant),
    use this output to author the canonical joint-name list.
  - **Deep-debugging a grounding-test failure.** When
    ``test_inspire_urdf_grounding`` reports a confusing mismatch, dump the
    full articulation here for human eyeballing.

For ongoing regression coverage of the URDF ↔ code contract, see
``tests/test_sim/test_inspire_urdf_grounding.py`` — that's the steady-state
check; this script is for one-off use.

Run inside Docker:
    ./docker/run_docker_grasp.sh python scripts/utils/inspire/inspect_inspire_joints.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from simulation.tasks.grasp_policy_inspire.config.robot_config import (  # noqa: E402
    UNITREE_G1_29DOF_INSPIRE_FTP_USD,
)

# The active local USD — the same one robot_config.py loads at sim startup.
# Audit doc §4.7 (workflows/rheo/docs/inspire/joint_spaces.md) explains why
# this must NOT be the Nucleus USD: that one merges fixed joints and drops
# d435_link, the wrist_cam mounts, and the IMU.
USD_PATH = UNITREE_G1_29DOF_INSPIRE_FTP_USD

# Inspire FTP actuated joints, by URDF naming suffix (6 per hand).
# Thumb has two actuated DOF (yaw=_1, pitch=_2); the other four fingers
# have one actuated proximal DOF (=_1).
ACTUATED_KEYWORDS = [
    "_thumb_1_joint", "_thumb_2_joint",
    "_index_1_joint", "_middle_1_joint", "_ring_1_joint", "_little_1_joint",
]

# Inspire FTP mimic joints, by URDF naming suffix (6 per hand).
# Thumb chains _3 (intermediate) → _4 (distal); other fingers have one
# mimic distal DOF (=_2). See mimic_action.py for the multipliers.
MIMIC_KEYWORDS = [
    "_thumb_3_joint", "_thumb_4_joint",
    "_index_2_joint", "_middle_2_joint", "_ring_2_joint", "_little_2_joint",
]


@configclass
class InspectSceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, fix_root_link=True
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=10.0,
                damping=1.0,
            ),
        },
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.0, 2.0, 2.0), target=(0.0, 0.0, 1.0))

    scene_cfg = InspectSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(dt=sim_cfg.dt)

    robot = scene["robot"]
    joint_names = robot.data.joint_names

    print(f"\n{'='*80}")
    print(f"USD: {USD_PATH}")
    print(f"Total joints: {len(joint_names)}")
    print(f"{'='*80}\n")

    # Classify joints
    body_indices = []
    hand_actuated_left = []
    hand_actuated_right = []
    hand_mimic_left = []
    hand_mimic_right = []
    hand_all_left = []
    hand_all_right = []

    for i, name in enumerate(joint_names):
        is_hand = any(k in name for k in ACTUATED_KEYWORDS + MIMIC_KEYWORDS)

        if not is_hand:
            body_indices.append(i)
        else:
            is_left = name.startswith("L_") or name.startswith("left_")
            is_actuated = any(k in name for k in ACTUATED_KEYWORDS)

            if is_left:
                hand_all_left.append(i)
                if is_actuated:
                    hand_actuated_left.append(i)
                else:
                    hand_mimic_left.append(i)
            else:
                hand_all_right.append(i)
                if is_actuated:
                    hand_actuated_right.append(i)
                else:
                    hand_mimic_right.append(i)

    # Print all joints
    print("ALL JOINTS (index: name):")
    print("-" * 60)
    for i, name in enumerate(joint_names):
        tag = ""
        if i in body_indices:
            tag = "[BODY]"
        elif i in hand_actuated_left + hand_actuated_right:
            tag = "[HAND-ACTUATED]"
        else:
            tag = "[HAND-MIMIC]"
        print(f"  {i:3d}: {name:<50s} {tag}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Body joints ({len(body_indices)}): {body_indices}")
    print(f"Left hand actuated ({len(hand_actuated_left)}): {hand_actuated_left}")
    print(f"  Names: {[joint_names[i] for i in hand_actuated_left]}")
    print(f"Right hand actuated ({len(hand_actuated_right)}): {hand_actuated_right}")
    print(f"  Names: {[joint_names[i] for i in hand_actuated_right]}")
    print(f"Left hand mimic ({len(hand_mimic_left)}): {hand_mimic_left}")
    print(f"  Names: {[joint_names[i] for i in hand_mimic_left]}")
    print(f"Right hand mimic ({len(hand_mimic_right)}): {hand_mimic_right}")
    print(f"  Names: {[joint_names[i] for i in hand_mimic_right]}")
    print(f"Left hand all ({len(hand_all_left)}): {hand_all_left}")
    print(f"Right hand all ({len(hand_all_right)}): {hand_all_right}")

    # Print copyable Python constants
    print(f"\n{'='*80}")
    print("COPY-PASTE CONSTANTS:")
    print(f"{'='*80}")
    print(f"BODY_JOINT_INDICES = {body_indices}")
    print(f"INSPIRE_ACTUATED_INDICES = {hand_actuated_left + hand_actuated_right}")
    print(f"INSPIRE_MIMIC_INDICES = {hand_mimic_left + hand_mimic_right}")
    print(f"INSPIRE_ALL_LEFT = {hand_all_left}")
    print(f"INSPIRE_ALL_RIGHT = {hand_all_right}")

    # Print the full joint_names list for env config
    print(f"\n{'='*80}")
    print("JOINT NAMES LIST (for env config):")
    print(f"{'='*80}")
    print("joint_names = [")
    for name in joint_names:
        print(f'    "{name}",')
    print("]")

    simulation_app.close()


if __name__ == "__main__":
    main()
