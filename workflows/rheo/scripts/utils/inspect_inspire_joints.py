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

"""Inspect the joint ordering of the G1 + Inspire FTP nucleus USD.

Run inside Docker:
    ./docker/run_docker.sh -g1.5 python scripts/utils/inspect_inspire_joints.py

Prints the full joint list with indices, and identifies body vs hand joints,
actuated vs mimic hand joints. Use this output to set hardcoded indices in
the grasp_policy_inspire observation functions.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof_inspire_hand.usd"

# Inspire FTP actuated joints (6 per hand)
ACTUATED_KEYWORDS = [
    "thumb_proximal_yaw", "thumb_proximal_pitch",
    "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal",
]

# Inspire FTP mimic joints (6 per hand)
MIMIC_KEYWORDS = [
    "thumb_intermediate", "thumb_distal",
    "index_intermediate", "middle_intermediate", "ring_intermediate", "pinky_intermediate",
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

    # Compare body indices with Dex3
    dex3_body_indices = [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
        2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
        12, 16, 20, 22, 24, 26, 28,
    ]
    print(f"\nDex3 body_joint_indices (for observation reorder): {dex3_body_indices}")
    print("Check if body joints are at the same indices in this USD.")
    print("Body joint names in this USD (first 29):")
    for idx in sorted(body_indices[:29]):
        print(f"  {idx}: {joint_names[idx]}")

    simulation_app.close()


if __name__ == "__main__":
    main()
