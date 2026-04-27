# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""D1 check: print the physical joint name at each GROUP_SIM_INDICES slot.

Boots the Inspire FTP grasp env, looks up the 41-D actuated joint list
from the action term, and prints what each canonical scatter index
actually lands on. Exits immediately after printing.
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Grasp-Policy-G129-InspireFTP-Joint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from simulation.tasks import grasp_policy_inspire  # noqa: F401,E402

# Inlined copy of GROUP_SIM_INDICES from utils/inspire_experiment_config.py
# (avoids the dataclass import-by-path quirk inside Isaac Sim's Python).
GROUP_SIM_INDICES = {
    "left_arm":   [11, 15, 19, 21, 23, 25, 27],
    "right_arm":  [12, 16, 20, 22, 24, 26, 28],
    "left_hand":  [33, 39, 29, 30, 32, 31],
    "right_hand": [38, 40, 34, 35, 37, 36],
}


CANONICAL_LABELS = {
    "left_arm":  ["L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw",
                  "L_elbow", "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw"],
    "right_arm": ["R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw",
                  "R_elbow", "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw"],
    "left_hand":  ["L_thumb_yaw", "L_thumb_pitch", "L_index", "L_middle", "L_ring", "L_pinky"],
    "right_hand": ["R_thumb_yaw", "R_thumb_pitch", "R_index", "R_middle", "R_ring", "R_pinky"],
}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    joint_ids = term._joint_ids.tolist() if hasattr(term._joint_ids, "tolist") else list(term._joint_ids)
    names_41 = [robot.joint_names[i] for i in joint_ids]

    print("\n=============================================")
    print("41-D actuated_joint_names (env runtime order):")
    print("=============================================")
    for i, n in enumerate(names_41):
        print(f"  {i:2d}: {n}")

    print("\n=============================================")
    print("Canonical slot -> physical joint (via GROUP_SIM_INDICES):")
    print("=============================================")
    for g in ("left_arm", "right_arm", "left_hand", "right_hand"):
        print(f"\n  {g}:")
        for label, idx in zip(CANONICAL_LABELS[g], GROUP_SIM_INDICES[g]):
            phys = names_41[idx]
            print(f"    slot '{label:<18}'  ->  env[{idx:2d}] = {phys}")

    # D2 check: observation-side joint layout.
    # extract_state slices body_87d[:, 22:29] for right_arm and
    # inspire_12d[:, 6:12] for right_hand. Verify these slices resolve to
    # the physical joints the canonical labels claim.
    from simulation.tasks.grasp_policy_inspire.mdp.observations import (
        _BODY_JOINT_NAMES_CANONICAL,
        _INSPIRE_ACTUATED_NAMES,
        get_robot_body_joint_states,
        get_robot_inspire_joint_states,
    )

    print("\n=============================================")
    print("D2: observation-side joint layout")
    print("=============================================")
    print("\n  _BODY_JOINT_NAMES_CANONICAL[15:29] (arm slice source):")
    for i in range(15, 29):
        print(f"    body[{i:2d}]: {_BODY_JOINT_NAMES_CANONICAL[i]}")
    print("\n  _INSPIRE_ACTUATED_NAMES (full 12-D):")
    for i, n in enumerate(_INSPIRE_ACTUATED_NAMES):
        print(f"    inspire[{i:2d}]: {n}")

    # Runtime obs numerical check: right_elbow should reset to -0.3,
    # right_shoulder_pitch to -0.5, all others (arm + hand) to 0.
    env.reset()
    body_87d = get_robot_body_joint_states(env).cpu().numpy()[0]
    inspire_12d = get_robot_inspire_joint_states(env).cpu().numpy()[0]

    print("\n  Runtime obs values after reset (env0):")
    print("    body[15:29] (arm positions — L then R):")
    for i in range(15, 29):
        print(f"      [{i:2d}] {_BODY_JOINT_NAMES_CANONICAL[i]:<28} = {body_87d[i]:+.4f}")
    print("\n    inspire[0:12] (hand positions — L then R):")
    for i in range(12):
        print(f"      [{i:2d}] {_INSPIRE_ACTUATED_NAMES[i]:<24} = {inspire_12d[i]:+.4f}")
    print("\n  Expected: *_shoulder_pitch = -0.5, *_elbow = -0.3, all others ≈ 0.0")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
