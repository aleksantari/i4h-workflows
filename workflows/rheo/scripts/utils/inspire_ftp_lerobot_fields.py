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

"""Helpers for mapping Inspire FTP HDF5 fields into canonical LeRobot 26-D tensors.

Parallel to ``assemble_trocar_lerobot_fields.py`` but for the Inspire FTP
hand (6 actuated DOF per hand instead of Dex3's 7).
"""

from __future__ import annotations

import numpy as np

STATE_26_GROUP_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")

# Canonical 26-D joint order for Inspire FTP LeRobot data.
# 14 arm joints (same as Dex3) + 12 actuated hand joints.
STATE_26_NAMES_ENV_ORDER = [
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    # Left hand — 6 actuated (Inspire FTP)
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
    # Right hand — 6 actuated (Inspire FTP)
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]

# Indices of the 12 actuated hand joints within the 53-D action space.
# Useful for extracting hand state from recorded 53-D HDF5 actions.
INSPIRE_ACTUATED_JOINT_INDICES = [
    33, 43, 29, 30, 32, 31,  # left: thumb_yaw, thumb_pitch, index, middle, ring, pinky
    38, 48, 34, 35, 37, 36,  # right: thumb_yaw, thumb_pitch, index, middle, ring, pinky
]

# Indices of the 14 arm joints within the 53-D action space.
ARM_JOINT_INDICES = [
    11, 15, 19, 21, 23, 25, 27,  # left arm
    12, 16, 20, 22, 24, 26, 28,  # right arm
]

# Elbow offset (same as Dex3 — env config uses offset_dict for elbows)
STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA = np.zeros(26, dtype=np.float64)
STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[3] = 0.3   # left_elbow_joint
STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[10] = 0.3  # right_elbow_joint

# Column indices into the 87-D body observation (29 pos | 29 vel | 29 torque)
# for extracting arm positions.  The canonical body observation order puts
# left arm at positions 15-21 and right arm at 22-28.
STATE_26_BODY_COL_LEFT_ARM = list(range(15, 22))
STATE_26_BODY_COL_RIGHT_ARM = list(range(22, 29))

# Column indices into the 12-D inspire hand observation (already canonical).
STATE_26_INSPIRE_COL_LEFT_HAND = list(range(0, 6))
STATE_26_INSPIRE_COL_RIGHT_HAND = list(range(6, 12))

# The full 53-joint env config order (= USD tree-traversal order).
# Updated from inspect_inspire_ftp_joints.py output.
# L/R are interleaved, and actuated/mimic hand joints are NOT contiguous.
RECORDED_ACTION_53_JOINT_NAMES = (
    # --- Body (29): L/R interleaved ---
    "left_hip_pitch_joint",        # 0
    "right_hip_pitch_joint",       # 1
    "waist_yaw_joint",             # 2
    "left_hip_roll_joint",         # 3
    "right_hip_roll_joint",        # 4
    "waist_roll_joint",            # 5
    "left_hip_yaw_joint",          # 6
    "right_hip_yaw_joint",         # 7
    "waist_pitch_joint",           # 8
    "left_knee_joint",             # 9
    "right_knee_joint",            # 10
    "left_shoulder_pitch_joint",   # 11
    "right_shoulder_pitch_joint",  # 12
    "left_ankle_pitch_joint",      # 13
    "right_ankle_pitch_joint",     # 14
    "left_shoulder_roll_joint",    # 15
    "right_shoulder_roll_joint",   # 16
    "left_ankle_roll_joint",       # 17
    "right_ankle_roll_joint",      # 18
    "left_shoulder_yaw_joint",     # 19
    "right_shoulder_yaw_joint",    # 20
    "left_elbow_joint",            # 21
    "right_elbow_joint",           # 22
    "left_wrist_roll_joint",       # 23
    "right_wrist_roll_joint",      # 24
    "left_wrist_pitch_joint",      # 25
    "right_wrist_pitch_joint",     # 26
    "left_wrist_yaw_joint",        # 27
    "right_wrist_yaw_joint",       # 28
    # --- Hands (24): L/R interleaved, actuated then mimic ---
    "L_index_proximal_joint",          # 29 [actuated]
    "L_middle_proximal_joint",         # 30 [actuated]
    "L_pinky_proximal_joint",          # 31 [actuated]
    "L_ring_proximal_joint",           # 32 [actuated]
    "L_thumb_proximal_yaw_joint",      # 33 [actuated]
    "R_index_proximal_joint",          # 34 [actuated]
    "R_middle_proximal_joint",         # 35 [actuated]
    "R_pinky_proximal_joint",          # 36 [actuated]
    "R_ring_proximal_joint",           # 37 [actuated]
    "R_thumb_proximal_yaw_joint",      # 38 [actuated]
    "L_index_intermediate_joint",      # 39 [mimic]
    "L_middle_intermediate_joint",     # 40 [mimic]
    "L_pinky_intermediate_joint",      # 41 [mimic]
    "L_ring_intermediate_joint",       # 42 [mimic]
    "L_thumb_proximal_pitch_joint",    # 43 [actuated]
    "R_index_intermediate_joint",      # 44 [mimic]
    "R_middle_intermediate_joint",     # 45 [mimic]
    "R_pinky_intermediate_joint",      # 46 [mimic]
    "R_ring_intermediate_joint",       # 47 [mimic]
    "R_thumb_proximal_pitch_joint",    # 48 [actuated]
    "L_thumb_intermediate_joint",      # 49 [mimic]
    "R_thumb_intermediate_joint",      # 50 [mimic]
    "L_thumb_distal_joint",            # 51 [mimic]
    "R_thumb_distal_joint",            # 52 [mimic]
)

# Map canonical 26-D joint names to their index in the 53-D action space.
_recorded_action_name_to_idx_53 = {
    name: i for i, name in enumerate(RECORDED_ACTION_53_JOINT_NAMES)
}
ACTION_HDF5_TO_ENV_26 = [
    _recorded_action_name_to_idx_53[name] for name in STATE_26_NAMES_ENV_ORDER
]


def _extract_26d(state_body: np.ndarray, state_inspire: np.ndarray) -> np.ndarray:
    """Extract canonical 26-D vector from body (87D) and hand (12D) observations."""
    parts = [
        state_body[:, STATE_26_BODY_COL_LEFT_ARM],
        state_body[:, STATE_26_BODY_COL_RIGHT_ARM],
        state_inspire[:, STATE_26_INSPIRE_COL_LEFT_HAND],
        state_inspire[:, STATE_26_INSPIRE_COL_RIGHT_HAND],
    ]
    return np.concatenate(parts, axis=1).astype(np.float64)


def convert_g1_state_action_to_lerobot_26d(
    state_body: np.ndarray,
    state_inspire: np.ndarray,
    action_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Inspire FTP HDF5 obs/action arrays into canonical 26-D state/action.

    If *action_full* is 53-D (joint-space recording), the 26 policy joints are
    extracted directly via ``ACTION_HDF5_TO_ENV_26``.

    If *action_full* is any other width (e.g. 38-D PinkIK from teleop) or
    ``None``, actions are derived from next-step observations:
    ``action[t] = state[t+1]``.  This is the standard approach for teleop
    recordings where the recorded commands are in a different action space.
    """
    full_26d = _extract_26d(state_body, state_inspire)  # (T, 26)
    state = full_26d[:-1]  # (T-1, 26)

    if action_full is not None and action_full.shape[1] == 53:
        action = action_full[:-1, ACTION_HDF5_TO_ENV_26].astype(np.float64)
        action += STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA
    else:
        # Teleop recording: action = next-step observed joint positions.
        # Elbow offset is already baked into the observed positions.
        action = full_26d[1:]  # (T-1, 26)

    return state, action
