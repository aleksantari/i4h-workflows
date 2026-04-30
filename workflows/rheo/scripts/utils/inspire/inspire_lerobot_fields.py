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

"""Helpers for mapping Inspire FTP HDF5 fields into canonical LeRobot 26-D / 13-D tensors.

Pure-Python conversion logic — runs in non-Kit contexts (just numpy +
``inspire_joint_constants``). All joint-identity constants are derived from
that module so the URDF anchor flows through to LeRobot without any
hand-authored duplicates.
"""

from __future__ import annotations

import numpy as np

from inspire_joint_constants import (
    BODY_JOINT_NAMES_CANONICAL,
    INSPIRE_ACTUATED_NAMES,
    JOINT_NAMES_NUCLEUS_HAND,
    MIMIC_JOINT_NAMES_NUCLEUS,
    URDF_TO_NUCLEUS,
)

# Group order axiom for the canonical 26-D layout.
STATE_26_GROUP_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")

# Canonical 26-D joint order for Inspire FTP LeRobot data.
# 14 arm joints (URDF naming, derived from the canonical body order's arm
# slices) + 12 actuated hand joints (Nucleus naming, derived from the
# canonical hand order via URDF_TO_NUCLEUS).
STATE_26_NAMES_ENV_ORDER: list[str] = (
    BODY_JOINT_NAMES_CANONICAL[15:22]                              # left arm (7)
    + BODY_JOINT_NAMES_CANONICAL[22:29]                            # right arm (7)
    + [URDF_TO_NUCLEUS[n] for n in INSPIRE_ACTUATED_NAMES[0:6]]    # left hand (6, Nucleus)
    + [URDF_TO_NUCLEUS[n] for n in INSPIRE_ACTUATED_NAMES[6:12]]   # right hand (6, Nucleus)
)

# 53-joint USD order with hand half renamed to Nucleus. Used by the legacy
# 53-D recorded action path. Sourced from joint_constants so it stays
# bit-equal to env_cfg.joint_names with the URDF→Nucleus rename applied.
RECORDED_ACTION_53_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES_NUCLEUS_HAND

# Indices of the 12 actuated hand joints within the 53-D action space.
# Derived from RECORDED_ACTION_53_JOINT_NAMES so it auto-tracks any USD reorder.
INSPIRE_ACTUATED_JOINT_INDICES: list[int] = [
    RECORDED_ACTION_53_JOINT_NAMES.index(URDF_TO_NUCLEUS[n])
    for n in INSPIRE_ACTUATED_NAMES
]

# Indices of the 14 arm joints within the 53-D action space.
# Derived from RECORDED_ACTION_53_JOINT_NAMES.
ARM_JOINT_INDICES: list[int] = [
    RECORDED_ACTION_53_JOINT_NAMES.index(n)
    for n in (BODY_JOINT_NAMES_CANONICAL[15:22] + BODY_JOINT_NAMES_CANONICAL[22:29])
]

# Elbow offset chain: parquet `action[t]` carries +0.3 at the elbow columns
# to cancel the env's offset_dict[*_elbow_joint] = -0.3. Indices are derived
# from STATE_26_NAMES_ENV_ORDER so they auto-track any layout reorder.
_LEFT_ELBOW_26 = STATE_26_NAMES_ENV_ORDER.index("left_elbow_joint")
_RIGHT_ELBOW_26 = STATE_26_NAMES_ENV_ORDER.index("right_elbow_joint")
STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA = np.zeros(26, dtype=np.float64)
STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[_LEFT_ELBOW_26] = 0.3
STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[_RIGHT_ELBOW_26] = 0.3

# Column indices into the 87-D body observation (29 pos | 29 vel | 29 torque)
# for extracting arm positions. Pinned by BODY_JOINT_NAMES_CANONICAL's slice
# contract: indices 15-21 are left arm, 22-28 are right arm.
STATE_26_BODY_COL_LEFT_ARM = list(range(15, 22))
STATE_26_BODY_COL_RIGHT_ARM = list(range(22, 29))

# Column indices into the 12-D inspire hand observation (already canonical).
# Pinned by INSPIRE_ACTUATED_NAMES's slice contract.
STATE_26_INSPIRE_COL_LEFT_HAND = list(range(0, 6))
STATE_26_INSPIRE_COL_RIGHT_HAND = list(range(6, 12))


# ---------------------------------------------------------------------------
# 41-D action space (29 body + 12 actuated hand, mimic joints removed).
# Used for new recordings from the refactored RL/eval env.
# ---------------------------------------------------------------------------

RECORDED_ACTION_41_JOINT_NAMES: tuple[str, ...] = tuple(
    name for name in RECORDED_ACTION_53_JOINT_NAMES if name not in MIMIC_JOINT_NAMES_NUCLEUS
)

# Map canonical 26-D joint names to their index in the 53-D / 41-D action spaces.
_recorded_action_name_to_idx_53 = {
    name: i for i, name in enumerate(RECORDED_ACTION_53_JOINT_NAMES)
}
_recorded_action_name_to_idx_41 = {
    name: i for i, name in enumerate(RECORDED_ACTION_41_JOINT_NAMES)
}
ACTION_HDF5_TO_ENV_26 = [
    _recorded_action_name_to_idx_53[name] for name in STATE_26_NAMES_ENV_ORDER
]
ACTION_HDF5_TO_ENV_26_FROM_41 = [
    _recorded_action_name_to_idx_41[name] for name in STATE_26_NAMES_ENV_ORDER
]


# ---------------------------------------------------------------------------
# 13-D right-arm-only subset (right_arm[7] + right_hand[6])
# ---------------------------------------------------------------------------

# Derived as a slice of the 26-D layout: right arm + right hand.
STATE_13_NAMES_ENV_ORDER: list[str] = (
    STATE_26_NAMES_ENV_ORDER[7:14]    # right arm (7)
    + STATE_26_NAMES_ENV_ORDER[20:26] # right hand (6)
)

# Elbow offset for 13-D: right elbow at index derived from STATE_13.
_RIGHT_ELBOW_13 = STATE_13_NAMES_ENV_ORDER.index("right_elbow_joint")
STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA = np.zeros(13, dtype=np.float64)
STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA[_RIGHT_ELBOW_13] = 0.3

# Index maps from 53-D and 41-D recorded actions into canonical 13-D order.
ACTION_HDF5_TO_ENV_13 = [
    _recorded_action_name_to_idx_53[name] for name in STATE_13_NAMES_ENV_ORDER
]
ACTION_HDF5_TO_ENV_13_FROM_41 = [
    _recorded_action_name_to_idx_41[name] for name in STATE_13_NAMES_ENV_ORDER
]


def _extract_13d(state_body: np.ndarray, state_inspire: np.ndarray) -> np.ndarray:
    """Extract canonical 13-D vector (right arm + right hand) from body (87D) and hand (12D)."""
    parts = [
        state_body[:, STATE_26_BODY_COL_RIGHT_ARM],
        state_inspire[:, STATE_26_INSPIRE_COL_RIGHT_HAND],
    ]
    return np.concatenate(parts, axis=1).astype(np.float64)


def convert_g1_state_action_to_lerobot_13d(
    state_body: np.ndarray,
    state_inspire: np.ndarray,
    action_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Inspire FTP HDF5 obs/action arrays into canonical 13-D (right arm + hand).

    Same logic as the 26-D variant but only extracts right_arm and right_hand.
    """
    full_13d = _extract_13d(state_body, state_inspire)  # (T, 13)
    state = full_13d[:-1]  # (T-1, 13)

    if action_full is not None and action_full.shape[1] == 53:
        action = action_full[:-1, ACTION_HDF5_TO_ENV_13].astype(np.float64)
        action += STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA
    elif action_full is not None and action_full.shape[1] == 41:
        action = action_full[:-1, ACTION_HDF5_TO_ENV_13_FROM_41].astype(np.float64)
        action += STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA
    else:
        # Teleop recording (38D PinkIK): action = next-step observed joint positions.
        action = full_13d[1:] + STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA

    return state, action


# ---------------------------------------------------------------------------
# 13-D left-arm-only subset (left_arm[7] + left_hand[6])
# ---------------------------------------------------------------------------

# Derived as a slice of the 26-D layout: left arm + left hand.
STATE_13_LEFT_NAMES_ENV_ORDER: list[str] = (
    STATE_26_NAMES_ENV_ORDER[0:7]     # left arm (7)
    + STATE_26_NAMES_ENV_ORDER[14:20] # left hand (6)
)

# Elbow offset for 13-D left: left elbow at index derived from STATE_13_LEFT.
_LEFT_ELBOW_13 = STATE_13_LEFT_NAMES_ENV_ORDER.index("left_elbow_joint")
STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA = np.zeros(13, dtype=np.float64)
STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA[_LEFT_ELBOW_13] = 0.3

# Index maps from 53-D and 41-D recorded actions into canonical left-arm 13-D order.
ACTION_HDF5_TO_ENV_13_LEFT = [
    _recorded_action_name_to_idx_53[name] for name in STATE_13_LEFT_NAMES_ENV_ORDER
]
ACTION_HDF5_TO_ENV_13_LEFT_FROM_41 = [
    _recorded_action_name_to_idx_41[name] for name in STATE_13_LEFT_NAMES_ENV_ORDER
]


def _extract_13d_left(state_body: np.ndarray, state_inspire: np.ndarray) -> np.ndarray:
    """Extract canonical 13-D vector (left arm + left hand) from body (87D) and hand (12D)."""
    parts = [
        state_body[:, STATE_26_BODY_COL_LEFT_ARM],
        state_inspire[:, STATE_26_INSPIRE_COL_LEFT_HAND],
    ]
    return np.concatenate(parts, axis=1).astype(np.float64)


def convert_g1_state_action_to_lerobot_13d_left(
    state_body: np.ndarray,
    state_inspire: np.ndarray,
    action_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Inspire FTP HDF5 obs/action arrays into canonical 13-D (left arm + hand).

    Same logic as the right-arm 13-D variant but extracts left_arm and left_hand.
    """
    full_13d = _extract_13d_left(state_body, state_inspire)  # (T, 13)
    state = full_13d[:-1]  # (T-1, 13)

    if action_full is not None and action_full.shape[1] == 53:
        action = action_full[:-1, ACTION_HDF5_TO_ENV_13_LEFT].astype(np.float64)
        action += STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA
    elif action_full is not None and action_full.shape[1] == 41:
        action = action_full[:-1, ACTION_HDF5_TO_ENV_13_LEFT_FROM_41].astype(np.float64)
        action += STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA
    else:
        # Teleop recording (38D PinkIK): action = next-step observed joint positions.
        action = full_13d[1:] + STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA

    return state, action


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
    elif action_full is not None and action_full.shape[1] == 41:
        action = action_full[:-1, ACTION_HDF5_TO_ENV_26_FROM_41].astype(np.float64)
        action += STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA
    else:
        # Teleop recording (38D PinkIK) or unknown width:
        # action = next-step observed joint positions.
        # Out-of-place add avoids aliasing full_26d (state is a view of it).
        action = full_26d[1:] + STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA

    return state, action
