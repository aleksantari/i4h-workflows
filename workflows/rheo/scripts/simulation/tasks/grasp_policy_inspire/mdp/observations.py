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

"""Observation functions for G1 + Inspire FTP hand.

Provides:
- ``get_robot_body_joint_states``: 87D body joint state (29 × [pos|vel|torque])
- ``get_robot_inspire_joint_states``: 12D actuated hand joint positions

Joint indices are resolved lazily from the articulation's joint_names on
first call, so this works regardless of the exact USD joint ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Names of the 29 body joints in the canonical output order.
# This must match the order used by the Dex3 body observation function
# so that downstream consumers (obs_processor, experiment config) see the
# same arm joint positions at the same slice.
_BODY_JOINT_NAMES_CANONICAL = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# 12 actuated hand joint names in canonical order (6 left + 6 right).
_INSPIRE_ACTUATED_NAMES = [
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_index_1_joint",
    "left_middle_1_joint",
    "left_ring_1_joint",
    "left_little_1_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
]


def _resolve_indices(joint_names: list[str], target_names: list[str], device: torch.device) -> torch.Tensor:
    """Map target joint names to their indices in the articulation's joint_names list."""
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    indices = []
    for name in target_names:
        idx = name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Joint '{name}' not found in articulation. Available: {joint_names}")
        indices.append(idx)
    return torch.tensor(indices, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Body joint observation (87D)
# ---------------------------------------------------------------------------

_body_obs_cache: dict = {
    "device": None,
    "batch": None,
    "idx_t": None,
    "idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "combined_buf": None,
}


def get_robot_body_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return body joint states as [pos(29) | vel(29) | torque(29)] = 87D.

    Resolves indices dynamically from the articulation on first call.
    """
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    device = joint_pos.device
    batch = joint_pos.shape[0]

    global _body_obs_cache
    if _body_obs_cache["device"] != device or _body_obs_cache["idx_t"] is None:
        art_joint_names = list(env.scene["robot"].data.joint_names)
        _body_obs_cache["idx_t"] = _resolve_indices(art_joint_names, _BODY_JOINT_NAMES_CANONICAL, device)
        _body_obs_cache["device"] = device
        _body_obs_cache["batch"] = None

    idx_t = _body_obs_cache["idx_t"]
    n = idx_t.numel()

    if _body_obs_cache["batch"] != batch or _body_obs_cache["idx_batch"] is None:
        _body_obs_cache["idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _body_obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["combined_buf"] = torch.empty(batch, n * 3, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["batch"] = batch

    idx_batch = _body_obs_cache["idx_batch"]
    pos_buf = _body_obs_cache["pos_buf"]
    vel_buf = _body_obs_cache["vel_buf"]
    torque_buf = _body_obs_cache["torque_buf"]
    combined_buf = _body_obs_cache["combined_buf"]

    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))

    combined_buf[:, 0:n].copy_(pos_buf)
    combined_buf[:, n : 2 * n].copy_(vel_buf)
    combined_buf[:, 2 * n : 3 * n].copy_(torque_buf)
    return combined_buf


# ---------------------------------------------------------------------------
# Inspire FTP actuated hand joint observation (12D)
# ---------------------------------------------------------------------------

_inspire_obs_cache: dict = {
    "device": None,
    "batch": None,
    "idx_t": None,
    "idx_batch": None,
    "pos_buf": None,
}


def get_robot_inspire_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return 12 actuated Inspire FTP hand joint positions [batch, 12]."""
    joint_pos = env.scene["robot"].data.joint_pos
    device = joint_pos.device
    batch = joint_pos.shape[0]

    global _inspire_obs_cache
    if _inspire_obs_cache["device"] != device or _inspire_obs_cache["idx_t"] is None:
        art_joint_names = list(env.scene["robot"].data.joint_names)
        _inspire_obs_cache["idx_t"] = _resolve_indices(art_joint_names, _INSPIRE_ACTUATED_NAMES, device)
        _inspire_obs_cache["device"] = device
        _inspire_obs_cache["batch"] = None

    idx_t = _inspire_obs_cache["idx_t"]
    n = idx_t.numel()

    if _inspire_obs_cache["batch"] != batch or _inspire_obs_cache["idx_batch"] is None:
        _inspire_obs_cache["idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _inspire_obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _inspire_obs_cache["batch"] = batch

    idx_batch = _inspire_obs_cache["idx_batch"]
    pos_buf = _inspire_obs_cache["pos_buf"]

    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))

    return pos_buf
