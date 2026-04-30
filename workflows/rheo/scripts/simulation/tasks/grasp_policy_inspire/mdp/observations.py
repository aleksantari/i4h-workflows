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

# Canonical body / hand obs ordering — sourced from the joint-identity anchor.
# `_BODY_JOINT_NAMES_CANONICAL` is intentionally a different *order* from the
# USD articulation (interleaved L/R by body part, so arm joints land at fixed
# slices [15:22] / [22:29]). `_INSPIRE_ACTUATED_NAMES` is the canonical 12-D
# hand obs order. See utils/inspire/joint_constants.py for definitions.
from inspire_joint_constants import (  # noqa: E402
    BODY_JOINT_NAMES_CANONICAL as _BODY_JOINT_NAMES_CANONICAL,
    INSPIRE_ACTUATED_NAMES as _INSPIRE_ACTUATED_NAMES,
)


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
