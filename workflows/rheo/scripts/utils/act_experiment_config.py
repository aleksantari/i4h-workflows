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

"""Config-driven camera and joint group selection for ACT experiments.

Single source of truth for which cameras and joint groups the ACT pipeline uses.
All downstream consumers (IL eval, RL obs/action converters, RL policy wrapper)
import from here instead of hardcoding camera maps and joint slices.

The ``experiment:`` section in ``act_config.yaml`` defines the selection.  The
default (all 3 cameras, all 4 joint groups = 28 DOF) reproduces the original
hardcoded behaviour exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Joint group constants (derived from env joint ordering)
# ---------------------------------------------------------------------------

# Indices into the 87-D ``robot_joint_state`` for each arm group.
ARM_BODY_RANGES: dict[str, tuple[int, int]] = {
    "left_arm": (15, 22),   # 7 joints
    "right_arm": (22, 29),  # 7 joints
}

# Indices into the 14-D ``robot_dex3_joint_state`` for each hand group.
HAND_DEX3_RANGES: dict[str, tuple[int, int]] = {
    "left_hand": (0, 7),   # 7 joints
    "right_hand": (7, 14),  # 7 joints
}

# Where each group's joints land in the 43-D sim action space.
GROUP_SIM_RANGES: dict[str, tuple[int, int]] = {
    "left_arm": (15, 22),
    "right_arm": (22, 29),
    "left_hand": (29, 36),
    "right_hand": (36, 43),
}

VALID_GROUPS = list(GROUP_SIM_RANGES.keys())
GROUP_SIZE = 7  # every group has exactly 7 DOF

# Mapping from sim camera key → RLinf video key (used for RL path).
_SIM_TO_RLINF_VIDEO: dict[str, str] = {
    "front_camera": "video.room_view",
    "left_wrist_camera": "video.left_wrist_view",
    "right_wrist_camera": "video.right_wrist_view",
}

# Mapping from sim camera key → RLinf state key prefix.
_SIM_TO_RLINF_STATE: dict[str, str] = {
    "left_arm": "state.left_arm",
    "right_arm": "state.right_arm",
    "left_hand": "state.left_hand",
    "right_hand": "state.right_hand",
}

# Default cameras (sim key → ACT feature key).
DEFAULT_CAMERAS: dict[str, str] = {
    "front_camera": "observation.images.cam_room",
    "left_wrist_camera": "observation.images.cam_left_wrist",
    "right_wrist_camera": "observation.images.cam_right_wrist",
}

DEFAULT_JOINT_GROUPS: list[str] = ["left_arm", "right_arm", "left_hand", "right_hand"]

@dataclass
class ACTExperimentConfig:
    """Defines which cameras and joint groups an ACT experiment uses.

    All derived indices are computed at ``__post_init__`` from the two
    user-facing fields (*cameras* and *joint_groups*).
    """

    cameras: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERAS))
    joint_groups: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_GROUPS))

    # Computed ---
    policy_dim: int = field(init=False)
    body_state_indices: list[int] = field(init=False)
    dex3_state_indices: list[int] = field(init=False)
    sim_scatter_indices: list[int] = field(init=False)

    def __post_init__(self) -> None:
        for g in self.joint_groups:
            if g not in VALID_GROUPS:
                raise ValueError(f"Unknown joint group {g!r}. Valid: {VALID_GROUPS}")

        self.policy_dim = len(self.joint_groups) * GROUP_SIZE

        # Indices into 87-D body state for selected arm groups.
        body_idx: list[int] = []
        for g in self.joint_groups:
            if g in ARM_BODY_RANGES:
                s, e = ARM_BODY_RANGES[g]
                body_idx.extend(range(s, e))
        self.body_state_indices = body_idx

        # Indices into 14-D dex3 state for selected hand groups.
        dex3_idx: list[int] = []
        for g in self.joint_groups:
            if g in HAND_DEX3_RANGES:
                s, e = HAND_DEX3_RANGES[g]
                dex3_idx.extend(range(s, e))
        self.dex3_state_indices = dex3_idx

        # Scatter indices: policy_dim → 43-D sim action.
        scatter: list[int] = []
        for g in self.joint_groups:
            s, e = GROUP_SIM_RANGES[g]
            scatter.extend(range(s, e))
        self.sim_scatter_indices = scatter

    # --- Constructors --------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> ACTExperimentConfig:
        """Load from the ``experiment:`` section of an ACT config YAML."""
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        exp = raw.get("experiment", {})
        cameras = exp.get("cameras", dict(DEFAULT_CAMERAS))
        joint_groups = exp.get("joint_groups", list(DEFAULT_JOINT_GROUPS))
        return cls(cameras=cameras, joint_groups=joint_groups)

    @classmethod
    def from_env_or_default(cls) -> ACTExperimentConfig:
        """Load from ``ACT_EXPERIMENT_CONFIG`` env var, or return defaults.

        The result is cached at module level so repeated calls are free.
        """
        global _cached_config  # noqa: PLW0603
        if _cached_config is not None:
            return _cached_config

        env_path = os.environ.get("ACT_EXPERIMENT_CONFIG")
        if env_path and Path(env_path).is_file():
            _cached_config = cls.from_yaml(env_path)
        else:
            _cached_config = cls()
        return _cached_config

    # --- State extraction / action scatter -----------------------------------

    def extract_state(self, body_87d: Any, dex3_14d: Any) -> Any:
        """Select and concatenate joint groups from raw env observations.

        Args:
            body_87d: Tensor of shape ``(B, 87)`` — ``robot_joint_state``.
            dex3_14d: Tensor of shape ``(B, 14)`` — ``robot_dex3_joint_state``.

        Returns:
            Tensor of shape ``(B, policy_dim)`` with selected groups concatenated
            in the order given by ``self.joint_groups``.
        """
        import torch

        parts: list[torch.Tensor] = []
        for g in self.joint_groups:
            if g in ARM_BODY_RANGES:
                s, e = ARM_BODY_RANGES[g]
                parts.append(body_87d[:, s:e])
            elif g in HAND_DEX3_RANGES:
                s, e = HAND_DEX3_RANGES[g]
                parts.append(dex3_14d[:, s:e])
        return torch.cat(parts, dim=-1)

    def scatter_to_sim(self, policy_action: Any) -> Any:
        """Place policy-dim actions at the correct 43-D sim positions.

        Args:
            policy_action: Tensor of shape ``(..., policy_dim)``.

        Returns:
            Tensor of shape ``(..., 43)`` with zeros elsewhere.
        """
        import torch

        leading = policy_action.shape[:-1]
        sim = torch.zeros(
            *leading, 43, dtype=policy_action.dtype, device=policy_action.device
        )
        idx = torch.tensor(
            self.sim_scatter_indices, device=policy_action.device
        )
        sim[..., idx] = policy_action
        return sim

    def scatter_to_sim_numpy(self, policy_action: Any) -> Any:
        """Numpy version of :meth:`scatter_to_sim` for RLinf action converter."""
        import numpy as np

        leading = policy_action.shape[:-1]
        sim = np.zeros((*leading, 43), dtype=policy_action.dtype)
        sim[..., self.sim_scatter_indices] = policy_action
        return sim

    # --- RLinf key helpers ---------------------------------------------------

    def rlinf_state_keys(self) -> list[str]:
        """Return RLinf state part keys for selected joint groups.

        Example: ``["state.right_arm", "state.right_hand"]``
        """
        return [_SIM_TO_RLINF_STATE[g] for g in self.joint_groups]

    def rlinf_video_keys(self) -> dict[str, str]:
        """Return mapping from RLinf video key → ACT feature key.

        Example: ``{"video.room_view": "observation.images.cam_room", ...}``
        """
        result: dict[str, str] = {}
        for sim_key, act_key in self.cameras.items():
            rlinf_key = _SIM_TO_RLINF_VIDEO.get(sim_key)
            if rlinf_key:
                result[rlinf_key] = act_key
        return result


# Module-level cache for ``from_env_or_default()``.
_cached_config: ACTExperimentConfig | None = None
