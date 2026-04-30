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

"""Config-driven camera and joint group selection for Inspire FTP ACT experiments.

Specifies which cameras and joint groups an ACT experiment consumes, and the
mapping from policy state space (26D dual-arm or 13D single-arm) to the 41D
sim action space. Hand groups are 6 actuated DOF each (12 total mimic joints
are driven internally by InspireJointPositionAction).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from inspire_joint_constants import (
    ACTUATED_JOINT_NAMES,
    BODY_JOINT_NAMES_CANONICAL,
    GROUP_NAMES,
    INSPIRE_ACTUATED_NAMES,
)

# ---------------------------------------------------------------------------
# Joint group constants — all derived from inspire_joint_constants.GROUP_NAMES.
# Hand-authoring is concentrated in joint_constants; this file only expresses
# how the group name lists are projected into the various index spaces.
# ---------------------------------------------------------------------------

# Per-group DOF counts.
GROUP_SIZES: dict[str, int] = {g: len(names) for g, names in GROUP_NAMES.items()}


def _contiguous_range(haystack: list[str], needle: list[str], group: str) -> tuple[int, int]:
    """Return (start, end) such that haystack[start:end] == needle.

    Raises if the group's names aren't contiguous in the host layout — this
    is an invariant of how BODY_JOINT_NAMES_CANONICAL and INSPIRE_ACTUATED_NAMES
    were authored, and the grounding tests pin it.
    """
    start = haystack.index(needle[0])
    end = start + len(needle)
    if haystack[start:end] != needle:
        raise RuntimeError(
            f"{group} joint names not contiguous in host layout — "
            "joint_constants.{BODY_JOINT_NAMES_CANONICAL,INSPIRE_ACTUATED_NAMES} drifted"
        )
    return (start, end)


# Indices into the 29-prefix of the ``robot_joint_state`` body observation,
# which observations.py orders to match BODY_JOINT_NAMES_CANONICAL.
ARM_BODY_RANGES: dict[str, tuple[int, int]] = {
    g: _contiguous_range(BODY_JOINT_NAMES_CANONICAL, GROUP_NAMES[g], g)
    for g in ("left_arm", "right_arm")
}

# Indices into the 12-D ``robot_inspire_joint_state`` (actuated-only) for each
# hand group.
HAND_INSPIRE_RANGES: dict[str, tuple[int, int]] = {
    g: _contiguous_range(INSPIRE_ACTUATED_NAMES, GROUP_NAMES[g], g)
    for g in ("left_hand", "right_hand")
}

# Where each group's joints land in the 41-D sim action space (env's
# actuated_joint_names == ACTUATED_JOINT_NAMES, pinned at runtime by the
# Layer-3 grounding test). Arm and hand joints are not contiguous in the
# 41-D layout, so the per-group indices are explicit lookups. A `ValueError`
# from .index() here would mean a name in GROUP_NAMES is not actuated at the
# articulation level — i.e. the joint identity contract is broken upstream.
GROUP_SIM_INDICES: dict[str, list[int]] = {
    g: [ACTUATED_JOINT_NAMES.index(n) for n in GROUP_NAMES[g]] for g in GROUP_NAMES
}

SIM_ACTION_DIM: int = len(ACTUATED_JOINT_NAMES)  # 41 = 29 body + 12 actuated hand

VALID_GROUPS: list[str] = list(GROUP_NAMES.keys())

# Mapping from sim camera key -> RLinf video key.
# Extension point: to enable RL post-training with wrist cameras, add
# "left_wrist_camera": "video.left_wrist_view" and
# "right_wrist_camera": "video.right_wrist_view" here (IL via LeRobot ignores
# this map; it reads input_features directly).
_SIM_TO_RLINF_VIDEO: dict[str, str] = {
    "front_camera": "video.room_view",
}

# Mapping from sim joint group -> RLinf state key prefix.
_SIM_TO_RLINF_STATE: dict[str, str] = {
    "left_arm": "state.left_arm",
    "right_arm": "state.right_arm",
    "left_hand": "state.left_hand",
    "right_hand": "state.right_hand",
}

# Default cameras (sim key -> ACT feature key).
DEFAULT_CAMERAS: dict[str, str] = {
    "front_camera": "observation.images.cam_room",
}

DEFAULT_JOINT_GROUPS: list[str] = ["left_arm", "right_arm", "left_hand", "right_hand"]


@dataclass
class InspireExperimentConfig:
    """Defines which cameras and joint groups an Inspire FTP ACT experiment uses.

    Per-group sizing: arm=7, hand=6. Scatters policy actions back into the
    41D sim action space using GROUP_SIM_INDICES.
    """

    cameras: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERAS))
    joint_groups: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_GROUPS))

    # Computed ---
    policy_dim: int = field(init=False)
    body_state_indices: list[int] = field(init=False)
    inspire_state_indices: list[int] = field(init=False)
    sim_scatter_indices: list[int] = field(init=False)

    def __post_init__(self) -> None:
        for g in self.joint_groups:
            if g not in VALID_GROUPS:
                raise ValueError(f"Unknown joint group {g!r}. Valid: {VALID_GROUPS}")

        self.policy_dim = sum(GROUP_SIZES[g] for g in self.joint_groups)

        # Indices into 87-D body state for selected arm groups.
        body_idx: list[int] = []
        for g in self.joint_groups:
            if g in ARM_BODY_RANGES:
                s, e = ARM_BODY_RANGES[g]
                body_idx.extend(range(s, e))
        self.body_state_indices = body_idx

        # Indices into 12-D inspire hand state for selected hand groups.
        inspire_idx: list[int] = []
        for g in self.joint_groups:
            if g in HAND_INSPIRE_RANGES:
                s, e = HAND_INSPIRE_RANGES[g]
                inspire_idx.extend(range(s, e))
        self.inspire_state_indices = inspire_idx

        # Scatter indices: policy_dim -> 41-D sim action.
        scatter: list[int] = []
        for g in self.joint_groups:
            scatter.extend(GROUP_SIM_INDICES[g])
        self.sim_scatter_indices = scatter

    # --- Constructors --------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> InspireExperimentConfig:
        """Load from the ``experiment:`` section of an ACT config YAML."""
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        exp = raw.get("experiment", {})
        cameras = exp.get("cameras", dict(DEFAULT_CAMERAS))
        joint_groups = exp.get("joint_groups", list(DEFAULT_JOINT_GROUPS))
        return cls(cameras=cameras, joint_groups=joint_groups)

    @classmethod
    def from_env_or_default(cls) -> InspireExperimentConfig:
        """Load from ``INSPIRE_EXPERIMENT_CONFIG`` env var, or return defaults."""
        global _cached_config  # noqa: PLW0603
        if _cached_config is not None:
            return _cached_config

        env_path = os.environ.get("INSPIRE_EXPERIMENT_CONFIG")
        if env_path and Path(env_path).is_file():
            _cached_config = cls.from_yaml(env_path)
        else:
            _cached_config = cls()
        return _cached_config

    # --- State extraction / action scatter -----------------------------------

    def extract_state(self, body_87d: Any, inspire_12d: Any) -> Any:
        """Select and concatenate joint groups from raw env observations.

        Args:
            body_87d: Tensor of shape ``(B, 87)`` — ``robot_joint_state``.
            inspire_12d: Tensor of shape ``(B, 12)`` — ``robot_inspire_joint_state``.

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
            elif g in HAND_INSPIRE_RANGES:
                s, e = HAND_INSPIRE_RANGES[g]
                parts.append(inspire_12d[:, s:e])
        return torch.cat(parts, dim=-1)

    def scatter_to_sim(self, policy_action: Any, base_action: Any = None) -> Any:
        """Place policy-dim actions at the correct 41-D sim positions.

        Args:
            policy_action: Tensor of shape ``(..., policy_dim)``.
            base_action: Optional tensor of shape ``(..., 41)`` or
                ``(num_envs, 41)`` used as the base for non-commanded joints.
                If ``None``, non-commanded joints get zeros (drives them to
                zero-angle posture). For single-arm eval, pass the env init
                pose so the non-controlled arm/hand holds its reset posture.

        Returns:
            Tensor of shape ``(..., 41)``.
            Mimic joints are not part of the 41-D action space — they are
            driven by InspireJointPositionAction.apply_actions().
        """
        import torch

        leading = policy_action.shape[:-1]
        if base_action is None:
            sim = torch.zeros(
                *leading, SIM_ACTION_DIM, dtype=policy_action.dtype, device=policy_action.device
            )
        else:
            # base_action is (num_envs, 41); broadcast to match leading dims
            # (typically (num_envs, chunk_size, 41)).
            base = base_action.to(dtype=policy_action.dtype, device=policy_action.device)
            while base.ndim < policy_action.ndim:
                base = base.unsqueeze(-2)
            sim = base.expand(*leading, SIM_ACTION_DIM).clone()
        idx = torch.tensor(
            self.sim_scatter_indices, device=policy_action.device
        )
        sim[..., idx] = policy_action
        return sim

    def scatter_to_sim_numpy(self, policy_action: Any, base_action: Any = None) -> Any:
        """Numpy version of :meth:`scatter_to_sim`."""
        import numpy as np

        leading = policy_action.shape[:-1]
        if base_action is None:
            sim = np.zeros((*leading, SIM_ACTION_DIM), dtype=policy_action.dtype)
        else:
            base = np.asarray(base_action, dtype=policy_action.dtype)
            while base.ndim < policy_action.ndim:
                base = np.expand_dims(base, -2)
            sim = np.broadcast_to(base, (*leading, SIM_ACTION_DIM)).copy()
        sim[..., self.sim_scatter_indices] = policy_action
        return sim

    # --- RLinf key helpers ---------------------------------------------------

    def rlinf_state_keys(self) -> list[str]:
        """Return RLinf state part keys for selected joint groups."""
        return [_SIM_TO_RLINF_STATE[g] for g in self.joint_groups]

    def rlinf_video_keys(self) -> dict[str, str]:
        """Return mapping from RLinf video key -> ACT feature key."""
        result: dict[str, str] = {}
        for sim_key, act_key in self.cameras.items():
            rlinf_key = _SIM_TO_RLINF_VIDEO.get(sim_key)
            if rlinf_key:
                result[rlinf_key] = act_key
        return result


# Module-level cache for ``from_env_or_default()``.
_cached_config: InspireExperimentConfig | None = None
