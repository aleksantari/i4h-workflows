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

"""Custom joint position action with Inspire FTP mimic joint enforcement.

The Inspire FTP hand has 12 joints per hand: 6 independently actuated and
6 mechanically coupled (mimic) joints. This action class accepts the full
joint position target vector and overwrites mimic joint targets with values
computed from the actuated joints using the URDF-specified multipliers.

Processing order matters for chained mimic joints (thumb):
  thumb_intermediate = thumb_proximal_pitch × 0.8024
  thumb_distal       = thumb_intermediate   × 0.9487
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Mimic relationship definitions
# ---------------------------------------------------------------------------
# Each tuple: (mimic_joint_name, parent_joint_name, multiplier)
# Order matters: thumb_intermediate must be computed before thumb_distal.

_MIMIC_RULES_PER_SIDE: list[tuple[str, str, float]] = [
    # Finger intermediates mimic their proximal at 1.0843×
    ("{side}_index_intermediate_joint", "{side}_index_proximal_joint", 1.0843),
    ("{side}_middle_intermediate_joint", "{side}_middle_proximal_joint", 1.0843),
    ("{side}_ring_intermediate_joint", "{side}_ring_proximal_joint", 1.0843),
    ("{side}_pinky_intermediate_joint", "{side}_pinky_proximal_joint", 1.0843),
    # Thumb chain: intermediate mimics proximal pitch, distal mimics intermediate
    ("{side}_thumb_intermediate_joint", "{side}_thumb_proximal_pitch_joint", 0.8024),
    ("{side}_thumb_distal_joint", "{side}_thumb_intermediate_joint", 0.9487),
]

# Expand for both hands
MIMIC_RULES: list[tuple[str, str, float]] = []
for side in ("L", "R"):
    for mimic_tmpl, parent_tmpl, mult in _MIMIC_RULES_PER_SIDE:
        MIMIC_RULES.append((mimic_tmpl.format(side=side), parent_tmpl.format(side=side), mult))


class InspireFTPJointPositionAction(JointPositionAction):
    """JointPositionAction with mimic joint enforcement for Inspire FTP hands.

    After the standard position target processing, this class reads the
    actuated hand joint targets and overwrites the mimic joint targets
    using the defined multipliers.
    """

    def __init__(self, cfg: InspireFTPJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._mimic_indices: list[tuple[int, int, float]] | None = None

    def _resolve_mimic_indices(self) -> list[tuple[int, int, float]]:
        """Lazily resolve joint name → action-space index mapping.

        Returns list of (mimic_idx, parent_idx, multiplier) in the
        action-space ordering defined by cfg.joint_names.
        """
        joint_names = list(self.cfg.joint_names)
        name_to_idx = {name: i for i, name in enumerate(joint_names)}

        resolved: list[tuple[int, int, float]] = []
        for mimic_name, parent_name, mult in MIMIC_RULES:
            mimic_idx = name_to_idx.get(mimic_name)
            parent_idx = name_to_idx.get(parent_name)
            if mimic_idx is not None and parent_idx is not None:
                resolved.append((mimic_idx, parent_idx, mult))

        return resolved

    def _apply_mimic(self, actions: torch.Tensor) -> torch.Tensor:
        """Overwrite mimic joint targets with parent × multiplier."""
        if self._mimic_indices is None:
            self._mimic_indices = self._resolve_mimic_indices()

        for mimic_idx, parent_idx, mult in self._mimic_indices:
            actions[:, mimic_idx] = actions[:, parent_idx] * mult

        return actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply mimic enforcement then delegate to parent."""
        actions = self._apply_mimic(actions.clone())
        super().process_actions(actions)


@configclass
class InspireFTPJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the Inspire FTP mimic-enforcing joint position action."""

    class_type: type = InspireFTPJointPositionAction
