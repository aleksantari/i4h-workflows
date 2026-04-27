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
6 mechanically coupled (mimic) joints.  This action class accepts a 41-D
action vector (29 body + 12 actuated hand) and drives the 12 mimic joints
separately on the articulation using the URDF-specified multipliers.

Processing order matters for chained mimic joints (thumb):
  thumb_intermediate = thumb_proximal_pitch x 0.8024
  thumb_distal       = thumb_intermediate   x 0.9487
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
    # Finger _2 joints mimic their _1 (proximal) at 1.0843x
    ("{side}_index_2_joint", "{side}_index_1_joint", 1.0843),
    ("{side}_middle_2_joint", "{side}_middle_1_joint", 1.0843),
    ("{side}_ring_2_joint", "{side}_ring_1_joint", 1.0843),
    ("{side}_little_2_joint", "{side}_little_1_joint", 1.0843),
    # Thumb chain: _3 mimics _2 (proximal pitch), _4 mimics _3 (intermediate)
    ("{side}_thumb_3_joint", "{side}_thumb_2_joint", 0.8024),
    ("{side}_thumb_4_joint", "{side}_thumb_3_joint", 0.9487),
]

# Expand for both hands
MIMIC_RULES: list[tuple[str, str, float]] = []
for side in ("left", "right"):
    for mimic_tmpl, parent_tmpl, mult in _MIMIC_RULES_PER_SIDE:
        MIMIC_RULES.append((mimic_tmpl.format(side=side), parent_tmpl.format(side=side), mult))


class InspireJointPositionAction(JointPositionAction):
    """JointPositionAction with mimic joint enforcement for Inspire FTP hands.

    Accepts a 41-D action (29 body + 12 actuated hand).  After setting
    position targets for those 41 joints, ``apply_actions`` computes and
    sets targets for the 12 mimic joints on the articulation.
    """

    def __init__(self, cfg: InspireJointPositionActionCfg, env):
        super().__init__(cfg, env)

        # Resolve mimic joint IDs from the full articulation.
        all_art_names = list(self._asset.data.joint_names)
        art_name_to_idx = {n: i for i, n in enumerate(all_art_names)}

        # Map actuated joint names in the 41-D action space to action indices.
        action_name_to_idx = {n: i for i, n in enumerate(cfg.joint_names)}

        mimic_art_ids: list[int] = []
        # Each entry: ("action" | "mimic", parent_index, multiplier)
        # "action" means parent is in the 41-D action tensor;
        # "mimic" means parent is another mimic joint (chained thumb case).
        mimic_parent_info: list[tuple[str, int, float]] = []
        mimic_name_to_local: dict[str, int] = {}

        for mimic_name, parent_name, mult in MIMIC_RULES:
            if mimic_name not in art_name_to_idx:
                continue  # joint not in this articulation

            mimic_art_ids.append(art_name_to_idx[mimic_name])
            local_idx = len(mimic_art_ids) - 1
            mimic_name_to_local[mimic_name] = local_idx

            if parent_name in action_name_to_idx:
                mimic_parent_info.append(("action", action_name_to_idx[parent_name], mult))
            elif parent_name in mimic_name_to_local:
                mimic_parent_info.append(("mimic", mimic_name_to_local[parent_name], mult))
            else:
                raise ValueError(
                    f"Mimic parent '{parent_name}' for '{mimic_name}' not found in "
                    f"action space or prior mimic joints."
                )

        self._mimic_art_ids = mimic_art_ids
        self._mimic_parent_info = mimic_parent_info
        self._n_mimic = len(mimic_art_ids)

    def apply_actions(self) -> None:
        """Set targets for actuated joints, then compute and set mimic targets."""
        # 1. Set targets for the 41 actuated joints (parent class logic).
        super().apply_actions()

        if self._n_mimic == 0:
            return

        # 2. Compute mimic joint targets from processed_actions.
        B = self.processed_actions.shape[0]
        mimic_vals = torch.zeros(
            B, self._n_mimic,
            dtype=self.processed_actions.dtype,
            device=self.processed_actions.device,
        )

        for i, (source, idx, mult) in enumerate(self._mimic_parent_info):
            if source == "action":
                mimic_vals[:, i] = self.processed_actions[:, idx] * mult
            else:  # "mimic" — chained (e.g. thumb_4 from thumb_3)
                mimic_vals[:, i] = mimic_vals[:, idx] * mult

        # 3. Set mimic joint targets on the articulation.
        self._asset.set_joint_position_target(mimic_vals, joint_ids=self._mimic_art_ids)


@configclass
class InspireJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the Inspire FTP mimic-enforcing joint position action."""

    class_type: type = InspireJointPositionAction
