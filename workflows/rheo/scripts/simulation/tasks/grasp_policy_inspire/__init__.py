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

# Gym ID naming notes:
#
# The canonical IDs use the `-Inspire-` infix (matching the rest of the
# codebase's `_ftp` cleanup). The legacy `-InspireFTP-` IDs are kept registered
# as **deprecated aliases** that point at the same env cfg entry, so HDF5 demos
# and ACT checkpoints recorded against the old names still load.
#
# Do not delete the alias registrations until every dataset/checkpoint
# referencing the old names has been re-recorded / re-trained.

import gymnasium as gym

from . import g1_grasp_policy_inspire_env_cfg, g1_grasp_policy_inspire_teleop_env_cfg


def _register_with_alias(canonical_id, deprecated_id, env_cfg_entry_point):
    """Register both the canonical gym ID and a deprecated alias against the same env cfg."""
    common_kwargs = {
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "kwargs": {"env_cfg_entry_point": env_cfg_entry_point},
        "disable_env_checker": True,
    }
    gym.register(id=canonical_id, **common_kwargs)
    gym.register(id=deprecated_id, **common_kwargs)


# RL training environment (random block placement).
_register_with_alias(
    canonical_id="Isaac-Grasp-Policy-G129-Inspire-Joint",
    deprecated_id="Isaac-Grasp-Policy-G129-InspireFTP-Joint",
    env_cfg_entry_point=g1_grasp_policy_inspire_env_cfg.G1GraspPolicyInspireEnvCfg,
)

# RL evaluation environment (deterministic placement).
_register_with_alias(
    canonical_id="Isaac-Grasp-Policy-G129-Inspire-Joint-Eval",
    deprecated_id="Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval",
    env_cfg_entry_point=g1_grasp_policy_inspire_env_cfg.G1GraspPolicyInspireEvalEnvCfg,
)

# Teleop environment (PinkIK 38D + AVP dex-retargeting).
_register_with_alias(
    canonical_id="Isaac-Grasp-Policy-G129-Inspire-Teleop",
    deprecated_id="Isaac-Grasp-Policy-G129-InspireFTP-Teleop",
    env_cfg_entry_point=g1_grasp_policy_inspire_teleop_env_cfg.G1GraspPolicyInspireTeleopEnvCfg,
)
