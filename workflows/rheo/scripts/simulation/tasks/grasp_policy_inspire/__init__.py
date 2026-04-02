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

import gymnasium as gym

from . import g1_grasp_policy_inspire_env_cfg

# RL training environment (random block placement)
gym.register(
    id="Isaac-Grasp-Policy-G129-InspireFTP-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": g1_grasp_policy_inspire_env_cfg.G1GraspPolicyInspireEnvCfg},
    disable_env_checker=True,
)

# RL evaluation environment (deterministic placement)
gym.register(
    id="Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": g1_grasp_policy_inspire_env_cfg.G1GraspPolicyInspireEvalEnvCfg},
    disable_env_checker=True,
)
