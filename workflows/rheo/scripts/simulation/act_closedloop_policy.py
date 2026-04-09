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

"""
ACT (Action Chunking Transformer) closed-loop policy wrapper for IsaacLab evaluation.

Loads a LeRobot-trained ACT checkpoint and wraps it in the PolicyBase interface
used by the rheo evaluation pipeline.
"""

import sys
from pathlib import Path
from typing import Any

# Resolve scripts/ directory for local imports (Isaac Sim's python.sh may reset PYTHONPATH)
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _import_from_utils(module_name: str):
    """Import a module from scripts/utils/ by absolute path, bypassing namespace collisions."""
    import importlib.util

    fqn = f"utils.{module_name}"
    if fqn in sys.modules:
        return sys.modules[fqn]

    # Ensure the parent 'utils' package is registered in sys.modules
    if "utils" not in sys.modules:
        import types

        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(_SCRIPTS_DIR / "utils")]
        utils_pkg.__package__ = "utils"
        sys.modules["utils"] = utils_pkg

    module_path = _SCRIPTS_DIR / "utils" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(fqn, str(module_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod

import numpy as np
import torch
import yaml
from isaaclab_arena.policy.policy_base import PolicyBase


class ACTClosedloopPolicy(PolicyBase):
    """Closed-loop policy wrapper for LeRobot ACT models.

    Implements the same action chunking pattern as CustomGr00tClosedloopPolicy
    but loads an ACT checkpoint from LeRobot's training pipeline.

    Camera selection and joint groups are driven by the ``experiment:`` section
    in the training config YAML (pointed to by ``experiment_config_path`` in
    the policy config).  See :class:`ACTExperimentConfig` for details.

    Supports both Dex3 (43D sim, 28D policy) and Inspire FTP (41D sim, 26D
    policy).  The hand type is selected via ``hand_type`` in the policy config
    YAML, or inferred from ``sim_action_dim`` if provided.

    Args:
        policy_config_yaml_path: Path to YAML config with model_path, action settings, etc.
        num_envs: Number of parallel environments.
        device: Device to run inference on.
    """

    def __init__(self, policy_config_yaml_path: Path, num_envs: int = 1, device: str = "cuda"):
        with open(policy_config_yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.num_envs = num_envs
        self.device = device
        self.model_path = Path(self.config["model_path"])
        self.action_chunk_length = self.config.get("action_chunk_length", 100)
        self.language_instruction = self.config.get("language_instruction", "pick up block and place in bin")

        # Image target size (H, W, C)
        self.target_image_size = tuple(self.config.get("target_image_size", [480, 640, 3]))

        # Determine hand type and load the appropriate experiment config.
        # "inspire_ftp" uses InspireFTPExperimentConfig (41D sim, 26D policy).
        # "dex3" (default) uses ACTExperimentConfig (43D sim, 28D policy).
        hand_type = self.config.get("hand_type", "dex3")
        if self.config.get("sim_action_dim") == 41:
            hand_type = "inspire_ftp"

        exp_cfg_path = self.config.get("experiment_config_path")
        if hand_type == "inspire_ftp":
            _mod = _import_from_utils("inspire_ftp_experiment_config")
            InspireFTPExperimentConfig = _mod.InspireFTPExperimentConfig

            if exp_cfg_path:
                exp_cfg_path = (Path(policy_config_yaml_path).parent / exp_cfg_path).resolve()
                self.exp_config = InspireFTPExperimentConfig.from_yaml(exp_cfg_path)
            else:
                self.exp_config = InspireFTPExperimentConfig()
            self.sim_action_dim = 41
        else:
            _mod = _import_from_utils("act_experiment_config")
            ACTExperimentConfig = _mod.ACTExperimentConfig

            if exp_cfg_path:
                exp_cfg_path = (Path(policy_config_yaml_path).parent / exp_cfg_path).resolve()
                self.exp_config = ACTExperimentConfig.from_yaml(exp_cfg_path)
            else:
                self.exp_config = ACTExperimentConfig()
            self.sim_action_dim = 43

        self.policy_action_dim = self.exp_config.policy_dim

        # Load ACT policy
        self.policy = self._load_policy()

        # Action chunking state
        self.current_action_chunk = torch.zeros(
            (num_envs, self.action_chunk_length, self.sim_action_dim),
            dtype=torch.float32,
            device=device,
        )
        self.env_requires_new_action_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def _load_policy(self):
        """Load ACT policy from LeRobot checkpoint."""
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        pretrained_path = str(self.model_path)
        policy = ACTPolicy.from_pretrained(pretrained_path)
        policy.eval()
        policy.to(self.device)
        print(f"[ACT] Loaded policy from {pretrained_path}")
        return policy

    def _extract_observations_from_raw(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert raw IsaacLab env observation dict to ACT input format.

        Used when calling the policy directly with raw env observations
        (e.g. from RL training paths).
        """
        body_state = observation["policy"]["robot_joint_state"]  # (B, 87)
        if self.sim_action_dim == 41:
            hand_state = observation["policy"]["robot_inspire_joint_state"]  # (B, 12)
        else:
            hand_state = observation["policy"]["robot_dex3_joint_state"]  # (B, 14)
        state = self.exp_config.extract_state(body_state, hand_state)  # (B, policy_dim)

        camera_obs = observation["camera_images"]
        act_obs = {"observation.state": state.to(self.device)}

        for sim_key, act_key in self.exp_config.cameras.items():
            img = camera_obs[sim_key]  # (B, H, W, C) uint8 on GPU
            img = img.float() / 255.0
            img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
            act_obs[act_key] = img.to(self.device)

        return act_obs

    def _extract_observations_from_processed(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert process_observation() output to ACT input format.

        process_observation() returns keys like "state.left_arm" (B,1,7),
        "video.room_view" (B,1,H,W,3). This method reassembles them into the
        flat ACT format: "observation.state" (B, policy_dim) and
        "observation.images.cam_room" (B, C, H, W).
        """
        # Reassemble joint state from pre-split groups.
        # process_observation adds a time dim (B,1,D) — squeeze it.
        state_parts = []
        for group in self.exp_config.joint_groups:
            key = f"state.{group}"
            s = observation[key]
            if isinstance(s, torch.Tensor):
                s = s.squeeze(1) if s.ndim == 3 else s  # (B,1,D) -> (B,D)
            state_parts.append(s.to(self.device))
        state = torch.cat(state_parts, dim=-1)  # (B, policy_dim)

        act_obs: dict[str, torch.Tensor] = {"observation.state": state}

        # Map video keys to ACT camera keys.
        # process_observation uses "video.room_view", ACT expects "observation.images.cam_room".
        _VIDEO_TO_ACT = {
            "video.room_view": "observation.images.cam_room",
            "video.left_wrist_view": "observation.images.cam_left_wrist",
            "video.right_wrist_view": "observation.images.cam_right_wrist",
        }
        for video_key, act_key in _VIDEO_TO_ACT.items():
            if video_key in observation and act_key in self.exp_config.cameras.values():
                img = observation[video_key]
                if isinstance(img, torch.Tensor):
                    img = img.squeeze(1) if img.ndim == 5 else img  # (B,1,H,W,C) -> (B,H,W,C)
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0
                    img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
                act_obs[act_key] = img.to(self.device)

        return act_obs

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Get a full action chunk for evaluate_episode().

        Accepts the pre-processed observation dict from ``process_observation()``
        (keys like ``state.left_arm``, ``video.room_view``).

        Returns:
            Dict with key ``"actions"`` containing a numpy array of shape
            ``(chunk_size, sim_action_dim)`` where chunk_size matches the ACT
            model's configured chunk_size (default 100, capped to 16 for the
            evaluate_episode action buffer).
        """
        act_obs = self._extract_observations_from_processed(observation)
        sim_actions = self._forward_action_chunk(act_obs)
        # Return first env's chunk as numpy (evaluate_episode handles multi-env via its own loop)
        return {"actions": sim_actions[0].cpu().numpy()}

    @torch.no_grad()
    def _forward_action_chunk(self, act_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run ACT forward pass to get a chunk of actions.

        Args:
            act_obs: Dict in ACT format (observation.state, observation.images.*).

        Returns:
            action_chunk: Shape (num_envs, chunk_size, sim_action_dim)
        """

        # ACT forward: returns (B, chunk_size, action_dim) or processes per-env
        # LeRobot ACT select_action returns (chunk_size, action_dim) for single env
        chunks = []
        for i in range(self.num_envs):
            single_obs = {k: v[i : i + 1] for k, v in act_obs.items()}
            action = self.policy.select_action(single_obs)  # (chunk_size, action_dim)
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            chunks.append(action)

        # Stack: (num_envs, chunk_size, policy_dim)
        policy_actions = torch.stack(chunks, dim=0).to(self.device)

        # Scatter policy_dim → sim_action_dim at correct sim joint positions
        sim_actions = self.exp_config.scatter_to_sim(policy_actions)

        # Truncate or pad to action_chunk_length
        chunk_len = sim_actions.shape[1]
        if chunk_len >= self.action_chunk_length:
            sim_actions = sim_actions[:, : self.action_chunk_length, :]
        else:
            # Pad with last action repeated
            repeat_count = self.action_chunk_length - chunk_len
            last_action = sim_actions[:, -1:, :].expand(-1, repeat_count, -1)
            sim_actions = torch.cat([sim_actions, last_action], dim=1)

        return sim_actions.float()

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset action chunking state for specified envs."""
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True
        # Reset ACT's internal action queue
        if hasattr(self.policy, "reset"):
            self.policy.reset()
