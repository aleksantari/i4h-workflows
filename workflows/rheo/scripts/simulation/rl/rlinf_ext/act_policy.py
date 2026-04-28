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
ACT (Action Chunking Transformer) wrapper for RLinf RL post-training.

Wraps a LeRobot ACT checkpoint into RLinf's BasePolicy interface so it can
be used with PPO/SAC training via the RLinf framework.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rlinf.models.embodiment.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """Simple MLP value head for critic estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _init_weights(self):
        """Initialize weights with small values for stable RL training."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ACTForRLActionPrediction(BasePolicy, nn.Module):
    """ACT policy wrapped for RLinf RL training.

    This class bridges LeRobot's ACTPolicy with RLinf's BasePolicy interface.
    It implements the required `default_forward` and `predict_action_batch`
    methods and optionally attaches a value head for critic-based RL algorithms.

    Args:
        act_policy: The loaded LeRobot ACT policy (nn.Module).
        action_dim: Output action dimension (26 for Inspire FTP dual-arm grasp).
        num_action_chunks: Number of action steps to predict per forward pass.
        add_value_head: Whether to attach a ValueHead for RL critic.
        obs_converter_type: Key for obs/action converter lookup in simulation_io.
    """

    def __init__(
        self,
        act_policy: nn.Module,
        action_dim: int = 26,
        num_action_chunks: int = 1,
        add_value_head: bool = True,
        obs_converter_type: str = "act_inspire",
    ):
        nn.Module.__init__(self)
        self.act_policy = act_policy
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.obs_converter_type = obs_converter_type

        # Value head for critic (used by PPO)
        if add_value_head:
            # Estimate latent dim from ACT's encoder output
            latent_dim = getattr(act_policy.config, "dim_model", 512)
            self.value_head = ValueHead(input_dim=latent_dim)
            self.value_head._init_weights()
        else:
            self.value_head = None

    def default_forward(self, **kwargs) -> dict[str, Any]:
        """Forward pass for RL training (computes actions + log_probs + values).

        Expected kwargs from RLinf:
            observations: dict with model-formatted observations
            actions: optional previous actions for log_prob computation

        Returns:
            Dict with keys: actions, log_probs, values, entropy
        """
        observations = kwargs.get("observations", {})

        # Convert RLinf observations to ACT input format
        act_obs = self._prepare_act_input(observations)

        # Get action predictions from ACT
        with torch.set_grad_enabled(self.training):
            action_output = self.act_policy(act_obs)

        # Extract actions (B, chunk_size, action_dim)
        if isinstance(action_output, dict):
            actions = action_output.get("action", action_output.get("actions"))
        else:
            actions = action_output

        # Take first num_action_chunks actions
        if actions.dim() == 3:
            actions = actions[:, : self.num_action_chunks, :]
        elif actions.dim() == 2:
            actions = actions.unsqueeze(1)

        result = {"actions": actions}

        # Compute value estimate if we have a value head
        if self.value_head is not None:
            # Use ACT's encoder features as value input
            # Fall back to flattened state if encoder features unavailable
            if hasattr(self.act_policy, "model") and hasattr(self.act_policy.model, "encoder"):
                with torch.no_grad():
                    encoder_out = self._get_encoder_features(act_obs)
                values = self.value_head(encoder_out)
            else:
                # Use state observation as fallback
                state = observations.get("states", torch.zeros(actions.shape[0], self.action_dim, device=actions.device))
                if state.dim() > 2:
                    state = state.squeeze(1)
                values = self.value_head(state)
            result["values"] = values.squeeze(-1)

        # Log probs and entropy: ACT uses deterministic actions during inference,
        # but for RL we need stochasticity. Use the VAE's latent distribution
        # if available, otherwise use a Gaussian approximation.
        if hasattr(self.act_policy, "model") and hasattr(self.act_policy.model, "latent_mean"):
            latent_mean = self.act_policy.model.latent_mean
            latent_log_var = self.act_policy.model.latent_log_var
            latent_std = torch.exp(0.5 * latent_log_var)
            dist = torch.distributions.Normal(latent_mean, latent_std)
            result["log_probs"] = dist.log_prob(dist.rsample()).sum(-1)
            result["entropy"] = dist.entropy().sum(-1)
        else:
            # Gaussian approximation for log_probs
            batch_size = actions.shape[0]
            result["log_probs"] = torch.zeros(batch_size, device=actions.device)
            result["entropy"] = torch.zeros(batch_size, device=actions.device)

        return result

    def predict_action_batch(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Inference-only action prediction for RLinf rollouts.

        Returns:
            Tuple of (actions_np, info_dict) where:
                - actions_np: numpy array of shape (B, chunk_size, action_dim)
                - info_dict: metadata including forward_inputs, prev_logprobs, prev_values
        """
        observations = kwargs.get("observations", {})

        act_obs = self._prepare_act_input(observations)

        with torch.no_grad():
            action_output = self.act_policy(act_obs)

        if isinstance(action_output, dict):
            actions = action_output.get("action", action_output.get("actions"))
        else:
            actions = action_output

        if actions.dim() == 3:
            actions = actions[:, : self.num_action_chunks, :]
        elif actions.dim() == 2:
            actions = actions.unsqueeze(1)

        actions_np = actions.cpu().numpy()

        # Build info dict expected by RLinf
        info = {
            "forward_inputs": observations,
            "prev_logprobs": np.zeros(actions_np.shape[0]),
            "prev_values": np.zeros(actions_np.shape[0]),
        }

        return actions_np, info

    def _prepare_act_input(self, observations: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert RLinf-format observations to ACT input format.

        RLinf observations (from obs converter):
            video.*: (B, T, H, W, C) numpy or tensor
            state.*: (B, T, D) numpy or tensor

        ACT expects:
            observation.images.*: (B, C, H, W) tensor
            observation.state: (B, D) tensor
        """
        act_obs = {}

        # Concatenate state parts into flat vector (config-driven joint groups)
        from utils.inspire.inspire_experiment_config import InspireExperimentConfig

        exp_config = InspireExperimentConfig.from_env_or_default()

        state_parts = []
        for key in exp_config.rlinf_state_keys():
            if key in observations:
                val = observations[key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                if val.dim() == 3:
                    val = val[:, 0, :]  # Remove temporal dim
                state_parts.append(val)
        if state_parts:
            act_obs["observation.state"] = torch.cat(state_parts, dim=-1)

        # Convert video observations to (B, C, H, W) (config-driven cameras)
        video_key_map = exp_config.rlinf_video_keys()
        for rlinf_key, act_key in video_key_map.items():
            if rlinf_key in observations:
                img = observations[rlinf_key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                # (B, T, H, W, C) -> (B, H, W, C) -> (B, C, H, W)
                if img.dim() == 5:
                    img = img[:, 0]
                if img.shape[-1] in (3, 4):  # HWC format
                    img = img.permute(0, 3, 1, 2)
                img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
                act_obs[act_key] = img

        return act_obs

    def _get_encoder_features(self, act_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract encoder features from ACT for value estimation."""
        # This is model-architecture dependent. For ACT, we can use the
        # encoder output before the action decoder.
        # Fall back to state concatenation if encoder is not accessible.
        if "observation.state" in act_obs:
            return act_obs["observation.state"]
        return torch.zeros(1, 512)


def get_model(cfg, torch_dtype=None):
    """Factory function for RLinf model loading.

    This is the entry point called by RLinf's model registry.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    from lerobot.common.policies.act.modeling_act import ACTPolicy

    model_path = cfg.model_path
    logger.info(f"Loading ACT model from {model_path}")

    act_policy = ACTPolicy.from_pretrained(model_path)
    act_policy.to(torch_dtype)

    model = ACTForRLActionPrediction(
        act_policy=act_policy,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        add_value_head=cfg.add_value_head,
        obs_converter_type=cfg.obs_converter_type,
    )
    model.to(torch_dtype)

    if cfg.add_value_head and model.value_head is not None:
        model.value_head._init_weights()

    logger.info("ACT model loaded for RL training")
    return model
