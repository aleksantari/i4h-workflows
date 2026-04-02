# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_registered = False


def register() -> None:
    """Register i4h extensions into RLinf.

    This function is called automatically by RLinf's Worker._load_user_extensions()
    when RLINF_EXT_MODULE=rlinf_ext is set in the environment.

    It performs the following registrations:
    1. Imports i4h's IsaacLab task packages (triggers gym.register calls)
    2. Registers task IDs into RLinf's REGISTER_ISAACLAB_ENVS map
    3. Registers GR00T obs/action converters for dex3
    4. Registers GR00T data config for new_embodiment
    5. Monkeypatches RLinf's get_model to support new_embodiment
    6. Registers ACT obs/action converters and model factory
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("rlinf_ext: Registering i4h extensions...")

    _register_gr00t_converters()
    _register_act_converters()
    _register_inspire_ftp_act_converters()

    _register_gr00t_data_config()

    _patch_gr00t_get_model()
    _register_act_model()

    _register_isaaclab_envs()

    logger.info("rlinf_ext: Registration complete.")


def _register_isaaclab_envs() -> None:
    """Register i4h task IDs into RLinf's REGISTER_ISAACLAB_ENVS map."""
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    # Use factory function to get the class with proper inheritance
    IsaaclabG129Dx3Env = _get_g129_dex3_env_class()

    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Assemble-Trocar-G129-Dex3-Joint", IsaaclabG129Dx3Env)
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Assemble-Trocar-G129-Dex3-Joint-Eval", IsaaclabG129Dx3Env)

    # Grasp policy task
    IsaaclabGraspPolicyEnv = _get_grasp_policy_env_class()
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Grasp-Policy-G129-Dex3-Joint", IsaaclabGraspPolicyEnv)
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Grasp-Policy-G129-Dex3-Joint-Eval", IsaaclabGraspPolicyEnv)

    # Inspire FTP grasp policy task
    IsaaclabGraspPolicyInspireEnv = _get_grasp_policy_inspire_env_class()
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Grasp-Policy-G129-InspireFTP-Joint", IsaaclabGraspPolicyInspireEnv)
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval", IsaaclabGraspPolicyInspireEnv)

    logger.debug(f"rlinf_ext: Registered ISAACLAB_ENVS: {list(REGISTER_ISAACLAB_ENVS.keys())}")


def _register_gr00t_converters() -> None:
    """Register GR00T obs/action converters for dex3."""
    from rlinf.models.embodiment.gr00t import simulation_io

    simulation_io.OBS_CONVERSION.setdefault("dex3", _convert_dex3_obs_to_gr00t_format)
    simulation_io.ACTION_CONVERSION.setdefault("dex3", _convert_to_dex3_action)
    logger.debug("rlinf_ext: Registered dex3 obs/action converters")


def _register_gr00t_data_config() -> None:
    """Register GR00T data config for new_embodiment (UnitreeG1Sim)."""
    # Import i4h's data config which adds to DATA_CONFIG_MAP
    import policy.gr00t_config  # noqa: F401

    logger.debug("rlinf_ext: Registered UnitreeG1SimDataConfig")


# ---------------------------------------------------------------------------
# IsaacLab env wrapper for G129 + Dex3
# ---------------------------------------------------------------------------


def _get_g129_dex3_env_class():
    """Factory function to create IsaaclabG129Dx3Env class with proper inheritance.

    This delays the import of IsaaclabBaseEnv until the class is actually needed,
    avoiding import-time side effects in worker processes that don't use IsaacLab.
    """

    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    class IsaaclabG129Dx3Env(IsaaclabBaseEnv):
        """Env wrapper for G1 (29DoF) + Dex3 tasks."""

        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _make_env_function(self):
            def make_env_isaaclab():
                from isaaclab.app import AppLauncher

                sim_app = AppLauncher(headless=True, enable_cameras=True).app
                import gymnasium as gym
                import simulation.tasks.assemble_trocar  # noqa: F401 - triggers gym.register()
                from isaaclab_tasks.utils import load_cfg_from_registry

                isaac_env_cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

                env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
                return env, sim_app

            return make_env_isaaclab

        def _wrap_obs(self, obs):
            left_wrist = obs["camera_images"]["left_wrist_camera"]
            right_wrist = obs["camera_images"]["right_wrist_camera"]
            front = obs["camera_images"]["front_camera"]

            dex3_states = obs["policy"]["robot_dex3_joint_state"]  # (B, 14)
            g129_shoulder_states = obs["policy"]["robot_joint_state"][:, 15:29]  # (B, 14)
            states = torch.concatenate([g129_shoulder_states, dex3_states], dim=-1)  # (B, 28)

            task_descriptions = [self.task_description] * self.num_envs
            extra_view_images = torch.stack([left_wrist, right_wrist], dim=1)  # (B, 2, H, W, C)

            return {
                "main_images": front,
                "extra_view_images": extra_view_images,
                "states": states,
                "task_descriptions": task_descriptions,
                "camera_images_raw": obs["camera_images"],
            }

        def add_image(self, obs):
            """Create a grid of images for video logging."""
            imgs = obs["camera_images"]["front_camera"].cpu().numpy()
            num_envs = imgs.shape[0]

            grid_cols = int(np.ceil(np.sqrt(num_envs)))
            grid_rows = int(np.ceil(num_envs / grid_cols))
            img_h, img_w = imgs.shape[1:3]

            grid_img = np.zeros((grid_rows * img_h, grid_cols * img_w, 3), dtype=np.uint8)

            for idx in range(num_envs):
                row, col = idx // grid_cols, idx % grid_cols
                y0, x0 = row * img_h, col * img_w
                grid_img[y0 : y0 + img_h, x0 : x0 + img_w] = imgs[idx]
                cv2.putText(
                    grid_img,
                    f"Env {idx}",
                    (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            return grid_img

    return IsaaclabG129Dx3Env


# ---------------------------------------------------------------------------
# GR00T obs/action converters
# ---------------------------------------------------------------------------


def _convert_dex3_obs_to_gr00t_format(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert RLinf env observations into the dict expected by GR00T transforms.

    Expected input schema comes from IsaaclabG129Dx3Env._wrap_obs():
      - main_images: (B, H, W, C) torch tensor
      - extra_view_images: (B, 2, H, W, C) torch tensor [left_wrist, right_wrist]
      - states: (B, 28) torch tensor [left_arm(7), right_arm(7), left_hand(7), right_hand(7)]
      - task_descriptions: list[str] length B
    """
    import torch

    main = env_obs["main_images"]
    extra = env_obs["extra_view_images"]
    states = env_obs["states"]

    if isinstance(main, torch.Tensor):
        # (B, H, W, C) -> (B, T=1, H, W, C)
        room_view = main.unsqueeze(1).cpu().numpy()
        left_wrist = extra[:, 0].unsqueeze(1).cpu().numpy()
        right_wrist = extra[:, 1].unsqueeze(1).cpu().numpy()
        st = states.unsqueeze(1).cpu().numpy()
    else:
        raise TypeError(f"Expected torch.Tensor observations, got {type(main)=}")

    return {
        "video.left_wrist_view": left_wrist,
        "video.right_wrist_view": right_wrist,
        "video.room_view": room_view,
        "state.left_arm": st[:, :, :7],
        "state.right_arm": st[:, :, 7:14],
        "state.left_hand": st[:, :, 14:21],
        "state.right_hand": st[:, :, 21:],
        "annotation.human.action.task_description": env_obs["task_descriptions"],
    }


def _convert_to_dex3_action(action_chunk: dict[str, Any], chunk_size: int = 1) -> Any:
    """Convert GR00T action dict into an action tensor for the IsaacLab env.

    Mirrors the padding behavior:
    - concatenate all action parts along last dim
    - pad 15 zeros at the *front* to align with the full robot joint action space
    """
    import numpy as np

    parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
    action_concat = np.concatenate(parts, axis=-1)
    return np.pad(
        action_concat,
        ((0, 0), (0, 0), (15, 0)),
        mode="constant",
        constant_values=0,
    )


# ---------------------------------------------------------------------------
# Monkeypatch RLinf's get_model to support new_embodiment
# ---------------------------------------------------------------------------


def _patch_embodiment_tags() -> None:
    """Add NEW_EMBODIMENT to RLinf's EmbodimentTag enum and mapping."""
    from rlinf.models.embodiment.gr00t import embodiment_tags

    # Check if NEW_EMBODIMENT already exists
    if not hasattr(embodiment_tags.EmbodimentTag, "NEW_EMBODIMENT"):
        from enum import Enum

        existing_members = {e.name: e.value for e in embodiment_tags.EmbodimentTag}
        existing_members["NEW_EMBODIMENT"] = "new_embodiment"
        NewEmbodimentTag = Enum("EmbodimentTag", existing_members)

        # Replace the old enum with the new one
        embodiment_tags.EmbodimentTag = NewEmbodimentTag

    # Add to mapping if not present
    if "new_embodiment" not in embodiment_tags.EMBODIMENT_TAG_MAPPING:
        embodiment_tags.EMBODIMENT_TAG_MAPPING["new_embodiment"] = 31


def _patch_gr00t_get_model() -> None:
    """Monkeypatch RLinf's GR00T get_model to support new_embodiment."""
    # First, ensure NEW_EMBODIMENT is in the enum
    _patch_embodiment_tags()

    import rlinf.models.embodiment.gr00t as rlinf_gr00t_mod

    original_get_model = rlinf_gr00t_mod.get_model

    def patched_get_model(cfg, torch_dtype=None):  # type: ignore[no-redef]
        import torch

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # If not new_embodiment, use original logic
        if cfg.embodiment_tag != "new_embodiment":
            return original_get_model(cfg, torch_dtype=torch_dtype)

        # Handle new_embodiment: use i4h's UnitreeG1SimDataConfig
        from pathlib import Path

        from gr00t.experiment.data_config import load_data_config
        from rlinf.models.embodiment.gr00t.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity
        from rlinf.utils.patcher import Patcher

        # Apply RLinf's standard EmbodimentTag patches
        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()

        # Load i4h's data config
        data_config = load_data_config("policy.gr00t_config:UnitreeG1SimDataConfig")
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
        model.to(torch_dtype)
        if cfg.rl_head_config.add_value_head:
            model.action_head.value_head._init_weights()
        if cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)

        logger.debug("rlinf_ext: Loaded GR00T model with new_embodiment")
        return model

    rlinf_gr00t_mod.get_model = patched_get_model  # type: ignore[assignment]
    logger.debug("rlinf_ext: Patched get_model for new_embodiment support")


# ---------------------------------------------------------------------------
# Grasp policy environment wrapper
# ---------------------------------------------------------------------------


def _get_grasp_policy_env_class():
    """Factory function to create IsaaclabGraspPolicyEnv class with proper inheritance."""

    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    class IsaaclabGraspPolicyEnv(IsaaclabBaseEnv):
        """Env wrapper for G1 (29DoF) + Dex3 grasp policy task."""

        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _make_env_function(self):
            def make_env_isaaclab():
                from isaaclab.app import AppLauncher

                sim_app = AppLauncher(headless=True, enable_cameras=True).app
                import gymnasium as gym
                import simulation.tasks.grasp_policy  # noqa: F401 - triggers gym.register()
                from isaaclab_tasks.utils import load_cfg_from_registry

                isaac_env_cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

                env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
                return env, sim_app

            return make_env_isaaclab

        def _wrap_obs(self, obs):
            left_wrist = obs["camera_images"]["left_wrist_camera"]
            right_wrist = obs["camera_images"]["right_wrist_camera"]
            front = obs["camera_images"]["front_camera"]

            dex3_states = obs["policy"]["robot_dex3_joint_state"]  # (B, 14)
            g129_shoulder_states = obs["policy"]["robot_joint_state"][:, 15:29]  # (B, 14)
            states = torch.concatenate([g129_shoulder_states, dex3_states], dim=-1)  # (B, 28)

            task_descriptions = [self.task_description] * self.num_envs
            extra_view_images = torch.stack([left_wrist, right_wrist], dim=1)  # (B, 2, H, W, C)

            return {
                "main_images": front,
                "extra_view_images": extra_view_images,
                "states": states,
                "task_descriptions": task_descriptions,
                "camera_images_raw": obs["camera_images"],
            }

        def add_image(self, obs):
            """Create a grid of images for video logging."""
            imgs = obs["camera_images"]["front_camera"].cpu().numpy()
            num_envs = imgs.shape[0]

            grid_cols = int(np.ceil(np.sqrt(num_envs)))
            grid_rows = int(np.ceil(num_envs / grid_cols))
            img_h, img_w = imgs.shape[1:3]

            grid_img = np.zeros((grid_rows * img_h, grid_cols * img_w, 3), dtype=np.uint8)

            for idx in range(num_envs):
                row, col = idx // grid_cols, idx % grid_cols
                y0, x0 = row * img_h, col * img_w
                grid_img[y0 : y0 + img_h, x0 : x0 + img_w] = imgs[idx]
                cv2.putText(
                    grid_img,
                    f"Env {idx}",
                    (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            return grid_img

    return IsaaclabGraspPolicyEnv


# ---------------------------------------------------------------------------
# ACT obs/action converters
# ---------------------------------------------------------------------------


def _register_act_converters() -> None:
    """Register ACT obs/action converters for dex3."""
    from rlinf.models.embodiment.gr00t import simulation_io

    simulation_io.OBS_CONVERSION.setdefault("act", _convert_dex3_obs_to_act_format)
    simulation_io.ACTION_CONVERSION.setdefault("act", _convert_act_action_to_sim)
    logger.debug("rlinf_ext: Registered ACT obs/action converters")


def _convert_dex3_obs_to_act_format(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert RLinf env observations into the dict expected by ACT.

    Uses :class:`ACTExperimentConfig` to select which cameras and joint-state
    subset to pass to the ACT model.  Falls back to all cameras / full 28-D
    state when no experiment config is set (backward compatible).

    Input from _wrap_obs():
      - camera_images_raw: dict[str, (B, H, W, C)] — raw camera tensors by sim key
      - states: (B, 28) torch tensor (full arms + hands)
      - (legacy) main_images / extra_view_images — used only when camera_images_raw absent

    Output (ACT format):
      - observation.images.<cam>: (B, C, H, W) float tensor  (per selected camera)
      - observation.state: (B, policy_dim) float tensor
    """
    from utils.act_experiment_config import ACTExperimentConfig

    exp_config = ACTExperimentConfig.from_env_or_default()

    # --- Camera images -------------------------------------------------------
    act_obs: dict[str, Any] = {}
    raw_cameras = env_obs.get("camera_images_raw")
    if raw_cameras is not None:
        # Config-driven path: pick cameras by sim key name
        for sim_key, act_key in exp_config.cameras.items():
            img = raw_cameras[sim_key]  # (B, H, W, C)
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for {sim_key}, got {type(img)}")
            act_obs[act_key] = img.float().permute(0, 3, 1, 2) / 255.0
    else:
        # Legacy fallback: positional extraction from main_images / extra_view_images
        main = env_obs["main_images"]
        extra = env_obs["extra_view_images"]
        if not isinstance(main, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor observations, got {type(main)=}")
        act_obs["observation.images.cam_room"] = main.float().permute(0, 3, 1, 2) / 255.0
        act_obs["observation.images.cam_left_wrist"] = extra[:, 0].float().permute(0, 3, 1, 2) / 255.0
        act_obs["observation.images.cam_right_wrist"] = extra[:, 1].float().permute(0, 3, 1, 2) / 255.0

    # --- Joint state ---------------------------------------------------------
    states = env_obs["states"]  # (B, 28) full state from _wrap_obs
    if exp_config.policy_dim == 28:
        # Using all groups — pass through directly
        act_obs["observation.state"] = states.float()
    else:
        # Subset: _wrap_obs produces [left_arm(7), right_arm(7), left_hand(7), right_hand(7)]
        # Select the groups requested by experiment config.
        group_slices = {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 21),
            "right_hand": slice(21, 28),
        }
        parts = [states[:, group_slices[g]] for g in exp_config.joint_groups]
        act_obs["observation.state"] = torch.cat(parts, dim=-1).float()

    return act_obs


def _convert_act_action_to_sim(action_chunk: dict[str, Any] | np.ndarray, chunk_size: int = 1) -> Any:
    """Convert ACT action output into an action tensor for the IsaacLab env.

    Uses :class:`ACTExperimentConfig` to scatter policy-dim actions into the
    correct positions of the 43-DOF sim action space.
    """
    from utils.act_experiment_config import ACTExperimentConfig

    if isinstance(action_chunk, dict):
        action = action_chunk.get("action", action_chunk.get("actions"))
        if action is None:
            parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
            action = np.concatenate(parts, axis=-1)
        else:
            action = action[:, :chunk_size, :]
    else:
        action = action_chunk[:, :chunk_size, :]

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    exp_config = ACTExperimentConfig.from_env_or_default()
    return exp_config.scatter_to_sim_numpy(action)


# ---------------------------------------------------------------------------
# Register ACT model factory into RLinf
# ---------------------------------------------------------------------------


def _register_act_model() -> None:
    """Register ACT's get_model factory so RLinf can load ACT models.

    When cfg.model_type == "act", RLinf dispatches to our get_model function.
    """
    try:
        from rlinf_ext.act_policy import get_model as act_get_model

        # Register into RLinf's model registry if it has one,
        # otherwise monkeypatch via the existing pattern
        import rlinf.models.embodiment as embodiment_mod

        if hasattr(embodiment_mod, "MODEL_REGISTRY"):
            embodiment_mod.MODEL_REGISTRY.setdefault("act", act_get_model)
        else:
            # Store reference for use in patched get_model dispatcher
            if not hasattr(embodiment_mod, "_act_get_model"):
                embodiment_mod._act_get_model = act_get_model

        logger.debug("rlinf_ext: Registered ACT model factory")
    except ImportError:
        logger.warning("rlinf_ext: Could not import act_policy. ACT RL training unavailable.")


# ---------------------------------------------------------------------------
# Inspire FTP grasp policy environment wrapper
# ---------------------------------------------------------------------------


def _get_grasp_policy_inspire_env_class():
    """Factory function to create IsaaclabGraspPolicyInspireEnv class."""

    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    class IsaaclabGraspPolicyInspireEnv(IsaaclabBaseEnv):
        """Env wrapper for G1 (29DoF) + Inspire FTP grasp policy task."""

        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _make_env_function(self):
            def make_env_isaaclab():
                from isaaclab.app import AppLauncher

                sim_app = AppLauncher(headless=True, enable_cameras=True).app
                import gymnasium as gym
                import simulation.tasks.grasp_policy_inspire  # noqa: F401 - triggers gym.register()
                from isaaclab_tasks.utils import load_cfg_from_registry

                isaac_env_cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

                env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
                return env, sim_app

            return make_env_isaaclab

        def _wrap_obs(self, obs):
            front = obs["camera_images"]["front_camera"]

            inspire_states = obs["policy"]["robot_inspire_joint_state"]  # (B, 12)
            g129_shoulder_states = obs["policy"]["robot_joint_state"][:, 15:29]  # (B, 14)
            states = torch.concatenate([g129_shoulder_states, inspire_states], dim=-1)  # (B, 26)

            task_descriptions = [self.task_description] * self.num_envs

            return {
                "main_images": front,
                "states": states,
                "task_descriptions": task_descriptions,
                "camera_images_raw": obs["camera_images"],
            }

        def add_image(self, obs):
            """Create a grid of images for video logging."""
            imgs = obs["camera_images"]["front_camera"].cpu().numpy()
            num_envs = imgs.shape[0]

            grid_cols = int(np.ceil(np.sqrt(num_envs)))
            grid_rows = int(np.ceil(num_envs / grid_cols))
            img_h, img_w = imgs.shape[1:3]

            grid_img = np.zeros((grid_rows * img_h, grid_cols * img_w, 3), dtype=np.uint8)

            for idx in range(num_envs):
                row, col = idx // grid_cols, idx % grid_cols
                y0, x0 = row * img_h, col * img_w
                grid_img[y0 : y0 + img_h, x0 : x0 + img_w] = imgs[idx]
                cv2.putText(
                    grid_img,
                    f"Env {idx}",
                    (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            return grid_img

    return IsaaclabGraspPolicyInspireEnv


# ---------------------------------------------------------------------------
# Inspire FTP ACT obs/action converters
# ---------------------------------------------------------------------------


def _register_inspire_ftp_act_converters() -> None:
    """Register Inspire FTP ACT obs/action converters."""
    from rlinf.models.embodiment.gr00t import simulation_io

    simulation_io.OBS_CONVERSION.setdefault("act_inspire_ftp", _convert_inspire_obs_to_act_format)
    simulation_io.ACTION_CONVERSION.setdefault("act_inspire_ftp", _convert_inspire_act_action_to_sim)
    logger.debug("rlinf_ext: Registered Inspire FTP ACT obs/action converters")


def _convert_inspire_obs_to_act_format(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert RLinf env observations into the dict expected by ACT (Inspire FTP).

    Input from _wrap_obs():
      - camera_images_raw: dict[str, (B, H, W, C)] — raw camera tensors
      - states: (B, 26) torch tensor (arms + actuated hands)

    Output (ACT format):
      - observation.images.cam_room: (B, C, H, W) float tensor
      - observation.state: (B, 26) float tensor
    """
    from utils.inspire_ftp_experiment_config import InspireFTPExperimentConfig

    exp_config = InspireFTPExperimentConfig.from_env_or_default()

    act_obs: dict[str, Any] = {}
    raw_cameras = env_obs.get("camera_images_raw")
    if raw_cameras is not None:
        for sim_key, act_key in exp_config.cameras.items():
            img = raw_cameras[sim_key]  # (B, H, W, C)
            act_obs[act_key] = img.float().permute(0, 3, 1, 2) / 255.0
    else:
        main = env_obs["main_images"]
        act_obs["observation.images.cam_room"] = main.float().permute(0, 3, 1, 2) / 255.0

    states = env_obs["states"]  # (B, 26) from _wrap_obs
    act_obs["observation.state"] = states.float()

    return act_obs


def _convert_inspire_act_action_to_sim(action_chunk: dict[str, Any] | np.ndarray, chunk_size: int = 1) -> Any:
    """Convert ACT action output into an action tensor for the Inspire FTP env.

    Uses InspireFTPExperimentConfig to scatter 26D policy actions into 53D sim space.
    """
    from utils.inspire_ftp_experiment_config import InspireFTPExperimentConfig

    if isinstance(action_chunk, dict):
        action = action_chunk.get("action", action_chunk.get("actions"))
        if action is None:
            parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
            action = np.concatenate(parts, axis=-1)
        else:
            action = action[:, :chunk_size, :]
    else:
        action = action_chunk[:, :chunk_size, :]

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    exp_config = InspireFTPExperimentConfig.from_env_or_default()
    return exp_config.scatter_to_sim_numpy(action)
