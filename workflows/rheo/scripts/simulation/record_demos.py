#!/usr/bin/env python3
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
Record demonstrations with Isaac Lab environments using human teleoperation.

Adapted from IsaacLab's ``scripts/tools/record_demos.py`` with project-specific
task registration (grasp_policy, assemble_trocar).

Features over the old ``record_demos_assemble_trocar.py``:
- **Auto-success detection** — evaluates the env's ``success`` termination every
  step; auto-saves after ``--num_success_steps`` consecutive successes.
- **VR gesture controls** — OpenXR hand-tracking device fires START / STOP / RESET
  callbacks via gestures (no keyboard needed from inside the headset).
- **XR UI overlays** — demo count shown as 3-D text in the headset.

Usage:
    # AVP hand-tracking (grasp_policy)
    python record_demos.py --task Isaac-Grasp-Policy-G129-Dex3-Teleop \\
        --teleop_device handtracking --enable_pinocchio --enable_cameras \\
        --dataset_file ./datasets/grasp_policy/demo.hdf5 --num_demos 5

    # AVP hand-tracking (trocar)
    python record_demos.py --task Isaac-Assemble-Trocar-G129-Dex3-Teleop \\
        --teleop_device handtracking --enable_pinocchio --enable_cameras \\
        --dataset_file ./datasets/trocar/demo.hdf5 --num_demos 5

    # Keyboard (desktop, any task)
    python record_demos.py --task Isaac-Grasp-Policy-G129-Dex3-Teleop \\
        --teleop_device keyboard --enable_pinocchio \\
        --dataset_file ./datasets/grasp_policy/test.hdf5 --num_demos 1
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--object",
    type=str,
    default="tool_0",
    choices=["tool_0", "tool_1", "tool_2", "tool_3", "tool_4"],
    help="Grasp object (default: tool_0).",
)
parser.add_argument(
    "--slot",
    type=int,
    default=4,
    choices=range(6),
    help="Tray slot index 0-5 for tool spawn (default: 4).",
)
parser.add_argument(
    "--arm",
    type=str,
    default="both",
    choices=["left", "right", "both"],
    help="Which arm(s) to control during teleop. Non-controlled arm is locked at idle pose (default: both).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import logging
import os
import time

import gymnasium as gym
import torch

import omni.ui as ui

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

from collections.abc import Callable

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Register project-specific gym environments
from simulation.tasks import assemble_trocar  # noqa: F401
from simulation.tasks import grasp_policy  # noqa: F401
from simulation.tasks import grasp_policy_inspire  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    # parse configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Copy camera observations into policy group so they get recorded
    if args_cli.enable_cameras and hasattr(env_cfg, "observations"):
        obs_cfg = env_cfg.observations
        if hasattr(obs_cfg, "camera_images") and obs_cfg.camera_images is not None:
            for name in ("front_camera", "left_wrist_camera", "right_wrist_camera"):
                if hasattr(obs_cfg.camera_images, name):
                    setattr(obs_cfg.policy, name, getattr(obs_cfg.camera_images, name))

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(callbacks: dict[str, Callable]) -> object:
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, handtracking")
                exit(1)

            # Add callbacks to fallback device
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


# 38D hand joint indices per hand. The USD ordering interleaves left/right,
# so a contiguous slice like [14:26] crosses hand boundaries. These explicit
# index lists match the USD articulation order from joint_names[29:].
_LEFT_HAND_38D_IDX = [14, 15, 16, 17, 18, 24, 25, 26, 27, 28, 34, 36]
_RIGHT_HAND_38D_IDX = [19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 35, 37]


def _read_frozen_wrist_fk(env, arm: str) -> torch.Tensor | None:
    """Read the actual FK wrist pose for the arm that should be frozen.

    Returns a 7D tensor [pos(3), quat(4)] in world frame, or None if arm=="both".
    Must be called after env.reset() so the robot is at its default joint state.
    """
    if arm == "both":
        return None
    robot = env.scene["robot"]
    # Ensure FK is computed from the current joint state
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    body_names = list(robot.data.body_names)
    if arm == "right":
        # Freeze LEFT arm — read left wrist FK
        idx = body_names.index("left_wrist_yaw_link")
    else:
        # Freeze RIGHT arm — read right wrist FK
        idx = body_names.index("right_wrist_yaw_link")
    pos = robot.data.body_pos_w[0, idx].clone()
    quat = robot.data.body_quat_w[0, idx].clone()
    return torch.cat([pos, quat])


def handle_reset(
    env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str
) -> int:
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
) -> int:
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = not args_cli.xr

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    # Set up teleoperation callbacks
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks)
    teleop_interface.add_callback("R", reset_recording_instance)

    # Reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()
    frozen_arm_wrist_target = _read_frozen_wrist_fk(env, args_cli.arm)

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Get teleop command
            action = teleop_interface.advance()
            # Lock frozen arm for single-arm teleop. Uses exact FK wrist pose
            # (not the approximate idle_action values) to prevent drift, and
            # explicit per-hand index lists because the USD ordering interleaves
            # left/right hand joints (contiguous slices cross hand boundaries).
            if args_cli.arm != "both" and frozen_arm_wrist_target is not None:
                idle = env_cfg.idle_action
                if args_cli.arm == "right":
                    # Freeze left arm
                    action[0:7] = frozen_arm_wrist_target
                    for i in _LEFT_HAND_38D_IDX:
                        action[i] = idle[i]
                elif args_cli.arm == "left":
                    # Freeze right arm
                    action[7:14] = frozen_arm_wrist_target
                    for i in _RIGHT_HAND_38D_IDX:
                        action[i] = idle[i]
            # Zero-pad if teleop device outputs fewer dims than the action space
            expected_dim = env.action_space.shape[-1]
            if action.shape[0] < expected_dim:
                action = torch.cat(
                    [action, torch.zeros(expected_dim - action.shape[0], device=action.device, dtype=action.dtype)]
                )
            # Expand to batch dimension
            actions = action.repeat(env.num_envs, 1)

            # Perform action on environment
            if running_recording_instance:
                # Compute actions based on environment
                obv = env.step(actions)
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)
                frozen_arm_wrist_target = _read_frozen_wrist_fk(env, args_cli.arm)
                should_reset_recording_instance = False

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # Override tool and slot from CLI args (Inspire FTP tasks)
    if hasattr(env_cfg.scene, "block"):
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg as _UsdFileCfg
        from simulation.assets.assets import SINUS_TOOL_USD_PATHS

        obj_name = args_cli.object
        slot_idx = getattr(args_cli, "slot", 4)

        if obj_name != "tool_0":
            env_cfg.scene.block.spawn = _UsdFileCfg(
                usd_path=SINUS_TOOL_USD_PATHS[obj_name],
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
        print(f"  Tool: {obj_name}")

        if slot_idx != 4 and hasattr(env_cfg, "events"):
            from simulation.tasks.grasp_policy_inspire.g1_grasp_policy_inspire_env_cfg import (
                TRAY_SLOT_POSITIONS,
            )

            slot_pos = TRAY_SLOT_POSITIONS[slot_idx]
            env_cfg.scene.block.init_state.pos = slot_pos
            env_cfg.events.reset_block_position.params["slot_pos"] = slot_pos
        print(f"  Slot: {slot_idx}")

    # Create environment
    env = create_environment(env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, None, success_term, rate_limiter)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
