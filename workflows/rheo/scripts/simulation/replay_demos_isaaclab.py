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
Replay demonstrations with Isaac Lab environments (IsaacLab-track).

Adapted from IsaacLab's ``scripts/tools/replay_demos.py`` with project-specific
task registration (grasp_policy, assemble_trocar).

Usage:
    # Replay all episodes
    python replay_demos_isaaclab.py --task Isaac-Grasp-Policy-G129-Dex3-Joint \
        --dataset_file /datasets/grasp_policy/demo.hdf5 --enable_cameras

    # Replay specific episodes with success validation
    python replay_demos_isaaclab.py --task Isaac-Grasp-Policy-G129-Dex3-Joint \
        --dataset_file /datasets/grasp_policy/demo.hdf5 --enable_cameras \
        --select_episodes 0 2 4 --validate_success_rate

    # Replay trocar demos
    python replay_demos_isaaclab.py --task Isaac-Assemble-Trocar-G129-Dex3-Joint \
        --dataset_file /datasets/trocar/demo.hdf5 --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Validate the replay success rate using the task environment termination criteria",
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
    help=(
        "Inspire FTP tool USD to spawn. Must match the tool used during recording —"
        " the HDF5 initial_state overrides pose but not geometry."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import gymnasium as gym
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Register project-specific gym environments
from simulation.tasks import assemble_trocar  # noqa: F401
from simulation.tasks import grasp_policy  # noqa: F401
from simulation.tasks import grasp_policy_inspire  # noqa: F401

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def main():
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Override tool USD for Inspire FTP tasks. The HDF5 initial_state restores
    # the block pose via env.reset_to(), but geometry is fixed at spawn time,
    # so the USD asset must be set to match the tool used during recording.
    if hasattr(env_cfg.scene, "block") and args_cli.object != "tool_0":
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg as _UsdFileCfg
        from simulation.assets.assets import SINUS_TOOL_USD_PATHS

        env_cfg.scene.block.spawn = _UsdFileCfg(
            usd_path=SINUS_TOOL_USD_PATHS[args_cli.object],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        print(f"Replay tool override: {args_cli.object}")

    # extract success checking function to invoke in the main loop
    success_term = None
    if args_cli.validate_success_rate:
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print(
                "No success termination term was found in the environment."
                " Will not be able to mark recorded demos as successful."
            )

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    recorded_episode_count = 0

    # Track current episode indices for each environment
    current_episode_indices = [None] * num_envs

    # Track failed demo IDs
    failed_demo_ids = []

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            episode_ended = [False] * num_envs
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                actions = idle_action
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        # check if the episode is successful after the whole episode_data is
                        if (
                            (success_term is not None)
                            and (current_episode_indices[env_id]) is not None
                            and (not episode_ended[env_id])
                        ):
                            if bool(success_term.func(env, **success_term.params)[env_id]):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""

                                print(
                                    f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                    f" of {replayed_episode_count} demos."
                                )
                            else:
                                # if not successful, add to failed demo IDs list
                                if (
                                    current_episode_indices[env_id] is not None
                                    and current_episode_indices[env_id] not in failed_demo_ids
                                ):
                                    failed_demo_ids.append(current_episode_indices[env_id])

                            episode_ended[env_id] = True

                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)

                            if next_episode_index < episode_count:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue
                env.step(actions)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1:4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
            break
    # Close environment after replay is complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")

    # Print success statistics only if validation was enabled
    if success_term is not None:
        print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")

        # Print failed demo IDs if any
        if failed_demo_ids:
            print(f"\nFailed demo IDs ({len(failed_demo_ids)} total):")
            print(f"  {sorted(failed_demo_ids)}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
