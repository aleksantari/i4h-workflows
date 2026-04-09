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
Evaluate a trained policy (ACT) on the Inspire FTP grasp_policy task.

Usage:
    # ACT evaluation
    python eval_grasp_policy_inspire.py --policy_type act --model_path /models/act_inspire_ftp

    # Test mode (dummy policy, no checkpoint needed)
    python eval_grasp_policy_inspire.py --test
"""

import argparse
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Policy on Inspire FTP Grasp Policy Task")
parser.add_argument(
    "--task", type=str, default="Isaac-Grasp-Policy-G129-InspireFTP-Joint", help="task name"
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="act",
    choices=["act", "test"],
    help="policy type: act (Action Chunking Transformer), test (dummy zero policy)",
)
parser.add_argument("--model_path", type=str, default=None, help="path to model checkpoint")
parser.add_argument(
    "--policy_config_yaml",
    type=str,
    default=None,
    help="path to policy config YAML (auto-generated if not provided)",
)
parser.add_argument("--num_episodes", type=int, default=10, help="number of evaluation episodes")
parser.add_argument("--max_steps", type=int, default=256, help="max steps per episode")
parser.add_argument("--seed", type=int, default=4, help="random seed")
parser.add_argument("--save_video", action="store_true", help="save video of evaluation")
parser.add_argument("--video_dir", type=str, default="./eval_videos", help="directory to save videos")
parser.add_argument("--num_envs", type=int, default=1, help="number of parallel simulation environments")
parser.add_argument("--video_env_id", type=int, default=0, help="which env index to record")
parser.add_argument("--save_video_all_envs", action="store_true", help="save videos for ALL envs")
parser.add_argument(
    "--action_chunk_size",
    type=int,
    default=1,
    help="number of actions to use from action chunk per observation (default: 1)",
)
parser.add_argument("--frequency", type=float, default=0.0, help="control frequency (Hz)")
parser.add_argument("--success_stage", type=int, default=3, help="success stage for the task")
parser.add_argument(
    "--task_description", type=str, default="pick up surgical tool from tray and place in bin", help="task description"
)
parser.add_argument("--test", action="store_true", help="run integration test with dummy policy")
parser.add_argument("--view", action="store_true", help="load scene and render without stepping (scene inspection mode)")
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
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for PinkIK teleop tasks).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401  — must be imported before AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from simulation.examples.utils import _MultiViewConcatWriter, evaluate_episode, set_viewport_camera  # noqa: E402
from simulation.tasks import grasp_policy_inspire  # noqa: F401


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Policy Evaluation: Inspire FTP Grasp Policy Task")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Policy Type: {args_cli.policy_type}")
    print(f"Model: {args_cli.model_path or '<test>'}")
    print("=" * 60)

    view_mode = bool(getattr(args_cli, "view", False))
    test_mode = bool(getattr(args_cli, "test", False)) or args_cli.policy_type == "test"
    if view_mode:
        print("View mode enabled (scene inspection, no policy)")
    elif test_mode:
        print("Test mode enabled (dummy policy, no checkpoint needed)")
    elif not args_cli.model_path:
        print("--model_path is required unless --test or --view is set")
        return

    # Parse environment configuration
    print("\n[1/4] Loading environment configuration...")
    num_envs = int(getattr(args_cli, "num_envs", 1) or 1)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed

    # Override tool and slot from CLI args
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg as _UsdFileCfg
    from simulation.assets.assets import SINUS_TOOL_USD_PATHS
    from simulation.tasks.grasp_policy_inspire.g1_grasp_policy_inspire_env_cfg import (
        TOOL_ROT,
        TRAY_SLOT_POSITIONS,
    )

    obj_name = args_cli.object
    slot_idx = args_cli.slot

    # Override tool USD (default: tool_0, scene cfg already has tool_0)
    if obj_name != "tool_0":
        env_cfg.scene.block.spawn = _UsdFileCfg(
            usd_path=SINUS_TOOL_USD_PATHS[obj_name],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    print(f"  Tool: {obj_name}")

    # Override slot position (default: slot 4)
    if slot_idx != 4:
        slot_pos = TRAY_SLOT_POSITIONS[slot_idx]
        env_cfg.scene.block.init_state.pos = slot_pos
        env_cfg.events.reset_block_position.params["slot_pos"] = slot_pos
    print(f"  Slot: {slot_idx}")

    # Create environment
    print("\n[2/4] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)

    set_viewport_camera("/World/envs/env_0/Robot/d435_link/front_cam")

    # View-only mode: load scene, render, no policy stepping
    if getattr(args_cli, "view", False):
        print("\n[VIEW MODE] Scene loaded. Rendering without stepping. Close the window to exit.")
        obs, _ = env.reset()
        while simulation_app.is_running():
            env.sim.render()
        env.close()
        simulation_app.close()
        return

    # Load policy
    print("\n[3/4] Loading policy...")
    if test_mode:
        class _DummyPolicy:
            def __init__(self, device: str):
                self.device = device

            def get_action(self, _obs):
                # 41D action space (29 body + 12 actuated hand)
                return {"actions": np.zeros((16, 41), dtype=np.float32)}

        policy = _DummyPolicy(args_cli.device)
        print("Dummy policy ready (41D zero actions)")

    elif args_cli.policy_type == "act":
        from simulation.act_closedloop_policy import ACTClosedloopPolicy

        if args_cli.policy_config_yaml:
            config_path = args_cli.policy_config_yaml
        else:
            import tempfile

            import yaml

            config = {
                "model_path": args_cli.model_path,
                "action_chunk_length": 100,
                "language_instruction": args_cli.task_description,
                "policy_action_dim": 26,
                "sim_action_dim": 41,
                "target_image_size": [480, 640, 3],
            }
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            yaml.dump(config, tmp)
            tmp.close()
            config_path = tmp.name

        policy = ACTClosedloopPolicy(config_path, num_envs=num_envs, device=args_cli.device)
        print("ACT policy loaded successfully")

    # Run evaluation
    print("\n[4/4] Running evaluation...")
    print("=" * 60)

    video_writer = None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args_cli.save_video:
        model_name_for_file = "test" if test_mode else Path(args_cli.model_path).stem
        base_name = f"{timestamp}_{args_cli.policy_type}_{model_name_for_file}"
        video_writer = _MultiViewConcatWriter(args_cli.video_dir, base_name=base_name, fps=50)

    results = evaluate_episode(
        env,
        policy,
        max_steps=args_cli.max_steps,
        num_episodes=args_cli.num_episodes,
        save_video=args_cli.save_video,
        action_chunk_size=args_cli.action_chunk_size,
        frequency_hz=float(getattr(args_cli, "frequency", 0.0) or 0.0),
        task_description=args_cli.task_description,
        success_stage=args_cli.success_stage,
        video_writer=video_writer,
        video_env_id=int(getattr(args_cli, "video_env_id", 0) or 0),
        save_video_all_envs=bool(getattr(args_cli, "save_video_all_envs", False)),
    )

    # Calculate statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    successes = sum(1 for r in results if r["success"])
    success_rate = successes / len(results) * 100

    success_steps = [r["steps"] for r in results if r["success"]]
    avg_success_steps = np.mean(success_steps) if success_steps else 0

    print("\nOverall Statistics:")
    print(f"  Policy Type: {args_cli.policy_type}")
    print(f"  Total Episodes: {len(results)}")
    print(f"  Success Number: {successes}")
    print(f"  Success Rate: {success_rate:.1f}%")
    if success_steps:
        print(f"  Average Steps (Success): {avg_success_steps:.1f}")

    print("\nEpisode-by-Episode Results:")
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  Ep {r['episode']+1:2d}: {status} | Steps: {r['steps']:4d} | Reward: {r['total_reward']:7.2f}")

    # Save results to file
    results_file = Path("./eval_results") / f"results_{timestamp}_{args_cli.policy_type}_inspire.txt"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w") as f:
        f.write("Evaluation Results — Inspire FTP Grasp Policy\n")
        f.write(f"{'='*60}\n")
        f.write(f"Task: {args_cli.task}\n")
        f.write(f"Policy Type: {args_cli.policy_type}\n")
        f.write(f"Model: {args_cli.model_path}\n")
        f.write(f"Episodes: {args_cli.num_episodes}\n")
        f.write(f"Action Chunk Size: {args_cli.action_chunk_size}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Average Success Steps: {avg_success_steps:.1f}\n")
        f.write("\nDetailed Results:\n")
        for r in results:
            status = "SUCCESS" if r["success"] else "FAILED"
            stage = r.get("final_stage", -1)
            f.write(
                f"  Episode {r['episode'] + 1}: {status}"
                f" | Stage: {stage}/{args_cli.success_stage}"
                f" | Steps: {r['steps']}"
                f" | Reward: {r['total_reward']:.3f}\n"
            )

    print(f"\nResults saved to: {results_file}")

    if args_cli.save_video and video_writer is not None:
        video_writer.close()
        print(f"\nVideo saved to: {args_cli.video_dir}/{base_name}_*.mp4")

    print("\nCleaning up...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
