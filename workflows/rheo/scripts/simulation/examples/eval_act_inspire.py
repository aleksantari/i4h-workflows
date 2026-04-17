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
Clean ACT evaluation script for Inspire FTP grasp task.

Purpose-built for ACT policies — no GR00T legacy code. Uses raw env
observations directly (bypasses process_observation() entirely).

Usage:
    # ACT evaluation
    python eval_act_inspire.py --model_path /models/act_inspire_ftp

    # Test mode (dummy zero-action policy)
    python eval_act_inspire.py --test --max_steps 100

    # With video recording
    python eval_act_inspire.py --model_path /models/... --save_video --enable_cameras
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import argparse
import time

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ACT Evaluation — Inspire FTP Grasp Task")
parser.add_argument("--task", type=str, default="Isaac-Grasp-Policy-G129-InspireFTP-Joint")
parser.add_argument("--model_path", type=str, default=None, help="path to ACT checkpoint")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--video_dir", type=str, default="./eval_videos")
parser.add_argument("--success_stage", type=int, default=3)
parser.add_argument(
    "--task_description", type=str, default="pick up surgical tool from tray and place in bin",
)
parser.add_argument("--test", action="store_true", help="run with dummy zero-action policy")
parser.add_argument(
    "--object", type=str, default="tool_0",
    choices=["tool_0", "tool_1", "tool_2", "tool_3", "tool_4"],
)
parser.add_argument("--slot", type=int, default=1, choices=range(6))
parser.add_argument(
    "--action_chunk_size", type=int, default=50,
    help="actions to execute per chunk before re-observing (default: 50, matching ACT chunk_size)",
)
parser.add_argument("--clamp_actions", type=float, default=0.0,
                    help="clamp action values to [-val, val] (0 = no clamping)")
parser.add_argument("--log_actions", action="store_true", help="log per-chunk action statistics")
parser.add_argument(
    "--arm", type=str, default="dual", choices=["dual", "left", "right"],
    help="which ACT config to load — dual-arm (26D) or single-arm (13D, left or right)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from simulation.examples.utils import (  # noqa: E402
    _MultiViewConcatWriter,
    _apply_cfg_default_pose,
    check_success,
    set_viewport_camera,
)
from simulation.tasks import grasp_policy_inspire  # noqa: F401


def main():
    print("=" * 60)
    print("ACT Evaluation — Inspire FTP Grasp Task")
    print("=" * 60)

    test_mode = args_cli.test
    if not test_mode and not args_cli.model_path:
        print("ERROR: --model_path required unless --test is set")
        return

    # --- Environment setup ---
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed

    # Override tool and slot
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg as _UsdFileCfg
    from simulation.assets.assets import SINUS_TOOL_USD_PATHS
    from simulation.tasks.grasp_policy_inspire.g1_grasp_policy_inspire_env_cfg import (
        TOOL_ROT,
        TRAY_SLOT_POSITIONS,
    )

    if args_cli.object != "tool_0":
        env_cfg.scene.block.spawn = _UsdFileCfg(
            usd_path=SINUS_TOOL_USD_PATHS[args_cli.object],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    if args_cli.slot != 1:
        slot_pos = TRAY_SLOT_POSITIONS[args_cli.slot]
        env_cfg.scene.block.init_state.pos = slot_pos
        env_cfg.events.reset_block_position.params["slot_pos"] = slot_pos

    print(f"  Tool: {args_cli.object}  |  Slot: {args_cli.slot}")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)
    set_viewport_camera("/World/envs/env_0/Robot/d435_link/front_cam")

    # --- Policy setup ---
    if test_mode:
        class _DummyPolicy:
            def __init__(self, device):
                self.device = device

            def get_action_from_raw(self, _obs):
                return np.zeros((50, 41), dtype=np.float32)

            def reset(self, env_ids=None):
                pass

        policy = _DummyPolicy(args_cli.device)
        print("Dummy policy (41D zeros)")
    else:
        import tempfile

        import yaml
        from simulation.act_closedloop_policy import ACTClosedloopPolicy

        # Resolve experiment config for policy_action_dim.
        # --arm selects dual-arm (26D) vs single-arm (13D) training recipe.
        _arm_to_yaml = {
            "dual": "act_config_inspire_ftp.yaml",
            "left": "act_config_inspire_ftp_left_arm.yaml",
            "right": "act_config_inspire_ftp_right_arm.yaml",
        }
        _act_cfg_path = Path(_SCRIPTS_DIR) / "policy" / _arm_to_yaml[args_cli.arm]
        _exp_policy_dim = 26 if args_cli.arm == "dual" else 13
        _exp_cfg_rel = None
        if _act_cfg_path.exists():
            with open(_act_cfg_path) as f:
                _act_raw = yaml.safe_load(f)
            _groups = _act_raw.get("experiment", {}).get("joint_groups", [])
            _group_sizes = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6}
            if _groups:
                _exp_policy_dim = sum(_group_sizes.get(g, 0) for g in _groups)
            _exp_cfg_rel = str(_act_cfg_path.resolve())

        config = {
            "model_path": args_cli.model_path,
            "action_chunk_length": 50,
            "language_instruction": args_cli.task_description,
            "policy_action_dim": _exp_policy_dim,
            "sim_action_dim": 41,
            "target_image_size": [480, 640, 3],
            "hand_type": "inspire_ftp",
            "experiment_config_path": _exp_cfg_rel,
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, tmp)
        tmp.close()
        policy = ACTClosedloopPolicy(tmp.name, num_envs=1, device=args_cli.device)
        print(f"ACT policy loaded ({_exp_policy_dim}D → 41D sim)")

    # --- Video setup ---
    video_writer = None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args_cli.save_video:
        model_name = "test" if test_mode else Path(args_cli.model_path).stem
        base_name = f"{timestamp}_act_{model_name}"
        video_writer = _MultiViewConcatWriter(args_cli.video_dir, base_name=base_name, fps=50)

    # --- Evaluation loop ---
    chunk_size = max(1, args_cli.action_chunk_size)
    clamp_val = float(args_cli.clamp_actions) if args_cli.clamp_actions > 0 else 0.0
    results = []

    for ep in range(args_cli.num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{args_cli.num_episodes}")
        print(f"{'=' * 60}")

        obs, _ = env.reset()
        # reset_scene_to_default writes joint state but NOT joint targets,
        # so PD targets from the previous episode persist. Force defaults,
        # settle physics, and refresh the observation buffer.
        _apply_cfg_default_pose(env, settle_steps=20, allow_sim_steps=True)
        if hasattr(env, "get_observations"):
            obs = env.get_observations()
        elif hasattr(env, "_get_observations"):
            obs = env._get_observations()

        # Single-arm eval: hold non-commanded joints at env init pose so the
        # uncontrolled arm / hand / waist don't drift to zero-angle posture.
        #
        # The action term applies `target = scale * raw + offset` per joint,
        # so hold_41d must be in RAW action space: subtract the term's offset
        # (and divide by scale) so that once the term re-applies them, each
        # joint lands at its default_joint_pos. Elbows carry -0.3 in the
        # env's offset_dict — without this subtraction they'd be double-
        # offset and the non-controlled forearm would droop past the init.
        if args_cli.arm != "dual" and hasattr(policy, "set_default_action"):
            robot = env.scene["robot"]
            action_term = env.action_manager.get_term("joint_pos")
            hold_41d = robot.data.default_joint_pos[:, action_term._joint_ids].clone()
            hold_41d = (hold_41d - action_term._offset) / action_term._scale
            policy.set_default_action(hold_41d)

        policy.reset()
        action_buffer = []
        total_reward = 0.0
        ep_success = False
        chunk_idx = 0

        for step in range(args_cli.max_steps):
            # Get new action chunk if buffer empty
            if not action_buffer:
                if test_mode:
                    chunk = policy.get_action_from_raw(obs)
                else:
                    chunk = policy.get_action_from_raw(obs)  # (chunk_size, 41)

                # Action diagnostics
                if args_cli.log_actions:
                    nonzero_mask = np.any(chunk != 0, axis=0)
                    active_dims = np.where(nonzero_mask)[0]
                    print(f"  [Chunk {chunk_idx}] shape={chunk.shape} "
                          f"min={chunk.min():.4f} max={chunk.max():.4f} "
                          f"active_dims={len(active_dims)}/{chunk.shape[-1]}")
                    # Per-active-dim stats
                    for d in active_dims:
                        col = chunk[:, d]
                        print(f"    dim {d:2d}: min={col.min():+.4f} max={col.max():+.4f} "
                              f"mean={col.mean():+.4f} std={col.std():.4f}")

                # Warn on extreme values
                abs_max = float(np.abs(chunk).max())
                if abs_max > 3.0:
                    print(f"  WARNING: extreme action value {abs_max:.2f} in chunk {chunk_idx}")

                # Optional clamping
                if clamp_val > 0:
                    chunk = np.clip(chunk, -clamp_val, clamp_val)

                # Buffer only chunk_size actions
                action_buffer = [chunk[t] for t in range(min(chunk_size, chunk.shape[0]))]
                chunk_idx += 1

            # Pop and apply action
            action = action_buffer.pop(0)
            action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32).unsqueeze(0)
            obs, reward, terminated, truncated, _info = env.step(action_tensor)

            # Accumulate reward
            if isinstance(reward, torch.Tensor):
                total_reward += reward.item()
            else:
                total_reward += float(reward)

            # Video recording (front camera view)
            if args_cli.save_video and video_writer is not None:
                env.sim.render()
                overlay = f"Ep {ep + 1}/{args_cli.num_episodes}  Step {step}"
                video_writer.write_from_scene_camera(
                    env, view="front", camera_key="front_camera",
                    env_index=0, overlay_text=overlay,
                )

            # Check termination
            success = check_success(env, success_stage=args_cli.success_stage)
            if isinstance(success, torch.Tensor):
                ep_success = bool(success[0].item())

            done = ep_success
            if isinstance(terminated, torch.Tensor):
                done = done or bool(terminated[0].item())
            if isinstance(truncated, torch.Tensor):
                done = done or bool(truncated[0].item())

            # Progress logging
            if step % 250 == 0 and step > 0:
                stage = env._task_stage[0].item() if hasattr(env, "_task_stage") else -1
                print(f"  Step {step}/{args_cli.max_steps} | Stage: {stage} | "
                      f"Reward: {total_reward:.2f} | Chunks: {chunk_idx}")

            if done:
                stage = env._task_stage[0].item() if hasattr(env, "_task_stage") else -1
                status = "SUCCESS" if ep_success else "TERMINATED"
                print(f"  {status} at step {step + 1} (stage: {stage})")
                break

        results.append({
            "episode": ep,
            "success": ep_success,
            "steps": step + 1,
            "total_reward": total_reward,
            "chunks_used": chunk_idx,
        })

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    successes = sum(1 for r in results if r["success"])
    success_rate = successes / len(results) * 100
    success_steps = [r["steps"] for r in results if r["success"]]

    print(f"  Episodes: {len(results)}")
    print(f"  Success: {successes}/{len(results)} ({success_rate:.1f}%)")
    if success_steps:
        print(f"  Avg steps (success): {np.mean(success_steps):.0f}")

    print("\nPer-episode:")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  Ep {r['episode']+1:2d}: {status} | Steps: {r['steps']:5d} | "
              f"Reward: {r['total_reward']:7.2f} | Chunks: {r['chunks_used']}")

    # Save results
    results_file = Path("./eval_results") / f"results_{timestamp}_act_inspire.txt"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        f.write(f"ACT Evaluation — Inspire FTP Grasp\n{'='*60}\n")
        f.write(f"Model: {args_cli.model_path}\n")
        f.write(f"Tool: {args_cli.object} | Slot: {args_cli.slot}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n\n")
        for r in results:
            status = "SUCCESS" if r["success"] else "FAILED"
            f.write(f"  Ep {r['episode']+1}: {status} | Steps: {r['steps']} | "
                    f"Reward: {r['total_reward']:.3f}\n")
    print(f"\nResults saved to: {results_file}")

    if video_writer is not None:
        video_writer.close()
        print(f"Video saved to: {args_cli.video_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
