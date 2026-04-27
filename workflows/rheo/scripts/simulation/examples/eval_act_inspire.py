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
parser.add_argument(
    "--task", type=str, default="Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval",
    help="gym id. Default -Joint-Eval has zero block XY/yaw noise — use -Joint for noisy training env.",
)
parser.add_argument("--model_path", type=str, default=None, help="path to ACT checkpoint")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=300)
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
    "--pin_block_from_hdf5", type=str, default=None,
    help="HDF5 path to read block initial pose from (overrides --slot). "
         "Use --pin_demo_key to select which demo inside the file.",
)
parser.add_argument(
    "--pin_block_frame_idx", type=int, default=0,
    help="Frame index into states/rigid_object/block/root_pose to pin from. "
         "0 (default) uses initial_state (pre-settle). >0 uses the per-timestep "
         "states trajectory at that index — pick a settled frame via "
         "scripts/utils/inspect_block_settle.py.",
)
parser.add_argument(
    "--pin_demo_key", type=str, default="demo_0",
    help="HDF5 demo key (e.g. 'demo_28') used with --pin_block_from_hdf5.",
)
parser.add_argument(
    "--action_chunk_size", type=int, default=50,
    help="actions to execute per chunk before re-observing (default: 50, matching ACT chunk_size)",
)
parser.add_argument("--clamp_actions", type=float, default=0.0,
                    help="clamp action values to [-val, val] (0 = no clamping)")
parser.add_argument("--log_actions", action="store_true", help="log per-chunk action statistics")
parser.add_argument(
    "--arm", type=str, default="right", choices=["dual", "left", "right"],
    help="which ACT config to load — dual-arm (26D) or single-arm (13D, left or right)",
)
parser.add_argument(
    "--temporal_ensemble_coeff", type=float, default=None,
    help="Enable LeRobot ACT temporal ensembling with this coefficient "
         "(e.g. 0.01 = ACT paper default, positive weights older actions more). "
         "When set, the policy is queried every env step and overlapping chunks "
         "are exponentially blended; --action_chunk_size is ignored.",
)
parser.add_argument(
    "--dump_first_obs", type=str, default=None,
    help="Directory to dump t=0 diagnostics (state_13d.npy, pred_action_*.npy, "
         "front_camera.png, right_wrist_camera.png, meta.json) before the first "
         "step. Used to isolate reset/initial-obs mismatch from closed-loop drift.",
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


def _dump_first_obs(env, policy, obs, out_dir: Path, args_cli) -> None:
    """Write reset-time state, images, and first prediction for diagnostics.

    Captures the exact tensors the policy sees at t=0 plus its first action
    prediction (before scatter and after scatter). Paired with
    scripts/utils/extract_ep28_frame0.py + diff_first_obs.py to decompose
    the t=0 MAE offset into reset-pose / image / prediction components.
    """
    import json

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)

    # Policy-space 13D state + CHW float [0,1] image tensors the model sees.
    act_obs = policy._extract_observations_from_raw(obs)
    state_13d = act_obs["observation.state"][0].detach().cpu().numpy()
    np.save(out_dir / "state_13d.npy", state_13d)

    for act_key, img_tensor in act_obs.items():
        if not act_key.startswith("observation.images."):
            continue
        # (1, C, H, W) float [0,1] -> (H, W, C) uint8
        img_np = img_tensor[0].detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        cam_short = act_key.split("observation.images.")[-1]
        Image.fromarray(img_np).save(out_dir / f"{cam_short}.png")

    # Probe first prediction without advancing the env or ensembler state.
    # Use predict_action_chunk directly (not select_action) so the temporal
    # ensembler buffer is not populated by this diagnostic call.
    with torch.no_grad():
        single_obs = {k: v[0:1] for k, v in act_obs.items()}
        pred_chunk = policy.policy.predict_action_chunk(single_obs)
        if isinstance(pred_chunk, np.ndarray):
            pred_chunk = torch.from_numpy(pred_chunk)
        if pred_chunk.ndim == 3:
            pred_chunk = pred_chunk.squeeze(0)  # (chunk, policy_dim)
    pred_action_13d = pred_chunk[0].detach().cpu().numpy()
    np.save(out_dir / "pred_action_13d.npy", pred_action_13d)

    # Scatter to the 41D sim-action space as eval would apply it.
    pred_policy = pred_chunk.unsqueeze(0).to(policy.device)  # (1, chunk, 13)
    scatter_kwargs = {}
    if policy._base_action is not None and policy.sim_action_dim == 41:
        scatter_kwargs["base_action"] = policy._base_action
    pred_41d_chunk = policy.exp_config.scatter_to_sim(pred_policy, **scatter_kwargs)
    pred_action_41d = pred_41d_chunk[0, 0].detach().cpu().numpy()
    np.save(out_dir / "pred_action_41d.npy", pred_action_41d)

    # If ensembling is on, the probe populated the ensembler buffer — undo it
    # so the real rollout starts clean.
    if getattr(policy, "_use_temporal_ensemble", False):
        policy.policy.reset()

    sim_cfg = env.cfg.sim if hasattr(env, "cfg") else None
    meta = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "pin_demo_key": getattr(args_cli, "pin_demo_key", None),
        "pin_block_from_hdf5": getattr(args_cli, "pin_block_from_hdf5", None),
        "temporal_ensemble_coeff": getattr(args_cli, "temporal_ensemble_coeff", None),
        "model_path": str(args_cli.model_path),
        "arm": args_cli.arm,
        "render_interval": getattr(sim_cfg, "render_interval", None),
        "decimation": getattr(env.cfg, "decimation", None) if hasattr(env, "cfg") else None,
        "sim_dt": getattr(sim_cfg, "dt", None),
        "state_shape": list(state_13d.shape),
        "pred_action_13d_shape": list(pred_action_13d.shape),
        "pred_action_41d_shape": list(pred_action_41d.shape),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[dump_first_obs] wrote diagnostics to {out_dir}")
    print(f"  state_13d {state_13d.shape}  pred_action_13d {pred_action_13d.shape}  "
          f"pred_action_41d {pred_action_41d.shape}")


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
    if args_cli.pin_block_from_hdf5:
        import h5py

        idx = args_cli.pin_block_frame_idx
        with h5py.File(args_cli.pin_block_from_hdf5, "r") as _hf:
            if idx > 0:
                _pin_path = f"data/{args_cli.pin_demo_key}/states/rigid_object/block/root_pose"
                _root_pose = _hf[_pin_path][idx]
                _src = f"{_pin_path}[{idx}]"
            else:
                _pin_path = f"data/{args_cli.pin_demo_key}/initial_state/rigid_object/block/root_pose"
                _root_pose = _hf[_pin_path][()]
                _src = _pin_path
        _p = np.asarray(_root_pose).reshape(-1)
        slot_pos = (float(_p[0]), float(_p[1]), float(_p[2]))
        slot_rot = (float(_p[3]), float(_p[4]), float(_p[5]), float(_p[6]))
        env_cfg.scene.block.init_state.pos = slot_pos
        env_cfg.scene.block.init_state.rot = slot_rot
        env_cfg.events.reset_block_position.params["slot_pos"] = slot_pos
        env_cfg.events.reset_block_position.params["slot_rot"] = slot_rot
        print(
            f"  Tool: {args_cli.object}  |  Block pinned from {_src}: "
            f"pos={slot_pos}, rot={slot_rot}"
        )
    else:
        slot_pos = TRAY_SLOT_POSITIONS[args_cli.slot]
        env_cfg.scene.block.init_state.pos = slot_pos
        env_cfg.scene.block.init_state.rot = TOOL_ROT
        env_cfg.events.reset_block_position.params["slot_pos"] = slot_pos
        env_cfg.events.reset_block_position.params["slot_rot"] = TOOL_ROT
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
        if args_cli.temporal_ensemble_coeff is not None:
            config["temporal_ensemble_coeff"] = args_cli.temporal_ensemble_coeff
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, tmp)
        tmp.close()
        policy = ACTClosedloopPolicy(tmp.name, num_envs=1, device=args_cli.device)
        print(f"ACT policy loaded ({_exp_policy_dim}D → 41D sim)")
        if args_cli.temporal_ensemble_coeff is not None:
            print(
                f"[Inference] Temporal ensembling ON "
                f"(coeff={args_cli.temporal_ensemble_coeff}) — querying policy every env step"
            )
        else:
            print(
                f"[Inference] Chunk-exhaustion mode (action_chunk_size={args_cli.action_chunk_size})"
            )

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
        applied_actions = [] if args_cli.log_actions else None

        # t=0 diagnostic dump — run once per-episode before the first step.
        # Captures the state/images the policy actually sees at reset + the
        # first predicted 13D and 41D actions, so we can decompose the t=0
        # MAE offset into (reset-pose, image, prediction) components.
        if ep == 0 and args_cli.dump_first_obs and not test_mode:
            _dump_first_obs(env, policy, obs, Path(args_cli.dump_first_obs), args_cli)

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
            if applied_actions is not None:
                applied_actions.append(np.asarray(action, dtype=np.float32).copy())
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

        if applied_actions is not None and len(applied_actions) > 0:
            actions_dir = Path("./eval_results")
            actions_dir.mkdir(exist_ok=True)
            actions_path = actions_dir / f"actions_{timestamp}_ep{ep:02d}.npy"
            np.save(actions_path, np.stack(applied_actions, axis=0))
            print(f"  Applied actions saved to: {actions_path}  shape={np.stack(applied_actions).shape}")

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
