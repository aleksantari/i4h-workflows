# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture a single-frame rendering of the reset scene for pixel-level diffs.

Boots the Inspire FTP grasp env, optionally overrides ``sim.render_interval``,
resets once, steps a few times to let DLAA temporal accumulation stabilize,
and writes the front + wrist camera frames to disk as PNGs.

Usage:
    # Production eval cadence (default)
    ./docker/run_docker_grasp.sh python scripts/utils/inspire/capture_reset_frame.py \
        --headless --enable_cameras --render_interval 4 --out_dir /tmp/frame_ri4

    # Teleop-recording cadence
    ./docker/run_docker_grasp.sh python scripts/utils/inspire/capture_reset_frame.py \
        --headless --enable_cameras --render_interval 2 --out_dir /tmp/frame_ri2

Then compare:
    python scripts/utils/inspire/capture_reset_frame.py --diff /tmp/frame_ri4 /tmp/frame_ri2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _diff_mode(dir_a: Path, dir_b: Path) -> int:
    """Pixel-diff every matching PNG between two capture dirs."""
    import numpy as np
    from PIL import Image

    names = sorted(p.name for p in dir_a.glob("*.png"))
    if not names:
        print(f"[diff] no PNGs in {dir_a}")
        return 1

    print(f"[diff] comparing {dir_a}  vs  {dir_b}")
    print(f"  {'camera':<22} {'MAE(uint8)':>12} {'max_abs':>10} {'p99(abs)':>10} {'%px|d|>8':>10}")
    for name in names:
        pa = np.asarray(Image.open(dir_a / name)).astype(np.int32)
        pb = np.asarray(Image.open(dir_b / name)).astype(np.int32)
        if pa.shape != pb.shape:
            print(f"  {name:<22} shape mismatch {pa.shape} vs {pb.shape}")
            continue
        d = np.abs(pa - pb)
        mae = d.mean()
        mx = d.max()
        p99 = np.percentile(d, 99)
        pct_big = (d > 8).mean() * 100.0
        print(f"  {name:<22} {mae:>12.4f} {mx:>10d} {p99:>10.1f} {pct_big:>9.2f}%")
    return 0


def _capture_mode() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str, default="Isaac-Grasp-Policy-G129-Inspire-Joint-Eval")
    parser.add_argument("--render_interval", type=int, default=None,
                        help="Override sim.render_interval after env_cfg is loaded.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Zero-action steps before capture so DLAA temporal history stabilizes.")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
    from PIL import Image  # noqa: E402

    from simulation.tasks import grasp_policy_inspire  # noqa: F401,E402

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval
        print(f"[capture] overrode sim.render_interval -> {args_cli.render_interval}")
    print(f"[capture] decimation={env_cfg.decimation}  "
          f"sim.dt={env_cfg.sim.dt}  render_interval={env_cfg.sim.render_interval}")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # Warmup: apply the reset joint pose as a hold action so the arm stays
    # in place while DLAA temporal accumulation stabilizes.
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    joint_ids = term._joint_ids.tolist() if hasattr(term._joint_ids, "tolist") else list(term._joint_ids)
    default_41d = robot.data.default_joint_pos[:, joint_ids].clone()
    # Undo the env's elbow offset so commanded target = default pose exactly.
    # (InspireJointPositionActionCfg applies offset={-0.3 elbows}; to hold
    # the raw default, command raw_default + 0.3 on elbow columns.)
    joint_names_41 = [robot.joint_names[i] for i in joint_ids]
    for i, n in enumerate(joint_names_41):
        if n.endswith("elbow_joint"):
            default_41d[:, i] += 0.3
    for _ in range(args_cli.warmup_steps):
        env.step(default_41d)

    # Re-read camera images (env.step returns obs dict with camera_images group).
    obs, _, _, _, _ = env.step(default_41d)
    cam_group = obs["camera_images"]

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] writing frames to {out_dir}")

    for cam_name in ("front_camera", "left_wrist_camera", "right_wrist_camera"):
        img = cam_group[cam_name]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.asarray(img)[0]  # first env -> (H, W, C)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1)
            img = (img * 255.0).astype(np.uint8)
        Image.fromarray(img).save(out_dir / f"{cam_name}.png")
        print(f"  wrote {cam_name}.png  shape={img.shape}  dtype={img.dtype}")

    env.close()
    simulation_app.close()
    return 0


def main() -> int:
    # Sniff for --diff first so we never boot Isaac Sim for a pure file-diff.
    if len(sys.argv) >= 4 and sys.argv[1] == "--diff":
        return _diff_mode(Path(sys.argv[2]), Path(sys.argv[3]))
    return _capture_mode()


if __name__ == "__main__":
    sys.exit(main())
