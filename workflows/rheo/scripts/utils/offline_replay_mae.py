# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline memorization-quality check for an ACT checkpoint.

Loads a trained ACT policy and the ep28 LeRobot dataset, feeds each frame's
observation (state + two camera images) through the model, and reports MAE
between predicted and recorded actions in both normalized (loss-space) and
raw (radian) units. Runs inside the grasp docker — no Isaac Sim needed.

Usage:
    ./docker/run_docker_grasp.sh python scripts/utils/offline_replay_mae.py \
        --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/right_arm_smoketest_20260419-203926/checkpoints/005000/pretrained_model \
        --dataset_path /datasets/inspire_right_arm/demo_ep28/lerobot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(args.model_path)
    policy.eval().to(args.device)
    print(f"[offline_replay] loaded policy from {args.model_path}")

    ds = LeRobotDataset(
        repo_id="local_ep28",
        root=Path(args.dataset_path),
    )
    print(f"[offline_replay] dataset frames: {len(ds)}  fps: {ds.fps}")

    # Pull normalization stats from the model for raw-space conversion.
    # LeRobot ACT stores stats inside policy.normalize_inputs / unnormalize_outputs.
    action_mean = None
    action_std = None
    for name, module in policy.named_modules():
        for key, buf in module.named_buffers(recurse=False):
            if name.endswith("unnormalize_outputs") and "action" in key:
                if key.endswith(".mean"):
                    action_mean = buf.detach().cpu().numpy().reshape(-1)
                elif key.endswith(".std"):
                    action_std = buf.detach().cpu().numpy().reshape(-1)

    if action_mean is None or action_std is None:
        # Fall back: load from stats.json
        import json
        stats_path = Path(args.dataset_path) / "meta" / "episodes_stats.jsonl"
        with open(stats_path) as f:
            line = f.readline()
        stats = json.loads(line)["stats"]
        action_mean = np.asarray(stats["action"]["mean"])
        action_std = np.asarray(stats["action"]["std"])

    print(f"[offline_replay] action std per-dim: {action_std}")

    n_frames = len(ds)
    action_dim = ds.features["action"]["shape"][0]

    pred_first = np.zeros((n_frames, action_dim))
    true_first = np.zeros((n_frames, action_dim))

    with torch.no_grad():
        for t in range(n_frames):
            sample = ds[t]
            obs = {
                "observation.state": sample["observation.state"].unsqueeze(0).to(args.device),
                "observation.images.cam_room": sample["observation.images.cam_room"].unsqueeze(0).to(args.device),
                "observation.images.cam_right_wrist": sample["observation.images.cam_right_wrist"].unsqueeze(0).to(args.device),
            }
            chunk = policy.predict_action_chunk(obs)  # (1, chunk, action_dim)
            if isinstance(chunk, np.ndarray):
                chunk = torch.from_numpy(chunk)
            pred_first[t] = chunk[0, 0].cpu().numpy()
            true_first[t] = sample["action"].cpu().numpy()
            if t % 25 == 0:
                print(f"  t={t:4d}  pred[:4]={pred_first[t, :4]}  true[:4]={true_first[t, :4]}")

    # --- Report -----
    raw_abs_err = np.abs(pred_first - true_first)                            # (T, D) in raw action units (post-stat normalization has already been un-done by predict_action_chunk)
    norm_abs_err = raw_abs_err / action_std[None, :]                         # (T, D) in "loss-space" std units

    raw_per_dim = raw_abs_err.mean(axis=0)
    norm_per_dim = norm_abs_err.mean(axis=0)
    raw_overall = raw_abs_err.mean()
    norm_overall = norm_abs_err.mean()

    feature_names = ds.features.get("action", {}).get("names", {})
    if isinstance(feature_names, dict):
        dim_names = feature_names.get("motors", [f"dim{i}" for i in range(action_dim)])
    else:
        dim_names = feature_names if feature_names else [f"dim{i}" for i in range(action_dim)]

    print("\n=============================================")
    print("Offline replay MAE — predicted action vs recorded action")
    print("=============================================")
    print(f"  Overall raw MAE  (rad) : {raw_overall:.6f}")
    print(f"  Overall norm MAE (std) : {norm_overall:.6f}   (compare to train/l1_loss)")
    print()
    print("  Per-dim:")
    print(f"  {'dim':<4} {'name':<32} {'raw_MAE(rad)':>14} {'norm_MAE(std)':>14}")
    for i in range(action_dim):
        nm = dim_names[i] if i < len(dim_names) else f"dim{i}"
        print(f"  {i:<4} {nm:<32} {raw_per_dim[i]:>14.6f} {norm_per_dim[i]:>14.6f}")

    # Worst frames
    per_frame_raw = raw_abs_err.mean(axis=1)
    worst = np.argsort(per_frame_raw)[::-1][:5]
    print("\n  Worst 5 frames by raw MAE:")
    for t in worst:
        print(f"    t={t:4d}  raw_MAE={per_frame_raw[t]:.6f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
