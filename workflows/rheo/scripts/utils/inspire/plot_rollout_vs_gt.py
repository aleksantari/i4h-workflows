# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlay ACT rollout actions on top of ep28 teleop ground-truth actions.

The rollout log is 41-D sim-space actions (one row per env step). The ep28
parquet stores 13-D policy-space actions (7 right-arm + 6 right-hand). We
map the 13 policy dims into the 41-D sim space via the Inspire FTP scatter
indices from inspire_experiment_config.py, then compare per-dim.

Usage (host, grasp conda env):
    python scripts/utils/inspire/plot_rollout_vs_gt.py \
        --rollout eval_results/actions_20260420_133933_ep00.npy \
        --gt_parquet datasets/inspire_right_arm/demo_ep28/lerobot/data/chunk-000/episode_000000.parquet \
        --out eval_results/rollout_vs_ep28.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RIGHT_ARM_SIM_IDX = [12, 16, 20, 22, 24, 26, 28]
RIGHT_HAND_SIM_IDX = [38, 40, 34, 36, 37, 35]
SCATTER_IDX = RIGHT_ARM_SIM_IDX + RIGHT_HAND_SIM_IDX

DIM_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
    "thumb_yaw", "thumb_pitch", "index", "middle", "ring", "pinky",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", required=True, help="npy (T, 41) applied actions")
    p.add_argument("--gt_parquet", required=True, help="ep28 LeRobot parquet")
    p.add_argument("--out", required=True, help="output PNG path")
    args = p.parse_args()

    rollout_41d = np.load(args.rollout)
    assert rollout_41d.ndim == 2 and rollout_41d.shape[1] == 41, rollout_41d.shape
    rollout_13d = rollout_41d[:, SCATTER_IDX]

    gt_df = pd.read_parquet(args.gt_parquet)
    gt_13d = np.stack(gt_df["action"].to_numpy())
    assert gt_13d.shape[1] == 13

    T_r, T_g = rollout_13d.shape[0], gt_13d.shape[0]
    print(f"rollout: {T_r} steps  gt: {T_g} frames")

    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1.4], hspace=0.55, wspace=0.25)
    axes = [fig.add_subplot(gs[r, c]) for r in range(5) for c in range(3)]
    ax_mae = fig.add_subplot(gs[5, :])

    t_r = np.arange(T_r) / 50.0
    t_g = np.arange(T_g) / 50.0

    for i, name in enumerate(DIM_NAMES):
        ax = axes[i]
        ax.plot(t_g, gt_13d[:, i], label="ep28 GT", color="tab:green", lw=1.5)
        ax.plot(t_r, rollout_13d[:, i], label="rollout", color="tab:red", lw=1.0, alpha=0.9)
        ax.set_title(f"dim {i} — {name}  (sim idx {SCATTER_IDX[i]})", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    for j in range(len(DIM_NAMES), len(axes)):
        axes[j].axis("off")

    # Per-timestep MAE trace (overlapping window).
    overlap = min(T_r, T_g)
    t_o = np.arange(overlap) / 50.0
    abs_err = np.abs(rollout_13d[:overlap] - gt_13d[:overlap])  # (overlap, 13)
    mae_per_step_all = abs_err.mean(axis=1)
    mae_per_step_arm = abs_err[:, :7].mean(axis=1)
    mae_per_step_hand = abs_err[:, 7:].mean(axis=1)

    ax_mae.plot(t_o, mae_per_step_all, label="mean (all 13)", color="tab:blue", lw=1.5)
    ax_mae.plot(t_o, mae_per_step_arm, label="arm (7)", color="tab:orange", lw=1.0, alpha=0.9)
    ax_mae.plot(t_o, mae_per_step_hand, label="hand (6)", color="tab:purple", lw=1.0, alpha=0.9)
    ax_mae.axhline(0.014, color="gray", ls="--", lw=1.0,
                   label="offline replay MAE (0.014)")
    ax_mae.set_title("per-timestep MAE: rollout vs ep28 GT (overlapping window)",
                     fontsize=10)
    ax_mae.set_xlabel("time (s, 50 Hz)")
    ax_mae.set_ylabel("|rollout − GT| (rad)")
    ax_mae.grid(alpha=0.3)
    ax_mae.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"ACT rollout (temporal_ensemble_coeff=0.01) vs ep28 GT\n"
                 f"rollout={Path(args.rollout).name}", fontsize=11)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")

    mae = abs_err.mean(axis=0)
    print("\nPer-dim MAE over overlapping window (rad):")
    for i, name in enumerate(DIM_NAMES):
        print(f"  dim {i:2d}  {name:<14}  MAE={mae[i]:.4f}")
    print(f"\n  overall MAE  = {mae.mean():.4f}")
    print(f"  MAE @ t=0    = {mae_per_step_all[0]:.4f}  (reset-pose error signature)")
    print(f"  MAE @ t=1s   = {mae_per_step_all[min(50, overlap-1)]:.4f}")
    print(f"  MAE @ t=2s   = {mae_per_step_all[min(100, overlap-1)]:.4f}")
    print(f"  MAE @ t_end  = {mae_per_step_all[-1]:.4f}")


if __name__ == "__main__":
    main()
