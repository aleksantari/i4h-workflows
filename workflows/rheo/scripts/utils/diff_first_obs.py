# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-way decomposition of the t=0 MAE offset.

Compares a rollout-time dump (from eval_act_inspire.py --dump_first_obs)
against an ep28 frame-0 GT dump (from extract_ep28_frame0.py):

  1. Joint-state diff     rollout 13D state  vs  gt_state_13d
  2. Image diff           rollout PNGs        vs  GT MP4-frame-0 PNGs
  3. First-prediction     rollout pred_13d   vs  gt_action_13d

Prints per-dim tables, camera image stats, and a rule-based verdict on
which of (reset-pose / image / model) dominates the t=0 offset.

Usage:
    python scripts/utils/diff_first_obs.py \
        --rollout /tmp/first_obs_ri4 \
        --gt /tmp/ep28_frame0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DIM_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
    "thumb_yaw", "thumb_pitch", "index", "middle", "ring", "pinky",
]

CAMERA_PAIRS = [
    ("cam_room.png", "gt_front.png"),
    ("cam_right_wrist.png", "gt_right_wrist.png"),
]

STATE_MAX_RAD = 0.05
IMAGE_MAE_THRESH = 5.0
IMAGE_P99_THRESH = 20.0
PRED_MAX_RAD = 0.05


def _print_state_diff(rollout_state: np.ndarray, gt_state: np.ndarray) -> float:
    print("== Joint state diff (rollout reset vs ep28 frame-0) ==")
    print(f"  {'dim':<4}{'name':<18}{'rollout':>12}{'gt':>12}{'|diff|':>12}")
    diff = np.abs(rollout_state - gt_state)
    for i, name in enumerate(DIM_NAMES):
        print(f"  {i:<4}{name:<18}{rollout_state[i]:>12.4f}{gt_state[i]:>12.4f}{diff[i]:>12.4f}")
    print(f"\n  overall state MAE = {diff.mean():.4f} rad")
    print(f"  per-dim max |diff| = {diff.max():.4f} rad  (at dim {int(diff.argmax())} = {DIM_NAMES[int(diff.argmax())]})")
    return float(diff.max())


def _print_image_diff(rollout_dir: Path, gt_dir: Path) -> tuple[float, float]:
    print("\n== Image diff (rollout reset frame vs ep28 video frame 0) ==")
    print(f"  {'camera':<22}{'MAE(uint8)':>14}{'p99':>10}{'max':>8}{'%px|d|>8':>12}")
    worst_mae = 0.0
    worst_p99 = 0.0
    for roll_name, gt_name in CAMERA_PAIRS:
        rp = rollout_dir / roll_name
        gp = gt_dir / gt_name
        if not rp.exists() or not gp.exists():
            print(f"  {roll_name:<22}  missing ({'rollout' if not rp.exists() else 'gt'})")
            continue
        a = np.asarray(Image.open(rp).convert("RGB")).astype(np.int32)
        b = np.asarray(Image.open(gp).convert("RGB")).astype(np.int32)
        if a.shape != b.shape:
            print(f"  {roll_name:<22}  shape mismatch {a.shape} vs {b.shape}")
            continue
        d = np.abs(a - b)
        mae = float(d.mean())
        mx = int(d.max())
        p99 = float(np.percentile(d, 99))
        pct = float((d > 8).mean() * 100.0)
        print(f"  {roll_name:<22}{mae:>14.3f}{p99:>10.1f}{mx:>8d}{pct:>11.2f}%")
        worst_mae = max(worst_mae, mae)
        worst_p99 = max(worst_p99, p99)
    return worst_mae, worst_p99


def _print_pred_diff(pred_13d: np.ndarray, gt_action: np.ndarray) -> float:
    print("\n== First-prediction diff (policy output on rollout reset obs vs ep28 a_0) ==")
    print(f"  {'dim':<4}{'name':<18}{'pred':>12}{'gt_a0':>12}{'|diff|':>12}")
    diff = np.abs(pred_13d - gt_action)
    for i, name in enumerate(DIM_NAMES):
        print(f"  {i:<4}{name:<18}{pred_13d[i]:>12.4f}{gt_action[i]:>12.4f}{diff[i]:>12.4f}")
    print(f"\n  overall pred MAE   = {diff.mean():.4f} rad")
    print(f"  per-dim max |diff| = {diff.max():.4f} rad  (at dim {int(diff.argmax())} = {DIM_NAMES[int(diff.argmax())]})")
    print("  (offline replay raw MAE on ep28 was 0.014 rad — compare above)")
    return float(diff.max())


def _verdict(state_max: float, img_mae: float, img_p99: float, pred_max: float) -> None:
    print("\n== Verdict ==")
    flags = []
    if state_max > STATE_MAX_RAD:
        flags.append(f"RESET-POSE mismatch (state per-dim max {state_max:.3f} > {STATE_MAX_RAD})")
    if img_mae > IMAGE_MAE_THRESH or img_p99 > IMAGE_P99_THRESH:
        flags.append(f"IMAGE mismatch (MAE {img_mae:.1f} or p99 {img_p99:.1f})")
    if pred_max > PRED_MAX_RAD:
        flags.append(f"MODEL prediction offset (pred per-dim max {pred_max:.3f} > {PRED_MAX_RAD})")

    if not flags:
        print("  No single component crosses threshold — the t=0 offset is "
              "distributed. Drill into per-dim tables above.")
        return
    for f in flags:
        print(f"  - {f}")
    print("\n  Next steps:")
    if any("RESET-POSE" in f for f in flags):
        print("    * Align env reset to ep28 frame-0 joint state (both body87 and inspire12).")
    if any("IMAGE" in f for f in flags):
        print("    * Match sim.render_interval at eval to teleop; check DLAA/lighting.")
    if any("MODEL" in f for f in flags):
        print("    * Pixel-diff ep28 frame-0 image against the same frame rerun through "
              "offline_replay_mae.py. If offline pred matches GT on the dataset image but "
              "rollout pred diverges on the sim image, it's an image-encoder distribution gap, "
              "not model capacity.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", required=True, help="dir written by eval_act_inspire.py --dump_first_obs")
    p.add_argument("--gt", required=True, help="dir written by extract_ep28_frame0.py")
    args = p.parse_args()

    rollout = Path(args.rollout)
    gt = Path(args.gt)

    rollout_state = np.load(rollout / "state_13d.npy").astype(np.float64)
    rollout_pred = np.load(rollout / "pred_action_13d.npy").astype(np.float64)
    gt_state = np.load(gt / "gt_state_13d.npy").astype(np.float64)
    gt_action = np.load(gt / "gt_action_13d.npy").astype(np.float64)

    state_max = _print_state_diff(rollout_state, gt_state)
    img_mae, img_p99 = _print_image_diff(rollout, gt)
    pred_max = _print_pred_diff(rollout_pred, gt_action)
    _verdict(state_max, img_mae, img_p99, pred_max)


if __name__ == "__main__":
    main()
