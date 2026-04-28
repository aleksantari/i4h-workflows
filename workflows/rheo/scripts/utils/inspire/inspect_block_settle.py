# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect a recorded HDF5 demo to find the frame at which the block has settled.

Reads `data/{demo_key}/states/rigid_object/block/root_pose` (T, 7) and
`.../root_velocity` (T, 6), prints per-frame position-delta and velocity
magnitudes for the first N frames, and recommends a `--pin_block_frame_idx`
based on a velocity threshold.

Usage (host, grasp conda env):
    python scripts/utils/inspire/inspect_block_settle.py \
        --hdf5 datasets/inspire_right_arm/demo.hdf5 \
        --demo_key demo_28
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

LIN_THRESH = 1e-2  # m/s
ANG_THRESH = 5e-2  # rad/s
HOLD_FRAMES = 3    # require thresholds met for this many consecutive frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, help="path to the demo HDF5")
    p.add_argument("--demo_key", required=True, help="e.g. demo_28")
    p.add_argument("--n_frames", type=int, default=30, help="how many frames to print")
    args = p.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        pose = f[f"data/{args.demo_key}/states/rigid_object/block/root_pose"][:]
        vel = f[f"data/{args.demo_key}/states/rigid_object/block/root_velocity"][:]

    T = pose.shape[0]
    n = min(args.n_frames, T)

    print(f"demo: {args.demo_key}  |  total frames: {T}  |  showing first {n}")
    print(f"settle thresholds: |linvel| < {LIN_THRESH}  |angvel| < {ANG_THRESH}  "
          f"sustained over {HOLD_FRAMES} frames\n")
    print(f"  {'frame':>5}  {'px':>8}  {'py':>8}  {'pz':>8}  "
          f"{'|Δpos|':>8}  {'|linvel|':>9}  {'|angvel|':>9}  settled?")

    settled = np.zeros(T, dtype=bool)
    for i in range(n):
        px, py, pz = pose[i, 0], pose[i, 1], pose[i, 2]
        lin = float(np.linalg.norm(vel[i, :3]))
        ang = float(np.linalg.norm(vel[i, 3:]))
        if i > 0:
            dpos = float(np.linalg.norm(pose[i, :3] - pose[i - 1, :3]))
            dpos_str = f"{dpos:8.4f}"
        else:
            dpos_str = "       -"
        is_settled = lin < LIN_THRESH and ang < ANG_THRESH
        settled[i] = is_settled
        flag = "yes" if is_settled else ""
        print(f"  {i:>5}  {px:8.4f}  {py:8.4f}  {pz:8.4f}  "
              f"{dpos_str}  {lin:9.4f}  {ang:9.4f}  {flag}")

    # Find first frame where settled is true for HOLD_FRAMES consecutive frames.
    rec = -1
    run = 0
    for i in range(T):
        if settled[i]:
            run += 1
            if run >= HOLD_FRAMES:
                rec = i - HOLD_FRAMES + 1
                break
        else:
            run = 0

    print()
    if rec >= 0:
        print(f"recommended --pin_block_frame_idx {rec}")
    else:
        print(f"no settled frame found within thresholds in first {n} frames; "
              "loosen thresholds or inspect a longer window")


if __name__ == "__main__":
    main()
