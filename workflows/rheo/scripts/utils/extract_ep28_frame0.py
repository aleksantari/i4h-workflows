# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract frame-0 ground truth from an ep LeRobot dataset for diff_first_obs.

Writes to an output dir:
  - gt_state_13d.npy        observation.state row 0 (13D, arm7+hand6)
  - gt_action_13d.npy       action row 0 (13D)
  - gt_front.png            frame 0 from cam_room MP4
  - gt_right_wrist.png      frame 0 from cam_right_wrist MP4
  - gt_meta.json            fps, resolution, shapes

Usage (host, grasp conda env):
    python scripts/utils/extract_ep28_frame0.py \
        --lerobot_root datasets/inspire_right_arm/demo_ep28/lerobot \
        --out /tmp/ep28_frame0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd


def _first_video_frame(mp4_path: Path) -> np.ndarray:
    """Decode frame 0 of an MP4 to a (H, W, 3) uint8 RGB array."""
    frame = iio.imread(mp4_path, index=0, plugin="FFMPEG")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"unexpected frame shape {frame.shape} from {mp4_path}")
    return frame.astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot_root", required=True,
                   help="path to the single-episode LeRobot dataset root")
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args()

    root = Path(args.lerobot_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(parquet)
    state_13d = np.asarray(df["observation.state"].iloc[0], dtype=np.float64)
    action_13d = np.asarray(df["action"].iloc[0], dtype=np.float64)
    assert state_13d.shape == (13,), state_13d.shape
    assert action_13d.shape == (13,), action_13d.shape
    np.save(out / "gt_state_13d.npy", state_13d)
    np.save(out / "gt_action_13d.npy", action_13d)

    video_dir = root / "videos" / "chunk-000"
    cam_map = {
        "gt_front.png": video_dir / "observation.images.cam_room" / "episode_000000.mp4",
        "gt_right_wrist.png": video_dir / "observation.images.cam_right_wrist" / "episode_000000.mp4",
    }
    for out_name, mp4 in cam_map.items():
        if not mp4.exists():
            print(f"[warn] missing video: {mp4}")
            continue
        frame = _first_video_frame(mp4)
        iio.imwrite(out / out_name, frame)
        print(f"  wrote {out / out_name}  shape={frame.shape}")

    info = json.loads((root / "meta" / "info.json").read_text())
    meta = {
        "lerobot_root": str(root),
        "fps": info.get("fps"),
        "total_frames": info.get("total_frames"),
        "state_shape": list(state_13d.shape),
        "action_shape": list(action_13d.shape),
        "state_row0": state_13d.tolist(),
        "action_row0": action_13d.tolist(),
    }
    (out / "gt_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote GT artifacts to {out}")


if __name__ == "__main__":
    main()
