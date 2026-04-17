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

"""Copy one episode out of a LeRobot v2.1 dataset as a new 1-episode dataset.

Workaround for the LeRobot 0.1.0 `episode_data_index[ep_idx]` bug that breaks
`dataset.episodes: [N]` for non-zero N. We rewrite the chosen episode as
`episode_000000.*` with its `episode_index` column set to 0 and regenerate
the meta files.

Usage:
    python scripts/utils/extract_single_episode.py \
        --src /datasets/inspire_right_arm/demo/lerobot \
        --dst /datasets/inspire_right_arm/demo_ep28/lerobot \
        --episode 28
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _rewrite_parquet(src_path: Path, dst_path: Path) -> int:
    table = pq.read_table(src_path)
    cols = {name: table.column(name) for name in table.column_names}

    n = len(table)
    if "episode_index" in cols:
        cols["episode_index"] = pa.array([0] * n, type=cols["episode_index"].type)
    if "index" in cols:
        cols["index"] = pa.array(list(range(n)), type=cols["index"].type)

    new_table = pa.table(cols)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Source lerobot dataset root")
    parser.add_argument("--dst", type=Path, required=True, help="Destination lerobot dataset root")
    parser.add_argument("--episode", type=int, required=True, help="Source episode index to extract")
    args = parser.parse_args()

    src, dst = args.src, args.dst
    ep = args.episode

    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst}. Remove it first or pick another path.")

    print(f"Extracting episode {ep} from {src} -> {dst}")

    # 1) Parquet
    src_pq = src / f"data/chunk-000/episode_{ep:06d}.parquet"
    dst_pq = dst / "data/chunk-000/episode_000000.parquet"
    if not src_pq.exists():
        raise SystemExit(f"Parquet not found: {src_pq}")
    n_frames = _rewrite_parquet(src_pq, dst_pq)
    print(f"  parquet rewritten ({n_frames} frames) -> {dst_pq}")

    # 2) Videos — copy every observation.images.* key present in the source
    src_videos_root = src / "videos/chunk-000"
    if src_videos_root.exists():
        for cam_dir in sorted(p for p in src_videos_root.iterdir() if p.is_dir()):
            src_mp4 = cam_dir / f"episode_{ep:06d}.mp4"
            if not src_mp4.exists():
                print(f"  WARN: no video for {cam_dir.name} at episode {ep}, skipping")
                continue
            dst_mp4 = dst / "videos/chunk-000" / cam_dir.name / "episode_000000.mp4"
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_mp4, dst_mp4)
            print(f"  video copied: {cam_dir.name}")

    # 3) meta/info.json — keep features/fps/video info, rewrite counts
    src_info = json.loads((src / "meta/info.json").read_text())
    info = dict(src_info)
    info["total_episodes"] = 1
    info["total_frames"] = n_frames
    info["total_videos"] = sum(
        1
        for k, v in info.get("features", {}).items()
        if v.get("dtype") in ("image", "video")
    )
    info["total_chunks"] = 1
    if "splits" in info:
        info["splits"] = {"train": "0:1"}
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))
    print("  info.json rewritten")

    # 4) meta/episodes.jsonl — one entry, remapped
    src_eps = _read_jsonl(src / "meta/episodes.jsonl")
    orig = next((e for e in src_eps if e.get("episode_index") == ep), None)
    if orig is None:
        raise SystemExit(f"Episode {ep} not found in source episodes.jsonl")
    new_ep = dict(orig)
    new_ep["episode_index"] = 0
    new_ep["length"] = n_frames
    _write_jsonl(dst / "meta/episodes.jsonl", [new_ep])
    print("  episodes.jsonl rewritten")

    # 5) meta/tasks.jsonl — copy as-is (tasks are shared across episodes)
    src_tasks = src / "meta/tasks.jsonl"
    if src_tasks.exists():
        shutil.copy2(src_tasks, dst / "meta/tasks.jsonl")
        print("  tasks.jsonl copied")

    # 6) meta/episodes_stats.jsonl — remap if present; otherwise let the
    # training launcher regenerate it.
    src_stats_path = src / "meta/episodes_stats.jsonl"
    if src_stats_path.exists():
        src_stats = _read_jsonl(src_stats_path)
        orig_stats = next((s for s in src_stats if s.get("episode_index") == ep), None)
        if orig_stats is not None:
            new_stats = dict(orig_stats)
            new_stats["episode_index"] = 0
            _write_jsonl(dst / "meta/episodes_stats.jsonl", [new_stats])
            print("  episodes_stats.jsonl remapped")
        else:
            print("  WARN: episodes_stats.jsonl present in source but no entry for chosen episode")

    print("\nDone. Train with:")
    print(f"  --dataset_path {dst}")


if __name__ == "__main__":
    main()
