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

"""Batch convert sinus toolkit .obj meshes to USD format for Isaac Sim.

Converts all tool_N.obj files in assets/sinus_toolkit_v1/ to USD, adding
physics properties (mass, collision, rigid body) for use as manipulation
objects in IsaacLab environments.

Usage (inside Docker):
    python scripts/simulation/assets/convert_sinus_toolkit.py

Output structure:
    assets/sinus_toolkit_v1/
    ├── tool_0.obj
    ├── tool_0/tool_0.usd      ← generated
    ├── tool_1.obj
    ├── tool_1/tool_1.usd      ← generated
    └── ...
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch convert sinus toolkit .obj meshes to USD.")
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    help="Collision approximation method (default: convexDecomposition)",
)
parser.add_argument("--mass", type=float, default=0.05, help="Mass in kg (default: 0.05)")
parser.add_argument("--force", action="store_true", help="Re-convert even if USD already exists")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from pathlib import Path

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

# Map collision approximation name to config class
COLLISION_CFG_MAP = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
    "none": None,
}

# Resolve paths relative to this script → assets/sinus_toolkit_v1/
SCRIPT_DIR = Path(__file__).resolve().parent
RHEO_DIR = SCRIPT_DIR.parent.parent.parent  # scripts/simulation/assets → rheo/
TOOLKIT_DIR = RHEO_DIR / "assets" / "sinus_toolkit_v1"

TOOL_NAMES = ["tool_0", "tool_1", "tool_2", "tool_3", "tool_4"]


def main():
    if not TOOLKIT_DIR.exists():
        raise FileNotFoundError(f"Toolkit directory not found: {TOOLKIT_DIR}")

    collision_cls = COLLISION_CFG_MAP.get(args_cli.collision_approximation)
    collision_cfg = collision_cls() if collision_cls is not None else None

    mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
    rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=args_cli.collision_approximation != "none"
    )

    converted = 0
    skipped = 0

    for name in TOOL_NAMES:
        obj_path = TOOLKIT_DIR / f"{name}.obj"
        usd_dir = TOOLKIT_DIR / name
        usd_path = usd_dir / f"{name}.usd"

        if not obj_path.exists():
            print(f"[SKIP] {obj_path} not found")
            continue

        if usd_path.exists() and not args_cli.force:
            print(f"[SKIP] {usd_path} already exists (use --force to reconvert)")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"Converting: {obj_path}")
        print(f"Output:     {usd_path}")
        print(f"{'='*60}")

        cfg = MeshConverterCfg(
            asset_path=str(obj_path),
            usd_dir=str(usd_dir),
            usd_file_name=f"{name}.usd",
            force_usd_conversion=True,
            make_instanceable=True,
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mesh_collision_props=collision_cfg,
        )

        converter = MeshConverter(cfg)
        print(f"[OK] Generated: {converter.usd_path}")
        converted += 1

    print(f"\n{'='*60}")
    print(f"Done. Converted: {converted}, Skipped: {skipped}, Total tools: {len(TOOL_NAMES)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    simulation_app.close()
