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

"""Convert the local Inspire FTP URDF to USD for IsaacLab simulation.

The Nucleus-hosted USD (g1_29dof_inspire_hand.usd) is missing the d435_link
camera mount that exists on the physical robot. This script converts our local
URDF — which includes d435_link — to USD so that camera transforms match the
real robot for sim-to-real transfer.

Usage (inside Docker):
    python scripts/utils/convert_inspire_urdf_to_usd.py

Output:
    assets/robots/g1-29dof-inspire-ftp-usd/g1_29dof_inspire_ftp.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert Inspire FTP URDF to USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

# Resolve paths relative to the rheo workflow root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RHEO_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")
_URDF_PATH = os.path.join(
    _RHEO_ROOT, "assets", "robots", "g1-29dof-inspire-ftp-urdf",
    "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf",
)
_USD_OUTPUT_DIR = os.path.join(_RHEO_ROOT, "assets", "robots", "g1-29dof-inspire-ftp-usd")

_URDF_PATH = os.path.abspath(_URDF_PATH)
_USD_OUTPUT_DIR = os.path.abspath(_USD_OUTPUT_DIR)

if not os.path.isfile(_URDF_PATH):
    raise FileNotFoundError(f"URDF not found: {_URDF_PATH}")

print(f"URDF source:  {_URDF_PATH}")
print(f"USD output:   {_USD_OUTPUT_DIR}/")

cfg = UrdfConverterCfg(
    asset_path=_URDF_PATH,
    usd_dir=_USD_OUTPUT_DIR,
    usd_file_name="g1_29dof_inspire_ftp.usd",
    force_usd_conversion=True,
    fix_base=True,
    # CRITICAL: keep d435_link, mid360_link, imu_in_pelvis as separate prims.
    # Without this, fixed joints are merged and d435_link disappears — which is
    # exactly why the Nucleus USD doesn't have it.
    merge_fixed_joints=False,
    # Mimic enforcement is handled by our custom InspireJointPositionAction,
    # not by the USD/physics engine.
    convert_mimic_joints_to_normal_joints=False,
    self_collision=False,
    make_instanceable=True,
    # PD gains set to 0 — actual control gains come from robot_config.py
    # actuator definitions (IdealPDActuatorCfg / ImplicitActuatorCfg).
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    ),
)

converter = UrdfConverter(cfg)
print(f"\nUSD generated at: {converter.usd_path}")
print("Done.")

simulation_app.close()
