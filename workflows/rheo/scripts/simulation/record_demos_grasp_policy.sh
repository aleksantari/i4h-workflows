#!/bin/bash

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

# Record demonstrations for the Grasp-Policy task using AVP hand tracking.
#
# This is a thin wrapper around record_demos_assemble_trocar.py with
# grasp_policy-specific defaults.
#
# Usage (inside Docker):
#   bash scripts/simulation/record_demos_grasp_policy.sh [OPTIONS]
#
# Options are forwarded to record_demos_assemble_trocar.py. Defaults:
#   --task              Isaac-Grasp-Policy-G129-Dex3-Teleop
#   --teleop_device     handtracking
#   --enable_pinocchio  (required for PINK IK with hand tracking)
#   --enable_cameras    (records camera observations)
#   --dataset_file      ./datasets/grasp_policy/demo.hdf5
#
# Examples:
#   # Record 10 demos with AVP hand tracking (default)
#   bash scripts/simulation/record_demos_grasp_policy.sh --num_demos 10 --xr
#
#   # Record with motion controllers instead
#   bash scripts/simulation/record_demos_grasp_policy.sh \
#       --teleop_device motion_controllers --num_demos 5 --xr
#
#   # Custom output path
#   bash scripts/simulation/record_demos_grasp_policy.sh \
#       --dataset_file /datasets/grasp_policy/session1.hdf5 --num_demos 10 --xr

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/record_demos_assemble_trocar.py" \
    --task Isaac-Grasp-Policy-G129-Dex3-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --dataset_file ./datasets/grasp_policy/demo.hdf5 \
    "$@"
