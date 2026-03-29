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

# ACT IL Training on grasp_policy task
# Usage: bash train_act_grasp_policy.sh --dataset_path /path/to/lerobot_dataset [OPTIONS]
#
# This script trains an ACT (Action Chunking Transformer) policy using LeRobot's
# native training pipeline on demonstration data from the grasp_policy task.
#
# Examples:
#   # Train with default settings
#   bash train_act_grasp_policy.sh --dataset_path /datasets/grasp_policy_lerobot
#
#   # Train with custom batch size and steps
#   bash train_act_grasp_policy.sh --dataset_path /datasets/grasp_policy_lerobot \
#       training.batch_size=32 training.offline_steps=50000
#
#   # Resume from checkpoint
#   bash train_act_grasp_policy.sh --dataset_path /datasets/grasp_policy_lerobot \
#       --resume_path /models/act_grasp_policy/checkpoint_50000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="/workspaces"
CONFIG_PATH="${SCRIPT_DIR}/act_config.yaml"

# Parse arguments
DATASET_PATH=""
RESUME_PATH=""
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --resume_path)
            RESUME_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --dataset_path PATH [OPTIONS] [HYDRA_OVERRIDES...]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH    Path to LeRobot-format dataset (required)"
            echo "  --resume_path PATH     Path to checkpoint to resume from"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Hydra Overrides (examples):"
            echo "  training.batch_size=32"
            echo "  training.offline_steps=50000"
            echo "  policy.chunk_size=50"
            echo "  policy.dim_model=256"
            exit 0
            ;;
        *)
            HYDRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

# Validate dataset path
if [[ -z "$DATASET_PATH" ]]; then
    echo "Error: --dataset_path is required"
    echo "Usage: $0 --dataset_path /path/to/lerobot_dataset [OPTIONS...]"
    exit 1
fi

# Setup logging directory
TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
OUTPUT_DIR="${SCRIPT_DIR}/../simulation/rl/results/act_grasp_policy/train_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"

# Add dataset and output to overrides
HYDRA_OVERRIDES+=("dataset_repo_id=${DATASET_PATH}")
HYDRA_OVERRIDES+=("output_dir=${OUTPUT_DIR}")

# Add resume path if specified
if [[ -n "$RESUME_PATH" ]]; then
    HYDRA_OVERRIDES+=("resume=${RESUME_PATH}")
fi

# Set environment
export PYTHONPATH="${WORKSPACE_ROOT}/workflows/rheo/scripts:${PYTHONPATH}"

echo "========================================"
echo "ACT IL Training: grasp_policy"
echo "========================================"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Overrides: ${HYDRA_OVERRIDES[*]}"
echo "========================================"

/isaac-sim/python.sh -m lerobot.scripts.train \
    --config-path "${CONFIG_PATH}" \
    "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================"
echo "Training complete. Log saved to: ${LOG_FILE}"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================"
