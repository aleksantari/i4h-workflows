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

# ACT IL Training on Inspire FTP grasp_policy task
# Usage: bash train_act_grasp_policy_inspire.sh --dataset_path /path/to/lerobot_dataset [OPTIONS]
#
# Parallel to train_act_grasp_policy.sh but for the Inspire FTP hand:
#   - 26D policy space (14 arm + 12 hand with 6 actuated DOF per hand)
#   - Uses act_config_inspire_ftp.yaml
#   - Sets INSPIRE_FTP_EXPERIMENT_CONFIG env var
#
# Examples:
#   # Train with default settings
#   bash train_act_grasp_policy_inspire.sh --dataset_path /datasets/grasp_policy_inspire_lerobot
#
#   # Train with custom batch size and steps
#   bash train_act_grasp_policy_inspire.sh --dataset_path /datasets/grasp_policy_inspire_lerobot \
#       --steps 50000 --batch_size 32
#
#   # Resume from checkpoint
#   bash train_act_grasp_policy_inspire.sh --dataset_path /datasets/grasp_policy_inspire_lerobot \
#       --resume_path /models/act_inspire_ftp/checkpoint_50000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="/workspaces"
CONFIG_PATH="${SCRIPT_DIR}/act_config_inspire_ftp.yaml"

# Parse arguments
DATASET_PATH=""
RESUME_PATH=""
EXTRA_ARGS=()

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
            echo "Usage: $0 --dataset_path PATH [OPTIONS] [EXTRA_ARGS...]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH    Path to LeRobot-format dataset (required)"
            echo "  --resume_path PATH     Path to checkpoint to resume from"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Extra args (passed directly to lerobot.scripts.train):"
            echo "  --steps 50000"
            echo "  --batch_size 32"
            echo "  --log_freq 50"
            echo "  --policy.chunk_size 50"
            echo "  --policy.dim_model 256"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
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

# Setup logging directory (only create parent — LeRobot requires output_dir to NOT exist)
TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
OUTPUT_DIR="${SCRIPT_DIR}/../simulation/rl/results/act_grasp_policy_inspire/train_${TIMESTAMP}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

# Strip the custom 'experiment:' section — LeRobot's TrainPipelineConfig
# rejects unknown top-level keys.  Our code reads experiment config via the
# INSPIRE_FTP_EXPERIMENT_CONFIG env var, so LeRobot never needs to see it.
FILTERED_CONFIG=$(mktemp /tmp/act_config_inspire_XXXXXX.yaml)
/isaac-sim/python.sh -c "
import yaml, sys
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
cfg.pop('experiment', None)
with open('${FILTERED_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

# Build command args
CMD_ARGS=(
    --config_path "${FILTERED_CONFIG}"
    --dataset.repo_id grasp_policy_inspire
    --dataset.root "${DATASET_PATH}"
    --dataset.video_backend pyav
    --output_dir "${OUTPUT_DIR}"
)

# Add resume path if specified
if [[ -n "$RESUME_PATH" ]]; then
    CMD_ARGS+=(--resume "${RESUME_PATH}")
fi

# Add any extra args
CMD_ARGS+=("${EXTRA_ARGS[@]}")

# Set environment
export PYTHONPATH="${WORKSPACE_ROOT}/workflows/rheo/scripts:${PYTHONPATH}"
# Expose Inspire FTP experiment config to downstream code
export INSPIRE_FTP_EXPERIMENT_CONFIG="${CONFIG_PATH}"

# Generate episodes_stats.jsonl if missing (required by LeRobot v2.1 loader)
STATS_FILE="${DATASET_PATH}/meta/episodes_stats.jsonl"
if [[ ! -f "${STATS_FILE}" ]]; then
    echo "Generating episodes_stats.jsonl..."
    /isaac-sim/python.sh -c "
import json, numpy as np, pyarrow.parquet as pq
from pathlib import Path

dataset_path = Path('${DATASET_PATH}')
info = json.loads((dataset_path / 'meta/info.json').read_text())
num_episodes = info['total_episodes']
numerical_features = {k: v for k, v in info['features'].items()
                      if v['dtype'] in ('float32', 'float64') and k not in ('timestamp',)}
video_features = {k: v for k, v in info['features'].items()
                  if v['dtype'] in ('image', 'video')}

stats = []
for ep_idx in range(num_episodes):
    pq_path = dataset_path / f'data/chunk-000/episode_{ep_idx:06d}.parquet'
    table = pq.read_table(pq_path)
    n_frames = len(table)
    ep_stats = {}
    for feat_name, feat_info in numerical_features.items():
        if feat_name in table.column_names:
            arr = table.column(feat_name).to_numpy()
            arr = np.stack(arr) if arr.dtype == object else arr
            ep_stats[feat_name] = {
                'mean': np.atleast_1d(arr.mean(axis=0)).tolist(),
                'std':  np.atleast_1d(arr.std(axis=0)).tolist(),
                'min':  np.atleast_1d(arr.min(axis=0)).tolist(),
                'max':  np.atleast_1d(arr.max(axis=0)).tolist(),
                'count': [len(arr)],
                'q01': np.atleast_1d(np.quantile(arr, 0.01, axis=0)).tolist(),
                'q10': np.atleast_1d(np.quantile(arr, 0.10, axis=0)).tolist(),
                'q50': np.atleast_1d(np.quantile(arr, 0.50, axis=0)).tolist(),
                'q90': np.atleast_1d(np.quantile(arr, 0.90, axis=0)).tolist(),
                'q99': np.atleast_1d(np.quantile(arr, 0.99, axis=0)).tolist(),
            }
    for feat_name, feat_info in video_features.items():
        c = feat_info['shape'][-1]
        ep_stats[feat_name] = {
            'mean': [[[0.5]]] * c, 'std': [[[0.25]]] * c,
            'min':  [[[0.0]]] * c, 'max': [[[1.0]]] * c,
            'count': [n_frames],
            'q01': [[[0.0]]] * c, 'q10': [[[0.1]]] * c,
            'q50': [[[0.5]]] * c, 'q90': [[[0.9]]] * c,
            'q99': [[[1.0]]] * c,
        }
    stats.append({'episode_index': ep_idx, 'stats': ep_stats})

with open(dataset_path / 'meta/episodes_stats.jsonl', 'w') as f:
    for s in stats:
        f.write(json.dumps(s) + '\n')
print(f'Generated episodes_stats.jsonl for {num_episodes} episodes')
"
fi

echo "========================================"
echo "ACT IL Training: Inspire FTP grasp_policy"
echo "========================================"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "========================================"

/isaac-sim/python.sh -m lerobot.scripts.train \
    "${CMD_ARGS[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"

echo ""
echo "========================================"
echo "Training complete. Log saved to: ${OUTPUT_DIR}/train.log"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================"
