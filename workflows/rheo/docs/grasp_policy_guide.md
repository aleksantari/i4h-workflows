<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Grasp Policy: End-to-End Training Guide

Complete guide for training and evaluating an ACT (Action Chunking Transformer) policy
on the **grasp_policy** task (pick up a block and place it in a bin) using the G1 robot
with Dex3 hands.

**Pipeline overview:**

```
Teleoperate (AVP)  -->  Record HDF5  -->  Convert to LeRobot  -->  Train ACT (IL)
                                                                       |
                                                                       v
                                          Evaluate  <--  RL Post-train (RLinf)
```

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Teleoperation with AVP](#2-teleoperation-with-avp)
3. [Recording Demonstrations](#3-recording-demonstrations)
4. [Data Conversion (HDF5 to LeRobot)](#4-data-conversion-hdf5-to-lerobot)
5. [ACT Imitation Learning Training](#5-act-imitation-learning-training)
6. [RL Post-Training (PPO via RLinf)](#6-rl-post-training-ppo-via-rlinf)
7. [Evaluation](#7-evaluation)
8. [Architecture Reference](#8-architecture-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### Docker Image

Build the Docker image with LeRobot enabled:

```bash
cd workflows/rheo
docker/build_docker.sh -g1.5 --build-arg INSTALL_LEROBOT=true
```

> **Code:** LeRobot install is configured in
> [`docker/Dockerfile.x86`](../docker/Dockerfile.x86) via the `INSTALL_LEROBOT`
> build arg. It installs LeRobot at the pinned commit `c75455a` using
> [`tools/env_setup/install_lerobot.sh`](../../tools/env_setup/install_lerobot.sh).

### Hardware

- NVIDIA GPU with >= 16 GB VRAM (RTX 4090 / A6000 recommended)
- Apple Vision Pro (AVP) for hand-tracking teleoperation
- Both devices on the same network (CloudXR requires port 48010 TCP + 47998-48012 UDP)

### Host Directory Layout

The Docker container mounts these host directories:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `~/datasets` | `/datasets` | Recorded HDF5 demos + converted LeRobot datasets |
| `~/models` | `/models` | Trained model checkpoints |
| `~/eval` | `/eval` | Evaluation results and videos |

---

## 2. Teleoperation with AVP

The grasp_policy task registers a dedicated teleop environment
(`Isaac-Grasp-Policy-G129-Dex3-Teleop`) that supports AVP hand tracking.

> **Code:** Task registration in
> [`scripts/simulation/tasks/grasp_policy/__init__.py`](../scripts/simulation/tasks/grasp_policy/__init__.py).
> Teleop environment config in
> [`scripts/simulation/tasks/grasp_policy/g1_grasp_policy_teleop_env_cfg.py`](../scripts/simulation/tasks/grasp_policy/g1_grasp_policy_teleop_env_cfg.py).

### How It Works

The teleop environment uses **Whole-Body Control + PINK IK** (23D action space):

| Indices | Dimensions | Control |
|---------|-----------|---------|
| 0-1 | 2 | Left/right gripper (pinch = close) |
| 2-8 | 7 | Left wrist pose (xyz + quat) |
| 9-15 | 7 | Right wrist pose (xyz + quat) |
| 16-18 | 3 | Navigation (fixed to 0) |
| 19 | 1 | Base height (fixed to 0.75) |
| 20-22 | 3 | Torso orientation (fixed to 0) |

Hand tracking converts AVP hand joints into wrist poses and pinch-gripper commands.
The lower body is locked in a standing posture.

> **Code:** Hand tracking retargeter in
> [`scripts/teleop_devices/handtracking.py`](../scripts/teleop_devices/handtracking.py).
> Gripper close threshold: 0.03m, open threshold: 0.05m (hysteresis).

### Launching XR Teleoperation

```bash
# Stop any external CloudXR container first
docker stop cloudxr-runtime 2>/dev/null

# Launch with AVP hand tracking + built-in CloudXR
./docker/run_docker.sh -g1.5 \
    python scripts/simulation/record_demos_assemble_trocar.py \
    --task Isaac-Grasp-Policy-G129-Dex3-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --xr
```

**In Isaac Sim:** AR Panel --> "CloudXR Runtime (5.0)" --> "Start AR"

**On AVP:** Launch "Isaac XR Teleop Sample Client" --> Enter workstation IP --> Connect --> Play

> See [`docs/avp_teleoperation.md`](avp_teleoperation.md) and
> [`docs/cloudxr.md`](cloudxr.md) for detailed CloudXR setup and troubleshooting.

---

## 3. Recording Demonstrations

A convenience wrapper script is provided for recording demos on the grasp_policy task.

> **Code:**
> [`scripts/simulation/record_demos_grasp_policy.sh`](../scripts/simulation/record_demos_grasp_policy.sh)
> wraps the generic
> [`scripts/simulation/record_demos_assemble_trocar.py`](../scripts/simulation/record_demos_assemble_trocar.py)
> with grasp_policy defaults.

### Quick Start

```bash
# Record 10 demos with AVP hand tracking
./docker/run_docker.sh -g1.5 \
    bash scripts/simulation/record_demos_grasp_policy.sh --num_demos 10 --xr
```

### Wrapper Defaults

The wrapper sets these defaults (all overridable via extra arguments):

| Argument | Default |
|----------|---------|
| `--task` | `Isaac-Grasp-Policy-G129-Dex3-Teleop` |
| `--teleop_device` | `handtracking` |
| `--enable_pinocchio` | enabled |
| `--enable_cameras` | enabled |
| `--dataset_file` | `./datasets/grasp_policy/demo.hdf5` |

### Recording Controls

| Key | Action |
|-----|--------|
| **B** | Start recording a demo |
| **S** | Save current demo (mark as success) |
| **R** | Reset environment (discard current demo) |

### Custom Recording Examples

```bash
# Record with motion controllers instead of hand tracking
./docker/run_docker.sh -g1.5 \
    bash scripts/simulation/record_demos_grasp_policy.sh \
    --teleop_device motion_controllers --num_demos 5 --xr

# Custom output path
./docker/run_docker.sh -g1.5 \
    bash scripts/simulation/record_demos_grasp_policy.sh \
    --dataset_file /datasets/grasp_policy/session1.hdf5 --num_demos 20 --xr
```

### HDF5 Output Format

Each recorded demo is stored as an HDF5 group:

```
/data/demo_N/
    obs/
        robot_joint_state          (T, 87)   Full body joint positions
        robot_dex3_joint_state     (T, 14)   Dex3 hand joint positions
        front_camera               (T, H, W, C)
        left_wrist_camera          (T, H, W, C)
        right_wrist_camera         (T, H, W, C)
    processed_actions              (T, 43)   WBC+PINK joint targets
```

> **Recommendation:** Record 20-50 high-quality demonstrations for initial ACT training.
> Quality matters more than quantity -- discard failed attempts with **R** and only save
> successful grasps with **S**.

---

## 4. Data Conversion (HDF5 to LeRobot)

Convert recorded HDF5 demonstrations to LeRobot format (Parquet + MP4) for ACT training.

> **Note:** This pipeline has been tested on the assemble_trocar task with identical
> robot configuration (G1 + Dex3, same 28D joint layout, same 3 cameras). It has not yet
> been tested end-to-end with grasp_policy data. The conversion should work identically
> since both tasks share the same robot, joint indices, and camera setup.

> **Code:**
> Conversion script:
> [`scripts/utils/convert_hdf5_to_lerobot.py`](../scripts/utils/convert_hdf5_to_lerobot.py).
> Dataset config:
> [`scripts/config/g1_grasp_policy_dataset.yaml`](../scripts/config/g1_grasp_policy_dataset.yaml).
> Field mappings (28D extraction):
> [`scripts/utils/assemble_trocar_lerobot_fields.py`](../scripts/utils/assemble_trocar_lerobot_fields.py).
> Modality definition:
> [`scripts/simulation/tasks/grasp_policy/modality_grasp_policy.json`](../scripts/simulation/tasks/grasp_policy/modality_grasp_policy.json).

### 28D Canonical Joint Order

Both state and action are extracted as a 28D vector:

| Index Range | Body Part | Joints |
|-------------|-----------|--------|
| 0-6 | Left arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| 7-13 | Right arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| 14-20 | Left hand | thumb 0/1/2, middle 0/1, index 0/1 |
| 21-27 | Right hand | thumb 0/1/2, middle 0/1, index 0/1 |

The arm joints are extracted from `robot_joint_state[:, 15:29]` (14 DOF) and hand joints
from `robot_dex3_joint_state` (14 DOF).

### Run Conversion

```bash
./docker/run_docker.sh -g1.5 \
    python scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/g1_grasp_policy_dataset.yaml \
    --hdf5_dir /datasets/grasp_policy \
    --output_dir /datasets/grasp_policy_lerobot
```

### Dataset Config Overview

The config at
[`scripts/config/g1_grasp_policy_dataset.yaml`](../scripts/config/g1_grasp_policy_dataset.yaml)
defines:

```yaml
language_instruction: "pick up block and place in bin"
use_rheo_converter: true            # Use rheo-specific HDF5 layout
rheo_action_key: "processed_actions"
rheo_28d_state_action: true         # Extract 28D canonical state/action
rheo_camera_mappings_obs:
  front_camera: "observation.images.cam_room"
  left_wrist_camera: "observation.images.cam_left_wrist"
  right_wrist_camera: "observation.images.cam_right_wrist"
```

### Output Structure

```
grasp_policy_lerobot/
├── data/chunk-000/episode_000000.parquet    # State/action per timestep
├── videos/chunk-000/
│   ├── observation.images.cam_room/episode_000000.mp4
│   ├── observation.images.cam_left_wrist/episode_000000.mp4
│   └── observation.images.cam_right_wrist/episode_000000.mp4
└── meta/
    ├── info.json           # Feature schemas (28D joint names)
    ├── tasks.jsonl         # Task descriptions
    ├── episodes.jsonl      # Episode metadata
    └── modality.json       # LeRobot modality config
```

---

## 5. ACT Imitation Learning Training

Train an ACT policy on the converted LeRobot dataset using LeRobot's native training
pipeline.

> **Code:**
> Training launcher:
> [`scripts/policy/train_act_grasp_policy.sh`](../scripts/policy/train_act_grasp_policy.sh).
> Training config:
> [`scripts/policy/act_config.yaml`](../scripts/policy/act_config.yaml).

### Quick Start

```bash
./docker/run_docker.sh -g1.5 \
    bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /datasets/grasp_policy_lerobot
```

### Training Configuration

Key parameters from
[`scripts/policy/act_config.yaml`](../scripts/policy/act_config.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy.chunk_size` | 100 | Action prediction horizon (timesteps) |
| `policy.dim_model` | 512 | Transformer hidden dimension |
| `policy.n_heads` | 8 | Attention heads |
| `policy.n_encoder_layers` | 4 | Encoder depth |
| `policy.n_decoder_layers` | 1 | Decoder depth |
| `policy.use_vae` | true | Use CVAE for stochastic training |
| `policy.latent_dim` | 32 | VAE latent dimension |
| `policy.kl_weight` | 10.0 | KL divergence loss weight |
| `policy.vision_backbone` | resnet18 | ImageNet-pretrained backbone |
| `training.offline_steps` | 100000 | Total training steps |
| `training.batch_size` | 64 | Batch size |
| `training.lr` | 1e-5 | Learning rate |

**Input dimensions:**

- State: 28D (arm + hand joints)
- Images: 3 cameras at 480x640 (left wrist, right wrist, front/room)
- Action output: 28D

### Common Overrides

```bash
# Smaller model for faster iteration
bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /datasets/grasp_policy_lerobot \
    policy.dim_model=256 policy.n_heads=4 training.batch_size=32

# Shorter training run
bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /datasets/grasp_policy_lerobot \
    training.offline_steps=50000

# Resume from checkpoint
bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /datasets/grasp_policy_lerobot \
    --resume_path /models/act_grasp_policy/checkpoint_50000
```

### Output

Checkpoints are saved to a timestamped directory:

```
scripts/simulation/rl/results/act_grasp_policy/train_YYYYMMDD-HHMMSS/
├── checkpoint_25000/
├── checkpoint_50000/
├── checkpoint_75000/
├── checkpoint_100000/    # Final checkpoint
└── train.log
```

Copy the final checkpoint to `/models/` for evaluation:

```bash
cp -r results/act_grasp_policy/train_*/checkpoint_100000 ~/models/act_grasp_policy
```

---

## 6. RL Post-Training (PPO via RLinf)

Fine-tune the IL-trained ACT checkpoint with PPO reinforcement learning in simulation.

> **Code:**
> RL launcher:
> [`scripts/simulation/rl/train_act_grasp_policy.sh`](../scripts/simulation/rl/train_act_grasp_policy.sh).
> Top-level config:
> [`scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml`](../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml).
> Model config:
> [`scripts/simulation/rl/rlinf_ext/config/model/act_dex3.yaml`](../scripts/simulation/rl/rlinf_ext/config/model/act_dex3.yaml).
> Env config:
> [`scripts/simulation/rl/rlinf_ext/config/env/isaaclab_grasp_policy.yaml`](../scripts/simulation/rl/rlinf_ext/config/env/isaaclab_grasp_policy.yaml).
> RLinf ACT wrapper:
> [`scripts/simulation/rl/rlinf_ext/act_policy.py`](../scripts/simulation/rl/rlinf_ext/act_policy.py).
> Extension registration:
> [`scripts/simulation/rl/rlinf_ext/__init__.py`](../scripts/simulation/rl/rlinf_ext/__init__.py).

### How It Works

RLinf wraps the ACT model in
[`ACTForRLActionPrediction`](../scripts/simulation/rl/rlinf_ext/act_policy.py), which:

1. Inherits from RLinf's `BasePolicy` ABC
2. Adds a `ValueHead` (3-layer MLP) for the PPO critic
3. Implements `default_forward()` for computing actions + log_probs + values
4. Implements `predict_action_batch()` for rollout inference
5. Converts between RLinf's observation format and ACT's expected input

The extension module
[`rlinf_ext/__init__.py`](../scripts/simulation/rl/rlinf_ext/__init__.py)
registers:

- Grasp policy gym IDs (`Isaac-Grasp-Policy-G129-Dex3-Joint` / `-Eval`)
- ACT observation converter (`_convert_dex3_obs_to_act_format`)
- ACT action converter (`_convert_act_action_to_sim` -- pads 28D to 43D)
- ACT model factory (`act_policy.get_model`)

### Quick Start

```bash
./docker/run_docker.sh -g1.5 \
    bash scripts/simulation/rl/train_act_grasp_policy.sh train \
    --model_path /models/act_grasp_policy
```

### PPO Configuration

Key parameters from the
[top-level config](../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `env.train.total_num_envs` | 64 | Parallel training environments |
| `env.eval.total_num_envs` | 64 | Parallel eval environments |
| `env.*.max_episode_steps` | 256 | Episode length |
| `algorithm.gamma` | 0.99 | Discount factor |
| `algorithm.gae_lambda` | 0.95 | GAE lambda |
| `algorithm.clip_ratio_high` | 0.2 | PPO clip ratio |
| `algorithm.rollout_epoch` | 8 | Rollouts per training epoch |
| `algorithm.update_epoch` | 4 | PPO update passes |
| `actor.optim.lr` | 5e-6 | Policy learning rate |
| `actor.optim.value_lr` | 1e-4 | Value head learning rate |
| `runner.max_epochs` | 1000 | Total training epochs |
| `runner.save_interval` | 2 | Checkpoint every N epochs |

The reward signal comes from the grasp_policy task's 3-stage sparse reward:
lift --> transport --> place.

### Common Overrides

```bash
# Fewer environments (lower VRAM)
bash scripts/simulation/rl/train_act_grasp_policy.sh train \
    --model_path /models/act_grasp_policy \
    env.train.total_num_envs=16 env.eval.total_num_envs=4

# Longer training
bash scripts/simulation/rl/train_act_grasp_policy.sh train \
    --model_path /models/act_grasp_policy \
    runner.max_epochs=2000

# Resume from RL checkpoint
bash scripts/simulation/rl/train_act_grasp_policy.sh train \
    --model_path /models/act_grasp_policy \
    runner.resume_dir=/path/to/rl_checkpoint
```

### Output

```
scripts/simulation/rl/results/act_grasp_policy/train_YYYYMMDD-HHMMSS/
├── checkpoints/
├── video/train/     # Training rollout videos
├── video/eval/      # Evaluation rollout videos
├── tensorboard/     # TensorBoard logs
└── train.log
```

Monitor training:

```bash
tensorboard --logdir scripts/simulation/rl/results/act_grasp_policy/
```

---

## 7. Evaluation

Evaluate trained ACT checkpoints (IL or RL) in simulation.

> **Code:**
> Evaluation script:
> [`scripts/simulation/examples/eval_grasp_policy.py`](../scripts/simulation/examples/eval_grasp_policy.py).
> ACT policy wrapper:
> [`scripts/simulation/act_closedloop_policy.py`](../scripts/simulation/act_closedloop_policy.py).
> Policy config:
> [`scripts/config/g1_act_closedloop_grasp_policy.yaml`](../scripts/config/g1_act_closedloop_grasp_policy.yaml).
> Shared base class:
> [`scripts/simulation/base_closedloop_policy.py`](../scripts/simulation/base_closedloop_policy.py).
> Observation processor:
> [`scripts/simulation/obs_processor.py`](../scripts/simulation/obs_processor.py).

### ACT IL Checkpoint

```bash
./docker/run_docker.sh -g1.5 \
    python scripts/simulation/examples/eval_grasp_policy.py \
    --policy_type act \
    --model_path /models/act_grasp_policy \
    --num_episodes 10 \
    --save_video
```

### ACT RL Checkpoint

```bash
./docker/run_docker.sh -g1.5 \
    python scripts/simulation/examples/eval_grasp_policy.py \
    --policy_type act \
    --model_path /path/to/rl_checkpoint \
    --num_episodes 20 \
    --save_video
```

### GR00T Checkpoint (for comparison)

```bash
./docker/run_docker.sh -g1.5 \
    python scripts/simulation/examples/eval_grasp_policy.py \
    --policy_type gr00t \
    --model_path /models/gr00t_grasp_policy \
    --num_episodes 10

# GR00T RL checkpoint (requires --rl_ckpt flag)
./docker/run_docker.sh -g1.5 \
    python scripts/simulation/examples/eval_grasp_policy.py \
    --policy_type gr00t \
    --model_path /path/to/gr00t_rl_ckpt \
    --rl_ckpt \
    --num_episodes 10
```

### Test Mode (No Checkpoint)

Verify the eval pipeline works without a trained model:

```bash
./docker/run_docker.sh -g1.5 \
    python scripts/simulation/examples/eval_grasp_policy.py --test
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Isaac-Grasp-Policy-G129-Dex3-Joint` | Gym task ID |
| `--policy_type` | `gr00t` | `gr00t`, `act`, or `test` |
| `--model_path` | (required) | Path to model checkpoint |
| `--policy_config_yaml` | (auto) | Policy config YAML (auto-generated for ACT) |
| `--num_episodes` | 10 | Number of evaluation episodes |
| `--max_steps` | 256 | Max steps per episode |
| `--action_chunk_size` | 1 | Actions per chunk step |
| `--save_video` | false | Save evaluation videos |
| `--video_dir` | `./eval_videos` | Video output directory |
| `--num_envs` | 1 | Parallel environments |
| `--success_stage` | 3 | Task success threshold |
| `--rl_ckpt` | false | Apply GR00T RL patch (gr00t only) |
| `--test` | false | Run with dummy zero policy |

### How ACT Evaluation Works

The [`ACTClosedloopPolicy`](../scripts/simulation/act_closedloop_policy.py) wrapper:

1. Loads the ACT checkpoint via `ACTPolicy.from_pretrained()`
2. Extracts observations from IsaacLab:
   - 28D joint state: arm joints from `robot_joint_state[:, 15:29]` + hand joints from `robot_dex3_joint_state`
   - 3 cameras normalized to float [0,1] in (B, C, H, W) format
3. Runs ACT forward pass, which predicts a **chunk** of 100 future actions
4. Caches the chunk and executes one action per step
5. When the chunk is exhausted, requests a new forward pass
6. Each action is 28D (arms + hands), padded with 15 leading zeros to 43D for the simulator

> **Code:** The action chunking lifecycle (chunk caching, index tracking, per-env reset)
> is implemented in
> [`base_closedloop_policy.py`](../scripts/simulation/base_closedloop_policy.py).
> The 28D-to-43D padding is in `pad_28d_to_43d()`.

### Output

```
eval_results/results_YYYYMMDD_HHMMSS_act.txt    # Per-episode results
eval_videos/<timestamp>_act_<model>_*.mp4        # Videos (if --save_video)
```

---

## 8. Architecture Reference

### Pipeline Architecture

```
                    SHARED DATA PIPELINE
                    ====================
  Teleop (AVP)  -->  HDF5 (43D actions, 101D state, 3 cameras)
                              |
                   convert_hdf5_to_lerobot.py
                              |
                    LeRobot Dataset (28D, Parquet + MP4)
                     /                          \
              +-----------+            +-----------+
              | GR00T SFT |            | ACT Train |
              | (existing)|            | (LeRobot) |
              +-----+-----+            +-----+-----+
                    |                         |
              GR00T Ckpt              ACT Checkpoint
                    |                         |
              +-----+-----+            +-----+-----+
              | GR00T RL  |            |  ACT RL   |
              |  (RLinf)  |            |  (RLinf)  |
              +-----+-----+            +-----+-----+
                    |                         |
              +-----+-----+            +-----+-----+
              | GR00T Eval|            | ACT Eval  |
              | (existing)|            |  (new)    |
              +-----------+            +-----------+
                     \                        /
                  PolicyBase interface (shared eval loop)
```

### Key Files

| Component | File | Description |
|-----------|------|-------------|
| **Task** | [`scripts/simulation/tasks/grasp_policy/`](../scripts/simulation/tasks/grasp_policy/) | Gym registration, env configs, MDP |
| **Recording** | [`scripts/simulation/record_demos_grasp_policy.sh`](../scripts/simulation/record_demos_grasp_policy.sh) | Demo recording wrapper |
| **Data Config** | [`scripts/config/g1_grasp_policy_dataset.yaml`](../scripts/config/g1_grasp_policy_dataset.yaml) | HDF5-to-LeRobot config |
| **Data Conversion** | [`scripts/utils/convert_hdf5_to_lerobot.py`](../scripts/utils/convert_hdf5_to_lerobot.py) | HDF5-to-LeRobot converter |
| **Field Mapping** | [`scripts/utils/assemble_trocar_lerobot_fields.py`](../scripts/utils/assemble_trocar_lerobot_fields.py) | 28D joint extraction |
| **IL Config** | [`scripts/policy/act_config.yaml`](../scripts/policy/act_config.yaml) | ACT training hyperparams |
| **IL Launcher** | [`scripts/policy/train_act_grasp_policy.sh`](../scripts/policy/train_act_grasp_policy.sh) | ACT IL training script |
| **RL Launcher** | [`scripts/simulation/rl/train_act_grasp_policy.sh`](../scripts/simulation/rl/train_act_grasp_policy.sh) | ACT RL training script |
| **RL Config** | [`scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml`](../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml) | PPO + env config |
| **RL Model** | [`scripts/simulation/rl/rlinf_ext/act_policy.py`](../scripts/simulation/rl/rlinf_ext/act_policy.py) | ACTForRLActionPrediction wrapper |
| **RL Extension** | [`scripts/simulation/rl/rlinf_ext/__init__.py`](../scripts/simulation/rl/rlinf_ext/__init__.py) | Model/env/converter registration |
| **Eval Script** | [`scripts/simulation/examples/eval_grasp_policy.py`](../scripts/simulation/examples/eval_grasp_policy.py) | Unified eval entry point |
| **Eval Policy** | [`scripts/simulation/act_closedloop_policy.py`](../scripts/simulation/act_closedloop_policy.py) | ACT closed-loop wrapper |
| **Eval Config** | [`scripts/config/g1_act_closedloop_grasp_policy.yaml`](../scripts/config/g1_act_closedloop_grasp_policy.yaml) | ACT inference config |
| **Base Policy** | [`scripts/simulation/base_closedloop_policy.py`](../scripts/simulation/base_closedloop_policy.py) | Action chunking base class |
| **Obs Processor** | [`scripts/simulation/obs_processor.py`](../scripts/simulation/obs_processor.py) | Model-agnostic obs extraction |

### Joint Dimension Flow

```
IsaacLab env step
    robot_joint_state: 87D (full body)
    robot_dex3_joint_state: 14D (hands)
            |
    Extract arm joints [15:29] = 14D
    Concatenate with hand joints = 14D
            |
    Policy input/output: 28D
    [left_arm(7) | right_arm(7) | left_hand(7) | right_hand(7)]
            |
    Pad with 15 leading zeros (legs + waist)
            |
    Simulator action: 43D
```

### RLinf Integration Pattern

```
RLINF_EXT_MODULE=rlinf_ext  (env var)
        |
    rlinf_ext/__init__.py:register()
        |
        +-- Register gym IDs: Isaac-Grasp-Policy-G129-Dex3-Joint[-Eval]
        +-- Register obs converter: _convert_dex3_obs_to_act_format()
        +-- Register action converter: _convert_act_action_to_sim()
        +-- Register model factory: act_policy.get_model()
        |
    RLinf training loop:
        env.step() --> _wrap_obs() --> obs converter --> ACT forward
                                                          |
        env.step(action) <-- action converter <-- 28D actions
```

---

## 9. Troubleshooting

### CloudXR / AVP

| Issue | Fix |
|-------|-----|
| AR panel shows `None` | Use `--xr` flag (built-in CloudXR) |
| Port 48010 already in use | `docker stop cloudxr-runtime` |
| Hand tracking doesn't move robot | Ensure `--teleop_device handtracking` |
| No video on AVP | Check firewall rules for UDP 47998-48012 |

See [`docs/cloudxr.md`](cloudxr.md) for more.

### Data Conversion

| Issue | Fix |
|-------|-----|
| `KeyError: 'processed_actions'` | Verify HDF5 was recorded with `--enable_cameras` |
| No camera videos in output | Check `rheo_camera_mappings_obs` in dataset YAML |
| Wrong number of joints | Verify `rheo_28d_state_action: true` in dataset YAML |

### Training

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: lerobot` | Rebuild Docker with `INSTALL_LEROBOT=true` |
| CUDA OOM during IL training | Reduce `training.batch_size` (try 32 or 16) |
| CUDA OOM during RL training | Reduce `env.train.total_num_envs` (try 16 or 8) |
| Poor IL convergence | Increase `training.offline_steps`, check demo quality |
| RL reward not improving | Verify IL checkpoint works in eval first |

### Evaluation

| Issue | Fix |
|-------|-----|
| `RuntimeError: ACTPolicy not found` | Rebuild Docker with `INSTALL_LEROBOT=true` |
| Wrong action dimensions | Check `policy_action_dim: 28` and `sim_action_dim: 43` in config |
| Robot doesn't move | Verify model checkpoint path is correct |
| GR00T RL checkpoint wrong output | Add `--rl_ckpt` flag (applies token padding patch) |
