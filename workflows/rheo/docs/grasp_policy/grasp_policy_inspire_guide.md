<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Grasp Policy (Inspire FTP): End-to-End Training Guide

Complete guide for training and evaluating an ACT (Action Chunking Transformer) policy
on the **grasp_policy_inspire** task (pick up a block and place it in a bin) using the
G1 robot with **Inspire FTP 5-finger hands**.

**Pipeline overview:**

```
Teleoperate (AVP dex-retargeting)  -->  Record HDF5  -->  Convert to LeRobot  -->  Train ACT (IL)
                                                                                        |
                                                                                        v
                                                           Evaluate  <--  RL Post-train (RLinf)
```

### Inspire FTP vs Dex3 — Key Differences

| Aspect | Dex3 | Inspire FTP |
|--------|------|-------------|
| Hand joints | 14 (7 per hand) | 24 (12 per hand: 6 actuated + 6 mimic) |
| Observed hand state | 14D (`robot_dex3_joint_state`) | 12D (`robot_inspire_joint_state`) |
| Policy action dim | 28D (14 arm + 14 hand) | 26D (14 arm + 12 hand) |
| Sim action dim | 43D | 53D (29 body + 24 hand) |
| Cameras | 3 (front + 2 wrist) | 1 (front only) |
| Teleop controller | WBC+PINK (23D) | PinkIK (38D) |
| Hand control | Binary gripper (pinch open/close) | Per-finger dex-retargeting |
| Mimic joints | None | 12 (with multiplier rules) |
| Gym IDs | `Isaac-Grasp-Policy-G129-Dex3-*` | `Isaac-Grasp-Policy-G129-InspireFTP-*` |

### Status Legend

Each section is marked with its current implementation status:

- **TESTED** — code runs and has been verified
- **UNTESTED** — code exists but has not been run end-to-end
- **TODO** — code or config files still need to be created

---

## Table of Contents

0. [Smoketest](#0-smoketest)
1. [Prerequisites](#1-prerequisites)
2. [Architecture — Key Differences from Dex3](#2-architecture--key-differences-from-dex3)
3. [Teleoperation with AVP](#3-teleoperation-with-avp)
4. [Recording Demonstrations](#4-recording-demonstrations)
5. [Data Conversion (HDF5 to LeRobot)](#5-data-conversion-hdf5-to-lerobot)
6. [ACT Imitation Learning Training](#6-act-imitation-learning-training)
7. [What You Can and Cannot Change Between IL and RL](#7-what-you-can-and-cannot-change-between-il-and-rl)
8. [RL Post-Training (PPO via RLinf)](#8-rl-post-training-ppo-via-rlinf)
9. [Evaluation](#9-evaluation)
10. [Key Files Reference](#10-key-files-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 0. Smoketest

**Status: TESTED**

Before anything else, verify the simulation environment loads and the robot spawns
correctly with the Inspire FTP hands.

```bash
./docker/run_docker_grasp.sh \
    python scripts/simulation/examples/eval_grasp_policy_inspire.py --test
```

**What to verify:**

- Environment creates without errors
- Robot spawns with 5-finger Inspire FTP hands visible
- 53D zero actions are accepted (dummy policy)
- Front camera renders in the viewport
- Episode completes and reports 0% success rate (expected with dummy policy)
- Action Manager shows `shape: 53` (direct joint control)
- Observation Manager shows `robot_joint_state (87,)` and `robot_inspire_joint_state (12,)`

> **Code:**
> [`scripts/simulation/examples/eval_grasp_policy_inspire.py`](../scripts/simulation/examples/eval_grasp_policy_inspire.py) —
> evaluation entry point supporting `--test` (dummy policy) and `--policy_type act`
> (ACT checkpoint).

---

## 1. Prerequisites

### Docker Image

The Inspire FTP task uses the same Docker image as the Dex3 grasp_policy task:

```bash
cd workflows/rheo
./docker/run_docker_grasp.sh -r    # build (first time or after changes)
```

> **Code:** Docker config in
> [`docker/Dockerfile.grasp`](../docker/Dockerfile.grasp) and
> [`docker/run_docker_grasp.sh`](../docker/run_docker_grasp.sh).
> Installs LeRobot, IsaacLab, and all dependencies.

### Hardware

- NVIDIA GPU with >= 16 GB VRAM (RTX 4090 / A6000 recommended)
- Apple Vision Pro (AVP) for hand-tracking teleoperation with dex-retargeting
- Both devices on the same network (CloudXR requires port 48010 TCP + 47998-48012 UDP)

### Host Directory Layout

The Docker container mounts these host directories:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `~/datasets` | `/datasets` | Recorded HDF5 demos + converted LeRobot datasets |
| `~/models` | `/models` | Trained model checkpoints |
| `~/eval` | `/eval` | Evaluation results and videos |

See [`docs/grasp_policy_guide.md`](grasp_policy_guide.md) Section 1 for full
prerequisites details (shared with Dex3).

---

## 2. Architecture — Key Differences from Dex3

### Joint Dimension Flow

```
IsaacLab env observations
    robot_joint_state: 87D (29 body joints x [pos|vel|torque])
    robot_inspire_joint_state: 12D (6 actuated per hand)
            |
    Extract arm joints from 87D [15:29] = 14D
    Concatenate with inspire hand joints = 12D
            |
    Policy input/output: 26D
    [left_arm(7) | right_arm(7) | left_hand(6) | right_hand(6)]
            |
    Scatter to 53D sim action (InspireFTPExperimentConfig.scatter_to_sim)
            |
    Mimic enforcement (InspireFTPJointPositionAction)
            |
    Simulator steps with 53D action (29 body + 24 hand)
```

> **Code:**
> Joint group definitions and scatter logic in
> [`scripts/utils/inspire_ftp_experiment_config.py`](../scripts/utils/inspire_ftp_experiment_config.py).
> Observation extraction in
> [`scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py`](../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py).

### 26D Canonical Joint Order

Both state and action use a 26D vector:

| Index Range | Body Part | Joints | DOF |
|-------------|-----------|--------|-----|
| 0-6 | Left arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw | 7 |
| 7-13 | Right arm | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw | 7 |
| 14-19 | Left hand | thumb_1 (yaw), thumb_2 (pitch), index_1, middle_1, ring_1, little_1 | 6 |
| 20-25 | Right hand | thumb_1 (yaw), thumb_2 (pitch), index_1, middle_1, ring_1, little_1 | 6 |

The arm joints are extracted from `robot_joint_state[:, 15:29]` (14 DOF) and hand
joints from `robot_inspire_joint_state` (12 DOF — actuated joints only).

### Mimic Joint Rules

The Inspire FTP hand has 12 mimic joints (6 per hand) that are mechanically coupled
to actuated joints. The env's `InspireFTPJointPositionAction` class enforces these
automatically — the policy only needs to output the 12 actuated joint targets.

| Mimic Joint | Parent Joint | Multiplier |
|-------------|-------------|------------|
| `{side}_index_2_joint` | `{side}_index_1_joint` | 1.0843 |
| `{side}_middle_2_joint` | `{side}_middle_1_joint` | 1.0843 |
| `{side}_ring_2_joint` | `{side}_ring_1_joint` | 1.0843 |
| `{side}_little_2_joint` | `{side}_little_1_joint` | 1.0843 |
| `{side}_thumb_3_joint` | `{side}_thumb_2_joint` | 0.8024 |
| `{side}_thumb_4_joint` | `{side}_thumb_3_joint` | 0.9487 |

> Processing order matters for the thumb chain: `_3` is computed from `_2`, then `_4`
> from `_3`.

> **Code:**
> [`scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py`](../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py).

### Single Camera

The Inspire FTP robot has no wrist camera mount links, so only the front camera
(`d435_link`) is used. This simplifies the vision pipeline but means the policy has
less spatial information compared to the 3-camera Dex3 setup.

### Teleop: PinkIK with Per-Finger Dex-Retargeting

| Aspect | Dex3 Teleop | Inspire FTP Teleop |
|--------|-------------|-------------------|
| Controller | WBC+PINK (23D) | PinkIK (38D) |
| Hand control | Binary gripper (pinch → open/close) | 24 direct joint targets |
| Retargeter | `G1HandtrackingGripperRetargeterCfg` | `UnitreeG1RetargeterCfg` |
| Action format | gripper(2) + wrist(14) + nav(3) + height(1) + torso(3) | wrist(14) + hand(24) |

The `UnitreeG1RetargeterCfg` maps 52 OpenXR hand joint poses (2 hands x 26 joints)
from the Apple Vision Pro to the 24 Inspire FTP finger actuators.

> **Note:** The retargeter uses Nucleus-style joint names (`L_index_proximal_joint`)
> internally, while our URDF-generated USD uses numbered names (`left_index_1_joint`).
> A name mapping (`_URDF_TO_NUCLEUS` dict) in the teleop env cfg bridges this gap.

> **Code:**
> [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py).

---

## 3. Teleoperation with AVP

**Status: TESTED** (launches, retargeter initializes; full AVP demo pending)

The teleop environment (`Isaac-Grasp-Policy-G129-InspireFTP-Teleop`) uses PinkIK with
full per-finger dex-retargeting from the Apple Vision Pro.

> **Code:** Task registration in
> [`scripts/simulation/tasks/grasp_policy_inspire/__init__.py`](../scripts/simulation/tasks/grasp_policy_inspire/__init__.py).
> Teleop environment config in
> [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py).

### 38D Action Space

| Indices | Dimensions | Control |
|---------|-----------|---------|
| 0-2 | 3 | Left wrist position (xyz) |
| 3-6 | 4 | Left wrist orientation (quat wxyz) |
| 7-9 | 3 | Right wrist position (xyz) |
| 10-13 | 4 | Right wrist orientation (quat wxyz) |
| 14-37 | 24 | Hand joints (all 24, interleaved L/R) |

PinkIK solves inverse kinematics for the 14 arm joints from the wrist pose targets.
The 24 hand joint targets are passed through directly.

### Launching XR Teleoperation

```bash
# Stop any external CloudXR container first
docker stop cloudxr-runtime 2>/dev/null

# Launch with AVP hand tracking + built-in CloudXR
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --num_demos 10 \
    --xr
```

**In Isaac Sim:** AR Panel --> "CloudXR Runtime (5.0)" --> "Start AR"

**On AVP:** Launch "Isaac XR Teleop Sample Client" --> Enter workstation IP --> Connect --> Play

> See [`docs/avp_teleoperation.md`](avp_teleoperation.md) and
> [`docs/cloudxr.md`](cloudxr.md) for detailed CloudXR setup and troubleshooting.

> **Note:** The `idle_action` tensor in the teleop env cfg contains approximate wrist
> positions. These may need empirical tuning at first run via FK at the default joint pose.

---

## 4. Recording Demonstrations

**Status: TESTED** (script launches; full recording with AVP pending)

Record demos using
[`scripts/simulation/record_demos.py`](../scripts/simulation/record_demos.py).
This is the same generic recording script used for Dex3 — no Inspire-specific
modifications needed.

### Quick Start

```bash
# Record 10 demos with AVP hand tracking
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --num_demos 10 \
    --xr
```

> **Important:** The `--enable_cameras` flag is required. Without it,
> `remove_camera_configs()` strips the front camera from the scene but leaves
> the observation term, causing a `front_camera does not exist` error.

### Recording Controls

When using XR (AVP), recording is controlled via **VR gestures** in the headset GUI:

| Gesture / Button | Action |
|------------------|--------|
| **START** | Start recording a demo |
| **STOP** | Stop and save current demo |
| **RESET** | Reset environment (discard current demo) |

**Auto-success detection:** When the block is placed on the target pad (stage 3),
the demo auto-saves after `--num_success_steps` consecutive successes (default: 1).

### HDF5 Output Format

The recorded HDF5 will contain:

| Key | Shape | Description |
|-----|-------|-------------|
| `processed_actions` | (T, 38) | PinkIK teleop actions (not 53D joint space) |
| `robot_joint_state` | (T, 87) | Full body state (29 joints x 3) |
| `robot_inspire_joint_state` | (T, 12) | Actuated hand joints only |
| `front_camera` | (T, 480, 640, 3) | Front camera RGB (single camera) |

> **Note:** No wrist camera images are recorded (Inspire FTP has no wrist mount links).

> **Recommendation:** Record 20-50 high-quality demonstrations. With auto-success
> detection, only successful grasps are saved automatically.

### Replaying Recorded Demos

**Status: TESTED**

```bash
# Replay all episodes
./docker/run_docker_grasp.sh \
    python scripts/simulation/replay_demos_isaaclab.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --enable_cameras --enable_pinocchio

# Replay with success validation
./docker/run_docker_grasp.sh \
    python scripts/simulation/replay_demos_isaaclab.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --enable_cameras --enable_pinocchio \
    --validate_success_rate
```

> **Note:** Use the `Teleop` task variant (not `Joint`) for replay since demos are
> recorded with the 38D PinkIK action space.

---

## 5. Data Conversion (HDF5 to LeRobot)

**Status: TODO** — converter needs Inspire FTP support

Convert recorded HDF5 demonstrations to LeRobot format (Parquet + MP4) for ACT training.

> **Code:**
> Conversion script:
> [`scripts/utils/convert_hdf5_to_lerobot.py`](../scripts/utils/convert_hdf5_to_lerobot.py) (Dex3-only, needs adaptation).
> Inspire field mappings:
> [`scripts/utils/inspire_ftp_lerobot_fields.py`](../scripts/utils/inspire_ftp_lerobot_fields.py) (exists).
> Modality definition:
> [`scripts/simulation/tasks/grasp_policy_inspire/modality_grasp_policy_inspire.json`](../scripts/simulation/tasks/grasp_policy_inspire/modality_grasp_policy_inspire.json) (exists).

### What Needs To Be Created

1. **Dataset config YAML** — `scripts/config/g1_grasp_policy_inspire_dataset.yaml`

   Parallel to
   [`scripts/config/g1_grasp_policy_dataset.yaml`](../scripts/config/g1_grasp_policy_dataset.yaml)
   (Dex3 version). Key changes:

   ```yaml
   language_instruction: "pick up block and place in bin"
   use_rheo_converter: true
   rheo_action_key: "processed_actions"
   rheo_26d_inspire_state_action: true    # Inspire FTP 26D extraction
   rheo_camera_mappings_obs:
     front_camera: "observation.images.cam_room"
     # No wrist cameras for Inspire FTP
   ```

2. **Converter adaptation** — `convert_hdf5_to_lerobot.py` needs a branch to:
   - Extract 26D state/action using `inspire_ftp_lerobot_fields.py` constants
   - Read `robot_inspire_joint_state` (12D) instead of `robot_dex3_joint_state` (14D)
   - Handle 38D PinkIK recorded actions (convert to 26D canonical policy space)

### Expected Output Structure

```
inspire_ftp_lerobot/
├── data/chunk-000/episode_000000.parquet    # 26D state/action per timestep
├── videos/chunk-000/
│   └── observation.images.cam_room/episode_000000.mp4   # Front camera only
└── meta/
    ├── info.json           # Feature schemas (26D joint names)
    ├── tasks.jsonl         # Task descriptions
    ├── episodes.jsonl      # Episode metadata
    └── modality.json       # From modality_grasp_policy_inspire.json
```

### Expected Command (Once Implemented)

```bash
./docker/run_docker_grasp.sh \
    python scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/g1_grasp_policy_inspire_dataset.yaml \
    --hdf5_dir /datasets/inspire_ftp \
    --output_dir /datasets/inspire_ftp_lerobot
```

---

## 6. ACT Imitation Learning Training

**Status: TODO** — config file needs to be created

Train an ACT policy on the converted LeRobot dataset.

> **Code:**
> Dex3 training config template:
> [`scripts/policy/act_config.yaml`](../scripts/policy/act_config.yaml).
> Training launcher:
> [`scripts/policy/train_act_grasp_policy.sh`](../scripts/policy/train_act_grasp_policy.sh).

### What Needs To Be Created

**ACT training config** — `scripts/policy/act_config_inspire.yaml`

Key changes from the Dex3 config:

```yaml
experiment:
  cameras:
    front_camera: "observation.images.cam_room"
    # No wrist cameras for Inspire FTP
  joint_groups:
    - left_arm    # 7 DOF
    - right_arm   # 7 DOF
    - left_hand   # 6 DOF (Inspire FTP actuated)
    - right_hand  # 6 DOF (Inspire FTP actuated)

policy:
  input_features:
    observation.state:
      type: STATE
      shape: [26]        # was [28] for Dex3
    observation.images.cam_room:
      type: VISUAL
      shape: [3, 480, 640]
    # No wrist camera features
  output_features:
    action:
      type: ACTION
      shape: [26]        # was [28] for Dex3
```

All other hyperparameters (chunk_size=100, dim_model=512, kl_weight=10, etc.)
remain the same as Dex3.

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy.chunk_size` | 100 | Action prediction horizon (timesteps) |
| `policy.dim_model` | 512 | Transformer hidden dimension |
| `policy.use_vae` | true | Use CVAE for stochastic training |
| `policy.kl_weight` | 10.0 | KL divergence loss weight |
| `policy.vision_backbone` | resnet18 | ImageNet-pretrained backbone |
| `steps` | 100000 | Total training steps |
| `batch_size` | 64 | Batch size |

**Input dimensions:**

- State: 26D (14 arm + 12 hand actuated joints)
- Images: 1 camera at 480x640 (front/room only)
- Action output: 26D

### Expected Command (Once Config Created)

```bash
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /datasets/inspire_ftp_lerobot \
    --config scripts/policy/act_config_inspire.yaml
```

---

## 7. What You Can and Cannot Change Between IL and RL

The same principles from the Dex3 guide apply. See
[`docs/grasp_policy_guide.md`](grasp_policy_guide.md) Section 6 for the full
explanation.

Updated for Inspire FTP:

### Locked After IL Demo Collection

| Thing | Inspire FTP Value |
|-------|-------------------|
| Policy observation dimensions | 26D joints + 1 camera |
| Camera resolution and mount point | 480x640 front camera on `d435_link` |
| Joint ordering | 26D canonical (arm14 + inspire_hand12) |
| Robot embodiment (URDF/USD) | G1 + Inspire FTP 5-finger hand |
| Action space dimensions | 26D |
| Control frequency / dt | 1/200s, decimation=4 |

### Free to Change for RL

Critic observations, reward functions, termination conditions, episode length,
number of parallel envs, PPO hyperparameters, object placement randomization.

### Mental Model

```
                    +---------------------+
  policy obs ------►|  IL Policy (frozen)  |------► actions
  (26D + 1 cam)    |  input/output locked |        (26D)
                    +---------------------+

                    +---------------------+
  critic obs ------►|  ValueHead (new)     |------► value estimate
  (anything)        |  trained from scratch|
                    +---------------------+

  rewards, terminations, stages -- only guide the RL optimizer
```

---

## 8. RL Post-Training (PPO via RLinf)

**Status: TODO** — runtime code is fully implemented; config files need to be created

The RLinf extension module has full Inspire FTP support:

- Environment wrapper (`IsaaclabGraspPolicyInspireEnv`) with correct 26D state extraction
- ACT obs/action converters (`act_inspire_ftp`) for 26D policy <-> 53D sim mapping
- Gym IDs registered: `Isaac-Grasp-Policy-G129-InspireFTP-Joint` and `-Joint-Eval`

> **Code:**
> [`scripts/simulation/rl/rlinf_ext/__init__.py`](../scripts/simulation/rl/rlinf_ext/__init__.py)
> — Inspire env wrapper (lines 565-636), ACT converters (lines 644-705).
> [`scripts/utils/inspire_ftp_experiment_config.py`](../scripts/utils/inspire_ftp_experiment_config.py)
> — 26D joint groups, scatter_to_sim (53D), state extraction.

### What Needs To Be Created

Three YAML config files plus a shell script, parallel to their Dex3 equivalents:

**1. Model config** — `scripts/simulation/rl/rlinf_ext/config/model/act_inspire_ftp.yaml`

Parallel to [`config/model/act_dex3.yaml`](../scripts/simulation/rl/rlinf_ext/config/model/act_dex3.yaml):

```yaml
model_type: "act"
model_path: "/path/to/act_inspire_ftp_checkpoint"
precision: "bf16"
action_dim: 26                    # was 28 for Dex3
num_action_chunks: 1
obs_converter_type: "act_inspire_ftp"   # was "act" for Dex3
add_value_head: True
rl_head_config:
  add_value_head: ${actor.model.add_value_head}
  disable_dropout: True
  noise_level: 0.3
```

**2. Env config** — `scripts/simulation/rl/rlinf_ext/config/env/isaaclab_grasp_policy_inspire.yaml`

Parallel to [`config/env/isaaclab_grasp_policy.yaml`](../scripts/simulation/rl/rlinf_ext/config/env/isaaclab_grasp_policy.yaml):

```yaml
env_type: isaaclab
init_params:
    id: "Isaac-Grasp-Policy-G129-InspireFTP-Joint"    # was Dex3
    task_description: "pick up block and place in bin"
```

**3. Main PPO config** — `scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml`

Parallel to [`config/isaaclab_ppo_act_grasp_policy.yaml`](../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml):

```yaml
defaults:
  - env/isaaclab_grasp_policy_inspire@env.train      # changed
  - env/isaaclab_grasp_policy_inspire@env.eval        # changed
  - model/act_inspire_ftp@actor.model                 # changed

runner:
  logger:
    experiment_name: "act_grasp_policy_inspire"       # changed

env:
  eval:
    init_params:
      id: "Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval"  # changed
```

All PPO hyperparameters (gamma=0.99, clip_ratio=0.2, etc.) remain the same.

**4. Training script** — `scripts/simulation/rl/train_act_grasp_policy_inspire.sh`

Copy of [`train_act_grasp_policy.sh`](../scripts/simulation/rl/train_act_grasp_policy.sh)
with `CONFIG_NAME="isaaclab_ppo_act_grasp_policy_inspire"`.

### Expected Command (Once Configs Created)

```bash
./docker/run_docker_grasp.sh \
    bash scripts/simulation/rl/train_act_grasp_policy_inspire.sh train \
    actor.model.model_path=/models/act_inspire_ftp \
    rollout.model.model_path=/models/act_inspire_ftp
```

### PPO Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_ratio` | 0.2 | PPO clip ratio |
| `update_epoch` | 4 | PPO update epochs |
| `rollout_epoch` | 8 | Rollout epochs |
| `actor.optim.lr` | 5e-6 | Actor learning rate |
| `actor.optim.value_lr` | 1e-4 | Critic learning rate |
| `env.train.total_num_envs` | 64 | Training environments |
| `env.train.max_episode_steps` | 256 | Steps per episode |

---

## 9. Evaluation

**Status: TESTED** (test mode with dummy policy)

### Dummy Policy (Smoketest)

```bash
./docker/run_docker_grasp.sh \
    python scripts/simulation/examples/eval_grasp_policy_inspire.py --test
```

### ACT IL Checkpoint

**Status: UNTESTED** (requires trained model + code fixes)

```bash
./docker/run_docker_grasp.sh \
    python scripts/simulation/examples/eval_grasp_policy_inspire.py \
    --policy_type act \
    --model_path /models/act_inspire_ftp \
    --num_episodes 10 \
    --save_video
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Isaac-Grasp-Policy-G129-InspireFTP-Joint` | Gym task ID |
| `--policy_type` | `act` | `act` or `test` |
| `--model_path` | None | Path to ACT checkpoint |
| `--num_episodes` | 10 | Number of evaluation episodes |
| `--max_steps` | 256 | Max steps per episode |
| `--action_chunk_size` | 1 | Actions per chunk to execute |
| `--save_video` | false | Save evaluation videos |
| `--success_stage` | 3 | Task success stage (grasp=1, transport=2, place=3) |
| `--enable_pinocchio` | false | Required for PinkIK (teleop task only) |

### Known Issues for ACT Evaluation

Two files need modifications before ACT checkpoint evaluation will work:

1. **`scripts/simulation/act_closedloop_policy.py`** — hardcodes `sim_action_dim = 43`
   (Dex3). Needs to read `sim_action_dim` from the policy config YAML (which
   `eval_grasp_policy_inspire.py` auto-generates with `sim_action_dim: 53`).

2. **`scripts/simulation/obs_processor.py`** — hardcodes `robot_dex3_joint_state`
   (14D). Needs a branch for `robot_inspire_joint_state` (12D).

> **Code:**
> [`scripts/simulation/examples/eval_grasp_policy_inspire.py`](../scripts/simulation/examples/eval_grasp_policy_inspire.py).
> [`scripts/simulation/act_closedloop_policy.py`](../scripts/simulation/act_closedloop_policy.py) (needs 53D fix).
> [`scripts/simulation/obs_processor.py`](../scripts/simulation/obs_processor.py) (needs Inspire branch).

---

## 10. Key Files Reference

### Inspire FTP Task Implementation

| File | Description |
|------|-------------|
| [`tasks/grasp_policy_inspire/__init__.py`](../scripts/simulation/tasks/grasp_policy_inspire/__init__.py) | Gym registrations (Joint, Joint-Eval, Teleop) |
| [`tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py`](../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py) | RL env config (53D action, 87D+12D obs, rewards, terminations) |
| [`tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py) | Teleop env config (38D PinkIK, XR, dex-retargeting) |
| [`tasks/grasp_policy_inspire/config/robot_config.py`](../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py) | Robot USD asset, actuator configs, default joint positions |
| [`tasks/grasp_policy_inspire/mdp/mimic_action.py`](../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py) | Mimic joint enforcement (12 rules) |
| [`tasks/grasp_policy_inspire/mdp/observations.py`](../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py) | Body (87D) + hand (12D) observation functions |
| [`tasks/grasp_policy_inspire/mdp/rewards.py`](../scripts/simulation/tasks/grasp_policy_inspire/mdp/rewards.py) | Reward functions (grasp, transport, place) — shared with Dex3 |
| [`tasks/grasp_policy_inspire/mdp/terminations.py`](../scripts/simulation/tasks/grasp_policy_inspire/mdp/terminations.py) | Termination conditions — shared with Dex3 |
| [`tasks/grasp_policy_inspire/mdp/events.py`](../scripts/simulation/tasks/grasp_policy_inspire/mdp/events.py) | Reset events — shared with Dex3 |

### Entry Points

| File | Description |
|------|-------------|
| [`examples/eval_grasp_policy_inspire.py`](../scripts/simulation/examples/eval_grasp_policy_inspire.py) | Evaluation (ACT/test modes) |
| [`simulation/record_demos.py`](../scripts/simulation/record_demos.py) | Demo recording (shared, generic) |
| [`simulation/replay_demos_isaaclab.py`](../scripts/simulation/replay_demos_isaaclab.py) | Demo replay (shared, generic) |

### Pipeline Utilities

| File | Description |
|------|-------------|
| [`utils/inspire_ftp_experiment_config.py`](../scripts/utils/inspire_ftp_experiment_config.py) | 26D joint groups, scatter_to_sim (53D), state extraction |
| [`utils/inspire_ftp_lerobot_fields.py`](../scripts/utils/inspire_ftp_lerobot_fields.py) | Joint index constants for HDF5 -> LeRobot conversion |
| [`utils/convert_hdf5_to_lerobot.py`](../scripts/utils/convert_hdf5_to_lerobot.py) | Dataset converter (Dex3-only, needs Inspire branch) |
| [`utils/inspect_inspire_ftp_joints.py`](../scripts/utils/inspect_inspire_ftp_joints.py) | Debug tool: USD joint ordering verification |

### RLinf Integration

| File | Description |
|------|-------------|
| [`rl/rlinf_ext/__init__.py`](../scripts/simulation/rl/rlinf_ext/__init__.py) | Inspire env wrapper (L565-636) + ACT converters (L644-705) |
| [`rl/rlinf_ext/act_policy.py`](../scripts/simulation/rl/rlinf_ext/act_policy.py) | ACT wrapper with ValueHead for RL (generic) |

### Docker

| File | Description |
|------|-------------|
| [`docker/Dockerfile.grasp`](../docker/Dockerfile.grasp) | Docker image (shared with Dex3) |
| [`docker/run_docker_grasp.sh`](../docker/run_docker_grasp.sh) | Docker launcher (shared with Dex3) |

---

## 11. Troubleshooting

### Inspire FTP-Specific Issues

| Issue | Fix |
|-------|-----|
| `ValueError: Not all regular expressions matched -- L_.*: []` | Joint names use URDF convention (`left_index_1_joint`), not Nucleus (`L_index_proximal_joint`). Check `robot_config.py` actuator patterns. |
| `KeyError: 'robot_dex3_joint_state'` | Wrong task ID or shared code assumes Dex3. Ensure using `InspireFTP` task variant. Check `examples/utils.py` handles both hand types. |
| `ValueError: Invalid action shape, expected: 38, received: 53` | You're running the eval script against the Teleop env. Use the `Joint` variant for eval, or `record_demos.py` for teleop. |
| `ValueError: Invalid action shape, expected: 53, received: 43` | Code assumes Dex3 43D sim action. Check `act_closedloop_policy.py` — needs 53D for Inspire. |
| `FrameNotFound: "g1_29dof_rev_1_0_left_wrist_yaw_link"` | PinkIK frame names use wrong prefix. URDF robot name produces prefix `g1_29dof_rev_1_0_with_inspire_hand_FTP_`. Update `FrameTask` link names in teleop env cfg. |
| `ValueError: 'L_index_proximal_joint' is not in list` | Retargeter uses Nucleus-style joint names. Ensure `RETARGETER_HAND_JOINT_NAMES` (Nucleus naming) is passed to the retargeter, not `HAND_JOINT_NAMES` (URDF naming). |
| `front_camera does not exist` | Pass `--enable_cameras` to `record_demos.py`. Without it, `remove_camera_configs()` strips the camera scene entity but leaves the observation term. |
| Only 1 camera image in dataset | Expected — Inspire FTP has front camera only (no wrist cameras). |
| Mimic joints not moving | Verify `InspireFTPJointPositionAction` is used in env cfg (not plain `JointPositionAction`). Check mimic rules in `mimic_action.py`. |
| USD warnings about `d435_link/visuals` unresolved | Cosmetic — sensor links in the URDF don't have visual meshes. Does not affect sim behavior. |

### Shared Issues

See [`docs/grasp_policy_guide.md`](grasp_policy_guide.md) Section 10 for
CloudXR/AVP issues, training issues, and general debugging.
