<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Grasp Policy (Inspire FTP): End-to-End Training Guide

Complete guide for training and evaluating an ACT (Action Chunking Transformer) policy
on the **grasp_policy_inspire** task (pick up a surgical tool from a tray and place it on a
target) using the G1 robot with **Inspire FTP 5-finger hands**. The scene spawns a surgical
tray with a single tool (default: tool_0) that the robot must grasp and place.

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
| Sim action dim | 43D | 41D (29 body + 12 actuated hand) |
| Cameras | 3 (front + 2 wrist) | 1 (front only) |
| Teleop controller | WBC+PINK (23D) | PinkIK (38D) |
| Hand control | Binary gripper (pinch open/close) | Full dex-retargeting (DexPilot IK) |
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
    python scripts/simulation/examples/eval_act_inspire.py \
    --test --enable_cameras --device cuda:0
```

> **Important:** `--enable_cameras` is required for all eval/recording scripts.
> Without it, camera scene entities are stripped but observation terms still reference
> them, causing runtime errors.
>
> **Important:** `--device cuda:0` is required for GPU acceleration. IsaacLab defaults
> to CPU when running with `--xr`, so always pass it explicitly.

**What to verify:**

- Environment creates without errors
- Robot spawns with 5-finger Inspire FTP hands visible
- A surgical tray appears on the table with a single tool (tool_0 by default)
- 41D zero actions are accepted (dummy policy)
- Front camera renders in the viewport
- Episode completes and reports 0% success rate (expected with dummy policy)
- Action Manager shows `shape: 41` (direct joint control, 29 body + 12 actuated hand)
- Observation Manager shows `robot_joint_state (87,)` and `robot_inspire_joint_state (12,)`

**Tool selection:** By default, `tool_0` is loaded in tray slot 1. Use `--object tool_1`
through `--object tool_4` to select a different tool. Use `--slot N` (0-5) to change the
tray slot.

> **Code:**
> [`scripts/simulation/examples/eval_act_inspire.py`](../../scripts/simulation/examples/eval_act_inspire.py) —
> the active evaluation entry point. Supports `--test` (dummy zero-action policy)
> and `--model_path` (ACT checkpoint).

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
    Scatter to 41D sim action (InspireFTPExperimentConfig.scatter_to_sim)
            |
    Simulator steps with 41D action (29 body + 12 actuated hand)
    Mimic joints (12) driven internally by InspireFTPJointPositionAction.apply_actions()
```

> **Code:**
> Joint group definitions and scatter logic in
> [`scripts/utils/inspire_experiment_config.py`](../../scripts/utils/inspire_experiment_config.py).
> Observation extraction in
> [`scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py).

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
> [`scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py).

### Cameras

Three cameras are mounted on the robot, matching the Dex3 setup:
- `front_camera` on `d435_link` (head-mounted, room view)
- `left_wrist_camera` on `left_hand_camera_base_link`
- `right_wrist_camera` on `right_hand_camera_base_link`

All three are published in `ObservationsCfg.CameraImagesCfg` and recorded into the
HDF5. Wrist camera mount links live in the `g1-29dof-inspire-ftp-usd-wrist_cam/`
USD variant; presets are `CameraPresets.left_inspire_wrist_camera` /
`right_inspire_wrist_camera` in `camera_config.py`.

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
> [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py).

---

## 3. Teleoperation with AVP

**Status: TESTED**

The teleop environment (`Isaac-Grasp-Policy-G129-InspireFTP-Teleop`) uses PinkIK with
full 5-finger dex-retargeting from the Apple Vision Pro via `UnitreeG1Retargeter`.
All fingers are individually tracked using DexPilot IK, which maps the operator's
hand pose to the 24 Inspire FTP finger joints via Nucleus hand-only URDFs.

> **Code:** Task registration in
> [`scripts/simulation/tasks/grasp_policy_inspire/__init__.py`](../../scripts/simulation/tasks/grasp_policy_inspire/__init__.py).
> Teleop environment config in
> [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py).

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

### Single-Arm Teleoperation

The `--arm` flag selects which arm(s) to control during teleop. The non-controlled
arm is locked at its initial pose using FK wrist readings (read once after each
reset, held constant throughout the episode). The non-controlled hand joints are
held at the idle pose.

| `--arm` value | Controlled | Locked |
|---------------|-----------|--------|
| `both` (default) | Both arms + both hands | Nothing |
| `right` | Right arm + right hand | Left arm + left hand |
| `left` | Left arm + left hand | Right arm + right hand |

> **Implementation detail:** Hand joint indices in the 38D action are interleaved
> L/R in USD articulation order — they are NOT contiguous per hand. The masking
> uses explicit per-hand index lists (`_LEFT_HAND_38D_IDX`, `_RIGHT_HAND_38D_IDX`)
> in `record_demos.py`. See the plan in the conversation history for the full
> index table.

### Launching XR Teleoperation

```bash
# Stop any external CloudXR container first
docker stop cloudxr-runtime 2>/dev/null

# Launch with AVP hand tracking + built-in CloudXR (dual-arm, default)
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --device cuda:0 \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --num_demos 10 \
    --xr

# Single right arm only (left arm locked)
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --device cuda:0 \
    --arm right \
    --dataset_file ./datasets/inspire_right_arm/demo.hdf5 \
    --num_demos 10 \
    --xr
```

**In Isaac Sim:** AR Panel --> "CloudXR Runtime (5.0)" --> "Start AR"

**On AVP:** Launch "Isaac XR Teleop Sample Client" --> Enter workstation IP --> Connect --> Play

> See [`docs/avp_teleoperation.md`](../utils/avp_teleoperation.md) and
> [`docs/cloudxr.md`](../utils/cloudxr.md) for detailed CloudXR setup and troubleshooting.

> **Note:** The `idle_action` tensor in the teleop env cfg contains approximate wrist
> positions. These may need empirical tuning at first run via FK at the default joint pose.

---

## 4. Recording Demonstrations

**Status: TESTED**

Record demos using
[`scripts/simulation/record_demos.py`](../../scripts/simulation/record_demos.py).
This is the same generic recording script used for Dex3 — no Inspire-specific
modifications needed.

### Quick Start

```bash
# Record 10 demos with AVP hand tracking — dual arm (default)
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --device cuda:0 \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --num_demos 10 \
    --xr

# Record 10 demos — right arm only (left arm locked at idle)
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --device cuda:0 \
    --arm right \
    --dataset_file ./datasets/inspire_right_arm/demo.hdf5 \
    --num_demos 10 \
    --xr
```

> **Important:** The `--enable_cameras` flag is required. Without it,
> `remove_camera_configs()` strips the front camera from the scene but leaves
> the observation term, causing a `front_camera does not exist` error.

> **Single-arm recording note:** The HDF5 always records the **full** body state
> (87D + 12D) and all available cameras regardless of `--arm`. The non-controlled
> arm will show constant joint values in the recording. The `--arm` flag only
> affects which arm responds to hand tracking during teleop — it does not change
> the observation or recording format. This means the same HDF5 can be converted
> to either 26D (dual-arm) or 13D (single-arm) LeRobot format after the fact.

### Tool and Slot Selection During Recording

By default, `tool_0` is spawned in tray slot 4. Use `--object` to select a
different tool and `--slot` to change the tray slot (0-5). On each reset, the
tool position is randomized +/-2 cm in X/Y with +/-15° yaw rotation.

```bash
# Record demos with a specific tool
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --device cuda:0 \
    --object tool_2 \
    --dataset_file ./datasets/inspire_ftp/tool_2_demos.hdf5 \
    --num_demos 10 \
    --xr

# Record demos with tool_3 in slot 1
./docker/run_docker_grasp.sh \
    python scripts/simulation/record_demos.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --teleop_device handtracking \
    --enable_pinocchio \
    --enable_cameras \
    --device cuda:0 \
    --object tool_3 --slot 1 \
    --dataset_file ./datasets/inspire_ftp/tool_3_slot1_demos.hdf5 \
    --num_demos 10 \
    --xr
```

Options: `--object` accepts `tool_0` (default), `tool_1`, `tool_2`, `tool_3`, `tool_4`.
`--slot` accepts 0-5 (default: 4).

### Recording Controls

When using XR (AVP), recording is controlled via **VR gestures** in the headset GUI:

| Gesture / Button | Action |
|------------------|--------|
| **START** | Start recording a demo |
| **STOP** | Stop and save current demo |
| **RESET** | Reset environment (discard current demo) |

**Auto-success detection:** When the tool is placed on the target pad (stage 3),
the demo auto-saves after `--num_success_steps` consecutive successes (default: 1).

### HDF5 Output Format

The recorded HDF5 will contain:

| Key | Shape | Description |
|-----|-------|-------------|
| `processed_actions` | (T, 38) | PinkIK teleop actions (not 41D joint space) |
| `robot_joint_state` | (T, 87) | Full body state (29 joints x 3) |
| `robot_inspire_joint_state` | (T, 12) | Actuated hand joints only |
| `front_camera` | (T, 480, 640, 3) | Front camera RGB |
| `left_wrist_camera` | (T, 480, 640, 3) | Left wrist camera RGB |
| `right_wrist_camera` | (T, 480, 640, 3) | Right wrist camera RGB |

> **Recommendation:** Record 20-50 high-quality demonstrations. With auto-success
> detection, only successful grasps are saved automatically.

### Replaying Recorded Demos

**Status: TESTED**

```bash
# Replay all episodes (default tool_0)
./docker/run_docker_grasp.sh \
    python scripts/simulation/replay_demos_isaaclab.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --enable_cameras --enable_pinocchio --device cuda:0

# Replay with success validation
./docker/run_docker_grasp.sh \
    python scripts/simulation/replay_demos_isaaclab.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --dataset_file ./datasets/inspire_ftp/demo.hdf5 \
    --enable_cameras --enable_pinocchio --device cuda:0 \
    --validate_success_rate

# Replay a dataset recorded with a non-default tool
./docker/run_docker_grasp.sh \
    python scripts/simulation/replay_demos_isaaclab.py \
    --task Isaac-Grasp-Policy-G129-InspireFTP-Teleop \
    --dataset_file ./datasets/inspire_tool2_slot1/single_arm.hdf5 \
    --object tool_2 \
    --enable_cameras --enable_pinocchio --device cuda:0
```

> **Note:** Use the `Teleop` task variant (not `Joint`) for replay since demos are
> recorded with the 38D PinkIK action space.

> **Tool selection (`--object`):** The HDF5 `initial_state` group restores per-episode
> poses (robot, block, target_pad) via `env.reset_to()`, so the `--slot` flag from
> `record_demos.py` has no equivalent on replay — the recorded block pose is played
> back regardless of the env's default slot. Geometry, however, is fixed at env-spawn
> time: if you recorded with a different tool, pass `--object <tool_N>` so replay
> spawns the matching USD. Mismatched geometry will leave the wrist trajectory
> physically correct in world space but grasping empty air relative to the loaded
> mesh. The HDF5 does not record which tool was used, so this flag must be set
> externally (e.g. from the dataset directory name).

---

## 5. Data Conversion (HDF5 to LeRobot)

**Status: TESTED**

Convert recorded HDF5 demonstrations to LeRobot format (Parquet + MP4) for ACT training.

> **Code:**
> Conversion script:
> [`scripts/utils/convert_hdf5_to_lerobot.py`](../../scripts/utils/convert_hdf5_to_lerobot.py).
> Inspire field mappings:
> [`scripts/utils/inspire_lerobot_fields.py`](../../scripts/utils/inspire_lerobot_fields.py).
> Dataset config:
> [`scripts/config/inspire/g1_grasp_policy_inspire_dataset.yaml`](../../scripts/config/inspire/g1_grasp_policy_inspire_dataset.yaml).
> Modality definition:
> [`scripts/simulation/tasks/grasp_policy_inspire/modality_grasp_policy_inspire.json`](../../scripts/simulation/tasks/grasp_policy_inspire/modality_grasp_policy_inspire.json).

### Running the Conversion

**Dual-arm (26D) — default:**

```bash
./docker/run_docker_grasp.sh \
    python scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/inspire/g1_grasp_policy_inspire_dataset.yaml
```

**Single-arm (13D) — right arm only:**

```bash
./docker/run_docker_grasp.sh \
    python scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/inspire/g1_grasp_policy_inspire_dataset_right_arm.yaml
```

**Single-arm (13D) — left arm only:**

```bash
./docker/run_docker_grasp.sh \
    python scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/inspire/g1_grasp_policy_inspire_dataset_left_arm.yaml
```

Three dataset configs are available:

| Config | Policy dim | Flag | Camera(s) | `data_root` |
|--------|-----------|------|-----------|-------------|
| `g1_grasp_policy_inspire_dataset.yaml` | 26D | `rheo_26d_state_action` | front | `datasets/inspire_ftp` |
| `g1_grasp_policy_inspire_dataset_right_arm.yaml` | 13D | `rheo_13d_state_action` | front | `datasets/inspire_right_arm` |
| `g1_grasp_policy_inspire_dataset_left_arm.yaml` | 13D | `rheo_13d_left_state_action` | front | `datasets/inspire_left_arm` |

The 13D configs extract only the controlled arm (7 joints) + hand (6 joints) from
the full-body HDF5 recording. The same HDF5 file can be converted with any of the
three configs — the `--arm` flag used during recording does not constrain which
conversion config you use. Key settings (example for 26D):

```yaml
use_rheo_converter: true
rheo_action_key: "processed_actions"
rheo_26d_state_action: true       # Triggers 26D Inspire extraction
rheo_camera_mappings_obs:
  front_camera: "observation.images.cam_room"
  left_wrist_camera: "observation.images.cam_left_wrist"
  right_wrist_camera: "observation.images.cam_right_wrist"
```

### How Actions Are Derived

The HDF5 `processed_actions` are 38D (PinkIK: 14D arm IK + 24D hand passthrough),
**not** 41D joint-space. The converter cannot directly extract 26D policy joints from
38D, so it uses **observation-derived actions**:

```
action[t] = state[t+1]    (next-step observed joint positions)
```

This is standard practice for teleop IL recordings. The fallback is automatic —
`convert_g1_state_action_to_lerobot_26d()` checks `action_full.shape[1]` — it handles
53D (legacy), 41D (current RL/eval), and falls back to observation-derived for anything
else (e.g. 38D teleop).

### Verified Output

Tested with 1 demo (434 timesteps). Output at `datasets/inspire_ftp/test/lerobot/`:

```
test/lerobot/
├── data/chunk-000/episode_000000.parquet    # 433 rows × 26D state + 26D action
├── videos/chunk-000/
│   └── observation.images.cam_room/episode_000000.mp4
└── meta/
    ├── info.json           # 26D joint names, feature schemas
    ├── tasks.jsonl         # "pick up block and place in bin"
    ├── episodes.jsonl      # Episode metadata
    └── modality.json       # 4 groups: left_arm(7), right_arm(7), left_hand(6), right_hand(6)
```

**State/action dimensions:** (433, 26) — 14 arm + 12 hand (6 actuated per hand).
433 frames = 434 timesteps minus 1 for the observation-derived shift.

**Joint names in `info.json`** (canonical 26D order):

```
 [0-6]   left arm:  left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/pitch/yaw
 [7-13]  right arm: right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw
[14-19]  left hand: L_thumb_proximal_yaw/pitch, L_index/middle/ring/pinky_proximal
[20-25]  right hand: R_thumb_proximal_yaw/pitch, R_index/middle/ring/pinky_proximal
```

---

## 6. ACT Imitation Learning Training

**Status: TESTED**

Train an ACT policy on the converted LeRobot dataset.

> **Code:**
> Training config:
> [`scripts/policy/act_config_inspire.yaml`](../../scripts/policy/act_config_inspire.yaml).
> Training launcher:
> [`scripts/policy/train_act_grasp_policy_inspire.sh`](../../scripts/policy/train_act_grasp_policy_inspire.sh).

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

### Key Differences from Dex3 Config

The Inspire FTP config (`act_config_inspire.yaml`) differs from the Dex3 config
(`act_config_dex3.yaml`) in:

- **`experiment.cameras`**: Currently front only (dataset also has both wrist
  cameras; wire them in when training with wrist vision)
- **`experiment.joint_groups`**: Hand groups have 6 DOF (not 7)
- **`input_features.observation.state.shape`**: [26] (not [28])
- **`output_features.action.shape`**: [26] (not [28])
- **`input_features`**: Currently no wrist camera features (can be added)

All other hyperparameters (chunk_size=100, dim_model=512, kl_weight=10, etc.)
remain the same as Dex3.

### Training Command

```bash
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy_inspire.sh \
    --dataset_path /workspaces/workflows/rheo/datasets/inspire_ftp/test/lerobot
```

Extra args are passed through to LeRobot's training script:

```bash
# Custom batch size and steps
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy_inspire.sh \
    --dataset_path /workspaces/workflows/rheo/datasets/inspire_ftp/test/lerobot \
    --steps 50000 --batch_size 32

# Resume from checkpoint
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy_inspire.sh \
    --dataset_path /workspaces/workflows/rheo/datasets/inspire_ftp/test/lerobot \
    --resume_path /models/act_inspire_ftp/checkpoint_50000
```

The script automatically:
- Generates `episodes_stats.jsonl` if missing (required by LeRobot v2.1)
- Strips the `experiment:` section before passing to LeRobot (it rejects unknown fields)
- Sets `INSPIRE_FTP_EXPERIMENT_CONFIG` env var for downstream code
- Saves output to `scripts/simulation/rl/results/act_grasp_policy_inspire/`

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

**Status: UNTESTED** — all config files and runtime code are implemented

The RLinf extension module has full Inspire FTP support:

- Environment wrapper (`IsaaclabGraspPolicyInspireEnv`) with correct 26D state extraction
- ACT obs/action converters (`act_inspire_ftp`) for 26D policy <-> 41D sim mapping
- Gym IDs registered: `Isaac-Grasp-Policy-G129-InspireFTP-Joint` and `-Joint-Eval`

> **Note (2026-04-13):** `-Joint-Eval` is now truly deterministic — `G1GraspPolicyInspireEvalEnvCfg.__post_init__` zeros `xy_noise` and `yaw_noise_deg` on `reset_block_position`, so every episode sees the identical block pose. Previously it was a `pass` stub that inherited the RL noise. `eval_act_inspire.py` still defaults to `-Joint`; future work: add a `--deterministic` flag (or switch the default) so the IL eval can select `-Joint-Eval` for checkpoint sweeps where you want variance to come from (tool, slot) grids instead of reset noise.

> **Code:**
> [`scripts/simulation/rl/rlinf_ext/__init__.py`](../../scripts/simulation/rl/rlinf_ext/__init__.py)
> — Inspire env wrapper (lines 565-636), ACT converters (lines 644-705).
> [`scripts/utils/inspire_experiment_config.py`](../../scripts/utils/inspire_experiment_config.py)
> — 26D joint groups, scatter_to_sim (41D), state extraction.

### RLinf Config Files

Three YAML configs (parallel to their Dex3 equivalents):

| Config | File | Key Difference from Dex3 |
|--------|------|--------------------------|
| **Model** | [`config/model/act_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/model/act_inspire.yaml) | `action_dim: 26` (was 28) |
| **Env** | [`config/env/isaaclab_grasp_policy_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/env/isaaclab_grasp_policy_inspire.yaml) | `id: "Isaac-Grasp-Policy-G129-InspireFTP-Joint"` |
| **PPO** | [`config/isaaclab_ppo_act_grasp_policy_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml) | References Inspire env/model configs, eval uses `-Joint-Eval` |

All PPO hyperparameters (gamma=0.99, clip_ratio=0.2, etc.) remain the same as Dex3.

### Training Command

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
    python scripts/simulation/examples/eval_act_inspire.py \
    --test --enable_cameras --device cuda:0
```

### ACT IL Checkpoint

```bash
./docker/run_docker_grasp.sh \
    python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /models/act_inspire_ftp \
    --arm right \
    --num_episodes 10 \
    --save_video \
    --enable_cameras --device cuda:0
```

### Tool and Slot Selection

By default, `tool_0` is loaded in tray slot 1. Use `--object` and `--slot` to override:

```bash
# Evaluate on tool_2 in default slot
./docker/run_docker_grasp.sh \
    python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /models/act_inspire_ftp \
    --object tool_2 --enable_cameras --device cuda:0

# Evaluate on tool_0 in slot 4
./docker/run_docker_grasp.sh \
    python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /models/act_inspire_ftp \
    --slot 4 --enable_cameras --device cuda:0
```

The eval script writes a temporary policy config YAML and the
`ACTClosedloopPolicy` wrapper loads `InspireFTPExperimentConfig` (26D / 13D
policy, 41D sim scatter, front camera by default).

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval` | Gym task ID. `-Joint-Eval` is deterministic (zero block XY/yaw noise); use `-Joint` for the noisy training env. |
| `--model_path` | None | Path to ACT checkpoint. Omit with `--test` for dummy zero-action policy. |
| `--test` | false | Run with dummy zero-action policy (no checkpoint needed). |
| `--num_episodes` | 1 | Number of evaluation episodes. |
| `--max_steps` | 300 | Max steps per episode. |
| `--action_chunk_size` | 50 | Actions per chunk to execute. |
| `--arm` | `right` | `dual`, `left`, or `right` — must match the dim of the trained policy (26D dual / 13D single). |
| `--object` | `tool_0` | Grasp tool: `tool_0`..`tool_4`. |
| `--slot` | 1 | Tray slot index (0-5). |
| `--save_video` | false | Save evaluation videos. |
| `--video_dir` | `./eval_videos` | Where to save videos. |
| `--success_stage` | 3 | Task success stage (grasp=1, transport=2, place=3). |
| `--seed` | 4 | RNG seed. |
| `--temporal_ensemble_coeff` | None | Enable LeRobot ACT temporal ensembling at this exponential coefficient. |
| `--clamp_actions` | 0.0 | Clip per-step action delta from the held base action (debugging). |
| `--log_actions` | false | Log per-chunk action mean/std statistics — primary diagnostic for chunk-collapse. |
| `--dump_first_obs` | None | Dump the first env observation to this dir; pair with `scripts/utils/diff_first_obs.py` to compare against a recorded reference. |
| `--pin_block_from_hdf5` | None | Pin the block to the pose at frame `--pin_block_frame_idx` from `--pin_demo_key` of the given HDF5. |
| `--device` | `cuda:0` | **Recommended.** Simulation device. XR mode overrides to CPU if not set explicitly. |
| `--enable_cameras` | false | **Required.** Enable camera rendering for observations. |

> **Code:**
> [`scripts/simulation/examples/eval_act_inspire.py`](../../scripts/simulation/examples/eval_act_inspire.py).
> [`scripts/simulation/act_closedloop_policy.py`](../../scripts/simulation/act_closedloop_policy.py)
> — Inspire-FTP-only ACT eval wrapper (26D / 13D policy → 41D sim scatter via
> `InspireFTPExperimentConfig`).

---

## 10. Key Files Reference

### Inspire FTP Task Implementation

| File | Description |
|------|-------------|
| [`tasks/grasp_policy_inspire/__init__.py`](../../scripts/simulation/tasks/grasp_policy_inspire/__init__.py) | Gym registrations (Joint, Joint-Eval, Teleop) |
| [`tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py) | RL env config (41D action, 87D+12D obs, rewards, terminations) |
| [`tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py) | Teleop env config (38D PinkIK, XR, dex-retargeting) |
| [`tasks/grasp_policy_inspire/config/robot_config.py`](../../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py) | Robot USD asset, actuator configs, default joint positions |
| [`tasks/grasp_policy_inspire/mdp/mimic_action.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py) | Mimic joint enforcement (12 rules) |
| [`tasks/grasp_policy_inspire/mdp/observations.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py) | Body (87D) + hand (12D) observation functions |
| [`tasks/grasp_policy_inspire/mdp/rewards.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/rewards.py) | Reward functions (grasp, transport, place) |
| [`tasks/grasp_policy_inspire/mdp/terminations.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/terminations.py) | Termination conditions |
| [`tasks/grasp_policy_inspire/mdp/events.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/events.py) | Reset events |

### Entry Points

| File | Description |
|------|-------------|
| [`examples/eval_act_inspire.py`](../../scripts/simulation/examples/eval_act_inspire.py) | Evaluation entry point (ACT / `--test` dummy modes) |
| [`simulation/record_demos.py`](../../scripts/simulation/record_demos.py) | Demo recording (shared, generic) |
| [`simulation/replay_demos_isaaclab.py`](../../scripts/simulation/replay_demos_isaaclab.py) | Demo replay (shared, generic) |

### Scene Assets

| File | Description |
|------|-------------|
| [`assets/sinus_toolkit_v1/`](../assets/sinus_toolkit_v1/) | 5 surgical tool .obj meshes (tool_0..tool_4) + surgical tray USD |
| [`simulation/assets/convert_sinus_toolkit.py`](../../scripts/simulation/assets/convert_sinus_toolkit.py) | Batch .obj→.usd converter (run inside Docker) |
| [`simulation/assets/assets.py`](../../scripts/simulation/assets/assets.py) | USD path constants (`SINUS_TOOL_USD_PATHS`) |

### Pipeline Utilities

| File | Description |
|------|-------------|
| [`utils/inspire_experiment_config.py`](../../scripts/utils/inspire_experiment_config.py) | 26D joint groups, scatter_to_sim (41D), state extraction |
| [`utils/inspire_lerobot_fields.py`](../../scripts/utils/inspire_lerobot_fields.py) | Joint index constants for HDF5 -> LeRobot conversion (26D, 13D right, 13D left) |
| [`utils/convert_hdf5_to_lerobot.py`](../../scripts/utils/convert_hdf5_to_lerobot.py) | Dataset converter (handles 53D, 41D, and 38D teleop) |
| [`config/g1_grasp_policy_inspire_dataset_right_arm.yaml`](../../scripts/config/inspire/g1_grasp_policy_inspire_dataset_right_arm.yaml) | 13D right-arm conversion config |
| [`config/g1_grasp_policy_inspire_dataset_left_arm.yaml`](../../scripts/config/inspire/g1_grasp_policy_inspire_dataset_left_arm.yaml) | 13D left-arm conversion config |
| [`utils/inspect_inspire_joints.py`](../../scripts/utils/inspect_inspire_joints.py) | Debug tool: USD joint ordering verification |

### ACT Training

| File | Description |
|------|-------------|
| [`policy/act_config_inspire.yaml`](../../scripts/policy/act_config_inspire.yaml) | IL training config (26D state/action, 1 camera) |
| [`policy/train_act_grasp_policy_inspire.sh`](../../scripts/policy/train_act_grasp_policy_inspire.sh) | IL training launcher (LeRobot) |
| [`simulation/act_closedloop_policy.py`](../../scripts/simulation/act_closedloop_policy.py) | ACT eval wrapper (Inspire FTP: 26D / 13D policy → 41D sim scatter) |

### RLinf Integration

| File | Description |
|------|-------------|
| [`rl/rlinf_ext/__init__.py`](../../scripts/simulation/rl/rlinf_ext/__init__.py) | Inspire env wrapper (L565-636) + ACT converters (L644-705) |
| [`rl/rlinf_ext/act_policy.py`](../../scripts/simulation/rl/rlinf_ext/act_policy.py) | ACT wrapper with ValueHead for RL (generic) |
| [`rl/rlinf_ext/config/model/act_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/model/act_inspire.yaml) | RLinf model config (action_dim=26) |
| [`rl/rlinf_ext/config/env/isaaclab_grasp_policy_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/env/isaaclab_grasp_policy_inspire.yaml) | RLinf env config (InspireFTP gym ID) |
| [`rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml) | RLinf PPO top-level config |

### Docker

| File | Description |
|------|-------------|
| [`docker/Dockerfile.grasp`](../docker/Dockerfile.grasp) | Docker image |
| [`docker/run_docker_grasp.sh`](../docker/run_docker_grasp.sh) | Docker launcher |

---

## 11. Troubleshooting

### Inspire FTP-Specific Issues

| Issue | Fix |
|-------|-----|
| `ValueError: Not all regular expressions matched -- L_.*: []` | Joint names use URDF convention (`left_index_1_joint`), not Nucleus (`L_index_proximal_joint`). Check `robot_config.py` actuator patterns. |
| `KeyError: 'robot_dex3_joint_state'` | Wrong task ID. Ensure you're on the `InspireFTP` gym variant — the Dex3 grasp task is no longer registered. |
| `ValueError: Invalid action shape, expected: 38, received: 41` | You're running the eval script against the Teleop env. Use the `Joint` variant for eval, or `record_demos.py` for teleop. |
| `ValueError: Invalid action shape, expected: 41, received: 43` | Stale policy config left over from the Dex3 era. `ACTClosedloopPolicy` is now Inspire-FTP-only and always emits a 41D action — regenerate the policy config YAML by re-running the eval script. |
| `FileExistsError: Output directory ... already exists` | LeRobot rejects pre-existing output dirs. Delete the old run: `rm -rf scripts/simulation/rl/results/act_grasp_policy_inspire/` |
| `DecodingError: fields 'experiment' are not valid for TrainPipelineConfig` | The `experiment:` section must be stripped before LeRobot sees the config. Use `train_act_grasp_policy_inspire.sh` (handles this automatically). |
| `PermissionError: ... episodes_stats.jsonl` | Dataset directory owned by root. Fix: `sudo chown -R $USER:$USER datasets/` |
| `FrameNotFound: "g1_29dof_rev_1_0_left_wrist_yaw_link"` | PinkIK frame names use wrong prefix. URDF robot name produces prefix `g1_29dof_rev_1_0_with_inspire_hand_FTP_`. Update `FrameTask` link names in teleop env cfg. |
| `ValueError: 'L_index_proximal_joint' is not in list` | Retargeter uses Nucleus-style joint names. Ensure `RETARGETER_HAND_JOINT_NAMES` (Nucleus naming) is passed to the retargeter, not `HAND_JOINT_NAMES` (URDF naming). |
| `front_camera does not exist` | Pass `--enable_cameras` to all scripts (`eval_act_inspire.py`, `record_demos.py`, `replay_demos_isaaclab.py`). Without it, `remove_camera_configs()` strips the camera scene entity but leaves the observation term. |
| `tool_N/tool_N.usd not found` | Run the mesh converter first: `./docker/run_docker_grasp.sh python scripts/simulation/assets/convert_sinus_toolkit.py`. The .obj files must be converted to .usd before the scene can load them. |
| Only 1 camera image in dataset | The dual-arm YAML (`g1_grasp_policy_inspire_dataset.yaml`) must map all three cameras under `rheo_camera_mappings_obs`. If only `front_camera` is mapped, wrist streams are skipped. |
| Mimic joints not moving | Verify `InspireFTPJointPositionAction` is used in env cfg (not plain `JointPositionAction`). Check mimic rules in `mimic_action.py`. |
| USD warnings about `d435_link/visuals` unresolved | Cosmetic — sensor links in the URDF don't have visual meshes. Does not affect sim behavior. |

### Adjacent References

For elbow-offset / pipeline-invariant issues see
[`docs/inspire/pipeline_contracts.md`](pipeline_contracts.md)
and [`docs/inspire/elbow_offset.md`](elbow_offset.md). For CloudXR / AVP setup see
[`docs/utils/cloudxr.md`](../utils/cloudxr.md) and
[`docs/utils/avp_teleoperation.md`](../utils/avp_teleoperation.md).
