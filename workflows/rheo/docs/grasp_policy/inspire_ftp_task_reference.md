# Inspire FTP Grasp Policy Task Reference

Technical reference for the G1 + Inspire FTP block pick-and-place task.
Covers both the **RL/eval** (53D joint control) and **teleop** (38D PinkIK) variants,
plus the 26D data conversion pipeline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Gym IDs and Registration](#2-gym-ids-and-registration)
3. [Scene Setup](#3-scene-setup)
4. [Action Space](#4-action-space)
   - [4a. RL/Eval: 53D Joint Position](#4a-rleval-53d-joint-position)
   - [4b. Teleop: 38D PinkIK](#4b-teleop-38d-pinkinverse-kinematics)
5. [Observation Space](#5-observation-space)
6. [Reward Structure](#6-reward-structure)
7. [Termination Conditions](#7-termination-conditions)
8. [Reset Events](#8-reset-events)
9. [Teleop Devices](#9-teleop-devices)
10. [Data Pipeline (26D Policy Format)](#10-data-pipeline-26d-policy-format)
11. [Actuator Configuration](#11-actuator-configuration)
12. [Comparison: Inspire FTP vs Dex3](#12-comparison-inspire-ftp-vs-dex3)
13. [RLinf Integration](#13-rlinf-integration)
14. [File Reference](#14-file-reference)

---

## 1. Overview

The task is a **block pick-and-place**: grasp a 5 cm red cube from the table, transport
it to a green target pad (bin), and place it inside. The robot is a Unitree G1 (29 body DOF)
with **Inspire FTP 5-finger hands** (24 hand joints: 12 actuated + 12 mimic, per pair of hands).

Three gym variants exist:

| Variant | Gym ID | Action Dim | Episode | Purpose |
|---------|--------|------------|---------|---------|
| **RL Training** | `Isaac-Grasp-Policy-G129-InspireFTP-Joint` | 53D | 20 s | Random block placement |
| **RL Evaluation** | `Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval` | 53D | 20 s | Deterministic block placement |
| **Teleoperation** | `Isaac-Grasp-Policy-G129-InspireFTP-Teleop` | 38D | 300 s | PinkIK + AVP hand tracking |

Key differences from the Dex3 variant: 53D actions (vs 43D), 12D hand observation (vs 14D),
front camera only (vs 3 cameras), 26D policy dim (vs 28D).

---

## 2. Gym IDs and Registration

**File:** `scripts/simulation/tasks/grasp_policy_inspire/__init__.py`

```python
"Isaac-Grasp-Policy-G129-InspireFTP-Joint"       # G1GraspPolicyInspireEnvCfg
"Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval"   # G1GraspPolicyInspireEvalEnvCfg
"Isaac-Grasp-Policy-G129-InspireFTP-Teleop"        # G1GraspPolicyInspireTeleopEnvCfg
```

All three register with `entry_point="isaaclab.envs:ManagerBasedRLEnv"`.
The Eval variant inherits the RL config but uses deterministic block placement.

---

## 3. Scene Setup

**Files:** `g1_grasp_policy_inspire_env_cfg.py` (lines 130-178), `config/robot_config.py`

| Entity | Size / Type | Init Position | Notes |
|--------|-------------|---------------|-------|
| **Robot** | G1 29DOF + Inspire FTP | (-1.849, 1.94, 0.812) | Fixed base, gravity disabled |
| **Block** | 5 cm cube, 0.1 kg | (-1.55, 1.90, 0.885) | Red, friction 0.8/0.6 |
| **Target Pad** | 15x15x0.5 cm, 0.3 kg | (-1.55, 1.61, 0.8375) | Green, friction 1.0/0.8 |
| **Front Camera** | RGB 640x480 | On robot head | focal_length=10.5 |
| **Scene** | Surgical room USD | - | Trocar assembly scene (table) |
| **Light** | Dome light | - | (0.75, 0.75, 0.75) intensity 1000 |

**Table height (TABLE_Z):** 0.855 m

No wrist cameras are available on the Inspire FTP hand (no camera mount links in the USD).

---

## 4. Action Space

### 4a. RL/Eval: 53D Joint Position

**File:** `g1_grasp_policy_inspire_env_cfg.py` (lines 185-195), `mdp/mimic_action.py`

The action is a 53D joint position target: **29 body + 24 hand joints**.
Uses `InspireFTPJointPositionAction`, which enforces mimic constraints after
standard position target processing.

#### Full 53-Joint List (USD Articulation Order)

| Idx | Joint Name | Group |
|-----|-----------|-------|
| 0 | `left_hip_pitch_joint` | Legs |
| 1 | `right_hip_pitch_joint` | Legs |
| 2 | `waist_yaw_joint` | Waist |
| 3 | `left_hip_roll_joint` | Legs |
| 4 | `right_hip_roll_joint` | Legs |
| 5 | `waist_roll_joint` | Waist |
| 6 | `left_hip_yaw_joint` | Legs |
| 7 | `right_hip_yaw_joint` | Legs |
| 8 | `waist_pitch_joint` | Waist |
| 9 | `left_knee_joint` | Legs |
| 10 | `right_knee_joint` | Legs |
| 11 | `left_shoulder_pitch_joint` | Arms |
| 12 | `right_shoulder_pitch_joint` | Arms |
| 13 | `left_ankle_pitch_joint` | Feet |
| 14 | `right_ankle_pitch_joint` | Feet |
| 15 | `left_shoulder_roll_joint` | Arms |
| 16 | `right_shoulder_roll_joint` | Arms |
| 17 | `left_ankle_roll_joint` | Feet |
| 18 | `right_ankle_roll_joint` | Feet |
| 19 | `left_shoulder_yaw_joint` | Arms |
| 20 | `right_shoulder_yaw_joint` | Arms |
| 21 | `left_elbow_joint` | Arms |
| 22 | `right_elbow_joint` | Arms |
| 23 | `left_wrist_roll_joint` | Arms |
| 24 | `right_wrist_roll_joint` | Arms |
| 25 | `left_wrist_pitch_joint` | Arms |
| 26 | `right_wrist_pitch_joint` | Arms |
| 27 | `left_wrist_yaw_joint` | Arms |
| 28 | `right_wrist_yaw_joint` | Arms |
| 29 | `left_index_1_joint` | Hand (actuated) |
| 30 | `left_little_1_joint` | Hand (actuated) |
| 31 | `left_middle_1_joint` | Hand (actuated) |
| 32 | `left_ring_1_joint` | Hand (actuated) |
| 33 | `left_thumb_1_joint` | Hand (actuated) |
| 34 | `right_index_1_joint` | Hand (actuated) |
| 35 | `right_little_1_joint` | Hand (actuated) |
| 36 | `right_middle_1_joint` | Hand (actuated) |
| 37 | `right_ring_1_joint` | Hand (actuated) |
| 38 | `right_thumb_1_joint` | Hand (actuated) |
| 39 | `left_index_2_joint` | Hand (mimic) |
| 40 | `left_little_2_joint` | Hand (mimic) |
| 41 | `left_middle_2_joint` | Hand (mimic) |
| 42 | `left_ring_2_joint` | Hand (mimic) |
| 43 | `left_thumb_2_joint` | Hand (actuated) |
| 44 | `right_index_2_joint` | Hand (mimic) |
| 45 | `right_little_2_joint` | Hand (mimic) |
| 46 | `right_middle_2_joint` | Hand (mimic) |
| 47 | `right_ring_2_joint` | Hand (mimic) |
| 48 | `right_thumb_2_joint` | Hand (actuated) |
| 49 | `left_thumb_3_joint` | Hand (mimic) |
| 50 | `right_thumb_3_joint` | Hand (mimic) |
| 51 | `left_thumb_4_joint` | Hand (mimic) |
| 52 | `right_thumb_4_joint` | Hand (mimic) |

**Note:** `thumb_1` = yaw, `thumb_2` = proximal pitch (both actuated).
`thumb_3` = intermediate, `thumb_4` = distal (both mimic).

#### Mimic Rules (12 total: 6 per hand)

| Mimic Joint | Parent Joint | Multiplier |
|-------------|-------------|------------|
| `{side}_index_2_joint` | `{side}_index_1_joint` | 1.0843 |
| `{side}_middle_2_joint` | `{side}_middle_1_joint` | 1.0843 |
| `{side}_ring_2_joint` | `{side}_ring_1_joint` | 1.0843 |
| `{side}_little_2_joint` | `{side}_little_1_joint` | 1.0843 |
| `{side}_thumb_3_joint` | `{side}_thumb_2_joint` | 0.8024 |
| `{side}_thumb_4_joint` | `{side}_thumb_3_joint` | 0.9487 |

**Chained dependency:** `thumb_4 = thumb_3 x 0.9487 = thumb_2 x 0.8024 x 0.9487`.
Processing order matters: thumb_3 must be computed before thumb_4.

#### Action Offsets

| Joint | Offset |
|-------|--------|
| `left_elbow_joint` | -0.3 |
| `right_elbow_joint` | -0.3 |

### 4b. Teleop: 38D PinkIK

**File:** `g1_grasp_policy_inspire_teleop_env_cfg.py` (lines 60-125)

The teleop variant uses **PinkIK** inverse kinematics for the arms and passes hand
joints through directly.

```
38D = [left_wrist_pos(3) | left_wrist_quat(4) | right_wrist_pos(3) | right_wrist_quat(4) | hand_joints(24)]
```

| Segment | Dims | Description |
|---------|------|-------------|
| Left wrist position | 3 | World-frame (x, y, z) |
| Left wrist quaternion | 4 | (w, x, y, z) |
| Right wrist position | 3 | World-frame (x, y, z) |
| Right wrist quaternion | 4 | (w, x, y, z) |
| Hand joints | 24 | Direct joint targets (all 24 including mimic) |

**PinkIK Configuration:**

| Parameter | Value |
|-----------|-------|
| Base link | `pelvis` |
| Controlled arm joints | 7 per side (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) |
| End-effector links | `left_wrist_yaw_link`, `right_wrist_yaw_link` |
| FrameTask position cost | 8.0 |
| FrameTask orientation cost | 2.0 |
| FrameTask damping | 10.0 |
| FrameTask gain | 0.5 |
| NullSpacePostureTask cost | 0.5 |
| NullSpacePostureTask gain | 0.3 |
| NullSpace controlled joints | 6 shoulder joints + 3 waist joints |

**Idle Action (38D default pose):**

```
Left wrist:  pos=(-1.9979, 2.1438, 1.0952), quat=(0.707, 0, 0, 0.707)
Right wrist: pos=(-1.7005, 2.1438, 1.0952), quat=(0.707, 0, 0, 0.707)
Hand joints: all 0.0 (open)
```

**Teleop overrides vs RL:**
- Episode length: **300 s** (vs 20 s)
- Render interval: **2** (vs 4, smoother XR visuals)
- URDF conversion at init for PinkIK solver

---

## 5. Observation Space

**File:** `mdp/observations.py`

Observations are returned as a nested dict (not concatenated):

```python
{
    "policy": {
        "robot_joint_state": (B, 87),           # 29 body x [pos|vel|torque]
        "robot_inspire_joint_state": (B, 12),   # 6 actuated per hand (positions only)
    },
    "camera_images": {
        "front_camera": (B, 480, 640, 3),       # RGB uint8
    }
}
```

### 5a. Body Joint State (87D)

`get_robot_body_joint_states()` returns 29 positions + 29 velocities + 29 torques
in **canonical order** (differs from USD order):

| Idx | Joint | Idx | Joint |
|-----|-------|-----|-------|
| 0 | `left_hip_pitch_joint` | 15 | `left_shoulder_pitch_joint` |
| 1 | `right_hip_pitch_joint` | 16 | `left_shoulder_roll_joint` |
| 2 | `left_hip_roll_joint` | 17 | `left_shoulder_yaw_joint` |
| 3 | `right_hip_roll_joint` | 18 | `left_elbow_joint` |
| 4 | `left_hip_yaw_joint` | 19 | `left_wrist_roll_joint` |
| 5 | `right_hip_yaw_joint` | 20 | `left_wrist_pitch_joint` |
| 6 | `left_knee_joint` | 21 | `left_wrist_yaw_joint` |
| 7 | `right_knee_joint` | 22 | `right_shoulder_pitch_joint` |
| 8 | `left_ankle_pitch_joint` | 23 | `right_shoulder_roll_joint` |
| 9 | `right_ankle_pitch_joint` | 24 | `right_shoulder_yaw_joint` |
| 10 | `left_ankle_roll_joint` | 25 | `right_elbow_joint` |
| 11 | `right_ankle_roll_joint` | 26 | `right_wrist_roll_joint` |
| 12 | `waist_yaw_joint` | 27 | `right_wrist_pitch_joint` |
| 13 | `waist_roll_joint` | 28 | `right_wrist_yaw_joint` |
| 14 | `waist_pitch_joint` | | |

Layout: `[positions(0:29) | velocities(29:58) | torques(58:87)]`

Left arm positions are at indices **15-21**, right arm at **22-28**.

### 5b. Inspire Hand Joint State (12D)

`get_robot_inspire_joint_states()` returns 12 actuated hand joint **positions only**:

| Idx | Joint | Type |
|-----|-------|------|
| 0 | `left_thumb_1_joint` | Thumb yaw |
| 1 | `left_thumb_2_joint` | Thumb pitch |
| 2 | `left_index_1_joint` | Index proximal |
| 3 | `left_middle_1_joint` | Middle proximal |
| 4 | `left_ring_1_joint` | Ring proximal |
| 5 | `left_little_1_joint` | Little proximal |
| 6 | `right_thumb_1_joint` | Thumb yaw |
| 7 | `right_thumb_2_joint` | Thumb pitch |
| 8 | `right_index_1_joint` | Index proximal |
| 9 | `right_middle_1_joint` | Middle proximal |
| 10 | `right_ring_1_joint` | Ring proximal |
| 11 | `right_little_1_joint` | Little proximal |

Mimic joints are **not observed** -- they are derived from actuated joints
during action processing (see mimic rules in Section 4a).

### 5c. Camera

Single front camera at 640x480 RGB (float32, not normalized).
No wrist cameras (Inspire FTP USD has no wrist camera mount links).

---

## 6. Reward Structure

**File:** `mdp/rewards.py` (shared with Dex3 `grasp_policy/mdp/rewards.py`)

Three-stage sparse rewards. Total possible reward per episode = **3.0**.

| Stage | Reward | Weight | Transition Condition |
|-------|--------|--------|---------------------|
| 0 -> 1 (Grasp) | 1.0 | 1.0 | Block z > TABLE_Z + 0.05 = **0.905 m** |
| 1 -> 2 (Transport) | 1.0 | 1.0 | Block over bin: x in [-1.625, -1.475], y in [1.535, 1.685] |
| 2 -> 3 (Place) | 1.0 | 1.0 | Block in bin: above floor (z > 0.785), below rim (z < 0.890), within x/y bounds |

- Rewards are **one-time per transition** (tracked by `_prev_stage_*` caches)
- With `use_sparse_reward=True`, rewards are scaled by `1/step_dt` giving ~1.0 total per transition
- Stages only advance forward, never backward within an episode

---

## 7. Termination Conditions

**File:** `mdp/terminations.py` (shared with Dex3)

| Condition | Trigger | `time_out` flag |
|-----------|---------|-----------------|
| **Time out** | Episode exceeds 20 s (RL) or 300 s (teleop) | Yes |
| **Success** | Task stage >= 3 (block placed in bin) | No |
| **Object drop** | Block z < 0.5 m | Yes |

---

## 8. Reset Events

**File:** `mdp/events.py` (shared with Dex3)

| Event | Action | Trigger |
|-------|--------|---------|
| `reset_scene` | Restore default joint/object poses | On reset |
| `reset_task_stage` | Set stage -> 0, clear reward caches | On reset |
| `reset_block_position` | Random offset: +/-3 cm in x, +/-3 cm in y from default | On reset |

The eval variant (`G1GraspPolicyInspireEvalEnvCfg`) inherits the RL config --
block randomization range is the same but placement is deterministic per env index.

---

## 9. Teleop Devices

**File:** `teleop_devices/inspire_gripper_retargeter.py`

The teleop variant uses Apple Vision Pro (AVP) hand tracking with the
`InspireGripperRetargeter`, a **binary gripper** that replaces per-finger dex-retargeting
with pinch-based open/close.

### Class Hierarchy

```
RetargeterBase
  -> UnitreeG1Retargeter      (provides _retarget_abs for wrist retargeting)
       -> InspireGripperRetargeter  (binary gripper, skips dex-retargeting init)
```

### Pinch Detection (Hysteresis)

| Parameter | Value |
|-----------|-------|
| Close threshold | 0.03 m (thumb-index distance) |
| Open threshold | 0.05 m |
| Hysteresis band | 0.02 m |
| Gripper closed angle | 1.0 rad (uniform for all actuated joints) |

State machine:
- **Currently open** (state < 0.5): close when distance < 0.03 m
- **Currently closed** (state >= 0.5): open when distance > 0.05 m

### Gripper Expansion: 1D -> 24D

When grip state = 1.0 (closed), all 6 actuated joints per hand are set to
`gripper_closed_angle` (1.0 rad), then mimic rules are applied sequentially:

| Joint Type | Closed Value |
|------------|-------------|
| index/middle/ring/little `_1` (proximal) | 1.0 rad |
| thumb `_1` (yaw) | 1.0 rad |
| thumb `_2` (pitch) | 1.0 rad |
| index/middle/ring/little `_2` (mimic) | 1.0843 rad |
| thumb `_3` (intermediate, mimic) | 0.8024 rad |
| thumb `_4` (distal, mimic) | 0.7614 rad |

When grip state = 0.0 (open), all joints are 0.0.

### XR Configuration

| Parameter | Value |
|-----------|-------|
| Anchor prim | `/World/envs/env_0/Robot/pelvis` |
| Fixed anchor height | True |
| Rotation mode | `FOLLOW_PRIM_SMOOTHED` |
| Anchor position offset | (0.0, 0.0, -1.0) |
| Anchor rotation | (0.70711, 0.0, 0.0, -0.70711) |

### Design Philosophy

The binary gripper simplifies teleop for **imitation learning (IL)**. The policy learns
binary grip patterns from demonstrations. **RL post-training** on the full Inspire env
(53D action space with per-finger control) can then unlock finer manipulation.

---

## 10. Data Pipeline (26D Policy Format)

**Files:** `utils/inspire_ftp_lerobot_fields.py`, `utils/convert_hdf5_to_lerobot.py`,
`config/g1_grasp_policy_inspire_dataset.yaml`

### Canonical 26D Joint Order (`STATE_26_NAMES_ENV_ORDER`)

| Idx | Joint Name | Group |
|-----|-----------|-------|
| 0 | `left_shoulder_pitch_joint` | Left arm |
| 1 | `left_shoulder_roll_joint` | Left arm |
| 2 | `left_shoulder_yaw_joint` | Left arm |
| 3 | `left_elbow_joint` | Left arm |
| 4 | `left_wrist_roll_joint` | Left arm |
| 5 | `left_wrist_pitch_joint` | Left arm |
| 6 | `left_wrist_yaw_joint` | Left arm |
| 7 | `right_shoulder_pitch_joint` | Right arm |
| 8 | `right_shoulder_roll_joint` | Right arm |
| 9 | `right_shoulder_yaw_joint` | Right arm |
| 10 | `right_elbow_joint` | Right arm |
| 11 | `right_wrist_roll_joint` | Right arm |
| 12 | `right_wrist_pitch_joint` | Right arm |
| 13 | `right_wrist_yaw_joint` | Right arm |
| 14 | `L_thumb_proximal_yaw_joint` | Left hand |
| 15 | `L_thumb_proximal_pitch_joint` | Left hand |
| 16 | `L_index_proximal_joint` | Left hand |
| 17 | `L_middle_proximal_joint` | Left hand |
| 18 | `L_ring_proximal_joint` | Left hand |
| 19 | `L_pinky_proximal_joint` | Left hand |
| 20 | `R_thumb_proximal_yaw_joint` | Right hand |
| 21 | `R_thumb_proximal_pitch_joint` | Right hand |
| 22 | `R_index_proximal_joint` | Right hand |
| 23 | `R_middle_proximal_joint` | Right hand |
| 24 | `R_ring_proximal_joint` | Right hand |
| 25 | `R_pinky_proximal_joint` | Right hand |

**Note:** Hand joint names in 26D use `L_`/`R_` prefixed Nucleus-style names
(matching `RECORDED_ACTION_53_JOINT_NAMES`), not the `left_`/`right_` URDF-style
names used in the env config and observations.

### State Extraction

From 87D body observation + 12D hand observation:

```
state_26d = [body[15:22], body[22:29], inspire[0:6], inspire[6:12]]
             left_arm     right_arm    left_hand     right_hand
```

### Action Derivation

**Two paths depending on recorded action format:**

| Source | Condition | Method |
|--------|-----------|--------|
| **53D joint-space** | `action_full.shape[1] == 53` | Extract via `ACTION_HDF5_TO_ENV_26` index mapping, add +0.3 elbow offset |
| **38D PinkIK (teleop)** | Any other width or `None` | `action[t] = state[t+1]` (observation-derived) |

The observation-derived approach works because for teleop recordings, the 38D
PinkIK commands cannot be directly mapped to 26D policy space. Instead, we use
the next-step observed joint positions as the action target.

### Elbow Offset Correction

The env config applies a -0.3 offset to elbow joints. When extracting actions from
53D recordings, +0.3 is added back to indices 3 (left elbow) and 10 (right elbow)
to get raw joint targets. For observation-derived actions, the offset is already
baked into the observed positions.

### Conversion Command

```bash
/isaac-sim/python.sh scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/g1_grasp_policy_inspire_dataset.yaml
```

**Output:** LeRobot dataset with (T-1) rows x 26D state/action + front camera video.

---

## 11. Actuator Configuration

**File:** `config/robot_config.py`

| Group | Type | Joints | Stiffness | Damping | Effort Limit |
|-------|------|--------|-----------|---------|-------------|
| **Legs** | `IdealPDActuatorCfg` | hip pitch/roll/yaw, knee | 150 (hip), 300 (knee) | 2 (hip), 4 (knee) | 88 (hip), 139 (knee) |
| **Feet** | `IdealPDActuatorCfg` | ankle pitch/roll | 40 | 2 | 50 |
| **Waist** | `ImplicitActuatorCfg` | yaw/roll/pitch | 10000 | 10000 | 1000 |
| **Arms** | `IdealPDActuatorCfg` | shoulder, elbow, wrist | 100 (shl p/r), 40 (shl yaw/elbow), 20 (wrist) | 15 (shl p/r), 8 (shl yaw/elbow), 4 (wrist) | 25 (shl/elbow), 5 (wrist p/y), 25 (wrist r) |
| **Hands** | `ImplicitActuatorCfg` | All 24 hand joints | 10 | 0.2 | 30 |

Hands use `ImplicitActuatorCfg` (not PD). Mimic enforcement is handled by
`InspireFTPJointPositionAction`, not the actuator model.

**Default joint positions:** All zero except elbows at -0.3 rad.

**Physics:** Gravity disabled on robot, fixed root link, self-collisions disabled.

---

## 12. Comparison: Inspire FTP vs Dex3

| Property | Inspire FTP | Dex3 |
|----------|-------------|------|
| **Gym ID prefix** | `Isaac-Grasp-Policy-G129-InspireFTP-*` | `Isaac-Grasp-Policy-G129-Dex3-*` |
| **Action dim (RL)** | 53 | 43 |
| **Body joints** | 29 | 29 |
| **Hand joints (total)** | 24 (12 actuated + 12 mimic) | 14 (7 per hand) |
| **Actuated hand joints** | 6 per hand | 7 per hand |
| **Mimic enforcement** | Yes (`InspireFTPJointPositionAction`) | No |
| **Hand observation dim** | 12 (`robot_inspire_joint_state`) | 14 (`robot_dex3_joint_state`) |
| **Body observation dim** | 87 (same) | 87 (same) |
| **Policy dim (26D/28D)** | 26 (14 arm + 12 hand) | 28 (14 arm + 14 hand) |
| **Cameras** | 1 (front only) | 3 (front, left wrist, right wrist) |
| **Teleop action dim** | 38 (PinkIK) | 23 (WBC+PINK) |
| **Teleop hand control** | Binary gripper (pinch) | Binary gripper (pinch) |
| **Action class** | `InspireFTPJointPositionActionCfg` | `JointPositionActionCfg` |
| **Eval script** | `eval_grasp_policy.py` (shared) | `eval_grasp_policy.py` (shared) |
| **Reward/termination/events** | Shared (`grasp_policy/mdp/`) | Same files |

---

## 13. RLinf Integration

**File:** `simulation/rl/rlinf_ext/__init__.py`

### Environment Registration

```python
IsaaclabGraspPolicyInspireEnv = _get_grasp_policy_inspire_env_class()
REGISTER_ISAACLAB_ENVS["Isaac-Grasp-Policy-G129-InspireFTP-Joint"] = IsaaclabGraspPolicyInspireEnv
REGISTER_ISAACLAB_ENVS["Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval"] = IsaaclabGraspPolicyInspireEnv
```

### Observation Wrapping

`IsaaclabGraspPolicyInspireEnv._wrap_obs()`:

```
Input:  obs["policy"]["robot_joint_state"][:, 15:29]  -> (B, 14) arm positions
        obs["policy"]["robot_inspire_joint_state"]     -> (B, 12) hand positions
Output: states = cat([arm, hand], dim=-1)              -> (B, 26) policy state
```

### ACT Converter

`_convert_inspire_obs_to_act_format`:
- Input: 26D states + front camera
- Policy output: (chunk_size, 26)
- Scatter to sim: 26D -> 53D (`scatter_to_sim_numpy()`)

### Data Flow (Eval)

```
env.step()
  -> obs: {robot_joint_state(87), robot_inspire_joint_state(12), front_camera}
  -> _wrap_obs(): {states(26), main_images, task_descriptions}
  -> ACT policy: select_action() -> (chunk_size, 26)
  -> scatter: 26 -> 53D
  -> env.step(action_53D)
```

---

## 14. File Reference

All paths relative to `scripts/`.

| File | Role |
|------|------|
| `simulation/tasks/grasp_policy_inspire/__init__.py` | Gym ID registration (3 variants) |
| `simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py` | RL/eval env config (53D, 20s) |
| `simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py` | Teleop env config (38D PinkIK, 300s) |
| `simulation/tasks/grasp_policy_inspire/mdp/__init__.py` | MDP module exports |
| `simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py` | `InspireFTPJointPositionAction` with mimic enforcement |
| `simulation/tasks/grasp_policy_inspire/mdp/observations.py` | 87D body + 12D hand observations |
| `simulation/tasks/grasp_policy/mdp/rewards.py` | 3-stage sparse rewards (shared) |
| `simulation/tasks/grasp_policy/mdp/terminations.py` | Drop/success/timeout (shared) |
| `simulation/tasks/grasp_policy/mdp/events.py` | Reset handlers (shared) |
| `simulation/tasks/grasp_policy_inspire/config/robot_config.py` | G1 + Inspire FTP articulation config |
| `teleop_devices/inspire_gripper_retargeter.py` | Binary gripper retargeter for AVP |
| `utils/inspire_ftp_lerobot_fields.py` | 26D state/action conversion logic |
| `utils/convert_hdf5_to_lerobot.py` | HDF5 -> LeRobot dataset converter |
| `config/g1_grasp_policy_inspire_dataset.yaml` | Dataset conversion config |
| `simulation/rl/rlinf_ext/__init__.py` | RLinf env registration + ACT converters |

---

## Timing

| Parameter | Value |
|-----------|-------|
| Simulation dt | 0.005 s (200 Hz) |
| Decimation | 4 |
| Control dt (step_dt) | 0.02 s (50 Hz) |
| RL episode length | 20 s (1000 steps) |
| Teleop episode length | 300 s (15000 steps) |
