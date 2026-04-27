# Inspire FTP Grasp Policy Task Reference

Technical reference for the G1 + Inspire FTP pick-and-place task.
The scene spawns a surgical tray with a single tool (default: tool_0) that the robot
must grasp and place on a target pad. Covers both the **RL/eval** (41D joint control)
and **teleop** (38D PinkIK) variants, plus the 26D data conversion pipeline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Gym IDs and Registration](#2-gym-ids-and-registration)
3. [Scene Setup](#3-scene-setup)
4. [Action Space](#4-action-space)
   - [4a. RL/Eval: 41D Joint Position](#4a-rleval-41d-joint-position)
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

The task is a **pick-and-place**: grasp a surgical tool from a tray, transport it to a green
target pad, and place it on the pad. The scene loads a surgical tray with a single tool
(default: tool_0 in slot 4, overridable via `--object` and `--slot` CLI args). On each
reset, the tool position is randomized +/-2 cm in X/Y with +/-15° yaw rotation.
The robot is a Unitree G1 (29 body DOF) with **Inspire FTP 5-finger hands** (24 hand
joints: 12 actuated + 12 mimic, per pair of hands).

Three gym variants exist:

| Variant | Gym ID | Action Dim | Episode | Purpose |
|---------|--------|------------|---------|---------|
| **RL Training** | `Isaac-Grasp-Policy-G129-InspireFTP-Joint` | 41D | 20 s | Random block placement |
| **RL Evaluation** | `Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval` | 41D | 20 s | Deterministic block placement |
| **Teleoperation** | `Isaac-Grasp-Policy-G129-InspireFTP-Teleop` | 38D | 300 s | PinkIK + AVP hand tracking |

Key differences from the Dex3 variant: 41D actions (vs 43D), 12D hand observation (vs 14D),
26D policy dim (vs 28D). Cameras match Dex3 (front + left/right wrist). The 12 mimic hand
joints are driven internally by the action class — they are not part of the action space.

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

**Files:** `g1_grasp_policy_inspire_env_cfg.py`, `config/robot_config.py`, `assets/assets.py`

| Entity | Size / Type | Init Position | Notes |
|--------|-------------|---------------|-------|
| **Robot** | G1 29DOF + Inspire FTP | (-1.849, 1.94, 0.812) | Fixed base, gravity disabled |
| **Surgical Tray** | Static prop (no physics) | (-1.549, 2.034, 0.846) | 90° CW rotation, from trocar task |
| **Grasp Tool** | Single sinus tool, 0.1 kg | Tray slot 4 (default) | `UsdFileCfg`, default: tool_0 |
| **Target Pad** | 15x15x0.5 cm, 0.3 kg | (-1.55, 1.61, 0.8375) | Green, friction 1.0/0.8 |
| **Front Camera** | RGB 640x480 | On robot head | focal_length=10.5 |
| **Scene** | Surgical room USD | - | Trocar assembly scene (table) |
| **Light** | Dome light | - | (0.75, 0.75, 0.75) intensity 1000 |

**Table height (TABLE_Z):** 0.855 m

### Surgical Tray

The tray (`SurgicalTray_endo.usd`) is a static `AssetBaseCfg` positioned to match the
trocar task's tray placement. It has 6 slots defined by Xform markers in the USD
(`/root/tool_0` through `/root/tool_5`). Slot positions are rotated 90° CW to match the
tray orientation and offset +3 cm in Z so tools rest on top of the tray surface.

### Grasp Tools (5 available)

| Name | Type | Source |
|------|------|--------|
| `tool_0` | Sinus surgical tool | `assets/sinus_toolkit_v1/tool_0/tool_0.usd` |
| `tool_1` | Sinus surgical tool | `assets/sinus_toolkit_v1/tool_1/tool_1.usd` |
| `tool_2` | Sinus surgical tool | `assets/sinus_toolkit_v1/tool_2/tool_2.usd` |
| `tool_3` | Sinus surgical tool | `assets/sinus_toolkit_v1/tool_3/tool_3.usd` |
| `tool_4` | Sinus surgical tool | `assets/sinus_toolkit_v1/tool_4/tool_4.usd` |

Only one tool is loaded at a time (default: `tool_0`). The .obj meshes must be converted
to .usd before first use:

```bash
./docker/run_docker_grasp.sh python scripts/simulation/assets/convert_sinus_toolkit.py
```

Both the eval script (`eval_act_inspire.py`) and the recording script
(`record_demos.py`) support `--object <name>` (default: `tool_0`) to select the tool
and `--slot N` to choose the tray slot (0-5).

Wrist cameras live on `left_hand_camera_base_link` / `right_hand_camera_base_link` in
the `g1-29dof-inspire-ftp-usd-wrist_cam/` USD variant (the same mount link layout as
Dex3). Camera presets: `CameraPresets.left_inspire_wrist_camera` /
`right_inspire_wrist_camera` in [`camera_config.py`](../../scripts/simulation/tasks/assemble_trocar/config/camera_config.py).

---

## 4. Action Space

### 4a. RL/Eval: 41D Joint Position

**File:** `g1_grasp_policy_inspire_env_cfg.py`, `mdp/mimic_action.py`

The action is a 41D joint position target: **29 body + 12 actuated hand joints**.
Uses `InspireJointPositionAction`, which sets targets for the 41 actuated joints
and then computes and sets targets for the 12 mimic joints separately on the
articulation via `apply_actions()`.

#### Full 41-Joint List (USD Articulation Order, Mimic Removed)

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
| 39 | `left_thumb_2_joint` | Hand (actuated) |
| 40 | `right_thumb_2_joint` | Hand (actuated) |

**Note:** `thumb_1` = yaw, `thumb_2` = proximal pitch (both actuated).
The 12 mimic joints (`*_2`, `thumb_3`, `thumb_4`) are NOT in the action space
but are driven by `InspireJointPositionAction.apply_actions()` using mimic rules.

The full 53-joint list (including mimic) is defined as `joint_names` in the env config
and used by the teleop variant's PinkIK controller.

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

**Single-arm masking (`--arm left|right`):**

When `record_demos.py` is launched with `--arm right` or `--arm left`, the
non-controlled arm is locked each step by overwriting its wrist pose and hand
joints in the 38D action before `env.step()`:

- **Wrist pose (7D):** Read once via FK (`robot.data.body_pos_w` /
  `body_quat_w` on the `*_wrist_yaw_link`) after each `env.reset()`, held
  constant for the episode. This avoids drift from the approximate positions in
  the `idle_action` tensor.
- **Hand joints:** Set to `idle_action` values using explicit per-hand index
  lists (`_LEFT_HAND_38D_IDX` / `_RIGHT_HAND_38D_IDX`). The indices are
  **not contiguous** because the USD articulation order interleaves left and
  right hand joints.

The full-body state (87D + 12D) is still recorded — only the teleop command is
masked, not the observations.

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

### 5c. Cameras

Three cameras at 480x640 RGB (float32, not normalized), matching Dex3:
- `front_camera` on `d435_link` (head-mounted, room view)
- `left_wrist_camera` on `left_hand_camera_base_link`
- `right_wrist_camera` on `right_hand_camera_base_link`

All three are published under `ObservationsCfg.CameraImagesCfg` and recorded to HDF5
by `record_demos.py`. Mount links are provided by the
`g1-29dof-inspire-ftp-usd-wrist_cam/` USD variant.

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
| `reset_block_position` | Teleport tool to slot position with +/-2 cm XY noise and +/-15° yaw noise | On reset |

The `reset_block_to_tray_slot` function accepts `slot_pos`, `slot_rot`, `xy_noise`,
and `yaw_noise_deg` parameters. The yaw noise composes a random Z-axis rotation with
the base `slot_rot` quaternion via Hamilton product.

The eval variant (`G1GraspPolicyInspireEvalEnvCfg`) inherits the RL config.

---

## 9. Teleop Devices

**File:** `tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`

The teleop variant uses Apple Vision Pro (AVP) hand tracking with the
`UnitreeG1Retargeter`, which provides **full 5-finger DexPilot IK retargeting**.
All fingers are individually tracked from the operator's hand pose.

### Class Hierarchy

```
RetargeterBase
  -> UnitreeG1Retargeter      (wrist retargeting + full dex-retargeting for all fingers)
```

### Dex-Retargeting

`UnitreeG1DexRetargeting` computes per-finger joint angles using DexPilot IK:
- Extracts 21 MANO joints from 26 OpenXR joints per hand
- Runs IK optimizer against Nucleus hand-only URDFs
- Returns 12D per hand (thumb yaw/pitch + 4 finger proximals + intermediates)
- Values are placed into the 24D hand joint array via positional name mapping

A `_URDF_TO_NUCLEUS` mapping dict in the teleop env cfg converts URDF-style joint
names (`left_thumb_1_joint`) to Nucleus-style (`L_thumb_proximal_yaw_joint`) so the
retargeter output aligns with PinkIK's expected joint order.

### XR Configuration

| Parameter | Value |
|-----------|-------|
| Anchor prim | `/World/envs/env_0/Robot/pelvis` |
| Fixed anchor height | True |
| Rotation mode | `FOLLOW_PRIM_SMOOTHED` |
| Anchor position offset | (0.0, 0.0, -1.0) |
| Anchor rotation | (0.70711, 0.0, 0.0, -0.70711) |

---

## 10. Data Pipeline (26D / 13D Policy Format)

**Files:** `utils/inspire_lerobot_fields.py`, `utils/convert_hdf5_to_lerobot.py`,
`config/g1_grasp_policy_inspire_dataset.yaml`,
`config/g1_grasp_policy_inspire_dataset_right_arm.yaml`,
`config/g1_grasp_policy_inspire_dataset_left_arm.yaml`

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

### 13D Single-Arm Variants

For single-arm policies, the converter extracts only one arm + hand from the
full-body HDF5 recording. Two 13D variants are available:

**Right arm (13D):** `rheo_13d_state_action: true`

| Idx | Joint Name | Group |
|-----|-----------|-------|
| 0-6 | `right_shoulder_pitch/roll/yaw`, `right_elbow`, `right_wrist_roll/pitch/yaw` | Right arm |
| 7-12 | `R_thumb_proximal_yaw/pitch`, `R_index/middle/ring/pinky_proximal` | Right hand |

**Left arm (13D):** `rheo_13d_left_state_action: true`

| Idx | Joint Name | Group |
|-----|-----------|-------|
| 0-6 | `left_shoulder_pitch/roll/yaw`, `left_elbow`, `left_wrist_roll/pitch/yaw` | Left arm |
| 7-12 | `L_thumb_proximal_yaw/pitch`, `L_index/middle/ring/pinky_proximal` | Left hand |

State extraction follows the same pattern as 26D but takes only one side:

```
# Right 13D
state_13d = [body[22:29], inspire[6:12]]
             right_arm    right_hand

# Left 13D
state_13d = [body[15:22], inspire[0:6]]
             left_arm     left_hand
```

Action derivation and elbow offset (+0.3) apply identically — the elbow is at
index 3 in both 13D variants.

### Conversion Commands

```bash
# 26D dual-arm (default)
/isaac-sim/python.sh scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/inspire/g1_grasp_policy_inspire_dataset.yaml

# 13D right arm
/isaac-sim/python.sh scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/inspire/g1_grasp_policy_inspire_dataset_right_arm.yaml

# 13D left arm
/isaac-sim/python.sh scripts/utils/convert_hdf5_to_lerobot.py \
    --config scripts/config/inspire/g1_grasp_policy_inspire_dataset_left_arm.yaml
```

**Output:** LeRobot dataset with (T-1) rows x 26D or 13D state/action + front camera video.

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
`InspireJointPositionAction`, not the actuator model.

**Default joint positions:** All zero except elbows at -0.3 rad.

**Physics:** Gravity disabled on robot, fixed root link, self-collisions disabled.

---

## 12. Comparison: Inspire FTP vs Dex3

| Property | Inspire FTP | Dex3 |
|----------|-------------|------|
| **Gym ID prefix** | `Isaac-Grasp-Policy-G129-InspireFTP-*` | `Isaac-Grasp-Policy-G129-Dex3-*` |
| **Action dim (RL)** | 41 | 43 |
| **Body joints** | 29 | 29 |
| **Hand joints (total)** | 24 (12 actuated + 12 mimic) | 14 (7 per hand) |
| **Actuated hand joints** | 6 per hand | 7 per hand |
| **Mimic enforcement** | Yes (`InspireJointPositionAction`) | No |
| **Hand observation dim** | 12 (`robot_inspire_joint_state`) | 14 (`robot_dex3_joint_state`) |
| **Body observation dim** | 87 (same) | 87 (same) |
| **Policy dim (26D/28D)** | 26 (14 arm + 12 hand) | 28 (14 arm + 14 hand) |
| **Cameras** | 1 (front only) | 3 (front, left wrist, right wrist) |
| **Teleop action dim** | 38 (PinkIK) | 23 (WBC+PINK) |
| **Teleop hand control** | Full dex-retargeting (DexPilot IK) | Binary gripper (pinch) |
| **Action class** | `InspireJointPositionActionCfg` | `JointPositionActionCfg` |
| **Eval script** | `eval_act_inspire.py` | (n/a — Dex3 grasp variant removed) |
| **Reward/termination/events** | `grasp_policy_inspire/mdp/` (self-contained) | (n/a) |

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
- Scatter to sim: 26D -> 41D (`scatter_to_sim_numpy()`)

### Data Flow (Eval)

```
env.step()
  -> obs: {robot_joint_state(87), robot_inspire_joint_state(12), front_camera}
  -> _wrap_obs(): {states(26), main_images, task_descriptions}
  -> ACT policy: select_action() -> (chunk_size, 26)
  -> scatter: 26 -> 41D
  -> env.step(action_41D)
```

---

## 14. File Reference

All paths relative to `scripts/`.

### Task Implementation

| File | Role |
|------|------|
| `simulation/tasks/grasp_policy_inspire/__init__.py` | Gym ID registration (3 variants) |
| `simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py` | RL/eval env config (41D, 20s) |
| `simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py` | Teleop env config (38D PinkIK, 300s) |
| `simulation/tasks/grasp_policy_inspire/mdp/__init__.py` | MDP module exports |
| `simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py` | `InspireJointPositionAction` with mimic enforcement |
| `simulation/tasks/grasp_policy_inspire/mdp/observations.py` | 87D body + 12D hand observations |
| `simulation/tasks/grasp_policy/mdp/rewards.py` | 3-stage sparse rewards (shared) |
| `simulation/tasks/grasp_policy/mdp/terminations.py` | Drop/success/timeout (shared) |
| `simulation/tasks/grasp_policy/mdp/events.py` | Reset handlers (shared) |
| `simulation/tasks/grasp_policy_inspire/config/robot_config.py` | G1 + Inspire FTP articulation config |
| `simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py` | Teleop env cfg with full dex-retargeting via `UnitreeG1Retargeter` |

### Data Pipeline

| File | Role |
|------|------|
| `utils/inspire_lerobot_fields.py` | 26D and 13D state/action conversion logic (handles 53D, 41D, 38D) |
| `utils/inspire_experiment_config.py` | 26D joint groups, scatter_to_sim (41D), state extraction |
| `utils/convert_hdf5_to_lerobot.py` | HDF5 -> LeRobot dataset converter |
| `config/g1_grasp_policy_inspire_dataset.yaml` | 26D dual-arm dataset conversion config |
| `config/g1_grasp_policy_inspire_dataset_right_arm.yaml` | 13D right-arm dataset conversion config |
| `config/g1_grasp_policy_inspire_dataset_left_arm.yaml` | 13D left-arm dataset conversion config |

### Scene Assets

| File | Role |
|------|------|
| `simulation/assets/assets.py` | USD path constants (`SINUS_TOOL_USD_PATHS`) |
| `simulation/assets/convert_sinus_toolkit.py` | Batch .obj→.usd converter (run inside Docker) |

### ACT Training and Evaluation

| File | Role |
|------|------|
| `policy/act_config_inspire.yaml` | IL training config (26D state/action, 1 camera) |
| `policy/train_act_grasp_policy_inspire.sh` | IL training launcher (LeRobot) |
| `simulation/policies/act.py` | ACT eval wrapper (Inspire FTP: 26D / 13D policy → 41D sim scatter) |
| `simulation/examples/eval_act_inspire.py` | Evaluation entry point (ACT / `--test` dummy modes) |

### RLinf RL Post-Training

| File | Role |
|------|------|
| `simulation/rl/rlinf_ext/__init__.py` | RLinf env registration + ACT converters |
| `simulation/rl/rlinf_ext/act_policy.py` | ACT wrapper with ValueHead for RL (generic) |
| `simulation/rl/rlinf_ext/config/model/act_inspire.yaml` | RLinf model config (action_dim=26) |
| `simulation/rl/rlinf_ext/config/env/isaaclab_grasp_policy_inspire.yaml` | RLinf env config (InspireFTP gym ID) |
| `simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml` | RLinf PPO top-level config |

---

## Timing

| Parameter | Value |
|-----------|-------|
| Simulation dt | 0.005 s (200 Hz) |
| Decimation | 4 |
| Control dt (step_dt) | 0.02 s (50 Hz) |
| RL episode length | 20 s (1000 steps) |
| Teleop episode length | 300 s (15000 steps) |
