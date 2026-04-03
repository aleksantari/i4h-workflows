<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# G1 Hand Configuration Reference

Reference for how the Unitree G1's hands are wired through the rheo pipeline.
Documents the Dex3 hand, the Inspire FTP hand specification, and the binary gripper
adaptation that simplifies Inspire FTP teleop for imitation learning.

---

## Table of Contents

1. [Current Hand: Dex3 Specification](#1-current-hand-dex3-specification)
2. [Target Hand: Inspire FTP Specification](#2-target-hand-inspire-ftp-specification)
3. [Dex3 → Inspire FTP Dimension Changes](#3-dex3--inspire-ftp-dimension-changes)
4. [43D Robot Joint Order (Dex3)](#4-43d-robot-joint-order-dex3)
5. [41D Robot Joint Order (Inspire FTP)](#5-41d-robot-joint-order-inspire-ftp)
6. [28D Policy State/Action Format (Dex3)](#6-28d-policy-stateaction-format-dex3)
7. [26D Policy State/Action Format (Inspire FTP)](#7-26d-policy-stateaction-format-inspire-ftp)
8. [Observation Pipeline](#8-observation-pipeline)
9. [Action Pipeline](#9-action-pipeline)
10. [USD Assets](#10-usd-assets)
11. [Camera Setup](#11-camera-setup)
12. [Experiment Config System](#12-experiment-config-system)
13. [Files That Change for Inspire FTP Swap](#13-files-that-change-for-inspire-ftp-swap)
14. [Inspire FTP Binary Gripper Adaptation](#14-inspire-ftp-binary-gripper-adaptation)

---

## 1. Current Hand: Dex3 Specification

> This section documents the **existing** Dex3 setup for reference. See
> [Section 2](#2-target-hand-inspire-ftp-specification) for the replacement.

The G1 uses **Unitree Dex3** dexterous hands — one per arm, 7 DOF each.

### Finger Structure (per hand)

| Finger | Joints | DOF |
|--------|--------|-----|
| Thumb  | `thumb_0`, `thumb_1`, `thumb_2` | 3 |
| Middle | `middle_0`, `middle_1` | 2 |
| Index  | `index_0`, `index_1` | 2 |
| **Total** | | **7** |

### Joint Names

Left hand:

```
left_hand_thumb_0_joint      left_hand_thumb_1_joint      left_hand_thumb_2_joint
left_hand_middle_0_joint     left_hand_middle_1_joint
left_hand_index_0_joint      left_hand_index_1_joint
```

Right hand: same pattern with `right_hand_` prefix.

### Default Joint Positions

| Joint | Left (rad) | Right (rad) |
|-------|-----------|------------|
| `index_0` | -1.047 (-60 deg) | 1.047 (60 deg) |
| `index_1` | -0.698 (-40 deg) | 0.698 (40 deg) |
| `middle_0` | -1.047 (-60 deg) | 1.047 (60 deg) |
| `middle_1` | -0.698 (-40 deg) | 0.698 (40 deg) |
| `thumb_0` | 0.0 | 0.0 |
| `thumb_1` | 0.0 | 0.0 |
| `thumb_2` | 0.0 | 0.0 |

> **Code:**
> [`scripts/simulation/tasks/assemble_trocar/config/robot_config.py`](../scripts/simulation/tasks/assemble_trocar/config/robot_config.py)
> — `G129_CFG_WITH_DEX3_BASE_FIX` → `init_state.joint_pos`.

### Actuator Parameters

```
effort_limit:   5.0 N·m
velocity_limit: 10.0 rad/s
stiffness:      8.0 N·m/rad
damping:        1.5 N·m·s/rad
armature:       0.03
friction:       0.5
```

Actuator type: `IdealPDActuatorCfg`. The hand actuator group is keyed as `"hands"` and
uses a regex pattern matching all `.*_hand_.*` joints.

> **Code:**
> [`scripts/simulation/tasks/assemble_trocar/config/robot_config.py`](../scripts/simulation/tasks/assemble_trocar/config/robot_config.py)
> — `G129_CFG_WITH_DEX3_BASE_FIX.actuators["hands"]`.

### DOF Summary

| Component | DOF |
|-----------|-----|
| Legs | 12 (hip pitch/roll/yaw, knee, ankle pitch/roll × 2) |
| Waist | 3 (yaw, roll, pitch) |
| Arms | 14 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw × 2) |
| **Dex3 hands** | **14 (7 × 2)** |
| **Total robot** | **43** |

---

## 2. Target Hand: Inspire FTP Specification

The **Inspire FTP** (also called DFTP) is a 5-finger dexterous hand. Each hand has
12 revolute joints in the URDF, but **only 6 are independently actuated** — the other
6 are mechanically coupled via `<mimic>` tags and move automatically on real hardware.

**URDF source:**
[`assets/robots/g1-29dof-inspire-ftp-urdf/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`](../assets/robots/g1-29dof-inspire-ftp-urdf/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf)

### Finger Structure (per hand)

| Finger | Actuated Joint | Limits (rad) | Mimic Joint | Multiplier |
|--------|---------------|-------------|-------------|------------|
| Thumb (yaw) | `thumb_1_joint` | 0 → 1.1641 | — | — |
| Thumb (pitch) | `thumb_2_joint` | 0 → 0.5864 | `thumb_3_joint` | 0.8024× |
| | | | `thumb_4_joint` | mimics thumb_3 @ 0.9487× (chain) |
| Index | `index_1_joint` | 0 → 1.4381 | `index_2_joint` | 1.0843× |
| Middle | `middle_1_joint` | 0 → 1.4381 | `middle_2_joint` | 1.0843× |
| Ring | `ring_1_joint` | 0 → 1.4381 | `ring_2_joint` | 1.0843× |
| Little | `little_1_joint` | 0 → 1.4381 | `little_2_joint` | 1.0843× |
| **Total actuated** | | | | **6** |

### Actuated Joint Names

Left hand:

```
left_thumb_1_joint       (yaw)
left_thumb_2_joint       (pitch)
left_index_1_joint
left_middle_1_joint
left_ring_1_joint
left_little_1_joint
```

Right hand: same pattern with `right_` prefix.

### Mimic Joint Names (not in action space)

Left hand:

```
left_thumb_3_joint       mimics left_thumb_2_joint  × 0.8024
left_thumb_4_joint       mimics left_thumb_3_joint  × 0.9487 (chained)
left_index_2_joint       mimics left_index_1_joint  × 1.0843
left_middle_2_joint      mimics left_middle_1_joint × 1.0843
left_ring_2_joint        mimics left_ring_1_joint   × 1.0843
left_little_2_joint      mimics left_little_1_joint × 1.0843
```

Right hand: same pattern with `right_` prefix.

### URDF Effort/Velocity Limits (all actuated joints)

```
effort:   10 N·m
velocity: 1 rad/s
```

### Hand Attachment

The hand connects to the arm via a fixed joint:

```
left_wrist_yaw_link  → (fixed: left_base_joint)  → left_base_link → finger joints
right_wrist_yaw_link → (fixed: right_base_joint) → right_base_link → finger joints
```

No camera links exist on the hand — the Dex3's `left_hand_camera_base_link` /
`right_hand_camera_base_link` are absent. The front camera on `d435_link` (torso)
is available.

### DOF Summary

| Component | DOF |
|-----------|-----|
| Legs | 12 (locked, zeroed in policy) |
| Waist | 3 (locked, zeroed in policy) |
| Arms | 14 (7 × 2) |
| **Inspire FTP hands (actuated only)** | **12 (6 × 2)** |
| **Total robot (actuated revolute)** | **41** |
| **Policy dim** | **26** (14 arm + 12 hand) |

---

## 3. Dex3 → Inspire FTP Dimension Changes

| Dimension | Dex3 | Inspire FTP | Notes |
|-----------|------|-------------|-------|
| Hand DOF (per hand) | 7 | 6 (actuated) | Inspire has 12 URDF joints but 6 mimic |
| Hand DOF (total) | 14 | 12 | |
| Total robot joints | 43 | 41 | 29 body + 12 hand (actuated only) |
| Policy dim | 28 | 26 | 14 arm + 12 hand |
| Hand obs key | `robot_dex3_joint_state` (14D) | new inspire obs (12D) | Must extract 6 actuated per hand |
| Body state | (B, 87) = 29×3 | (B, 87) = 29×3 | Unchanged (body is identical) |
| Sim action dim | 43 | 41 | Or 53 if mimic joints are in articulation |
| Cameras | 3 (front + 2 wrist) | 1 (front only) | No wrist camera links on Inspire |
| `GROUP_SIZE` in experiment config | 7 | varies (arm=7, hand=6) | **Cannot use single constant** |

**Key architectural impact:** The current `ACTExperimentConfig` assumes every joint
group has exactly 7 DOF (`GROUP_SIZE = 7`). With the Inspire FTP hand at 6 DOF per
hand, this must become per-group sizing.

---

## 4. 43D Robot Joint Order (Dex3)

This is the canonical joint ordering used in env configs, action spaces, and joint
space YAML files. Hand joints occupy indices **29–42**.

```
Index  Joint
-----  -----
 0     left_hip_pitch_joint
 1     right_hip_pitch_joint
 2     left_hip_roll_joint
 3     right_hip_roll_joint
 4     left_hip_yaw_joint
 5     right_hip_yaw_joint
 6     left_knee_joint
 7     right_knee_joint
 8     left_ankle_pitch_joint
 9     right_ankle_pitch_joint
10     left_ankle_roll_joint
11     right_ankle_roll_joint
12     waist_yaw_joint
13     waist_roll_joint
14     waist_pitch_joint
15     left_shoulder_pitch_joint
16     left_shoulder_roll_joint
17     left_shoulder_yaw_joint
18     left_elbow_joint
19     left_wrist_roll_joint
20     left_wrist_pitch_joint
21     left_wrist_yaw_joint
22     right_shoulder_pitch_joint
23     right_shoulder_roll_joint
24     right_shoulder_yaw_joint
25     right_elbow_joint
26     right_wrist_roll_joint
27     right_wrist_pitch_joint
28     right_wrist_yaw_joint
 ── hand joints (Dex3) ──
29     left_hand_thumb_0_joint
30     left_hand_thumb_1_joint
31     left_hand_thumb_2_joint
32     left_hand_middle_0_joint
33     left_hand_middle_1_joint
34     left_hand_index_0_joint
35     left_hand_index_1_joint
36     right_hand_thumb_0_joint
37     right_hand_thumb_1_joint
38     right_hand_thumb_2_joint
39     right_hand_middle_0_joint
40     right_hand_middle_1_joint
41     right_hand_index_0_joint
42     right_hand_index_1_joint
```

> **Code:**
> [`scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py`](../scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py)
> lines 39–83 (action joint list).
> Also
> [`scripts/simulation/tasks/grasp_policy/g1_grasp_policy_env_cfg.py`](../scripts/simulation/tasks/grasp_policy/g1_grasp_policy_env_cfg.py)
> lines 41–85.

**Note:** The 43D order above is the **env config / action space** order. The raw
USD joint order differs — the env config `joint_names` list defines the mapping.

---

## 5. 41D Robot Joint Order (Inspire FTP)

Proposed canonical joint ordering for the Inspire FTP hand. Body joints (0–28) are
identical to the Dex3 layout. Hand joints occupy indices **29–40** (6 actuated per
hand, mimic joints excluded from the action space).

```
Index  Joint
-----  -----
 0     left_hip_pitch_joint
 1     right_hip_pitch_joint
 2     left_hip_roll_joint
 3     right_hip_roll_joint
 4     left_hip_yaw_joint
 5     right_hip_yaw_joint
 6     left_knee_joint
 7     right_knee_joint
 8     left_ankle_pitch_joint
 9     right_ankle_pitch_joint
10     left_ankle_roll_joint
11     right_ankle_roll_joint
12     waist_yaw_joint
13     waist_roll_joint
14     waist_pitch_joint
15     left_shoulder_pitch_joint
16     left_shoulder_roll_joint
17     left_shoulder_yaw_joint
18     left_elbow_joint
19     left_wrist_roll_joint
20     left_wrist_pitch_joint
21     left_wrist_yaw_joint
22     right_shoulder_pitch_joint
23     right_shoulder_roll_joint
24     right_shoulder_yaw_joint
25     right_elbow_joint
26     right_wrist_roll_joint
27     right_wrist_pitch_joint
28     right_wrist_yaw_joint
 ── hand joints (Inspire FTP, actuated only) ──
29     left_thumb_1_joint
30     left_thumb_2_joint
31     left_index_1_joint
32     left_middle_1_joint
33     left_ring_1_joint
34     left_little_1_joint
35     right_thumb_1_joint
36     right_thumb_2_joint
37     right_index_1_joint
38     right_middle_1_joint
39     right_ring_1_joint
40     right_little_1_joint
```

**Important:** The URDF contains 12 revolute joints per hand (24 total), but the 6
mimic joints per hand must **not** appear in the action space. In simulation, mimic
joints should either:

- Be enforced by the physics engine (if the USD converter preserves mimic constraints)
- Be driven programmatically in the env step (apply multiplier to parent joint value)

Either way, the policy only commands the 6 actuated joints.

---

## 6. 28D Policy State/Action Format (Dex3)

Both GR00T and ACT policies operate on a 28D vector: 14 arm joints + 14 hand joints.
Legs and waist (15 DOF) are excluded from the policy and zeroed in sim actions.

### Canonical 28D Order

```
Index  Group       Joint
-----  ----------  -----
 0     left_arm    left_shoulder_pitch_joint
 1     left_arm    left_shoulder_roll_joint
 2     left_arm    left_shoulder_yaw_joint
 3     left_arm    left_elbow_joint
 4     left_arm    left_wrist_roll_joint
 5     left_arm    left_wrist_pitch_joint
 6     left_arm    left_wrist_yaw_joint
 7     right_arm   right_shoulder_pitch_joint
 8     right_arm   right_shoulder_roll_joint
 9     right_arm   right_shoulder_yaw_joint
10     right_arm   right_elbow_joint
11     right_arm   right_wrist_roll_joint
12     right_arm   right_wrist_pitch_joint
13     right_arm   right_wrist_yaw_joint
14     left_hand   left_hand_thumb_0_joint
15     left_hand   left_hand_thumb_1_joint
16     left_hand   left_hand_thumb_2_joint
17     left_hand   left_hand_middle_0_joint
18     left_hand   left_hand_middle_1_joint
19     left_hand   left_hand_index_0_joint
20     left_hand   left_hand_index_1_joint
21     right_hand  right_hand_thumb_0_joint
22     right_hand  right_hand_thumb_1_joint
23     right_hand  right_hand_thumb_2_joint
24     right_hand  right_hand_middle_0_joint
25     right_hand  right_hand_middle_1_joint
26     right_hand  right_hand_index_0_joint
27     right_hand  right_hand_index_1_joint
```

> **Code:**
> [`scripts/utils/assemble_trocar_lerobot_fields.py`](../scripts/utils/assemble_trocar_lerobot_fields.py)
> — `STATE_28_NAMES_ENV_ORDER`.

---

## 7. 26D Policy State/Action Format (Inspire FTP)

The Inspire FTP policy operates on a 26D vector: 14 arm joints + 12 hand joints
(6 actuated per hand). Legs and waist (15 DOF) are excluded.

### Canonical 26D Order

```
Index  Group       Joint
-----  ----------  -----
 0     left_arm    left_shoulder_pitch_joint
 1     left_arm    left_shoulder_roll_joint
 2     left_arm    left_shoulder_yaw_joint
 3     left_arm    left_elbow_joint
 4     left_arm    left_wrist_roll_joint
 5     left_arm    left_wrist_pitch_joint
 6     left_arm    left_wrist_yaw_joint
 7     right_arm   right_shoulder_pitch_joint
 8     right_arm   right_shoulder_roll_joint
 9     right_arm   right_shoulder_yaw_joint
10     right_arm   right_elbow_joint
11     right_arm   right_wrist_roll_joint
12     right_arm   right_wrist_pitch_joint
13     right_arm   right_wrist_yaw_joint
14     left_hand   left_thumb_1_joint
15     left_hand   left_thumb_2_joint
16     left_hand   left_index_1_joint
17     left_hand   left_middle_1_joint
18     left_hand   left_ring_1_joint
19     left_hand   left_little_1_joint
20     right_hand  right_thumb_1_joint
21     right_hand  right_thumb_2_joint
22     right_hand  right_index_1_joint
23     right_hand  right_middle_1_joint
24     right_hand  right_ring_1_joint
25     right_hand  right_little_1_joint
```

---

## 8. Observation Pipeline

How hand joint state flows from the IsaacLab environment to the policy input.

### Raw Environment Observations

| Obs key | Shape | Contents |
|---------|-------|----------|
| `robot_joint_state` | (B, 87) | Full body: pos(29) + vel(29) + torque(29) |
| `robot_dex3_joint_state` | (B, 14) | Dex3 hand positions only (7 left + 7 right) |

### Dex3 Joint Extraction

The `get_robot_dex3_joint_states()` function extracts 14 hand joint positions from
the full 43-joint robot state using hardcoded indices:

```python
dex3_joint_indices = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
```

These indices map into the raw robot articulation state (which uses USD joint order,
**not** the canonical 43D order above). The reordering produces the 14D hand vector
in the canonical order used by the 28D format.

> **Code:**
> [`scripts/simulation/tasks/assemble_trocar/mdp/observations.py`](../scripts/simulation/tasks/assemble_trocar/mdp/observations.py)
> — `get_robot_dex3_joint_states()`.

### State Assembly (obs → policy input)

```
robot_joint_state[:, 15:29]   →  arm_joints (B, 14)     ← left_arm(7) + right_arm(7)
robot_dex3_joint_state        →  hand_joints (B, 14)    ← left_hand(7) + right_hand(7)
                                      │
                              concatenate dim=-1
                                      │
                              state_28d (B, 28)          ← policy input
```

The `ObsProcessor` handles this via `ProcessedObservation`:

```python
ARM_JOINT_SLICE = slice(15, 29)   # positions 15-28 in the 87D body state

arm_joints  = body_state[:, ARM_JOINT_SLICE]   # (B, 14)
hand_joints = dex3_state                        # (B, 14)
state_28d   = torch.cat([arm_joints, hand_joints], dim=-1)  # (B, 28)
```

> **Code:**
> [`scripts/simulation/obs_processor.py`](../scripts/simulation/obs_processor.py)
> — `ObsProcessor`, `ProcessedObservation`.

### With Experiment Config

The `ACTExperimentConfig` can select a subset of joint groups. For the default config
(all 4 groups), the result is identical to the hardcoded 28D path:

```python
exp_config.extract_state(body_87d, dex3_14d)  # → (B, policy_dim)
```

This method slices into `body_87d` for arm groups and `dex3_14d` for hand groups,
then concatenates in group order.

> **Code:**
> [`scripts/utils/act_experiment_config.py`](../scripts/utils/act_experiment_config.py)
> — `ACTExperimentConfig.extract_state()`.

---

## 9. Action Pipeline

How policy output is mapped back to the 43D simulator action space.

### Policy Output → Sim Action

```
Policy output: (B, 28) or (B, policy_dim)
    │
    ├─ Hardcoded path (GR00T): pad 15 leading zeros → (B, 43)
    │   pad_28d_to_43d(): sim_action = [zeros(15) | policy_action(28)]
    │
    └─ Config path (ACT): scatter to indexed positions → (B, 43)
        exp_config.scatter_to_sim(policy_action):
          sim = zeros(B, 43)
          sim[:, scatter_indices] = policy_action
```

### Scatter Index Mapping

For the default experiment config (all 4 groups), `scatter_indices` are:

```python
GROUP_SIM_RANGES = {
    "left_arm":   (15, 22),   # indices 15-21
    "right_arm":  (22, 29),   # indices 22-28
    "left_hand":  (29, 36),   # indices 29-35
    "right_hand": (36, 43),   # indices 36-42
}
# → scatter_indices = [15, 16, ..., 42] (28 indices)
```

Indices 0–14 (legs + waist) are always zero.

> **Code:**
> [`scripts/simulation/act_closedloop_policy.py`](../scripts/simulation/act_closedloop_policy.py)
> — `ACTClosedloopPolicy._get_action_chunk()`.
> [`scripts/utils/act_experiment_config.py`](../scripts/utils/act_experiment_config.py)
> — `scatter_to_sim()`, `GROUP_SIM_RANGES`.

---

## 10. USD Assets

### Dex3 (current, remote S3)

| Constant | USD | Use |
|----------|-----|-----|
| `UNITREE_G1_29DOF_USD` | `g1_29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd` | Whole-body tasks (Arena track) |
| `UNITREE_G1_29DOF_BASE_FIX_USD` | `g1_29dof_with_dex3_base_fix/g1_29dof_with_dex3_base_fix.usd` | Fixed-base manipulation (trocar, grasp) |

Both are hosted on NVIDIA's Omniverse S3 bucket under `Assets/Isaac/Healthcare/0.5.0/`.

### Inspire FTP (local, needs URDF→USD conversion)

| Asset | Path |
|-------|------|
| URDF | `assets/robots/g1-29dof-inspire-ftp-urdf/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` |
| Meshes | `assets/robots/g1-29dof-inspire-ftp-urdf/meshes/` (STL files) |
| USD (to generate) | `assets/robots/g1-29dof-inspire-ftp-urdf/g1_29dof_inspire_ftp.usd` |

The URDF→USD conversion can be done with IsaacLab's `omni.isaac.urdf` extension
or `isaaclab.app.AppLauncher` asset converter. After conversion, add:

```python
# In assets/assets.py
UNITREE_G1_29DOF_INSPIRE_FTP_USD = str(
    Path(__file__).resolve().parents[2]
    / "assets/robots/g1-29dof-inspire-ftp-urdf/g1_29dof_inspire_ftp.usd"
)
```

### Reference configs in other repos

Existing Inspire FTP configs in `~/repos/unitree_sim_isaaclab/` can serve as reference
for actuator tuning and joint defaults:

- `robots/unitree.py` — `G129_CFG_WITH_INSPIRE_HAND`
- `tasks/common_config/robot_configs.py` — `G1RobotPresets.g1_29dof_inspire_base_fix()`
- `tasks/common_observations/inspire_state.py` — Inspire hand observation extraction

> **Code:**
> [`scripts/simulation/assets/assets.py`](../scripts/simulation/assets/assets.py)
> — asset URL constants.

---

## 11. Camera Setup

### Dex3 (current): 3 cameras

| Camera | Prim Path | Mount Link |
|--------|-----------|------------|
| Front | `/World/envs/env_.*/Robot/d435_link/front_cam` | `d435_link` (torso) |
| Left wrist | `/World/envs/env_.*/Robot/left_hand_camera_base_link/left_wrist_camera` | `left_hand_camera_base_link` |
| Right wrist | `/World/envs/env_.*/Robot/right_hand_camera_base_link/right_wrist_camera` | `right_hand_camera_base_link` |

### Inspire FTP: 1 camera (front only)

The Inspire FTP URDF has no wrist camera links (`left_hand_camera_base_link` /
`right_hand_camera_base_link` do not exist). Only the front camera is available:

```
Prim path:  /World/envs/env_.*/Robot/d435_link/front_cam
Mount link: d435_link (torso, fixed joint from torso_link)
Focal len:  12.0
```

> **Code:**
> [`scripts/simulation/tasks/assemble_trocar/config/camera_config.py`](../scripts/simulation/tasks/assemble_trocar/config/camera_config.py)
> — `CameraPresets`.

---

## 12. Experiment Config System

The `ACTExperimentConfig` dataclass abstracts camera and joint group selection so that
downstream code (IL eval, RL converters, RL policy) doesn't hardcode dimensions.

### How It Works

```yaml
# In act_config.yaml
experiment:
  cameras:
    front_camera: "observation.images.cam_room"
    left_wrist_camera: "observation.images.cam_left_wrist"
    right_wrist_camera: "observation.images.cam_right_wrist"
  joint_groups:
    - left_arm      # 7 DOF, body state indices 15-21
    - right_arm     # 7 DOF, body state indices 22-28
    - left_hand     # 7 DOF, dex3 state indices 0-6
    - right_hand    # 7 DOF, dex3 state indices 7-13
```

From these two fields, `__post_init__` computes:

| Attribute | Value (default) | Meaning |
|-----------|----------------|---------|
| `policy_dim` | 28 | `len(joint_groups) × 7` |
| `body_state_indices` | [15..28] | Indices into 87D body state for arm groups |
| `dex3_state_indices` | [0..13] | Indices into 14D dex3 state for hand groups |
| `sim_scatter_indices` | [15..42] | Where to place actions in 43D sim space |

### Key Assumption: GROUP_SIZE = 7

Every joint group is assumed to have exactly 7 DOF. This works for Dex3 (7 per hand)
and for each arm (7 per arm). **If a new hand has a different DOF count, this constant
and the range mappings must change.**

### Loading

```python
# IL eval: from YAML path
exp = ACTExperimentConfig.from_yaml("scripts/policy/act_config.yaml")

# RL training: from env var (set by launcher script)
# export ACT_EXPERIMENT_CONFIG=scripts/policy/act_config.yaml
exp = ACTExperimentConfig.from_env_or_default()
```

> **Code:**
> [`scripts/utils/act_experiment_config.py`](../scripts/utils/act_experiment_config.py).

---

## 13. Files That Change for Inspire FTP Swap

Since we are creating a **new task** (not modifying the existing Dex3 tasks), most
changes are new files rather than edits. The existing assemble_trocar and grasp_policy
tasks remain untouched.

### New files to create

| File | Purpose |
|------|---------|
| USD model (converted from URDF) | Inspire FTP robot for IsaacLab |
| `assets/assets.py` | Add `UNITREE_G1_29DOF_INSPIRE_FTP_USD` constant |
| `tasks/<new_task>/config/robot_config.py` | Inspire FTP joint defaults, actuator params (6 DOF/hand) |
| `tasks/<new_task>/config/camera_config.py` | Front camera only (no wrist cameras) |
| `tasks/<new_task>/mdp/observations.py` | New hand joint extraction (12 actuated indices from full articulation) |
| `tasks/<new_task>/g1_<task>_env_cfg.py` | 41D joint_names list, new scene, rewards, terminations |
| `tasks/<new_task>/__init__.py` | Gym registration |
| New LeRobot field mapping module | 26D canonical joint names, inspire joint indices |
| New dataset config YAML | 26D state/action, front camera only |
| New eval script | Entry point for the new task |

### Existing files that need adaptation (copies, not edits)

| File | What changes |
|------|-------------|
| `act_experiment_config.py` | `GROUP_SIZE` cannot be a single constant (arm=7, hand=6). Must support per-group sizing or define new group ranges for Inspire FTP hand. |
| `act_closedloop_policy.py` | `sim_action_dim` = 41 (not 43). Scatter indices change. |
| `obs_processor.py` | Hand observation is 12D (not 14D). `state_26d` replaces `state_28d`. |
| `act_config.yaml` | New experiment section: 1 camera, hand groups with 6 DOF each. |
| `g1_act_closedloop_grasp_policy.yaml` | `policy_action_dim: 26`, `sim_action_dim: 41`. |

### Mimic joint handling (simulation concern)

IsaacLab / PhysX does **not** natively enforce URDF `<mimic>` constraints. Two options:

1. **USD authoring:** Configure mimic joints as driven joints in the USD with
   position targets computed from parent joint × multiplier (requires USD editing
   after URDF→USD conversion).
2. **Env step callback:** In the env cfg's `post_action` or a custom event, read
   parent joint positions and set mimic joint targets each step.

Either way, the **action space only includes the 6 actuated joints per hand**.
The mimic joints exist in the articulation but are not policy-controlled.

### Data pipeline (fresh start)

Changing the robot embodiment means all data collection starts fresh:

1. Collect teleoperation demos with the Inspire FTP hand
2. Convert HDF5 → LeRobot (with new 26D field mappings, front camera only)
3. Train ACT IL from scratch
4. RL post-training with new reward functions

---

## 14. Inspire FTP Binary Gripper Adaptation

This section documents the **implemented** binary gripper retargeter that simplifies
Inspire FTP teleop for imitation learning. Instead of per-finger dex-retargeting
(where each of the 24 hand joints moves independently based on AVP hand tracking),
the gripper retargeter reduces hand control to a single binary open/close per hand.

---

### 14.1 Motivation and Strategy

The Inspire FTP hand has 6 independently actuated joints per hand (12 total) plus
6 mechanically coupled mimic joints per hand (12 total, 24 grand total). Teaching a
policy to predict all 12 actuated dimensions from imitation learning alone is hard —
especially when the teleoperator's pinch gesture only conveys "grasp" or "release."

The two-phase training strategy:

| Phase | Hand control | Policy dim | Env variant |
|-------|-------------|------------|-------------|
| **IL (teleop + ACT training)** | Binary gripper — all fingers open/close together | 26D (14 arm + 12 hand, but hand values are binary patterns) | `Isaac-Grasp-Policy-G129-InspireFTP-Teleop` |
| **RL (post-training)** | Full per-finger control — each of the 6 actuated joints moves independently | 26D (14 arm + 12 hand, truly independent) | `Isaac-Grasp-Policy-G129-InspireFTP-Joint` |

**Why this works:** IL learns the hard part (arm trajectories, approach angles, grasp
timing) with a simple hand model. RL then fine-tunes grasp quality by unlocking
per-finger control, starting from a policy that already knows *when* and *where* to
grasp. This mirrors the Dex3 pipeline where teleop uses a binary gripper
(`G1HandtrackingGripperRetargeter`) and the recorded joint positions are what the
policy learns.

---

### 14.2 Architecture: How the Gripper Retargeter Works

#### Class Hierarchy

```
RetargeterBase                         (isaaclab.devices.retargeter_base)
  └── UnitreeG1Retargeter              (isaaclab...inspire.g1_upper_body_retargeter)
        │   - _retarget_abs()          wrist OpenXR → USD frame transform
        │   - _hands_controller        UnitreeG1DexRetargeting (per-finger)
        │
        └── InspireGripperRetargeter   (teleop_devices.inspire_gripper_retargeter)
              - Skips _hands_controller init (no dex-retargeting engine)
              - Inherits _retarget_abs() for wrist retargeting
              - Adds binary pinch detection + 24D gripper expansion
```

`InspireGripperRetargeter` calls `RetargeterBase.__init__()` directly, bypassing
`UnitreeG1Retargeter.__init__()` which would instantiate `UnitreeG1DexRetargeting`
(requires pinocchio and Nucleus retargeting URDFs). The wrist retargeting method
`_retarget_abs()` is a pure method on `UnitreeG1Retargeter` and is inherited without
any init dependency.

#### Data Flow

```
AVP Hand Tracking (OpenXR)
    │
    ├── wrist pose (7D each) ──→ _retarget_abs() ──→ left_wrist(7), right_wrist(7)
    │                             (inherited from UnitreeG1Retargeter)
    │
    └── thumb_tip + index_tip ──→ pinch distance ──→ hysteresis ──→ binary 0/1
                                                                        │
                                                          ┌─────────────┘
                                                          ▼
                                                 gripper expansion
                                                 0 → open_joints (24D zeros)
                                                 1 → closed_joints (24D pre-computed)
                                                          │
                                                          ▼
                                              torch.cat([left_wrist(7),
                                                         right_wrist(7),
                                                         hand_joints(24)])
                                                          │
                                                      = 38D output
                                                          │
                                                    PinkIK action
                                              (unchanged — same 38D format)
```

#### Before vs After

| Aspect | Before (dex-retargeting) | After (binary gripper) |
|--------|-------------------------|----------------------|
| Retargeter class | `UnitreeG1Retargeter` | `InspireGripperRetargeter` |
| Hand controller | `UnitreeG1DexRetargeting` (pinocchio) | None (pre-computed arrays) |
| Hand joint names | Nucleus-style via `_URDF_TO_NUCLEUS` bridge | URDF-style directly (`joint_names[29:]`) |
| Finger control | Independent per-finger from AVP tracking | Binary open/close from pinch distance |
| Output format | 38D `[wrist(7), wrist(7), hand(24)]` | 38D `[wrist(7), wrist(7), hand(24)]` (identical) |
| PinkIK config | Unchanged | Unchanged |
| Dependencies | pinocchio, dex-retargeting URDFs | None (numpy only) |

> **Code:**
> [`scripts/teleop_devices/inspire_gripper_retargeter.py`](../../scripts/teleop_devices/inspire_gripper_retargeter.py)
> — `InspireGripperRetargeter`, `InspireGripperRetargeterCfg`.

---

### 14.3 Pinch Detection (Hysteresis)

The gripper uses the same pinch-based detection as the Dex3 teleop. The user's
thumb tip and index finger tip positions (from OpenXR hand tracking) are compared:

```
distance = ||thumb_tip[:3] - index_tip[:3]||
```

A hysteresis state machine prevents oscillation when the user's fingers are near
the threshold boundary:

```
                    distance < 0.03m
         ┌──────────────────────────────────┐
         │                                  ▼
     ┌───────┐                         ┌────────┐
     │ OPEN  │                         │ CLOSED │
     │ (0.0) │                         │ (1.0)  │
     └───────┘                         └────────┘
         ▲                                  │
         │                                  │
         └──────────────────────────────────┘
                    distance > 0.05m
```

| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| `pinch_close_distance` | 0.03 | meters | Pinch tighter than this to close |
| `pinch_open_distance` | 0.05 | meters | Spread wider than this to open |

The 2cm dead zone (0.03–0.05m) prevents flickering. If the user's fingers are at
0.04m, the gripper holds its previous state.

> **Dex3 reference:**
> [`scripts/teleop_devices/handtracking.py`](../../scripts/teleop_devices/handtracking.py)
> — `G1HandtrackingGripperRetargeter._compute_pinch_gripper()` uses identical
> thresholds (0.03m / 0.05m).

---

### 14.4 Gripper Expansion: 1D to 24D

When the gripper closes, all 12 actuated joints are set to `gripper_closed_angle`
(default: 1.0 rad), and the 12 mimic joints are computed from the actuated values
using the URDF multiplier rules. When the gripper opens, all 24 joints go to 0.

These two 24D arrays (`_open_joints` and `_closed_joints`) are pre-computed at init
time since they are deterministic for a given `gripper_closed_angle`.

#### Mimic Rules (applied sequentially)

Order matters for the thumb chain — `thumb_3` depends on `thumb_2`, and `thumb_4`
depends on `thumb_3`:

| Mimic joint | Parent joint | Multiplier | Closed value (at 1.0 rad) |
|-------------|-------------|------------|--------------------------|
| `{side}_index_2_joint` | `{side}_index_1_joint` | 1.0843 | 1.0843 |
| `{side}_middle_2_joint` | `{side}_middle_1_joint` | 1.0843 | 1.0843 |
| `{side}_ring_2_joint` | `{side}_ring_1_joint` | 1.0843 | 1.0843 |
| `{side}_little_2_joint` | `{side}_little_1_joint` | 1.0843 | 1.0843 |
| `{side}_thumb_3_joint` | `{side}_thumb_2_joint` | 0.8024 | 0.8024 |
| `{side}_thumb_4_joint` | `{side}_thumb_3_joint` | 0.9487 | 0.7613 |

> **Note:** `thumb_4` is a chained mimic — it mimics `thumb_3` (not `thumb_2`
> directly), so its closed value is `1.0 × 0.8024 × 0.9487 = 0.7613`.

#### Full 24D Joint Table (at default `gripper_closed_angle = 1.0`)

The 24 hand joints are in USD articulation order (`joint_names[29:]`):

```
Idx  Joint name              Type      Open   Closed
───  ──────────────────────  ────────  ─────  ──────
 0   left_index_1_joint      actuated  0.0    1.0000
 1   left_little_1_joint     actuated  0.0    1.0000
 2   left_middle_1_joint     actuated  0.0    1.0000
 3   left_ring_1_joint       actuated  0.0    1.0000
 4   left_thumb_1_joint      actuated  0.0    1.0000
 5   right_index_1_joint     actuated  0.0    1.0000
 6   right_little_1_joint    actuated  0.0    1.0000
 7   right_middle_1_joint    actuated  0.0    1.0000
 8   right_ring_1_joint      actuated  0.0    1.0000
 9   right_thumb_1_joint     actuated  0.0    1.0000
10   left_index_2_joint      mimic     0.0    1.0843
11   left_little_2_joint     mimic     0.0    1.0843
12   left_middle_2_joint     mimic     0.0    1.0843
13   left_ring_2_joint       mimic     0.0    1.0843
14   left_thumb_2_joint      actuated  0.0    1.0000
15   right_index_2_joint     mimic     0.0    1.0843
16   right_little_2_joint    mimic     0.0    1.0843
17   right_middle_2_joint    mimic     0.0    1.0843
18   right_ring_2_joint      mimic     0.0    1.0843
19   right_thumb_2_joint     actuated  0.0    1.0000
20   left_thumb_3_joint      mimic     0.0    0.8024
21   right_thumb_3_joint     mimic     0.0    0.8024
22   left_thumb_4_joint      mimic     0.0    0.7613
23   right_thumb_4_joint     mimic     0.0    0.7613
```

> **Code (mimic rules):**
> [`scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py)
> — `_MIMIC_RULES_PER_SIDE`, `MIMIC_RULES`.
>
> **Code (expansion):**
> [`scripts/teleop_devices/inspire_gripper_retargeter.py`](../../scripts/teleop_devices/inspire_gripper_retargeter.py)
> — `_compute_closed_joints()`.

---

### 14.5 Tunable Parameters

All parameters are fields on `InspireGripperRetargeterCfg` and can be adjusted in the
teleop env config without modifying the retargeter code.

#### `gripper_closed_angle` (default: 1.0 rad)

The uniform angle applied to **all 12 actuated joints** when the gripper closes.
Mimic joints scale from this value via their multipliers.

**URDF joint limits (from the Inspire FTP URDF):**

| Joint | Lower | Upper |
|-------|-------|-------|
| `thumb_1_joint` (yaw) | 0 | 1.1641 rad |
| `thumb_2_joint` (pitch) | 0 | 0.5864 rad |
| `index/middle/ring/little_1_joint` | 0 | 1.4381 rad |

**Tuning notes:**

- The thumb pitch (`thumb_2_joint`) has the tightest limit at **0.5864 rad**. At the
  default `gripper_closed_angle = 1.0`, thumb_2 is commanded to 1.0 rad which exceeds
  its URDF limit. IsaacLab's actuator will clamp to the joint limit, but you may want
  to reduce `gripper_closed_angle` to stay within bounds for all joints.
- Setting `gripper_closed_angle = 0.5864` would respect the thumb_2 limit, but the
  four-finger joints (limit 1.4381) would only close ~40% of their range.
- **Recommended approach for tuning:** Start with the default (1.0 rad) and observe
  the sim. If the thumb looks over-extended or the grip is too weak, reduce toward
  0.5–0.7. The four-finger joints are forgiving (large range), so thumb_2 is the
  binding constraint.
- **Future extension:** To use per-finger closed angles (e.g., different angle for
  thumb vs fingers), modify `_compute_closed_joints()` to accept a dict mapping joint
  names to angles instead of a single float. The expansion logic is already
  name-based, so this is a straightforward change.

#### `pinch_close_distance` (default: 0.03m) and `pinch_open_distance` (default: 0.05m)

Control how sensitive the gripper is to the user's thumb-index pinch.

**Tuning notes:**

- **Too sensitive (close too large, e.g., 0.05m):** Gripper closes when the user
  isn't intending to grasp. Accidental closures during reaching.
- **Too insensitive (close too small, e.g., 0.01m):** User must pinch very hard.
  Fatiguing for long recording sessions. Missed grasps.
- **Dead zone too small (close ≈ open):** Oscillation / flickering between states.
  Minimum recommended gap: 1.5cm.
- **Dead zone too large:** Sluggish response. User must exaggerate gestures.
- The Dex3 defaults (0.03 / 0.05) are well-tested with AVP hand tracking and are
  a good starting point. Adjust if your tracking environment differs (e.g., different
  lighting, gloves, hand size).

#### Thumb Yaw vs Pitch Angles

The thumb has two actuated joints with very different roles:

- `thumb_1_joint` (yaw): rotates the thumb in/out. At 0 the thumb is abducted; at
  1.16 rad it's adducted against the palm.
- `thumb_2_joint` (pitch): curls the thumb. At 0 the thumb is straight; at 0.59 rad
  the tip is flexed.

Currently both get the same `gripper_closed_angle`. For a more natural grip, you might
want the yaw to close more than the pitch. This would require the per-finger extension
described above.

---

### 14.6 Files Involved

| File | Role |
|------|------|
| [`scripts/teleop_devices/inspire_gripper_retargeter.py`](../../scripts/teleop_devices/inspire_gripper_retargeter.py) | `InspireGripperRetargeter` class + cfg. Pinch detection, 24D expansion, mimic rules. |
| [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py) | Teleop env config. Wires `InspireGripperRetargeterCfg` into the OpenXR device. Defines `HAND_JOINT_NAMES` (24 joints from `joint_names[29:]`). |
| [`scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py) | Canonical mimic rules (`_MIMIC_RULES_PER_SIDE`, `MIMIC_RULES`). The retargeter duplicates these rules — keep them in sync. |
| [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py) | Base env config. Defines `joint_names` (53D) — the 24 hand joints at indices 29–52 are the source of truth for joint ordering. |
| [`scripts/teleop_devices/handtracking.py`](../../scripts/teleop_devices/handtracking.py) | Dex3 gripper retargeter (`G1HandtrackingGripperRetargeter`). Reference implementation for pinch detection. |
| [`scripts/utils/inspire_ftp_lerobot_fields.py`](../../scripts/utils/inspire_ftp_lerobot_fields.py) | 26D LeRobot field definitions. `convert_g1_state_action_to_lerobot_26d()` handles observation-derived actions. |
| [`scripts/utils/convert_hdf5_to_lerobot.py`](../../scripts/utils/convert_hdf5_to_lerobot.py) | HDF5 → LeRobot converter. The `rheo_26d_state_action` branch calls the 26D conversion. |
| [`scripts/config/g1_grasp_policy_inspire_dataset.yaml`](../../scripts/config/g1_grasp_policy_inspire_dataset.yaml) | Dataset conversion config. Sets `rheo_26d_state_action: true`, front camera only. |
| IsaacLab: `g1_upper_body_retargeter.py` | Parent class `UnitreeG1Retargeter` — provides `_retarget_abs()` for wrist retargeting. Located in `third_party/IsaacLab/source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/unitree/inspire/`. |

---

### 14.7 Data Pipeline Impact

The binary gripper changes how teleop data flows from recording to training.

#### Why PinkIK Actions Can't Be Used Directly

During teleop, PinkIK receives 38D input:

```
38D input = [left_wrist_pose(7), right_wrist_pose(7), hand_joints(24)]
```

PinkIK solves arm IK (wrist poses → arm joint positions) but **passes hand joints
through unchanged**. The resulting 38D `processed_actions` in HDF5 contain:

```
38D processed_actions = [arm_IK_output(14), hand_joints_passthrough(24)]
```

This is **not** in the 53D joint-space format that the converter expects. The arm
portion is IK-solved positions (14D), not the full 53D articulation order. Extracting
26D policy joints from 38D requires knowing which 14 of the 38 are arm joints and
remapping them — which the current converter doesn't do.

#### Observation-Derived Actions (the actual approach)

Instead, the converter uses the standard IL approach for teleop recordings:

```
action[t] = state[t+1]    (next-step observed joint positions)
```

This works because:
1. The observations (`robot_joint_state` 87D + `robot_inspire_joint_state` 12D)
   contain the **actual** joint positions the robot achieved
2. These positions already reflect the gripper expansion (the sim drove the fingers
   to the commanded open/closed positions)
3. The 26D extraction pulls 14 arm + 12 hand from observations in canonical order

The converter in `convert_g1_state_action_to_lerobot_26d()` automatically falls back
to observation-derived actions when `processed_actions.shape[1] != 53`:

```python
if action_full is not None and action_full.shape[1] == 53:
    action = action_full[:-1, ACTION_HDF5_TO_ENV_26]  # direct extraction
else:
    action = full_26d[1:]  # observation-derived fallback
```

#### End-to-End Data Flow

```
Recording (in sim):
  AVP pinch ──→ InspireGripperRetargeter ──→ 38D [wrist(14), hand(24)]
       │                                           │
       │                                    PinkIK resolves arms
       │                                           │
       │                                    38D processed_actions
       │                                    (saved to HDF5)
       │
       └──→ Robot moves ──→ Observations recorded:
                              obs/robot_joint_state       (T, 87) = 29×3
                              obs/robot_inspire_joint_state (T, 12) = 12 actuated
                              obs/front_camera              (T, H, W, 3)

Conversion (convert_hdf5_to_lerobot.py):
  obs/robot_joint_state[:, 15:29]        ──→ arm positions (14D)
  obs/robot_inspire_joint_state          ──→ hand positions (12D)
  concatenate                            ──→ full_26d (T, 26)
  state = full_26d[:-1]                  ──→ (T-1, 26)
  action = full_26d[1:]                  ──→ (T-1, 26) [observation-derived]
                                              │
                                         LeRobot dataset
                                         (26D state + 26D action + video)

Training (ACT):
  Policy input:  26D state + front camera image
  Policy output: 26D action (14 arm + 12 hand)
  Hand values in training data are binary patterns (near 0 or near closed_angle)
```

---

### 14.8 Comparison: Dex3 Gripper vs Inspire Gripper

| Aspect | Dex3 (`G1HandtrackingGripperRetargeter`) | Inspire (`InspireGripperRetargeter`) |
|--------|-------|---------|
| **File** | `scripts/teleop_devices/handtracking.py` | `scripts/teleop_devices/inspire_gripper_retargeter.py` |
| **Parent class** | `RetargeterBase` | `UnitreeG1Retargeter` (for wrist retargeting) |
| **IK solver** | WBC+PINK (whole-body, 23D input) | PinkIK (arm-only, 38D input) |
| **Wrist retargeting** | Done by WBC+PINK solver | Inherited `_retarget_abs()` from parent |
| **Pinch thresholds** | 0.03m close / 0.05m open | 0.03m close / 0.05m open (identical) |
| **Gripper expansion** | `get_hand_joint_pos()` → 7D per hand | Pre-computed 24D arrays (12 actuated + 12 mimic) |
| **Expansion method** | Fixed per-joint amplitudes | Uniform angle + URDF mimic multipliers |
| **Output format** | 16D `[grip(1), grip(1), wrist(7), wrist(7)]` | 38D `[wrist(7), wrist(7), hand(24)]` |
| **Processed actions dim** | 43D (WBC+PINK resolves full body) | 38D (PinkIK resolves arms only) |
| **Data conversion** | Direct extraction from 43D → 28D | Observation-derived: `action[t] = state[t+1]` → 26D |
| **Hand DOF (actuated)** | 7 per hand (14 total) | 6 per hand (12 total) |
| **Policy dim** | 28D (14 arm + 14 hand) | 26D (14 arm + 12 hand) |
| **Mimic joints** | None (all 7 are independent) | 6 per hand (computed from actuated) |

The key architectural difference: Dex3 uses WBC+PINK which resolves the entire
43-joint body, giving the converter clean joint-space actions. Inspire uses PinkIK
which only resolves the 14 arm joints, so the converter must derive actions from
observations instead. Both approaches produce correct training data — the observation-
derived method is standard practice for teleop IL.
