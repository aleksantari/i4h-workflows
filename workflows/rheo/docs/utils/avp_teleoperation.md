# AVP Teleoperation for Rheo

How Apple Vision Pro hand tracking teleoperation works with rheo's G1 trocar assembly
task, and how it connects to IsaacLab's OpenXR pipeline.

---

## Quick Start

```bash
# Ensure CloudXR Runtime container is stopped (built-in mode handles it)
docker stop cloudxr-runtime 2>/dev/null

# Run trocar task with AVP hand tracking
./workflows/rheo/docker/run_docker.sh -g1.5 \
  python scripts/simulation/record_demos_assemble_trocar.py \
  --task Isaac-Assemble-Trocar-G129-Dex3-Teleop \
  --teleop_device handtracking \
  --enable_pinocchio \
  --enable_cameras \
  --num_demos 1 \
  --xr

# In Isaac Sim UI: AR Panel -> "CloudXR Runtime (5.0)" -> Start AR
# On AVP: Isaac XR Teleop Sample Client -> enter workstation IP -> Connect -> Play
```

For Quest controllers, use `--teleop_device motion_controllers` instead.

---

## Architecture Overview

```
Apple Vision Pro
    | (hand tracking: 26 joints per hand, each with 7D pose)
    v
CloudXR Runtime (built-in to Isaac Sim 5.1)
    | (WebRTC stream + tracking data)
    v
OpenXRDevice._get_raw_data()
    | Returns: {HAND_LEFT: {joint_name: [x,y,z,qw,qx,qy,qz], ...}, HAND_RIGHT: {...}}
    v
G1HandtrackingGripperRetargeter.retarget()
    | - Wrist pose: hand_poses["wrist"] -> coordinate transform -> 7D
    | - Gripper: thumb-index pinch distance -> binary 0/1 with hysteresis
    | Output: [gripper_L(1), gripper_R(1), left_wrist(7), right_wrist(7)] = 16D
    v
G1LowerBodyStandingMotionControllerRetargeter.retarget()
    | Output: [nav_x=0, nav_y=0, nav_yaw=0, hip_height=0.72] = 4D
    v
torch.cat() -> 20D action (zero-padded to 23D by record_demos script)
    v
G1AssembleTrocarFixedLegsWBCPinkAction.process_actions()
    | - Wrist poses -> PINK IK solver -> shoulder/elbow/wrist joint angles
    | - Gripper -> all fingers open/close in unison
    | - Lower body fixed (nav=0, height=0.75, torso=0)
    v
G1 Robot simulation
```

---

## Teleop Device Comparison

| | AVP Hand Tracking | Quest Motion Controllers |
|---|---|---|
| `--teleop_device` | `handtracking` | `motion_controllers` |
| Input source | 26 OpenXR hand joints per hand | Controller pose + trigger/buttons |
| Gripper control | Thumb-index pinch distance | Trigger analog value |
| Wrist rotation offset | `(0, +/-90, +/-90)` euler | `(0, -75, 90)` euler |
| Requirement | `HAND_TRACKING` | `MOTION_CONTROLLER` |
| Output format | Same 16D: `[grip(2), wrist(14)]` | Same 16D: `[grip(2), wrist(14)]` |

Both devices produce the same 20D action vector (16D upper + 4D lower), so the action
term and demo recording format are identical. Demos recorded with either device are
interchangeable.

---

## Gripper Control Details

The gripper uses pinch detection with hysteresis to prevent flickering:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Close threshold | 0.03 m | Pinch closer than this = close gripper (1.0) |
| Open threshold | 0.05 m | Pinch wider than this = open gripper (0.0) |
| Measurement | `\|\|thumb_tip[:3] - index_tip[:3]\|\|` | Euclidean distance between fingertips |

The G1 TriHand has 7 independently controllable joints per hand (thumb x3, index x2,
middle x2), but in this binary mode all fingers open or close together. Individual finger
control requires switching to the `G1TriHandUpperBodyRetargeter` with a different action
term (see Future Work below).

---

## Wrist Rotation Transforms

Hand tracking and motion controllers use different rotation offsets because the natural
hand pose differs from how you hold a controller:

**Hand tracking** (from `G1TriHandUpperBodyRetargeter`):
- Left hand: euler `(0, 90, 90)` -> quat `[0.7071, 0, 0.7071, 0]`
- Right hand: euler `(0, -90, -90)` -> quat `[0, -0.7071, 0, 0.7071]`

**Motion controllers** (from `G1TriHandUpperBodyMotionControllerGripperRetargeter`):
- Both hands: euler `(0, -75, 90)` -> quat `[0.5358, -0.4619, 0.5358, 0.4619]`

Transform is applied via `PoseUtils.pose_in_A_to_pose_in_B()`.

---

## Trocar Robot-Origin Offset

Both trocar retargeters (`TrocarG1MotionControllerGripperRetargeter` and
`TrocarG1HandtrackingGripperRetargeter`) subtract the robot's world-space origin from
wrist positions to convert to robot-local coordinates:

```python
_ROBOT_ORIGIN_W = [-1.84919, 1.94, 0.81168]

# Applied to left wrist (indices 2:5) and right wrist (indices 9:12)
action[2:5] -= _ROBOT_ORIGIN_W
action[9:12] -= _ROBOT_ORIGIN_W
```

---

## Action Space (23D)

```
Index  Dim  Component
0      1    Left gripper (0.0=open, 1.0=closed)
1      1    Right gripper
2-4    3    Left wrist position (x, y, z)
5-8    4    Left wrist quaternion (qw, qx, qy, qz)
9-11   3    Right wrist position (x, y, z)
12-15  4    Right wrist quaternion (qw, qx, qy, qz)
16-18  3    Navigation (vx, vy, vyaw) -- fixed to 0 for trocar
19     1    Base height -- fixed to 0.75 for trocar
20-22  3    Torso orientation (roll, pitch, yaw) -- fixed to 0 for trocar
```

The teleop device outputs 20D (indices 0-19). The recording script zero-pads to 23D.
The `G1AssembleTrocarFixedLegsWBCPinkAction` then overrides indices 16-22 with fixed values.

---

## OpenXR Hand Joints (26 per hand)

What the AVP sends via CloudXR → OpenXR:

```
palm, wrist,
thumb_metacarpal, thumb_proximal, thumb_distal, thumb_tip,
index_metacarpal, index_proximal, index_intermediate, index_distal, index_tip,
middle_metacarpal, middle_proximal, middle_intermediate, middle_distal, middle_tip,
ring_metacarpal, ring_proximal, ring_intermediate, ring_distal, ring_tip,
little_metacarpal, little_proximal, little_intermediate, little_distal, little_tip
```

Each joint has a 7D pose: `[x, y, z, qw, qx, qy, qz]` in world coordinates.

Currently only `wrist`, `thumb_tip`, and `index_tip` are used (for wrist pose and pinch
detection). All 26 joints are available for future individual finger control.

---

## Key Source Files

### Rheo (this repo)

| File | Purpose |
|------|---------|
| `scripts/teleop_devices/handtracking.py` | AVP hand tracking device + gripper retargeter |
| `scripts/teleop_devices/motion_controllers.py` | Quest controller device + gripper retargeter |
| `scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_teleop_env_cfg.py` | Trocar env config (registers both devices) |
| `scripts/simulation/record_demos_assemble_trocar.py` | Demo recording script |

### IsaacLab (third_party)

| File | Purpose |
|------|---------|
| `isaaclab/devices/openxr/openxr_device.py` | OpenXR device: captures hand joints from XR runtime |
| `isaaclab/devices/openxr/common.py` | `HAND_JOINT_NAMES` constant (26 joints) |
| `isaaclab/devices/device_base.py` | `DeviceBase.advance()`: calls retargeters, concatenates output |
| `isaaclab/devices/retargeter_base.py` | `RetargeterBase`: interface for all retargeters |
| `isaaclab/devices/openxr/retargeters/humanoid/unitree/trihand/g1_upper_body_retargeter.py` | Full hand tracking retargeter (28D with individual fingers) |
| `isaaclab/devices/openxr/retargeters/humanoid/unitree/trihand/g1_upper_body_motion_ctrl_gripper.py` | Motion controller retargeter (16D with binary gripper) |
| `isaaclab/devices/openxr/retargeters/manipulator/gripper_retargeter.py` | Generic pinch-based gripper retargeter |
| `isaaclab/devices/openxr/retargeters/humanoid/unitree/trihand/g1_dex_retargeting_utils.py` | Pinocchio IK for individual finger joints |
| `isaaclab/devices/openxr/retargeters/humanoid/unitree/g1_lower_body_standing.py` | Fixed standing lower body retargeter |

### IsaacLab Environment Reference

| File | Purpose |
|------|---------|
| `isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py` | G1 env that registers both `handtracking` (28D) and `motion_controllers` (16D) |

---

## Future Work: Individual Finger Control

The current implementation uses binary gripper (all fingers open/close together). To
upgrade to individual finger control:

1. **Use `G1TriHandUpperBodyRetargeter`** instead of `G1HandtrackingGripperRetargeter`
   - Output: `[left_wrist(7), right_wrist(7), hand_joints(14)]` = 28D
   - Requires Pinocchio + dex-retargeting for finger IK

2. **Change the action term** from `G1DecoupledWBCPinkAction` (23D with binary gripper)
   to `PinkInverseKinematicsAction` (28D with individual finger joints)

3. **Update the action space** from 23D to 32D (28D upper + 4D lower)

4. **Controlled G1 finger joints** (7 per hand, 14 total):
   ```
   thumb_0_joint (yaw), thumb_1_joint (pitch), thumb_2_joint (tip)
   index_0_joint (proximal), index_1_joint (distal)
   middle_0_joint (proximal), middle_1_joint (distal)
   ```

Reference implementation: `locomanipulation_g1_env_cfg.py` lines 221-254 in IsaacLab.

---

## CloudXR Configuration

See [cloudxr.md](cloudxr.md) for full CloudXR setup details. Key points for AVP teleop:

- **Built-in mode** (recommended for Docker): Select "CloudXR Runtime (5.0)" in AR panel
- **External Docker mode** (for local conda): Requires `XDG_RUNTIME_DIR` and
  `XR_RUNTIME_JSON` env vars
- **Port 48010** must not be in use by another CloudXR container
- **`NV_GPU_INDEX=0`** required for external mode to fix "tracking but no video" issue
