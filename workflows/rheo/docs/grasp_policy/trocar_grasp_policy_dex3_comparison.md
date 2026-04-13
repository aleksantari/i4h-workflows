# Assemble Trocar vs Grasp Policy: Full Pipeline Comparison

Both tasks live on the **IsaacLab track** (not Arena). They share the same robot (G1 29-DOF + Dex3 14-DOF hands), the same 28D state/action representation at the policy level, and the same 43D simulation action space. The key differences are the **policy architecture** (GR00T vs ACT), the **reward design**, the **teleop device**, and the **data augmentation** availability.

---

## At a Glance

| Aspect | Assemble Trocar | Grasp Policy |
|--------|----------------|--------------|
| **IL Model** | GR00T N1.5 (DiT, TensorRT) | ACT (CVAE, LeRobot) |
| **Action Chunk Size** | 16 | 100 |
| **Reward Stages** | 4 (lift, align, insert, place) | 3 (grasp, transport, place) |
| **Max Reward** | 4.0 | 3.0 |
| **Teleop Device** | Meta Quest controllers | Apple Vision Pro hand tracking |
| **Data Augmentation** | Training-time only (no Mimic) | Training-time only (no Mimic) |
| **Arena Used?** | Teleop only (WBC+PINK action) | Teleop only (WBC+PINK action) |
| **Task Description** | "install trocar from box" | "pick up block and place in bin" |
| **Scene Objects** | Two trocars + tray | One block + bin |

---

## 1. Task Definition & Scene

### Assemble Trocar

- **Goal**: Pick up two trocars from a box, align tips, insert one into the other, place assembled trocar into tray.
- **Scene objects**: Red trocar, white trocar, surgical tray (with random yaw rotation 0-10 deg).
- **Gym IDs**:
  - `Isaac-Assemble-Trocar-G129-Dex3-Joint` (RL training)
  - `Isaac-Assemble-Trocar-G129-Dex3-Joint-Eval` (deterministic tray rotation)
  - `Isaac-Assemble-Trocar-G129-Dex3-Teleop` (VR teleoperation)

### Grasp Policy

- **Goal**: Pick up a block from a table and place it into a bin.
- **Scene objects**: Red block (5cm cube, 0.1kg, friction mu_s=0.8/mu_k=0.6), green target pad, blue table, bin.
- **Gym IDs**:
  - `Isaac-Grasp-Policy-G129-Dex3-Joint` (RL training, random block placement)
  - `Isaac-Grasp-Policy-G129-Dex3-Joint-Eval` (deterministic placement)
  - `Isaac-Grasp-Policy-G129-Dex3-Teleop` (VR teleoperation, extended episode, fixed legs)

### Shared

- Robot init position: `(-1.84919, 1.94, 0.81168)`, quaternion `(1, 0, 0, 0)`
- Decimation: 4, sim dt: 1/200s, episode length: 20s
- Physics: PhysX, bounce_threshold_velocity=0.01

---

## 2. Action Space

### Simulation Action (both tasks): 43D

Both tasks use `JointPositionActionCfg` with 43 joints: hip(6) + knee(2) + ankle(4) + waist(3) + shoulder(6) + elbow(2) + wrist(6) + hand(14). Elbow offset: -0.3 for both arms.

### Policy Action: 28D

Both tasks strip legs/waist at the policy level:

```
28D = left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7)
```

The 15 leg/waist joints are padded with zeros when converting 28D policy output back to 43D sim input.

### Teleop Action

| | Assemble Trocar | Grasp Policy |
|--|----------------|--------------|
| **Device** | Meta Quest motion controllers | AVP hand tracking (OpenXR) |
| **Action dim** | 23D (WBC+PINK) | 23D (WBC+PINK) |
| **Controller** | `motion_controllers.py` | `handtracking.py` |
| **Gripper** | Trigger button | Pinch distance (hysteresis: 0.03m close / 0.05m open) |
| **Wrist control** | Controller pose | PINK IK from hand pose |
| **Output** | Gripper(2) + wrist(14) + nav(3) + height(1) + torso(3) | Same 23D format |

**Key difference**: Grasp policy teleop uses `G1HandtrackingGripperRetargeter` which outputs 16D raw (gripper(2) + wrist_poses(14)), fed into the WBC+PINK action class (`G1DecoupledWBCPinkAction` from Arena). Both tasks import this Arena action class for teleop only.

---

## 3. Observation Space

### Identical for both tasks

| Observation | Shape | Source |
|-------------|-------|--------|
| `robot_joint_state` | (B, 87) | 29 body joints x 3 (pos, vel, torque) |
| `robot_dex3_joint_state` | (B, 14) | 14 Dex3 hand joint positions |
| `front_camera` | (B, 480, 640, 3) | Front view, focal_length=10.5 |
| `left_wrist_camera` | (B, 480, 640, 3) | Left Dex3 wrist, focal_length=12.0 |
| `right_wrist_camera` | (B, 480, 640, 3) | Right Dex3 wrist, focal_length=12.0 |

Both import the same observation functions from `assemble_trocar.mdp.observations`:
- `get_robot_body_joint_states()` - 87D with custom joint index reordering
- `get_robot_dex3_joint_states()` - 14D with Dex3-specific indices `[31,37,41,30,36,29,35,34,40,42,33,39,32,38]`

The 28D state vector for policies is extracted as:
- Arm joints: `robot_joint_state[:, 15:29]` (14D)
- Hand joints: `robot_dex3_joint_state` (14D)

Camera presets are also shared (imported from `assemble_trocar.config`).

---

## 4. Reward Design

### Assemble Trocar: 4-Stage Sparse (max 4.0)

| Stage | Transition | Condition | Reward |
|-------|-----------|-----------|--------|
| 0 -> 1 | Lift | Both trocars z > table + 0.15m (1.00483m) | 1.0 |
| 1 -> 2 | Find hole | Trocar tip distance < 0.015m | 1.0 |
| 2 -> 3 | Insert | Angle < 0.15 rad AND center distance < 0.05m | 1.0 |
| 3 -> 4 | Place | Both trocars in tray zone (x/y/z bounds) | 1.0 |

Dense fallbacks available (Gaussian distance rewards) but sparse mode is default.

### Grasp Policy: 3-Stage Sparse (max 3.0)

| Stage | Transition | Condition | Reward |
|-------|-----------|-----------|--------|
| 0 -> 1 | Grasp | Block z > table_height + 0.05m (0.905m) | 1.0 |
| 1 -> 2 | Transport | Block x,y within bin bounds (x: [-1.90, -1.60], y: [1.55, 1.85]) | 1.0 |
| 2 -> 3 | Place | Block within bin AND z < rim (0.935m) AND z > floor (0.80m) | 1.0 |

Both use `1.0 / step_dt` scaling for the transition step. Stages only advance forward, never backward. Each reward function tracks its own `_prev_stage_*` variable.

### Termination Conditions

| Condition | Assemble Trocar | Grasp Policy |
|-----------|----------------|--------------|
| Timeout | 20s | 20s |
| Success | stage >= 4 | stage >= 3 |
| Object drop | trocar z < 0.5m | block z < 0.5m |

### Randomization / Events

| Event | Assemble Trocar | Grasp Policy |
|-------|----------------|--------------|
| Scene reset | `reset_scene_to_default` | `reset_scene_to_default` |
| Stage reset | `reset_task_stage` (zeros stage + reward caches) | `reset_task_stage` (same pattern) |
| Object randomization | `reset_tray_with_random_rotation` (0-10 deg yaw) | `reset_block_random_position` (+-3cm x,y) |

---

## 5. Data Collection

### Assemble Trocar

- **Script**: `record_demos_assemble_trocar.py` (legacy, task-specific)
- **Teleop**: Meta Quest controllers (`motion_controllers`)
- **Controls**: B=start, S=save, R=reset (keyboard)
- **Recording rate**: 50 Hz (update_period=0.02s)
- **Export mode**: `EXPORT_SUCCEEDED_ONLY`
- **HDF5 fields**: `obs/robot_joint_state(87)`, `obs/robot_dex3_joint_state(14)`, `obs/front_camera`, `obs/left_wrist_camera`, `obs/right_wrist_camera`, `processed_actions(43)`

### Grasp Policy

- **Script**: `record_demos.py` (generic, replaces task-specific scripts)
- **Teleop**: AVP hand tracking (`handtracking`) or keyboard/spacemouse/gamepad
- **Controls**: VR gesture-based (OpenXR START/STOP/RESET) + keyboard fallback
- **Recording rate**: 30 Hz (configurable via `--step_hz`)
- **Auto-success**: Detects success termination and auto-saves after N consecutive success frames (default 10)
- **XR UI**: Demo count displayed as 3D text in headset
- **Export mode**: `EXPORT_SUCCEEDED_ONLY`
- **HDF5 fields**: Same structure as trocar

---

## 6. Data Conversion (HDF5 -> LeRobot)

### Shared Pipeline

Both use `convert_hdf5_to_lerobot.py` with the `use_rheo_converter: true` path and `rheo_28d_state_action: true`. The 43D recorded actions are converted to 28D using the joint mapping in `assemble_trocar_lerobot_fields.py` (shared by both tasks despite the name). Elbow offset (+0.3) is applied during conversion.

### Dataset Config Differences

| Field | Assemble Trocar | Grasp Policy |
|-------|----------------|--------------|
| Config file | `g1_assemble_trocar_dataset.yaml` | `g1_grasp_policy__dex3_dataset.yaml` |
| Language instruction | "install trocar from box" | "pick up block and place in bin" |
| Modality template | `modality_assemble_trocar.json` | `modality_grasp_policy_dex3.json` |

The modality JSON files are **identical in structure** (same 28D state/action grouping, same camera keys). Only the language annotation differs.

---

## 7. Data Augmentation

### Neither Task Has Mimic-Based Data Augmentation

`generate_dataset.py` and `annotate_demos.py` are built on Arena's environment framework and **require** the environment to:
1. Be registered in `register_and_patch.py` → `ExampleEnvironments`
2. Inherit from `ManagerBasedRLMimicEnv` (hard check — `generate_dataset.py` raises `ValueError` otherwise)
3. Implement the mimic API (`datagen_config`, `subtask_configs`, `get_subtask_term_signals()`, etc.)

**Neither assemble_trocar nor grasp_policy meets these requirements.** Both inherit from plain `ManagerBasedRLEnvCfg` and are registered only as gym IDs, not in `ExampleEnvironments`. Only the **locomanip Arena-track tasks** (tray pick-and-place, cart push) have Mimic support.

### What IS Available

| Augmentation | Assemble Trocar | Grasp Policy |
|-------------|----------------|--------------|
| Cosmos Transfer 2.5 (external synthetic) | Referenced in docs | Not yet |
| Training-time image augmentation | GR00T: VideoColorJitter, VideoCrop, VideoResize | ACT: random crop (456x608), mean/std norm |

### What Would Be Needed for Mimic Support (Either Task)

1. Create an `ExampleEnvironmentBase` subclass with a `get_mimic_env_cfg()` method
2. Define `datagen_config` and `subtask_configs` in the env configuration
3. Register in `register_and_patch.py` → `register_workflow_cli()`
4. Implement mimic API methods (`target_eef_pose_to_action`, `get_object_poses()`, `get_robot_eef_pose()`, etc.)

---

## 8. IL Training

### Assemble Trocar: GR00T SFT

| Parameter | Value |
|-----------|-------|
| Model | GR00T N1.5 (DiT, TensorRT at inference) |
| Framework | GR00T fine-tuning scripts |
| Data config | `UnitreeG1SimDataConfig` in `gr00t_config.py` |
| Action horizon | 16 steps |
| State horizon | 1 step |
| Max steps | 30,000 |
| Batch size | 32 |
| Image processing | Resize to 224x224, ColorJitter, Crop (scale=0.95) |
| State encoding | Sin/cos + min-max normalization |
| Visual backbone | Tuned (--tune_visual) |
| Embodiment tag | `new_embodiment` |

### Grasp Policy: ACT (LeRobot)

| Parameter | Value |
|-----------|-------|
| Model | ACT (Action Chunking Transformer, CVAE) |
| Framework | LeRobot training pipeline |
| Config | `act_config_dex3.yaml` |
| Chunk size | 100 steps |
| Observation steps | 1 |
| Offline steps | 100,000 |
| Batch size | 64 |
| Learning rate | 1e-5 (both backbone and policy) |
| Architecture | dim_model=512, n_heads=8, 4 encoder layers, 1 decoder layer |
| CVAE | latent_dim=32, kl_weight=10.0 |
| Vision backbone | ResNet18 (ImageNet pretrained) |
| Image processing | Random crop (456x608), mean/std normalization |
| State normalization | mean_std |
| Input state dim | 28D |
| Output action dim | 28D |
| Cameras | 3 (cam_left_wrist, cam_right_wrist, cam_room), 480x640 |

### Key IL Differences

- GR00T uses **16-step action chunks** with a DiT backbone and TensorRT acceleration.
- ACT uses **100-step action chunks** with a smaller CVAE-based transformer and no TensorRT.
- GR00T takes images at 224x224; ACT uses full 480x640 with random cropping.
- GR00T uses sin/cos encoding for joint states; ACT uses mean/std normalization.

---

## 9. RL Post-Training

### Shared Infrastructure

Both use **RLinf PPO** with chunk-level reward/logprob/entropy computation. Both register through `rlinf_ext/__init__.py` and share the same obs/action converter interface (28D <-> 43D padding).

### Assemble Trocar RL

| Parameter | Value |
|-----------|-------|
| Script | `train_gr00t_assemble_trocar.sh` |
| Config | `isaaclab_ppo_gr00t_assemble_trocar.yaml` |
| Model | GR00T (DiT) with value head |
| Env wrapper | `IsaaclabG129Dx3Env` |
| Train envs | 64 |
| Eval envs | 64 |
| Max episode steps | 256 |
| Actor LR | 5e-6 |
| Critic LR | 1e-4 |
| PPO clip | 0.2 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Update epochs | 4 |
| Rollout epochs | 8 |
| Num action chunks | 1 |
| Precision | bfloat16 |
| Requires RL patch | Yes (`gr00t_policy_padding_dropout.patch`) |

### Grasp Policy RL

| Parameter | Value |
|-----------|-------|
| Script | `train_act_grasp_policy_dex3.sh` (in `rl/`) |
| Config | `isaaclab_ppo_act_grasp_policy_dex3.yaml` |
| Model | ACT with `ValueHead` (MLP: latent -> 256 -> 1) |
| Env wrapper | `IsaaclabGraspPolicyEnv` |
| Train envs | 64 |
| Eval envs | 64 |
| Max episode steps | 256 |
| Actor LR | 5e-6 |
| Critic LR | 1e-4 |
| PPO clip | 0.2 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Update epochs | 4 |
| Rollout epochs | 8 |
| Num action chunks | 1 |
| Precision | bfloat16 |
| Requires RL patch | No |

### Key RL Differences

- **GR00T RL requires a git patch** (`apply_gr00t_rl_patch.py` context manager) to replace dropout with Identity and pad eagle inputs to 850 tokens. ACT does not need any patch.
- **ACT value head** is a simple MLP with orthogonal init (gain=0.01). GR00T value head is added via RLinf's built-in mechanism.
- **ACT RL wrapper** (`act_policy.py`) implements `ACTForRLActionPrediction(BasePolicy, nn.Module)` with `default_forward()` and `predict_action_batch()` methods.

---

## 10. Policy Evaluation

### Assemble Trocar

- **Script**: `eval_assemble_trocar.py`
- **Policy wrapper**: `CustomGr00tClosedloopPolicy` (in `gr00t_closedloop_policy.py`)
- **Policy type**: GR00T only
- **RL checkpoint flag**: `--rl_ckpt` (applies patch, required for RL checkpoints)
- **Action chunk consumption**: 1-16 configurable via `--action_chunk_size`
- **Observation prep**: Manual extraction per camera/state group into GR00T modality format

### Grasp Policy

- **Script**: `eval_grasp_policy_dex3.py`
- **Policy wrapper**: `ACTClosedloopPolicy` (in `act_closedloop_policy.py`) or `CustomGr00tClosedloopPolicy`
- **Policy types**: `--policy_type {gr00t, act, test}`
- **RL checkpoint flag**: Not needed for ACT
- **Action chunk consumption**: 1 action per step from 100-step chunk
- **Observation prep**: Via `ObsProcessor` -> `ProcessedObservation` dataclass (model-agnostic)

### Policy Wrapper Hierarchy

```
BaseClosedloopPolicy (abstract)
  - Manages per-env action chunk state
  - pad_28d_to_43d() static helper
  |
  +-- CustomGr00tClosedloopPolicy (GR00T N1.5/N1.6)
  |     - 16-step chunks, 43D native output
  |     - Joint remapping via joint_conversion.py
  |
  +-- ACTClosedloopPolicy (ACT/LeRobot)
        - 100-step chunks, 28D -> 43D padding
        - Loads via ACTPolicy.from_pretrained()
        - Per-env inference (processes each env separately)
```

---

## 11. Arena Usage

Arena (IsaacLab-Arena) is used in **three** places across the pipeline:

### 1. Teleop (both tasks)
Both tasks import `G1DecoupledWBCPinkAction` and `G1DecoupledWBCPinkActionCfg` from `isaaclab_arena_g1` for the `-Teleop` gym variant. This provides the whole-body controller + PINK IK used during VR teleoperation.

### 2. Data Augmentation / Synthetic Generation (trocar only, currently)
`generate_dataset.py` and `annotate_demos.py` are **built on Arena's environment framework**:
- Both use `get_arena_builder_from_cli()` to construct environments
- Both call `register_workflow_cli()` and `register_workflow_assets()` from `register_and_patch.py`
- Both require the environment to be a `ManagerBasedRLMimicEnv` (from `isaaclab_mimic`)
- `generate_dataset.py` uses `isaaclab_mimic.datagen.generation` for synthetic demo generation
- `annotate_demos.py` uses Arena's builder to replay demos and add subtask annotations

Currently, only the **locomanip environments** are registered in `register_and_patch.py`. For grasp_policy to use `generate_dataset.py` or `annotate_demos.py`, it would need to:
1. Be registered in `register_and_patch.py` → `register_workflow_cli()`
2. Inherit from or be wrapped as a `ManagerBasedRLMimicEnv`
3. Implement the mimic API (`target_eef_pose_to_action`, `get_object_poses()`, `get_subtask_term_signals()`, etc.)

### 3. RL Training and Evaluation (neither task)
RL training and policy evaluation do **not** use Arena. Both tasks use standard `gymnasium.register()` with `ManagerBasedRLEnv` as the entry point.

---

## 12. Pipeline Stage Summary

### Assemble Trocar (Complete Pipeline)

```
Meta Quest Teleop               GR00T SFT            RLinf PPO
     |                              |                    |
record_demos_assemble_trocar.py     |                    |
     |                              |                    |
     v                              |                    |
  HDF5 (43D actions, 87D state)     |                    |
     |                              |                    |
  annotate_demos.py (subtasks)      |                    |
     |                              |                    |
  generate_dataset.py (Mimic)       |                    |
     |                              |                    |
  convert_hdf5_to_lerobot.py -----> |                    |
  (28D state/action)           gr00t_finetune.py         |
                                    |                    |
                               GR00T ckpt -----> train_gr00t_assemble_trocar.sh
                                                         |
                                                    RL ckpt
                                                         |
                                              eval_assemble_trocar.py (--rl_ckpt)
```

### Grasp Policy (Current Pipeline)

```
AVP Hand Tracking              ACT IL (LeRobot)     RLinf PPO
     |                              |                    |
  record_demos.py                   |                    |
  (auto-success, VR gestures)       |                    |
     |                              |                    |
     v                              |                    |
  HDF5 (43D actions, 87D state)     |                    |
     |                              |                    |
  [annotate_demos.py] (TODO)        |                    |
     |                              |                    |
  [generate_dataset.py] (TODO)      |                    |
     |                              |                    |
  convert_hdf5_to_lerobot.py -----> |                    |
  (28D state/action)           train_act_grasp_policy_dex3.sh |
                               (lerobot.scripts.train)        |
                                    |                         |
                               ACT ckpt -------> rl/train_act_grasp_policy_dex3.sh
                                                              |
                                                         RL ckpt
                                                              |
                                              eval_grasp_policy_dex3.py (--policy_type act)
```

Steps marked `[TODO]` have generic tooling available but haven't been configured for grasp_policy yet.
