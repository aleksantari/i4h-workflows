# Building a Custom Manipulation Task in Rheo

A developer reference and tutorial for creating your own manipulation task (no locomotion),
collecting demonstration data, training with imitation learning (GR00T SFT), and refining
with reinforcement learning (RLinf PPO). Written for someone who wants to pick up tools and
place them in a bin using the G1 humanoid with Dex3 hands.

Everything here is based on how the existing **Assemble Trocar** task was built. That task
is the closest analog: pure bimanual manipulation, no locomotion, same robot (G1 + Dex3),
same pipeline end-to-end.

---

## Table of Contents

- [How the Repo Is Structured](#how-the-repo-is-structured)
- [Phase 1: Define Your Task (IsaacLab Environment)](#phase-1-define-your-task)
- [Phase 2: Collect Demonstrations](#phase-2-collect-demonstrations)
- [Phase 3: Process Data for GR00T Fine-Tuning](#phase-3-process-data-for-groot-fine-tuning)
- [Phase 4: Imitation Learning (GR00T SFT)](#phase-4-imitation-learning-groot-sft)
- [Phase 5: Evaluate the IL Policy](#phase-5-evaluate-the-il-policy)
- [Phase 6: RL Post-Training (RLinf PPO)](#phase-6-rl-post-training-rlinf-ppo)
- [Phase 7: Evaluate the RL Policy](#phase-7-evaluate-the-rl-policy)
- [Quick Reference: Key Files](#quick-reference-key-files)
- [Quick Reference: Dimensions and Conventions](#quick-reference-dimensions-and-conventions)

---

## How the Repo Is Structured

### Execution Model

All simulation runs go through Docker. Never run simulation scripts on the host.

```bash
# For manipulation tasks (GR00T N1.5)
./docker/run_docker.sh -g1.5 python <script> ...

# Interactive shell
./docker/run_docker.sh -g1.5
```

Inside the container:
- `python` is aliased to `/isaac-sim/python.sh`
- Working directory: `/workspaces/workflows/rheo`
- Host mounts: `$HOME/datasets` -> `/datasets`, `$HOME/models` -> `/models`
- PYTHONPATH includes `workflows/rheo/scripts` (so `from simulation.X` works)

### Two Simulation Tracks

The repo has two tracks. **You want the IsaacLab track** (not Arena):

| Track | Purpose | GR00T Version | When to Use |
|-------|---------|---------------|-------------|
| IsaacLab-Arena | Locomotion + manipulation | N1.6 | Full-body tasks (walk to tray, pick up, walk to cart) |
| **IsaacLab** | **Manipulation only** | **N1.5** | **Tabletop/fixed-base tasks (your use case)** |

The Arena track composes scenes from swappable objects/backgrounds/embodiments. The IsaacLab
track defines the scene explicitly as a `ManagerBasedRLEnvCfg` — robot, cameras, objects,
reward, termination, events — all in one configclass hierarchy. This is what assemble_trocar
uses and what you should follow.

### Code Layout (What Matters for You)

```
scripts/
├── simulation/
│   ├── tasks/
│   │   └── assemble_trocar/              # <-- REFERENCE: copy this pattern
│   │       ├── __init__.py               # gym.register() calls
│   │       ├── g1_assemble_trocar_env_cfg.py        # Training env config
│   │       ├── g1_assemble_trocar_teleop_env_cfg.py # Teleoperation env config
│   │       ├── config/
│   │       │   ├── robot_config.py       # G1 + Dex3 articulation preset
│   │       │   └── camera_config.py      # Front + wrist camera presets
│   │       └── mdp/
│   │           ├── observations.py       # Joint state extraction (cached tensors)
│   │           ├── rewards.py            # Stage-based sparse rewards
│   │           ├── terminations.py       # Success + failure conditions
│   │           └── events.py            # Reset randomization
│   ├── record_demos_assemble_trocar.py   # XR data collection script
│   ├── gr00t_closedloop_policy.py        # Policy wrapper (action chunking + joint remap)
│   ├── examples/
│   │   └── eval_assemble_trocar.py       # Evaluation with success metrics
│   └── rl/
│       ├── rlinf_ext/                    # RLinf extension (env + converter registration)
│       │   ├── __init__.py               # register() — the entry point
│       │   └── config/                   # Hydra YAML configs for PPO
│       └── train_gr00t_assemble_trocar.sh
├── policy/
│   ├── gr00t_config.py                   # GR00T data config (modalities, transforms)
│   └── apply_gr00t_rl_patch.py           # RL patch context manager
├── utils/
│   ├── convert_hdf5_to_lerobot.py        # HDF5 -> LeRobot parquet + video
│   ├── assemble_trocar_lerobot_fields.py # 28-D joint mapping definitions
│   └── joint_conversion.py              # Policy-to-sim joint remapping
└── config/
    └── g1_assemble_trocar_dataset.yaml   # Dataset conversion config
```

### The Pipeline at a Glance

```
Define Task ──> Collect Demos ──> Convert to LeRobot ──> Fine-Tune GR00T (IL)
                     │                                         │
                     │ (optional: annotate + Mimic Gen)        │
                     │                                         ▼
                     │                                   Evaluate IL
                     │                                         │
                     │                                         ▼
                     └─────────────────────────────────> RL Post-Training
                                                               │
                                                               ▼
                                                         Evaluate RL
```

---

## Gripper Control Through the Pipeline

This is a critical detail that is not obvious from the code structure: **teleop uses
binary gripper signals, but the training data contains individual finger joints.**

### The Conversion Chain

```
Teleop input        WBC controller        HDF5 recording      LeRobot/GR00T
(binary 0/1)  ──>  (7-joint presets) ──>  (43D joints)   ──>  (28D joints)
                         │                                         │
                    get_hand_joint_pos()                    select 14 arm +
                    in g1_wbc_upperbody_                    14 hand joints
                    controller.py:223-238
```

### What Actually Happens

All three teleop input methods (Meta Quest controllers, AVP hand tracking, keyboard)
output a **binary gripper value** — either 0.0 (open) or 1.0 (closed). No intermediate
values, no per-finger control.

The **WBC (Whole Body Controller)** then converts this binary signal into **7 fixed
joint presets per hand** via `get_hand_joint_pos()`:

```
g1_wbc_upperbody_controller.py:223-238

Open  (hand_state == 0):  all 7 joints = 0.0 rad (flat hand)
Close (hand_state == 1):  thumb_1 = +0.7,  thumb_2 = +0.7
                          index_0 = -0.6,  index_1 = -1.2
                          middle_0 = -0.6, middle_1 = -1.2
```

The **recorded `processed_actions` (43D)** contain these joint presets — not the binary
signal. So the HDF5 files have continuous joint values, but they only ever appear in
exactly two configurations.

### Implications for IL and RL

| Stage | Gripper Representation | Consequence |
|-------|----------------------|-------------|
| **Teleop** | Binary (0 or 1) | Human can only open/close the whole hand |
| **HDF5 recording** | 7 joint presets per hand (from WBC) | Data looks continuous but has only 2 configs |
| **LeRobot / GR00T IL** | 14 continuous joint values (28D total) | IL learns to reproduce the 2 presets |
| **RL (direct joint control)** | 14 independent joints (43D action) | Policy can output **any** finger values |

**Key insight for RL:** The RL environment uses `JointPositionActionCfg` with direct
control over all 43 DOF. The policy is not constrained to the two WBC presets. With
appropriate reward shaping, RL can learn finger behaviors (partial grasps, sequential
finger closure, repositioning) that never appeared in the IL demonstrations.

**Source files:**
- WBC gripper conversion: `third_party/IsaacLab-Arena/.../g1_wbc_upperbody_controller.py:223-238`
- RL 43D action space: `scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py:162-172`
- Dex3 observations: `scripts/simulation/tasks/assemble_trocar/mdp/observations.py:137-167`

---

## Phase 1: Define Your Task

You need to create a new task package under `scripts/simulation/tasks/`. Here is the
anatomy of the assemble_trocar task, which you should use as a template.

### 1.1 Create the Task Package

```
scripts/simulation/tasks/your_task/
├── __init__.py
├── your_task_env_cfg.py
├── your_task_teleop_env_cfg.py
├── config/
│   ├── __init__.py
│   ├── robot_config.py
│   └── camera_config.py
└── mdp/
    ├── __init__.py
    ├── observations.py
    ├── rewards.py
    ├── terminations.py
    └── events.py
```

### 1.2 Register Gym Environments (`__init__.py`)

You need three gym IDs: training, evaluation, and teleoperation.

```python
import gymnasium as gym
from . import your_task_env_cfg

# Training environment (random resets)
gym.register(
    id="Isaac-YourTask-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": your_task_env_cfg.YourTaskEnvCfg},
    disable_env_checker=True,
)

# Evaluation environment (deterministic resets for reproducible metrics)
gym.register(
    id="Isaac-YourTask-G129-Dex3-Joint-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": your_task_env_cfg.YourTaskEvalEnvCfg},
    disable_env_checker=True,
)

# Teleop environment (VR control, extended episode, fixed legs)
gym.register(
    id="Isaac-YourTask-G129-Dex3-Teleop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": your_task_teleop_env_cfg.YourTaskTeleopEnvCfg},
    disable_env_checker=True,
)
```

### 1.3 Define the Environment Config (`your_task_env_cfg.py`)

This is the core file. It defines your scene and all MDP components via the
`ManagerBasedRLEnvCfg` configclass hierarchy.

**Key components** (see `g1_assemble_trocar_env_cfg.py` for the full reference):

#### Scene Configuration

```python
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg

@configclass
class YourTaskSceneCfg(InteractiveSceneCfg):
    # Robot: G1 29-DOF + Dex3 hands
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(x, y, z),
        init_rot=(w, x, y, z),
    )
    # Your objects (tools, bins, etc.)
    tool_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/tool_1",
        spawn=UsdFileCfg(usd_path="/path/to/tool.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z)),
    )
    bin: RigidObjectCfg = RigidObjectCfg(...)
    # Cameras: front + two wrist cameras
    front_camera: CameraCfg = CameraPresets.g1_front_camera()
    left_wrist_camera: CameraCfg = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera: CameraCfg = CameraPresets.right_dex3_wrist_camera()
```

The existing robot and camera presets in `config/robot_config.py` and `config/camera_config.py`
are reusable. You mainly need to define your **objects** and **scene layout**.

#### Actions — Two Modes

The task has **two action modes** that share the same scene but differ in how the robot
is controlled. Understanding this split is essential.

**RL mode** (`g1_assemble_trocar_env_cfg.py`): Direct joint position control over all
43 DOF. The policy outputs raw joint angle targets — no inverse kinematics, no WBC.

```python
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,  # all 43 joints explicitly listed
        scale=1.0,
        preserve_order=True,
    )
```

**Teleop mode** (`g1_assemble_trocar_teleop_env_cfg.py`): WBC+PINK inverse kinematics
that accepts a 23D high-level command and outputs 43D joint targets internally.

```
23D teleop input:
  [gripper_L(1), gripper_R(1),                    # binary open/close
   left_wrist_xyz(3), left_wrist_quat(4),          # task-space wrist pose
   right_wrist_xyz(3), right_wrist_quat(4),        # task-space wrist pose
   nav_x(1), nav_y(1), nav_yaw(1),                 # navigation (zeroed)
   base_height(1),                                  # hip height (0.75m fixed)
   torso_roll(1), torso_pitch(1), torso_yaw(1)]     # torso orientation (zeroed)
```

The two components that process this:
- **PINK** (Pinocchio Inverse Kinematics) — solves wrist poses → 14 arm joint angles.
  Only handles the arms; does not touch fingers.
- **WBC** (Whole Body Controller) — converts binary gripper → 7 finger joint presets
  (see [Gripper Control Through the Pipeline](#gripper-control-through-the-pipeline)),
  handles lower body balance, and orchestrates everything into a unified 43D joint command.

The teleop variant also **fixes the lower body** so only the arms move — implemented via
`G1AssembleTrocarFixedLegsWBCPinkAction` which zeros out navigation, height, and torso
commands before passing to the WBC.

#### Observations

Two observation groups — **policy** (state) and **camera_images** (visual):

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=get_robot_body_joint_states)  # 87-D
        robot_dex3_joint_state = ObsTerm(func=get_robot_dex3_joint_states)  # 14-D

    @configclass
    class CameraImagesCfg(ObsGroup):
        front_camera = ObsTerm(func=mdp.image, params={"sensor_cfg": ...})
        left_wrist_camera = ObsTerm(func=mdp.image, params={"sensor_cfg": ...})
        right_wrist_camera = ObsTerm(func=mdp.image, params={"sensor_cfg": ...})

    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()
```

The observation functions in `mdp/observations.py` use **cached tensor buffers** for
performance. They extract specific joint indices from the full robot state and reorder
them into the canonical body (29 joints) and dex3 (14 joints) ordering.

#### Rewards — The Stage-Based Pattern

This is the most important design pattern to understand. The assemble trocar task uses
**multi-stage sparse rewards**:

```
Stage 0 ──(condition met)──> Stage 1 ──> Stage 2 ──> ... ──> Stage N (success)
```

Each stage transition gives a reward of 1.0 (one-time, sparse). Stages only advance
forward, never backward. The task maintains a `_task_stage` tensor `(num_envs,)` that
tracks each environment's current stage.

**For your pick-and-place task, you might define stages like:**

| Stage | Condition | Description |
|-------|-----------|-------------|
| 0 -> 1 | Tool lifted above table by 0.1m | Successful grasp |
| 1 -> 2 | Tool moved over bin (x,y within bounds) | Transport |
| 2 -> 3 | Tool placed in bin (z below threshold, in bounds) | Placement |

Each reward function:
1. Calls `update_task_stage()` to check all transitions
2. Returns 1.0 on the specific transition it tracks, 0.0 otherwise
3. Can optionally provide **dense** rewards (continuous shaping) in addition to sparse

**Key pattern from `mdp/rewards.py`:**

```python
def grasp_tool_reward(env, ...):
    """Reward for stage 0 -> 1 transition (grasping)."""
    update_task_stage(env, ...)  # Check all stage transitions

    # Sparse: 1.0 on transition, 0.0 otherwise
    current = env._task_stage
    prev = env._prev_stage_grasp
    reward = ((current >= 1) & (prev < 1)).float()
    env._prev_stage_grasp = current.clone()
    return reward
```

The `_task_stage` and all per-stage caches are stored as attributes on the `env` object
and reset in `mdp/events.py:reset_task_stage()`.

#### Terminations

Two standard termination conditions:

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_success = DoneTerm(
        func=task_success_termination,
        params={"success_stage": 3},  # Your final stage number
        time_out=False,  # time_out=False means "this is success, not failure"
    )
    object_drop = DoneTerm(
        func=object_drop_termination,
        params={"min_height": 0.5},
        time_out=True,  # Treated as failure
    )
```

#### Events (Reset Randomization)

```python
@configclass
class EventCfg:
    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_task_stage = EventTerm(func=reset_task_stage, mode="reset")
    # Add your randomization (e.g., random object placement):
    reset_objects = EventTerm(
        func=reset_objects_random_position,
        mode="reset",
        params={"position_range": ...},
    )
```

#### Simulation Settings

```python
@configclass
class YourTaskEnvCfg(ManagerBasedRLEnvCfg):
    scene = YourTaskSceneCfg(num_envs=1, env_spacing=5.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    sim = SimulationCfg(
        dt=1 / 200,          # Physics at 200 Hz
        render_interval=4,    # Control at 50 Hz (200/4)
    )
    episode_length_s = 20.0   # 20 second episodes
```

### 1.4 Teleop Variant (`your_task_teleop_env_cfg.py`)

The teleop config inherits from the training config and overrides:

1. **Actions**: Replace joint position control with WBC+PINK (inverse kinematics) —
   this lets VR controllers drive end-effectors while the WBC handles joint-level control.
   The key trick is **fixing the lower body** so only the arms move:

   ```python
   class FixedLegsWBCPinkAction(G1DecoupledWBCPinkAction):
       def process_actions(self, actions):
           # Zero out navigation, base height, torso commands
           # Only pass through arm + hand commands
           ...
   ```

2. **Episode length**: Extended to 300s (vs 20s for training) — humans need time.
3. **Render interval**: 2 (higher visual quality for the VR headset).
4. **XR config**: Meta Quest anchor attached to robot pelvis.

### 1.5 Test Your Environment

```bash
# Quick sanity check — does the env create and step?
./docker/run_docker.sh -g1.5 python -c "
import gymnasium as gym
import simulation.tasks.your_task  # triggers gym.register()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
from isaaclab_tasks.utils import load_cfg_from_registry
cfg = load_cfg_from_registry('Isaac-YourTask-G129-Dex3-Joint', 'env_cfg_entry_point')
cfg.scene.num_envs = 1
env = gym.make('Isaac-YourTask-G129-Dex3-Joint', cfg=cfg).unwrapped
obs, _ = env.reset()
print('Obs keys:', {k: {kk: vv.shape for kk, vv in v.items()} for k, v in obs.items()})
for i in range(10):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    print(f'Step {i}: reward={rew.item():.2f}, term={term.item()}, trunc={trunc.item()}')
env.close()
"
```

---

## Phase 2: Collect Demonstrations

### 2.1 XR Teleoperation (Recommended for Manipulation)

Two teleop devices are supported. Both produce the same 20D output format (binary
gripper + wrist poses + lower body), so the downstream pipeline is identical regardless
of which device you use.

| Device | Flag | Input Method | Gripper Control |
|--------|------|-------------|-----------------|
| Meta Quest controllers | `--teleop_device motion_controllers` | Controller pose + trigger | Trigger analog > threshold |
| Apple Vision Pro hand tracking | `--teleop_device handtracking` | Hand joint poses | Thumb-index pinch distance |

See `docs/avp_teleoperation.md` for AVP setup details (CloudXR streaming, wrist
rotation transforms, pinch thresholds).

```bash
# Meta Quest controllers
./docker/run_docker.sh -g1.5 \
  python scripts/simulation/record_demos_assemble_trocar.py \
  --task Isaac-YourTask-G129-Dex3-Teleop \
  --teleop_device motion_controllers \
  --enable_pinocchio \
  --enable_cameras \
  --num_demos 50 \
  --xr

# Apple Vision Pro hand tracking
./docker/run_docker.sh -g1.5 \
  python scripts/simulation/record_demos_assemble_trocar.py \
  --task Isaac-YourTask-G129-Dex3-Teleop \
  --teleop_device handtracking \
  --enable_pinocchio \
  --enable_cameras \
  --num_demos 50 \
  --xr
```

**What gets recorded** (HDF5 format):
- `obs/robot_joint_state` — 87-D body joint state (pos + vel + torque)
- `obs/robot_dex3_joint_state` — 14-D hand joint positions
- `obs/front_camera`, `obs/left_wrist_camera`, `obs/right_wrist_camera` — RGB images
- `processed_actions` — 43-D joint position targets after WBC+PINK processing
- Episode metadata: success flag, timestamps, initial state

**Important:** The recorded `processed_actions` are the **WBC+PINK output** (43D joint
targets), not the raw teleop input (23D). This means:
- The binary gripper signal has already been converted to 7 finger joint presets
  (see [Gripper Control Through the Pipeline](#gripper-control-through-the-pipeline))
- Wrist poses have been solved into arm joint angles by PINK
- The recording captures the full kinematics solution, not the human's commands
- This is why IL trains on individual finger joints even though teleop is binary

**Controls during collection:**
- XR controllers drive the arms/hands via WBC+PINK inverse kinematics
- 'S' key marks the episode as successful
- 'R' key resets the environment
- Only successful episodes are exported (`EXPORT_SUCCEEDED_ONLY`)

You will likely want to **modify `record_demos_assemble_trocar.py`** (or write your own
variant) to reference your task's gym ID and adjust the data collection settings.

### 2.2 How Many Demos?

The assemble trocar pipeline uses ~50-100 human demos, then amplifies with synthetic
generation. For a simpler pick-and-place task, ~30-50 good demos may be enough for an
initial IL model, which RL can then refine.

### 2.3 Optional: Annotate and Generate Synthetic Data

If you want to amplify your dataset with Mimic Gen:

```bash
# 1. Annotate demos with subtask completion signals
./docker/run_docker.sh -g1.5 \
  python scripts/simulation/annotate_demos.py \
  --input_file /datasets/your_demos.hdf5 \
  --output_file /datasets/your_demos_annotated.hdf5 \
  --enable_cameras --mimic \
  <your_task_arena_env_args>

# 2. Generate synthetic variations
./docker/run_docker.sh -g1.5 \
  python scripts/simulation/generate_dataset.py \
  --input_file /datasets/your_demos_annotated.hdf5 \
  --output_file /datasets/your_demos_generated.hdf5 \
  --generation_num_trials 100 \
  --enable_cameras --mimic --headless \
  <your_task_arena_env_args>

# 3. Merge all datasets
./docker/run_docker.sh -g1.5 \
  python scripts/simulation/merge_demos.py \
  --input /datasets/your_demos*.hdf5 \
  --output /datasets/your_merged.hdf5
```

**Note:** The annotation and Mimic Gen pipeline currently has stronger support for
Arena-track (locomanip) environments. For IsaacLab-track tasks, you may need to
adapt the annotation scripts or skip straight to converting your human demos to LeRobot
format and rely on RL to handle the distribution gap.

---

## Phase 3: Process Data for GR00T Fine-Tuning

### 3.1 Define Your Field Mappings

Create a file like `scripts/utils/your_task_lerobot_fields.py` modeled on
`assemble_trocar_lerobot_fields.py`. This defines how the 43-D HDF5 joint data
maps to the canonical state/action dimensions GR00T expects.

For a manipulation task, you likely want the same **28-D canonical space** (no legs):

| Group | DOF | Joints |
|-------|-----|--------|
| Left Arm | 7 | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| Right Arm | 7 | (same) |
| Left Hand | 7 | thumb (3), middle (2), index (2) |
| Right Hand | 7 | (same) |
| **Total** | **28** | |

The existing `assemble_trocar_lerobot_fields.py` already defines this 28-D mapping
with all the index permutations (`STATE_28_BODY_COL_LEFT_ARM`, `ACTION_HDF5_TO_ENV_28`,
etc.). You can likely **reuse it directly** since you're using the same robot.

**Important detail:** The action offset. The trocar env applies a -0.3 rad offset to
elbows in `JointPositionActionCfg`. When converting recorded `processed_actions` back
to LeRobot format, this offset must be **added back** (+0.3 to indices 3 and 10).
Check whether your env uses the same offset or a different one.

### 3.2 Create a Dataset Config YAML

Create `scripts/config/your_task_dataset.yaml` based on `g1_assemble_trocar_dataset.yaml`:

```yaml
hdf5_name: "your_demos.hdf5"
data_root: "/datasets"
output_root: "/datasets/your_task_lerobot"

use_rheo_converter: true
rheo_28d_state_action: true
rheo_action_key: "processed_actions"

rheo_camera_mappings_obs:
  front_camera: "observation.images.cam_room"
  left_wrist_camera: "observation.images.cam_left_wrist"
  right_wrist_camera: "observation.images.cam_right_wrist"

state_name_lerobot: "observation.state"
action_name_lerobot: "action"
task_description_lerobot: "annotation.human.task_description"
language_instruction: "pick up the tool and place it in the bin"

chunks_size: 1000
fps: 30
robot_type: "unitree_g1"
```

The `language_instruction` is critical — it becomes the language conditioning for GR00T
and must match what you use during inference.

### 3.3 Convert HDF5 to LeRobot

```bash
./docker/run_docker.sh -g1.5 \
  python scripts/utils/convert_hdf5_to_lerobot.py \
  --config scripts/config/your_task_dataset.yaml
```

This produces a LeRobot dataset:
```
your_task_lerobot/
├── data/chunk-000/episode_000000.parquet  # State (28-D), action (28-D), metadata
├── videos/chunk-000/
│   ├── observation.images.cam_room/episode_000000.mp4
│   ├── observation.images.cam_left_wrist/episode_000000.mp4
│   └── observation.images.cam_right_wrist/episode_000000.mp4
└── meta/
    ├── info.json          # Feature shapes, joint names
    ├── tasks.jsonl        # Task index -> language instruction
    ├── episodes.jsonl     # Episode metadata
    └── modality.json      # Multimodal structure
```

---

## Phase 4: Imitation Learning (GR00T SFT)

### 4.1 Understand the Data Config

GR00T needs a **data config** that tells it how to interpret the LeRobot dataset.
For manipulation tasks, this is `UnitreeG1SimDataConfig` in `scripts/policy/gr00t_config.py`.

Key settings:
- **Video keys**: `left_wrist_view`, `right_wrist_view`, `room_view` (3 cameras)
- **State keys**: `left_arm(7)`, `right_arm(7)`, `left_hand(7)`, `right_hand(7)` = 28-D
- **Action keys**: same 4 groups = 28-D
- **Action horizon**: 16 steps (model predicts 16 future actions per observation)
- **Transforms**: video crop+resize to 224x224, color jitter, state sin/cos encoding, action min-max normalization

If your task uses the same 28-D state/action space and 3 cameras, you can **reuse this
config as-is**. If you have a different camera setup or DOF, you'll need to create your own
data config class.

### 4.2 Run Fine-Tuning

Fine-tuning runs inside the Docker container, using the GR00T repo at
`/workspaces/third_party/Isaac-GR00T`.

```bash
./docker/run_docker.sh -g1.5

# Inside container:
cd /workspaces/third_party/Isaac-GR00T
python scripts/gr00t_finetune.py \
  --dataset-path /datasets/your_task_lerobot \
  --num-gpus 1 \
  --batch-size 32 \
  --output-dir /models/GR00T-N1.5-YourTask \
  --data-config policy.gr00t_config:UnitreeG1SimDataConfig \
  --video_backend decord \
  --report_to tensorboard \
  --max_steps 30000 \
  --save-steps 5000 \
  --tune_visual
```

**Key parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--data-config` | `policy.gr00t_config:UnitreeG1SimDataConfig` | Module:Class syntax. Uses your PYTHONPATH. |
| `--max_steps` | 30000 | Trocar recipe uses 30K. Simpler tasks may converge faster. |
| `--tune_visual` | (flag) | Fine-tunes the visual encoder. Usually needed for sim domains. |
| `--batch-size` | 32 | Requires ~80GB GPU memory (H100). Reduce for smaller GPUs. |

**GPU requirement:** Single H100 (80GB HBM) or equivalent. For your RTX 5090 (32GB), you
will likely need to reduce `--batch-size` (try 8-16) and possibly enable gradient
checkpointing or mixed precision.

### 4.3 What Happens During Fine-Tuning

GR00T is a Vision-Language-Action (VLA) model. During SFT:

1. **Visual encoder** processes 3 camera views (224x224 each) into visual tokens
2. **Language encoder** processes your task description into language tokens
3. **Diffusion action head** predicts 16-step action chunks conditioned on visual + language tokens
4. Training minimizes the denoising loss on the action predictions

The `--tune_visual` flag unfreezes the visual encoder (important for sim-to-sim transfer
since the pretrained encoder was trained on real-world data).

**Note on hand joints in IL data:** GR00T trains on 28D actions that include 14
individual finger joint values (7 per hand). However, because teleop uses binary
gripper control and the WBC maps this to fixed joint presets, the training data only
contains **two distinct hand configurations** — fully open and fully closed. The IL
model learns to predict continuous joint values, but in practice it reproduces the
same two presets it was trained on. Individual finger dexterity is not learned at the
IL stage — that is a capability RL can unlock (see Phase 6).

---

## Phase 5: Evaluate the IL Policy

### 5.1 Create a Policy Config YAML

Create `scripts/config/your_task_closedloop_config.yaml`:

```yaml
model_path: /models/GR00T-N1.5-YourTask
language_instruction: "pick up the tool and place it in the bin"
action_horizon: 16
embodiment_tag: new_embodiment
video_backend: decord
data_config: unitree_g1_sim
policy_joints_config_path: /workspaces/third_party/IsaacLab-Arena/isaaclab_arena_gr00t/config/g1/gr00t_43dof_joint_space.yaml
action_joints_config_path: /workspaces/third_party/IsaacLab-Arena/isaaclab_arena_gr00t/config/g1/43dof_joint_space.yaml
state_joints_config_path: /workspaces/third_party/IsaacLab-Arena/isaaclab_arena_gr00t/config/g1/43dof_joint_space.yaml
action_chunk_length: 16
pov_cam_name_sim: front_camera
task_mode_name: dexterous
```

The `task_mode_name: dexterous` is key — this tells `CustomGr00tClosedloopPolicy` to use
the tabletop manipulation code path (28-D actions, no locomotion commands).

### 5.2 Run Evaluation

You can adapt `eval_assemble_trocar.py` for your task:

```bash
./docker/run_docker.sh -g1.5 \
  python -u scripts/simulation/examples/eval_assemble_trocar.py \
    --enable_cameras \
    --task Isaac-YourTask-G129-Dex3-Joint-Eval \
    --model_path /models/GR00T-N1.5-YourTask \
    --num_episodes 50 \
    --max_steps 500
```

**Note:** You'll likely need to modify `eval_assemble_trocar.py` to import your task
module instead of `simulation.tasks.assemble_trocar`. The core evaluation loop
(load policy -> step env -> collect metrics) is the same.

### 5.3 Understanding the Policy Wrapper

`CustomGr00tClosedloopPolicy` handles:

1. **Action chunking**: GR00T predicts 16 actions per observation. The wrapper maintains
   a `current_action_chunk` buffer and `current_action_index` per environment, only
   requesting a new forward pass when the chunk is exhausted.

2. **Joint remapping**: GR00T outputs actions in its own joint ordering (grouped by
   body part). The wrapper remaps to the simulator's joint ordering using YAML configs.

3. **Task mode routing**: For `dexterous` mode, only arm+hand actions are extracted.
   For `g1_locomanipulation`, navigation commands are appended.

---

## Phase 6: RL Post-Training (RLinf PPO)

RL post-training takes your IL-trained checkpoint and refines it with online PPO in
simulation. This is especially valuable for multi-stage tasks where IL alone gets the
general behavior right but fails on precision transitions.

**Key difference from IL:** The RL environment uses `JointPositionActionCfg` with
**direct control over all 43 DOF** — there is no WBC, no PINK, no binary gripper
presets. The policy outputs raw joint angle targets for every joint including
individual finger joints. This means:

- The RL policy can output **arbitrary finger joint values**, not just the two
  preset configurations (open/closed) that appeared in IL training data
- With appropriate reward shaping (e.g., rewards for partial grasps, finger
  repositioning, sequential finger closure), RL can learn dexterous finger
  behaviors that were impossible to demonstrate via binary teleop
- This is a significant capability upgrade: IL establishes the general manipulation
  strategy, RL refines it with finer motor control

### 6.1 Register Your Task in RLinf

Edit `scripts/simulation/rl/rlinf_ext/__init__.py`:

```python
def _register_isaaclab_envs() -> None:
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
    IsaaclabG129Dx3Env = _get_g129_dex3_env_class()

    # Existing trocar registrations...
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Assemble-Trocar-G129-Dex3-Joint", IsaaclabG129Dx3Env)
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Assemble-Trocar-G129-Dex3-Joint-Eval", IsaaclabG129Dx3Env)

    # ADD YOUR TASK:
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-YourTask-G129-Dex3-Joint", IsaaclabG129Dx3Env)
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-YourTask-G129-Dex3-Joint-Eval", IsaaclabG129Dx3Env)
```

Also update `_make_env_function()` in the `IsaaclabG129Dx3Env` class to import your task:

```python
def _make_env_function(self):
    def make_env_isaaclab():
        ...
        import simulation.tasks.assemble_trocar  # noqa: F401
        import simulation.tasks.your_task        # noqa: F401  <-- ADD THIS
        ...
    return make_env_isaaclab
```

**If your task has the same observation/action structure as assemble_trocar** (28-D state,
3 cameras, same joint groups), the existing `_wrap_obs()` and `_convert_dex3_obs_to_gr00t_format`
will work as-is. If not, you need custom converters.

### 6.2 Create Hydra Configs

**Environment config** (`rlinf_ext/config/env/your_task.yaml`):

```yaml
env_type: isaaclab
total_num_envs: null
auto_reset: False
use_rel_reward: True
max_episode_steps: 256
max_steps_per_rollout_epoch: 10

init_params:
  id: "Isaac-YourTask-G129-Dex3-Joint"
  task_description: "pick up the tool and place it in the bin"
```

**Main PPO config** (`rlinf_ext/config/isaaclab_ppo_gr00t_your_task.yaml`):

Copy `isaaclab_ppo_gr00t_assemble_trocar.yaml` and change:
- `defaults` section: point to your env config
- `env.eval.init_params.id`: point to your eval gym ID
- Optionally adjust `total_num_envs`, `max_episode_steps`, learning rates

Key PPO parameters (from the assemble trocar benchmark):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `env.train.total_num_envs` | 64 (start here, benchmark used 512) | More = faster but needs more VRAM |
| `algorithm.gamma` | 0.99 | Discount factor |
| `algorithm.gae_lambda` | 0.95 | GAE lambda |
| `algorithm.clip_ratio_high/low` | 0.2 | PPO clip range |
| `actor.optim.lr` | 5e-6 | Policy LR (very conservative for fine-tuning) |
| `actor.optim.value_lr` | 1e-4 | Value head LR |
| `algorithm.sampling_params.temperature_train` | 1.0 | Exploration |
| `algorithm.sampling_params.temperature_eval` | 0.6 | Exploitation |

### 6.3 Create a Launch Script

Copy `train_gr00t_assemble_trocar.sh` to `train_gr00t_your_task.sh` and change
`CONFIG_NAME` to your new config.

### 6.4 Run RL Training

```bash
./docker/run_docker.sh -g1.5

# Inside container:
bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_your_task.sh train \
    --model_path /models/GR00T-N1.5-YourTask

# Low memory? Reduce envs and batch size:
bash ... train --model_path /models/GR00T-N1.5-YourTask \
    env.train.total_num_envs=8 actor.micro_batch_size=2
```

### 6.5 What Happens During RL Training

The RLinf framework orchestrates three types of workers via Ray:

1. **Environment workers** run parallel IsaacLab simulations. Each worker creates
   `num_envs` environments that step in lockstep.

2. **Rollout workers** run GR00T inference (forward pass only, no gradients). They
   receive observations from env workers, predict action chunks, and return them.

3. **Actor workers** compute PPO updates. They receive rollout trajectories, compute
   advantages via GAE, and update the policy + value head with FSDP.

The training loop:
```
For each epoch:
  1. Collect 8 rollout epochs of experience (env step + inference)
  2. Compute GAE advantages (gamma=0.99, lambda=0.95)
  3. Run 4 PPO update epochs on the collected data
  4. Checkpoint every 2 epochs
```

The **GR00T RL patch** is applied automatically during model loading:
- Eagle input IDs padded to 850 tokens (fixed sequence length)
- All dropout layers replaced with Identity (deterministic)
- Tensors cast to bfloat16

### 6.6 Curriculum Training (Optional but Powerful)

The assemble trocar benchmarks show that **training each stage independently** yields the
best results (+53% on hard stages). To do this:

1. Set your reward config to only give reward for the target stage
2. Set termination to trigger when that stage is reached
3. Train until convergence, then use that checkpoint for the next stage

This is not built into the configs as a first-class feature — you achieve it by modifying
the reward/termination configs in your env cfg between training runs.

---

## Phase 7: Evaluate the RL Policy

### 7.1 Evaluate via RLinf

```bash
bash /workspaces/workflows/rheo/scripts/simulation/rl/train_gr00t_your_task.sh eval \
    --model_path /path/to/rl_checkpoint
```

### 7.2 Evaluate via eval script (--rl_ckpt flag)

```bash
./docker/run_docker.sh -g1.5 \
  python -u scripts/simulation/examples/eval_assemble_trocar.py \
    --task Isaac-YourTask-G129-Dex3-Joint-Eval \
    --model_path /path/to/rl_checkpoint \
    --rl_ckpt \
    --num_episodes 100 \
    --max_steps 500
```

**Critical: the `--rl_ckpt` flag.** When evaluating an RL-trained checkpoint, you MUST
pass this flag. It applies the same GR00T patch (eagle padding + dropout removal) that
was active during training. Without it, inference silently produces wrong results because
the model expects padded inputs and deterministic forward passes.

The patch is applied via a context manager in `apply_gr00t_rl_patch.py` that runs
`git apply` on the GR00T source code and auto-reverts when done.

---

## Quick Reference: Key Files

### Task Definition
| File | Purpose |
|------|---------|
| `scripts/simulation/tasks/assemble_trocar/__init__.py` | Gym registration (3 gym IDs) |
| `scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py` | Scene, actions, obs, rewards, terminations, events |
| `scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_teleop_env_cfg.py` | VR teleoperation variant (fixed legs, extended episode) |
| `scripts/simulation/tasks/assemble_trocar/config/robot_config.py` | G1+Dex3 articulation preset (reusable) |
| `scripts/simulation/tasks/assemble_trocar/config/camera_config.py` | Front + wrist camera presets (reusable) |
| `scripts/simulation/tasks/assemble_trocar/mdp/rewards.py` | Stage-based sparse reward pattern |

### Data Pipeline
| File | Purpose |
|------|---------|
| `scripts/simulation/record_demos_assemble_trocar.py` | XR data collection |
| `scripts/utils/convert_hdf5_to_lerobot.py` | HDF5 -> LeRobot conversion |
| `scripts/utils/assemble_trocar_lerobot_fields.py` | 28-D joint mapping (reusable for same robot) |
| `scripts/config/g1_assemble_trocar_dataset.yaml` | Dataset conversion config |

### Training (IL)
| File | Purpose |
|------|---------|
| `scripts/policy/gr00t_config.py` | GR00T data config: modalities, transforms, dimensions |
| `docs/assemble_trocar_finetuning.md` | Fine-tuning recipe and commands |

### Training (RL)
| File | Purpose |
|------|---------|
| `scripts/simulation/rl/rlinf_ext/__init__.py` | RLinf extension: env registration, converters, model patching |
| `scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_gr00t_assemble_trocar.yaml` | Full PPO config |
| `scripts/simulation/rl/rlinf_ext/config/env/isaaclab_assemble_trocar.yaml` | Env config for RLinf |
| `scripts/simulation/rl/rlinf_ext/config/model/gr00t_dex3.yaml` | Model config (value head, patch settings) |
| `scripts/simulation/rl/train_gr00t_assemble_trocar.sh` | Training launcher |
| `scripts/policy/apply_gr00t_rl_patch.py` | RL patch context manager |
| `docs/assemble_trocar_rl_guide.md` | Full RL guide with benchmarks |

### Inference & Evaluation
| File | Purpose |
|------|---------|
| `scripts/simulation/gr00t_closedloop_policy.py` | Policy wrapper: action chunking, joint remap, task mode |
| `scripts/simulation/examples/eval_assemble_trocar.py` | Evaluation script |
| `scripts/utils/policy_tasks.py` | TensorRT wrapper, success-hold logic |
| `scripts/utils/joint_conversion.py` | Policy-to-sim joint reordering |

---

## Quick Reference: Dimensions and Conventions

### State/Action Dimensions

| Space | Dim | Contents |
|-------|-----|----------|
| Teleop input | 23 | gripper(2) + wrist_poses(14) + nav(3) + height(1) + torso(3) |
| WBC+PINK output / HDF5 recording | 43 | Full joint targets (after IK + gripper preset conversion) |
| Full robot joints (RL action) | 43 | 29 body (legs + waist + arms) + 14 Dex3 hand |
| Body observation | 87 | 29 pos + 29 vel + 29 torque |
| Hand observation | 14 | 14 Dex3 joint positions |
| **Canonical manipulation (LeRobot/GR00T)** | **28** | **7 left arm + 7 right arm + 7 left hand + 7 right hand** |
| RLinf action (GR00T → sim) | 43 | 15 zeros (legs/waist padding) + 28 manipulation joints |
| GR00T action horizon | 16 | 16 future action steps per forward pass |

### Joint Group Ordering (28-D canonical)

```
[0:7]   Left arm:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                    wrist_roll, wrist_pitch, wrist_yaw
[7:14]  Right arm: (same)
[14:21] Left hand: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
[21:28] Right hand: (same)
```

### RLinf Action Conversion

GR00T outputs 28-D actions. RLinf pads **15 zeros at the front** to create a 43-D
action for the full robot (the zeros correspond to legs + waist which are held fixed).

### Important Thresholds (from assemble trocar)

| Parameter | Value |
|-----------|-------|
| Physics DT | 1/200s (5ms) |
| Control DT | 4x physics = 20ms (50 Hz) |
| Training episode | 20s (1000 physics steps, 250 control steps) |
| Teleop episode | 300s |
| Camera resolution | 640x480 (resized to 224x224 for GR00T) |
| Camera update rate | 50 Hz |

### Environment Variables (Inside Container)

| Variable | Value | Purpose |
|----------|-------|---------|
| `DATASET_DIR` | `/datasets` | Dataset root |
| `MODELS_DIR` | `/models` | Model checkpoint root |
| `ISAACLAB_PATH` | `/workspaces/third_party/IsaacLab` | IsaacLab installation |
| `RLINF_EXT_MODULE` | `rlinf_ext` | Set by RL training script only |
| `OMNI_PASS` | (your NGC token) | Must be set on host before Docker run |
