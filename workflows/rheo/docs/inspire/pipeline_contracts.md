# Inspire FTP Grasp Pipeline — Contract Reference

> Last validated against `i4h-workflows` commit `991e4e4`. Re-validate when
> any cited file changes.

This document is a **contract reference**, not a tutorial. It pins down, per
pipeline stage, exactly what the data looks like, what the joint/action
conventions are, where offsets are applied and cancelled, and how to
sanity-check each stage in isolation. If you are learning the pipeline for the
first time, read [`grasp_policy_guide.md`](grasp_policy_guide.md) first, then come back
here.

The mental model is **three layers** — Embodiment, Data, Policy. Most bugs in
this pipeline live at the *seams* between layers, not inside any single stage.
Treat every contract below as something that must hold for rollouts to work at
all.

---

## Pipeline at a glance

```
AVP hand skeleton (52 bone poses)  →  UnitreeG1RetargeterCfg (DexPilot, 38D)
      →  env.step (Teleop variant, PinkIK solves wrist IK + hand passthrough)
      →  record_demos.py HDF5        [--arm left|right masks non-controlled arm]
          obs:     robot_joint_state  (T, 87)   body pos/vel/torque  (always full)
                   robot_inspire_joint_state  (T, 12)   hand pos     (always full)
                   front_camera images
          action:  processed_actions  (T, 38)   PinkIK wrist+hand command
      →  convert_hdf5_to_lerobot.py  →  LeRobot parquet
          26D path (--config g1_grasp_policy_inspire_dataset.yaml):
            observation.state  (T, 26)   14 arm + 12 hand positions
            action             (T, 26)   next-step observed positions + (+0.3 elbow)
          13D path (--config g1_grasp_policy_inspire_dataset_{left,right}_arm.yaml):
            observation.state  (T, 13)   7 arm + 6 hand positions (single arm)
            action             (T, 13)   next-step observed positions + (+0.3 elbow)
          observation.images.cam_room  (mp4, 50 fps, 480×640)
          meta/episodes_stats.jsonl    mean/std normalization stats
      →  train_act_grasp_policy_inspire.sh  →  ACT checkpoint
          input:   observation.state (26 or 13) + cam_room image
          output:  50-step action chunks, each 26D or 13D
      →  ACTClosedloopPolicy  →  scatter 26D → 41D sim action
      →  env.step (Joint variant, 41D joint positions with -0.3 elbow offset)
```

---

## Pipeline invariants

Short list of facts that must never drift. If any of these turns out to be
violated in practice, something is broken.

- Frame rate is **50 Hz** everywhere (sim, recording, video, training).
- Canonical policy dim is **26** for dual-arm or **13** for single-arm
  (never 28 — that's the Dex3 path).
- Sim action dim is **41** for Inspire FTP (never 43 — that's Dex3).
- Elbow offset is **+0.3 in parquet**, **−0.3 in env `offset_dict`**. Exactly
  one compensation on each side of the chain — never both, never neither.
  For 13D, the elbow is at index **3** (same relative position within the arm).
- Parquet `state[t]` is the joint position at frame t. Parquet `action[t]` is
  the **next-step** joint position (`state[t+1]`) plus the elbow offset.
- Cameras: **3 cameras** (`front_camera` + `left_wrist_camera` + `right_wrist_camera`), matching Dex3. Wrist mount links live in the `g1-29dof-inspire-ftp-usd-wrist_cam/` USD variant.
- Normalization stats are **frozen at conversion time** in
  `meta/episodes_stats.jsonl`. Regenerating the parquet regenerates the
  stats; retraining uses the new stats automatically.
- There are **two joint-name namespaces** in play — the env uses
  `left_thumb_1_joint` / `left_index_1_joint` / …, while the LeRobot canonical
  uses `L_thumb_proximal_yaw_joint` / `L_index_proximal_joint` / …. They
  refer to the same physical joints in the same order; the converter mapping
  is **index-based**, not name-based, so the namespaces never actually meet.
  Don't try to grep the env for `L_thumb_proximal_yaw_joint` — it won't be
  there.

---

## Layer 1 — Embodiment & Environment

> The *physical* contract: robot, sim, actions, observations, task.
> Source of truth: [`g1_grasp_policy_inspire_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py).
> See [`joint_spaces.md`](joint_spaces.md) for the full joint-ordering / naming-convention audit (anchor, derivations, drift hotspots) that underpins this section and Layer 2.

### Gym registrations

All three are registered in
[`grasp_policy_inspire/__init__.py`](../../scripts/simulation/tasks/grasp_policy_inspire/__init__.py):

| Gym ID | Action space | Used by |
|---|---|---|
| `Isaac-Grasp-Policy-G129-Inspire-Joint` | 41D joint positions | `eval_act_inspire.py`, RL training, this doc's L1 healthcheck |
| `Isaac-Grasp-Policy-G129-Inspire-Joint-Eval` | 41D, deterministic slot placement | reproducibility sweeps |
| `Isaac-Grasp-Policy-G129-Inspire-Teleop` | 38D PinkIK (wrist poses + hand joints) | `record_demos.py` |

> **Deprecated alias:** the legacy `Isaac-Grasp-Policy-G129-InspireFTP-{Joint,Joint-Eval,Teleop}` IDs are still registered (pointing at the same env cfgs) for backward compat with HDF5 demos and ACT checkpoints recorded under the old names. Don't use them in new code; the alias will be removed in a future cleanup once all artifacts have been re-recorded / re-trained.

### Action space — 41D joint positions

Defined in [`ActionsCfg`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L253-L264):

```python
joint_pos = mdp.InspireJointPositionActionCfg(
    asset_name="robot",
    joint_names=actuated_joint_names,   # 29 body + 12 actuated hand = 41
    scale=1.0,
    use_default_offset=False,
    offset=offset_dict,                  # {"left_elbow_joint": -0.3, "right_elbow_joint": -0.3}
    preserve_order=True,
)
```

- **41 joints** = 29 body joints + 12 actuated hand joints. See
  `actuated_joint_names` ([env_cfg.py:139](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L139)),
  which is `joint_names` minus the 12 mimic joints in `_MIMIC_JOINT_NAMES`.
- **Order**: USD articulation tree traversal (interleaved L/R). See the full
  53-entry `joint_names` list at
  [env_cfg.py:53-110](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L53-L110).
- **Elbow `−0.3` offset** is applied here, at rollout time, before the command
  reaches the articulation. Source:
  [env_cfg.py:112-115](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L112-L115).
  This is the **−0.3 side** of the elbow offset chain.
- **Mimic joints (12 passive)** are driven inside
  [`InspireJointPositionAction.apply_actions()`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py).
  They are **not part of the 41D action** — the action manager commands only
  the 12 actuated hand joints (`*_{index,middle,ring,little}_1_joint`,
  `*_thumb_1_joint`, `*_thumb_2_joint`), and mimic targets are derived from
  those as a second phase inside `apply_actions()`.

  Multipliers (from the real Inspire FTP URDF `<mimic>` tags):

  | Mimic joint | Parent | Multiplier |
  |---|---|---|
  | `*_{index,middle,ring,little}_2_joint` | `*_{…}_1_joint` | `1.0843` |
  | `*_thumb_3_joint` | `*_thumb_2_joint` | `0.8024` |
  | `*_thumb_4_joint` | `*_thumb_3_joint` *(itself a mimic)* | `0.9487` |

  **`MIMIC_RULES` list order is load-bearing** for the thumb chain: `thumb_3`
  must be computed before `thumb_4` because `thumb_4`'s parent is itself a
  mimic joint, not an action-tensor entry. The action class handles this by
  tagging each rule as `"action"` or `"mimic"` source in `_mimic_parent_info`.
  Reordering `_MIMIC_RULES_PER_SIDE` alphabetically silently zeros the distal
  thumb segment.

  **Do not bypass `InspireJointPositionAction`.** If you write joint
  targets directly to the articulation (e.g. a debug script that calls
  `robot.set_joint_position_target` against `_MIMIC_JOINT_NAMES` by hand),
  you lose mimic enforcement and grasps collapse. Same "don't bypass the
  action manager" rule as the elbow offset, for a different reason.

### Observation space — `obs["policy"]`

Defined in
[`ObservationsCfg`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L267-L291).
The actual observation functions live in
[`mdp/observations.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py).

**`robot_joint_state` — shape `(B, 87)`:**

Layout: `[pos(29) | vel(29) | torque(29)]`, where the 29 body joints are
arranged in the **canonical body order** defined by `_BODY_JOINT_NAMES_CANONICAL`
([observations.py:39-69](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py#L39-L69)).
Crucially, this is **NOT** the USD order — the observation function gathers
the positions/velocities/torques at explicit name-resolved indices so that
arm joints are always at fixed, contiguous slices:

| Slice | Joints |
|---|---|
| 0–11 | 12 leg joints (hips×3, knees, ankles×2, L/R interleaved) |
| 12–14 | waist yaw / roll / pitch |
| **15–21** | **left arm** (shoulder pitch / roll / yaw, elbow, wrist roll / pitch / yaw) |
| **22–28** | **right arm** (same 7) |

The Dex3 body observation uses the same canonical order, which is why the
shared joint-index constants in [`inspire_lerobot_fields.py`](../../scripts/utils/inspire/inspire_lerobot_fields.py)
(`STATE_26_BODY_COL_LEFT_ARM = range(15, 22)`,
`STATE_26_BODY_COL_RIGHT_ARM = range(22, 29)`) work identically across the
two hand variants.

**`robot_inspire_joint_state` — shape `(B, 12)`:**

Layout: `[L_thumb_1, L_thumb_2, L_index_1, L_middle_1, L_ring_1, L_little_1, R_thumb_1, R_thumb_2, R_index_1, R_middle_1, R_ring_1, R_little_1]`.
Source: `_INSPIRE_ACTUATED_NAMES` at
[observations.py:72-85](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py#L72-L85).
Indices 0–5 are the left hand, 6–11 are the right hand.

**`obs["camera_images"]`:**

Three keys — `front_camera`, `left_wrist_camera`, `right_wrist_camera` — all
RGB, unnormalized. Configured via `CameraPresets.g1_front_camera(focal_length=10.5)`,
`CameraPresets.left_inspire_wrist_camera(focal_length=12.0)`, and
`CameraPresets.right_inspire_wrist_camera(focal_length=12.0)` in
[env_cfg.py](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py).
`act_config_inspire.yaml` currently declares only `observation.images.cam_room`;
the wrist cameras exist in the dataset and can be added to ACT `input_features`
when training with wrist vision.

### Frame rate & control

Set in `__post_init__` at
[env_cfg.py:390-401](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L390-L401):

- `sim.dt = 1/200` → 200 Hz physics
- `decimation = 4` → **50 Hz** effective control
- `render_interval = 4` → rendered frame per env step
- `episode_length_s = 200.0`

Everything downstream (recording, video encoding, training fps metadata) must
match 50 Hz. If you change any of these, re-record demos.

### Task stage machine

Stages 0→3 defined in
[`grasp_policy_inspire/mdp/rewards.py:35-95`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/rewards.py#L35-L95).
State cached on `env._task_stage`; `check_success`
and `eval_act_inspire.py`'s progress logger both read it directly.

| From | To | Trigger |
|---|---|---|
| 0 | 1 | block z > `table_height + lift_threshold` (0.855 + 0.05) |
| 1 | 2 | block (x, y) inside bin footprint |
| 2 | 3 | block below bin rim AND above bin floor AND in footprint |

Stages advance **forward only** — no regressions. Success termination is
`env._task_stage >= success_stage`; the env config sets
`success_stage=3` at [env_cfg.py:303](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L303).

### Reset behavior

- **Tray** spawned at `TRAY_POS = (-1.49919, 2.03365, 0.84554)` with
  90° CW rotation `TRAY_ROT`.
- **Tools** share `TOOL_ROT` with the tray (they're rotated before placement).
- **`TRAY_SLOT_POSITIONS`**: 6 slots, a 2×3 grid
  ([env_cfg.py:165-178](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L165-L178)).
  Env default is **slot 4** (`ACTIVE_SLOT_IDX = 4`).
- **Reset event** `reset_block_to_tray_slot` ([env_cfg.py:355-365](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L355-L365))
  teleports the block to `slot_pos` with **XY noise ±0.02 m** and **yaw
  noise ±15°**. `eval_act_inspire.py` overrides `slot_pos` at runtime based on
  `--slot`.
- **Eval default is slot 1** — set because the overfit demo was recorded at
  slot 1. Check `eval_act_inspire.py:63` if this changes.

---

## Layer 2 — Data

> The *representation* contract: how embodiment state and actions become
> something the model can train on.
> Source of truth:
> [`inspire_lerobot_fields.py`](../../scripts/utils/inspire/inspire_lerobot_fields.py)
> and [`convert_hdf5_to_lerobot.py`](../../scripts/utils/convert_hdf5_to_lerobot.py).

### AVP → teleop action (38D)

The Inspire FTP Teleop env cfg
([`g1_grasp_policy_inspire_teleop_env_cfg.py:228-243`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py#L228-L243))
registers its own `"handtracking"` device directly with a
`UnitreeG1RetargeterCfg`. When `record_demos.py --teleop_device handtracking`
runs, it finds this key in `env_cfg.teleop_devices.devices` and uses it —
**`handtracking.py` is NOT in the Inspire FTP pipeline**. That file provides a
fallback retargeter (`G1HandtrackingGripperRetargeterCfg`) for Arena-track
tasks (trocar, locomanip) that use binary gripper control; the Inspire task
bypasses it entirely because it needs full per-finger dexterity.

`UnitreeG1RetargeterCfg` reads the full AVP hand skeleton (52 OpenXR
skeletal joints, 2 hands × 26 bone poses each) and produces **38D**:

- **Wrist extraction**: reads each hand's wrist bone pose → 7D Cartesian
  (position + quaternion) per hand = 14D
- **DexPilot**: takes all finger bone poses and solves an optimization to find
  the 24 robot hand joint angles (12 per hand) that best approximate the
  observed human hand shape, given the robot hand's kinematic model

The retargeter config at
[`teleop_env_cfg.py:232-237`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py#L232-L237)
bridges the two sides: `num_open_xr_hand_joints=52` tells the retargeter how
many skeletal poses to expect from AVP, and `hand_joint_names` (24 entries in
Nucleus-style naming) tells it which robot joints to produce values for.
DexPilot loads a hand-only URDF internally and runs the optimization each frame.

The retargeter output is the **38D teleop command**:
`[L_wrist_pos(3) | L_wrist_quat(4) | R_wrist_pos(3) | R_wrist_quat(4) | hand_joints(24)]`.

**PinkIK is a separate, downstream step** — it lives in `TeleopActionsCfg`
([`teleop_env_cfg.py:100-162`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py#L100-L162)),
not in the retargeter. It takes the 14D Cartesian wrist poses and solves
inverse kinematics → 14D arm joint angles (7 per arm). The 24D hand joints
pass through unchanged. The result is what actually commands the articulation.

This is what the Teleop gym variant steps with, and what `record_demos.py`
writes to HDF5 as `processed_actions` (the 38D pre-IK form, not the
post-IK joint angles).

### HDF5 recording format

[`record_demos.py`](../../scripts/simulation/record_demos.py) saves per-trajectory
HDF5 with these keys for the Inspire FTP Teleop variant:

| Key | Shape | Meaning |
|---|---|---|
| `obs.robot_joint_state` | (T, 87) | canonical body obs (Layer 1) |
| `obs.robot_inspire_joint_state` | (T, 12) | canonical hand obs (Layer 1) |
| `processed_actions` | (T, **38**) | PinkIK command (not executed joint positions!) |
| front camera frames | varies | stored alongside for video re-encode |

**38D is a command in a different space from the env's 41D joint action.**
You cannot "just read" actions from HDF5 for teleop recordings — the
converter has to derive actions a different way (see below).

**Single-arm recording (`--arm left|right`):** The HDF5 always stores the
**full** 87D + 12D observations regardless of `--arm`. The non-controlled arm
will show constant joint values in the recording. Only the 38D
`processed_actions` are masked (non-controlled wrist overwritten with FK pose,
non-controlled hand joints zeroed). Since the converter derives actions from
next-step *observations* (not from `processed_actions`), the 13D extraction
works correctly on these recordings — the non-controlled arm's constant
values are simply not selected.

The FK wrist pose for the locked arm is read once from
`robot.data.body_pos_w` / `body_quat_w` after each `env.reset()` and held
constant for the episode, avoiding drift from the approximate values in
`idle_action`. Hand joint masking uses explicit per-hand index lists
(`_LEFT_HAND_38D_IDX` / `_RIGHT_HAND_38D_IDX`) because the USD articulation
order **interleaves** left and right hand joints — contiguous slices cross
hand boundaries.

### HDF5 → LeRobot conversion

Entry point:
[`convert_hdf5_to_lerobot.py`](../../scripts/utils/convert_hdf5_to_lerobot.py) →
`convert_trajectory_to_df_rheo()`. Per-row state/action construction is in
[`convert_g1_state_action_to_lerobot_26d()`](../../scripts/utils/inspire/inspire_lerobot_fields.py#L270-L301).

**Canonical 26D layout** (`STATE_26_NAMES_ENV_ORDER`,
[inspire_lerobot_fields.py:30-61](../../scripts/utils/inspire/inspire_lerobot_fields.py#L30-L61)):

| Slice | Joints |
|---|---|
| 0–6 | left arm (7 joints) |
| 7–13 | right arm (7 joints) |
| 14–19 | left hand (6 actuated) |
| 20–25 | right hand (6 actuated) |

State is built by index-slicing the raw obs:

```python
state[:, 0:7]   = obs.robot_joint_state[:, 15:22]   # left arm
state[:, 7:14]  = obs.robot_joint_state[:, 22:29]   # right arm
state[:, 14:20] = obs.robot_inspire_joint_state[:, 0:6]   # left hand
state[:, 20:26] = obs.robot_inspire_joint_state[:, 6:12]  # right hand
```

**Action derivation for 38D teleop recordings** — the only case currently in
use:

```python
action = full_26d[1:] + STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA
```

That is:
1. `action[t] = state[t+1]` — the action label is the *observed* next-step
   joint position, not the teleop command. The teleop command is in a
   different space (PinkIK wrist poses) and cannot be mapped back to joint
   targets directly.
2. Then **`+0.3`** is added to indices **3** (left elbow) and **10** (right
   elbow) via `STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA`
   ([inspire_lerobot_fields.py:77-79](../../scripts/utils/inspire/inspire_lerobot_fields.py#L77-L79))
   so the saved action compensates the env's `offset_dict -0.3` at rollout
   time. This is the **+0.3 side** of the elbow offset chain.

**The add MUST be out-of-place.** `state = full_26d[:-1]` and
`action_pre = full_26d[1:]` are both NumPy views into the same buffer; an
in-place `+=` on `action_pre` silently mutates `state` on elbow columns from
row 1 onward. This is the aliasing bug fixed in April 2026 — don't reintroduce
it.

**Other conversion modes** (present but unused): 53-D and 41-D recorded
actions are handled by `ACTION_HDF5_TO_ENV_26` /
`ACTION_HDF5_TO_ENV_26_FROM_41` index tables. These exist for legacy
recordings; current AVP teleop always hits the 38-D path.

### 13D single-arm conversion

For single-arm policies, two 13D extraction functions mirror the 26D logic
but select only one arm + hand:

| Config flag | Function | Extracts |
|---|---|---|
| `rheo_13d_state_action` | `convert_g1_state_action_to_lerobot_13d()` | right_arm(7) + right_hand(6) |
| `rheo_13d_left_state_action` | `convert_g1_state_action_to_lerobot_13d_left()` | left_arm(7) + left_hand(6) |

**13D layout** (same for both sides, just different source columns):

| Slice | Joints |
|---|---|
| 0-6 | arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) |
| 7-12 | hand (thumb yaw/pitch, index, middle, ring, pinky proximal) |

State is built by slicing one side of the raw obs:

```python
# Right 13D
state[:, 0:7]  = obs.robot_joint_state[:, 22:29]   # right arm
state[:, 7:13] = obs.robot_inspire_joint_state[:, 6:12]  # right hand

# Left 13D
state[:, 0:7]  = obs.robot_joint_state[:, 15:22]   # left arm
state[:, 7:13] = obs.robot_inspire_joint_state[:, 0:6]   # left hand
```

Action derivation is identical to 26D: `action[t] = state[t+1] + delta`,
where delta is `+0.3` at index 3 (elbow). The elbow offset delta vectors
are `STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA` (right) and
`STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA` (left).

**Dataset configs:**
- Right arm: `scripts/config/g1_grasp_policy_inspire_dataset_right_arm.yaml`
- Left arm: `scripts/config/g1_grasp_policy_inspire_dataset_left_arm.yaml`

Both use `data_root` pointing to arm-specific directories
(`datasets/inspire_right_arm`, `datasets/inspire_left_arm`). The same HDF5
recording can be converted with any of the three configs (26D, 13D right,
13D left) — the `--arm` flag used during teleop does not constrain which
conversion config is applicable.

### LeRobot dataset layout

Under `$HOME/datasets/inspire_tool0_slot1/…/lerobot/`:

```
lerobot/
├── data/chunk-000/episode_NNNNNN.parquet
├── videos/chunk-000/observation.images.cam_room/episode_NNNNNN.mp4
└── meta/
    ├── info.json                  # fps=50, feature shapes, joint names
    ├── episodes.jsonl             # per-episode length + task
    ├── episodes_stats.jsonl       # per-episode mean/std/min/max/quantiles
    └── tasks.jsonl                # task index → language instruction
```

Parquet columns (one row per frame):

- `observation.state` — 26D f64
- `action` — 26D f64
- `observation.images.cam_room` — video reference (path + frame index)
- `timestamp`, `episode_index`, `frame_index`, `task_index`
- `next.reward`, `next.done`, `annotation.human.task_description`

Video: h264 at **50 fps**, **480 × 640** RGB.

**`meta/episodes_stats.jsonl` drives training normalization.** Mean/std are
computed per episode at conversion time and aggregated by LeRobot at training
start; `act_config_inspire.yaml` uses `MEAN_STD` for `STATE`, `ACTION`,
and `VISUAL`. If you regenerate the parquet, these stats change; if you then
resume training from an old checkpoint, the normalization shift can silently
corrupt rollouts. **Regenerate = retrain.**

---

## Layer 3 — Policy

> The *learning* contract: model, inputs, outputs, inference wrapper.
> Source of truth:
> [`act_config_inspire.yaml`](../../scripts/policy/act_config_inspire.yaml),
> [`simulation/policies/act.py`](../../scripts/simulation/policies/act.py),
> and the experiment config module
> [`inspire_experiment_config.py`](../../scripts/utils/inspire/inspire_experiment_config.py).
> Joint identity (which joints land in which 26D / 41D slots) is anchored in
> [`inspire_joint_constants.py`](../../scripts/inspire_joint_constants.py) — the
> experiment config module's group-size, range, and scatter-index dicts are all
> derived from `GROUP_NAMES` defined there. See
> [`docs/inspire/joint_spaces.md`](joint_spaces.md) for the full audit.

### Base config — `act_config_inspire.yaml`

The `experiment:` section is **not** a LeRobot key — it's a rheo-local block
that the training launcher strips out and the inference wrapper re-reads.

- **`experiment.cameras`**: maps sim camera keys to ACT model keys.
  Inspire FTP: `front_camera → observation.images.cam_room`.
- **`experiment.joint_groups`**: list that determines the 26D state layout.
  `[left_arm, right_arm, left_hand, right_hand]` → 7 + 7 + 6 + 6 = 26.

The rest is standard LeRobot:

- `policy.type: act`, `chunk_size: 50`, `n_action_steps: 50`, `n_obs_steps: 1`
- Transformer: `dim_model 256`, `n_heads 8`, `dim_feedforward 1600`,
  `n_encoder_layers 2`, `n_decoder_layers 1`
- VAE: `use_vae: true`, `latent_dim 32`, `kl_weight 10.0`
- Vision: `resnet18`, ImageNet pretrained, no dilation
- Normalization: `STATE`/`ACTION`/`VISUAL` all `MEAN_STD`
- Input features: `observation.state [26]`,
  `observation.images.cam_room [3, 480, 640]`
- Output feature: `action [26]`

### Training launcher

[`train_act_grasp_policy_inspire.sh`](../../scripts/policy/train_act_grasp_policy_inspire.sh):

1. Reads `act_config_inspire.yaml`.
2. **Strips the `experiment:` block** into a temp file (LeRobot's
   `TrainPipelineConfig` rejects unknown keys).
3. Exports `INSPIRE_EXPERIMENT_CONFIG` pointing back to the original so
   the eval-time wrapper can read it.
4. Forwards `EXTRA_ARGS` (unknown CLI flags) verbatim to
   `lerobot.scripts.train`.

Any LeRobot CLI override works without editing the script, e.g.:

```bash
./docker/run_docker_grasp.sh bash scripts/policy/train_act_grasp_policy_inspire.sh \
    --dataset_path /datasets/inspire_tool0_slot1/lerobot \
    --dataset.episodes "[0]" \
    --policy.kl_weight 0 \
    --policy.optimizer_lr 1e-4 \
    --steps 8000
```

### Closed-loop inference wrapper

[`ACTClosedloopPolicy`](../../scripts/simulation/policies/act.py) loads
the LeRobot checkpoint (`ACTPolicy.from_pretrained`), picks the Inspire FTP
experiment config, and exposes two observation paths:

- **`get_action_from_raw(obs)`** — used by `eval_act_inspire.py`. Reads
  `obs["policy"]["robot_joint_state"]` and
  `obs["policy"]["robot_inspire_joint_state"]` directly, slices out the
  canonical 26D state, normalizes, and runs the policy.
- **`get_action(obs)`** — used when observations come from an `obs_processor`
  with pre-split `state.left_arm` / `video.room_view` keys.

**Forward path:**

```python
action = self.policy.predict_action_chunk(single_obs)   # (1, 50, 26)
action = action.squeeze(0)                              # (50, 26)
# stack per-env → (num_envs, 50, 26), then:
sim_actions = self.exp_config.scatter_to_sim(policy_actions)  # (num_envs, 50, 41)
```

**DO NOT use `select_action()`.** It pops one entry from an internal action
queue and returns `(1, 26)`. The wrapper's pad-to-chunk-length code then pads
with `last_action.expand(-1, 49, -1)`, producing **50 identical actions per
chunk**. This is the chunk-collapse bug fixed in April 2026 — the symptom is
`std = 0` on every active dim in `eval_act_inspire.py --log_actions` output.
Use `predict_action_chunk()` and squeeze.

### 26D → 41D scatter

`InspireExperimentConfig.scatter_to_sim` places the 26D policy output at
the canonical sim-joint indices and leaves everything else at its env default
(legs, waist, mimic hand joints). **The `-0.3` elbow offset is not applied
here** — it's applied downstream by `InspireJointPositionActionCfg.offset`
at articulation time. This is important: if you bypass the action manager and
write joint targets directly, you must apply the `-0.3` yourself.

---

## Debug playbook: symptom → likely layer → first diagnostic

| Symptom in eval | Likely layer | First diagnostic |
|---|---|---|
| Arm drifts monotonically into a pose you never trained | L2 (elbow offset missing, doubled, or wrong sign) | Read a few parquet rows; `action[t] - state[t+1]` on cols 3, 10 should be `+0.3`, everything else `~0`. |
| Every action in a chunk is identical (`std=0` per dim in `--log_actions`) | L3 inference | `select_action` vs `predict_action_chunk`. Dump `policy.predict_action_chunk(obs).shape` — it should be `(1, 50, 26)`. |
| Training loss collapses, but rollout goes nowhere | L3 obs-path mismatch | Dump the first ACT input at eval time, compare element-wise to a matching row of `observation.state` from the training parquet. Means/stds should match post-normalization. |
| Training loss never drops | L2 (regenerated parquet is wrong) or L1 (obs order changed) | Re-run the L2 spot-check (elbow delta +0.3, state row matches raw body cols 15-28). |
| Block never reaches stage 1 even with a known-good policy | L1 (reset state doesn't match training) | Check `TRAY_SLOT_POSITIONS[slot]` matches the slot used at recording; confirm `TOOL_ROT` hasn't changed. |
| Eval works for one slot/tool but not another | Training coverage, not a bug | Expected — 30 teleop demos is borderline. Record more, or restrict eval. |
| Extreme action values (`|a| > 3`) warning in eval | L3 normalization | `meta/episodes_stats.jsonl` may be stale relative to the checkpoint. Regenerate parquet → retrain. |
| Hand fingers move but mimic finger segments don't | L1 mimic | Confirm action is going through `InspireJointPositionActionCfg`, not a raw articulation write — mimic is applied in `apply_actions()`. |
| Video is fine in playback but policy sees black frames | L3 image path | Check `_extract_observations_from_raw()` — `obs["camera_images"]["front_camera"]` must be uint8 HWC before the wrapper divides by 255. |
| Non-controlled arm drifts during single-arm teleop | L1 teleop masking | FK wrist pose may not be reading correctly after reset. Check `_read_frozen_wrist_fk()` — verify `body_names.index("*_wrist_yaw_link")` resolves and `body_pos_w` / `body_quat_w` are populated. |
| Only proximal finger joints move in single-arm teleop | L1 teleop masking | Hand index lists are likely using contiguous slices instead of interleaved indices. Verify `_LEFT_HAND_38D_IDX` / `_RIGHT_HAND_38D_IDX` match the USD joint order in `env_cfg.py:53-110`. |

---

## Canonical sanity checks

Copy-pasteable commands — the "known-good artifacts" to keep warm. If any of
these fails, fix it before touching anything else.

### L1: env steps end-to-end (no policy)

```bash
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --test --max_steps 50 --enable_cameras
```

Dummy zero-action policy steps the env for 50 frames. If this fails, Layer 1
is broken (env config, USD assets, camera, reset).

### L2: parquet elbow bias + state sanity

Inside the grasp container:

```python
import pyarrow.parquet as pq, numpy as np
t = pq.read_table('/datasets/inspire_tool0_slot1/lerobot/data/chunk-000/episode_000000.parquet')
a = np.stack(t.column('action').to_numpy())
s = np.stack(t.column('observation.state').to_numpy())

# Elbow offset — should be ~0.3
print('L elbow:', a[0, 3]  - s[1, 3])
print('R elbow:', a[0, 10] - s[1, 10])

# Non-elbow diff — should be ~0
non_elbow = [i for i in range(26) if i not in (3, 10)]
print('max non-elbow |a-s[t+1]|:', np.max(np.abs(a[0, non_elbow] - s[1, non_elbow])))
```

Expected: both elbow deltas `≈ 0.3`, non-elbow max `≈ 0`. If the deltas
print `≈ 0.0` the aliasing bug is back.

### L3: policy rollout with action logging

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path <path_to_pretrained_model> \
    --slot 1 --object tool_0 \
    --num_episodes 1 --max_steps 1000 \
    --log_actions --save_video --enable_cameras \
    2>&1 | tee eval_logs/healthcheck_$(date +%Y%m%d_%H%M%S).log
```

In the log, within each `[Chunk N]` block:

- Every active dim should show **non-zero `std`** (if not → chunk collapse
  bug is back).
- Elbow dims **21** and **22** (sim indices, not policy indices) should sit
  in `-0.5 .. -0.1 rad` throughout. Monotonically drifting more negative
  means the elbow offset isn't being cancelled at eval time.

---

## What this doc is not

- **Not a tutorial.** First-time readers use [`grasp_policy_guide.md`](grasp_policy_guide.md).
- **Not a changelog.** Bug fixes and commit history live in git.
- **Not a style guide.** No opinions on how the pipeline *should* work — only
  on how it currently does.
- **Not exhaustive.** Anything already enforced by code (mimic joint ratios,
  camera focal length, scene USD paths) is linked to its source rather than
  duplicated here. If you need a value that isn't in this doc, read the file
  it cites.
