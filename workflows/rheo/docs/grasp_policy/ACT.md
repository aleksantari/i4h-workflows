<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ACT Pipeline Architecture — Inspire FTP Grasp

Technical reference for the ACT (Action Chunking Transformer) imitation learning
pipeline as it currently runs on the **Inspire FTP surgical-tool grasp task**
in the rheo workflow. For step-by-step usage instructions, see
[`grasp_policy_guide.md`](grasp_policy_guide.md).

> **Scope:** This doc covers the Inspire FTP variant (G1 + Inspire FTP hands,
> 26D policy → 41D sim action, surgical tool pickup). An earlier revision of
> this doc targeted the original Dex3-hand block-grasping task. The Dex3
> codepath still exists (see [`act_experiment_config.py`](../../scripts/utils/act_experiment_config.py),
> [`act_config_dex3.yaml`](../../scripts/policy/act_config_dex3.yaml),
> [`train_act_grasp_policy_dex3.sh`](../../scripts/policy/train_act_grasp_policy_dex3.sh)),
> but it is not the active focus and is not documented here.

---

## Table of Contents

1. [Overview](#1-overview)
2. [ACT Model Architecture](#2-act-model-architecture)
3. [Pipeline Stages](#3-pipeline-stages)
4. [Joint & Action Dimension Flow](#4-joint--action-dimension-flow)
5. [Mimic Joint Enforcement](#5-mimic-joint-enforcement)
6. [Experiment Config Propagation](#6-experiment-config-propagation)
7. [RLinf Integration](#7-rlinf-integration)
8. [Task Definition](#8-task-definition)
9. [Docker](#9-docker)
10. [Key Files Reference](#10-key-files-reference)

---

## 1. Overview

ACT is a CVAE-based imitation learning policy that predicts multi-step action
chunks from visual and proprioceptive observations. In the Inspire FTP
pipeline it is the primary policy used to solve the surgical-tool pick-and-place
task: **pick up a surgical tool from the tray and place it in the bin.**
The robot is G1 29-DoF with Inspire FTP hands (6 actuated DOF + 6 mimic per
hand).

### End-to-end pipeline

```
AVP teleoperation (hand tracking + PinkIK wrist retarget + DexPilot hand retarget)
    |
    v
record_demos.py
    |  obs:  robot_joint_state (T, 87)            -- 29 body joints x [pos|vel|torque]
    |        robot_inspire_joint_state (T, 12)    -- 6 actuated x 2 hands
    |        front_camera (T, 480, 640, 3)        -- single RGB
    |  action: processed_actions (T, 53 or 41)    -- post-teleop targets
    v
HDF5 demo files
    |
    v
convert_hdf5_to_lerobot.py  (rheo_26d_state_action: true)
    |
    v
LeRobot v2.1 dataset  (26D state + 26D action + 1 MP4 per episode)
    |
    v
train_act_grasp_policy_inspire.sh   (LeRobot ACT IL training)
    |
    v
ACT checkpoint (LeRobot HF-format folder)
    |
    v
eval_act_inspire.py  -->  ACTClosedloopPolicy  -->  IsaacLab env  -->  video + results
    |
    v
(optional) RLinf PPO post-training  -->  refined checkpoint
```

### Two eval paths

Two scripts can evaluate an Inspire ACT checkpoint today:

| Script | Status | Notes |
|-|-|-|
| [`eval_act_inspire.py`](../../scripts/simulation/examples/eval_act_inspire.py) | **Primary / canonical** | ~230-line clean rewrite purpose-built for ACT + Inspire FTP. Uses `get_action_from_raw()` to read obs directly from the env without going through the legacy `process_observation()` reassembly layer. Adds `--log_actions` and `--clamp_actions` diagnostic flags. |
| [`eval_grasp_policy_inspire.py`](../../scripts/simulation/examples/eval_grasp_policy_inspire.py) | Legacy | Unified evaluator inherited from the GR00T workflow; routes through `evaluate_episode()` and `process_observation()`. Kept working but harder to debug. |

The rest of this doc walks through the **primary** path.

---

## 2. ACT Model Architecture

ACT (Action Chunking with Transformers) is a Conditional Variational Autoencoder
that predicts a sequence of future joint targets from a single observation.
Architecture values below are for the Inspire FTP configuration, taken from
[`act_config_inspire_ftp.yaml`](../../scripts/policy/act_config_inspire_ftp.yaml).

### Inputs

| Input | Shape | Source |
|-|-|-|
| Joint state | `(B, 26)` | `left_arm(7) + right_arm(7) + left_hand(6) + right_hand(6)` |
| Front camera | `(B, 3, 480, 640)` | `front_camera` (RGB), mapped to ACT feature key `observation.images.cam_room` |

The env provides 3 cameras (front + left/right wrist) identical to Dex3. The
default ACT config consumes only the front camera; wrist cameras are available
in the dataset and can be added to `input_features` when training with wrist vision.

### Output

| Output | Shape | Description |
|-|-|-|
| Action chunk | `(B, 50, 26)` | 50 future joint position targets in policy-dim order |

The 50-step chunk length flows end-to-end:
`act_config_inspire_ftp.yaml:chunk_size=50` → `ACTClosedloopPolicy.action_chunk_length` →
`eval_act_inspire.py --action_chunk_size` default.

### Architecture (from `policy:` block of the YAML)

```
                  Observation
                  /           \
           ResNet18            Linear(26 -> dim_model)
           (front cam)              |
               \                   /
                Transformer Encoder (2 layers, dim_model=256, 8 heads, FFN=1600)
                         |
                VAE encoder (2 layers, latent_dim=32)
                         |
                 z ~ N(mu, sigma)        <-- sampled from prior at inference
                         |
                Transformer Decoder (1 layer)
                         |
                 Action sequence (50 x 26)
```

### Hyperparameters (Inspire variant)

| Parameter | Value | Note |
|-|-|-|
| `chunk_size` | 50 | Actions per forward pass |
| `n_action_steps` | 50 | Executed per observation |
| `n_obs_steps` | 1 | Single-frame observation |
| `dim_model` | 256 | Transformer hidden dim |
| `n_heads` | 8 | |
| `dim_feedforward` | 1600 | |
| `n_encoder_layers` | 2 | |
| `n_decoder_layers` | 1 | |
| `use_vae` | true | CVAE enabled |
| `latent_dim` | 32 | |
| `n_vae_encoder_layers` | 2 | |
| `kl_weight` | 10.0 | Strong regularization |
| `vision_backbone` | resnet18 | ImageNet-pretrained |
| `steps` | 50000 | Default training length |
| `batch_size` | 16 | |
| `optimizer_lr` | 1e-5 | Same for backbone and heads |
| `optimizer_weight_decay` | 1e-4 | AdamW |

### Loss

```
L = L1(action_pred, action_gt) + kl_weight * D_KL(q(z|obs, action) || p(z|obs))
```

- `kl_weight = 10.0` encourages diverse behaviors.
- At inference, `z` is sampled from the observation-conditional prior
  `p(z|obs)` — no teacher forcing.

---

## 3. Pipeline Stages

### 3.1 Data collection

> **Code:** [`record_demos.py`](../../scripts/simulation/record_demos.py),
> [`g1_grasp_policy_inspire_teleop_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py)

The teleop env uses the full 53-joint G1 articulation (29 body + 24 hand,
including all 12 mimic joints) so that VR teleop can drive the hand through
PinkIK wrist retargeting + DexPilot finger retargeting.

Per-episode HDF5 structure recorded by `record_demos.py`:

```
episode_NNNNNN/
  obs/
    robot_joint_state           (T, 87)              29 body joints x [pos|vel|torque]
    robot_inspire_joint_state   (T, 12)              6 actuated x 2 hands
    front_camera                (T, 480, 640, 3)     uint8 RGB
  processed_actions             (T, 53 or 41)        post-teleop targets
  metadata/
    success                     bool
```

Auto-success detection saves episodes after N consecutive frames where the
tool reaches stage 3, so the operator does not need to hand-label successes.

### 3.2 Dataset conversion

> **Code:** [`convert_hdf5_to_lerobot.py`](../../scripts/utils/convert_hdf5_to_lerobot.py)
> (see the `rheo_26d_state_action` branch at lines 65–82),
> [`g1_grasp_policy_inspire_dataset.yaml`](../../scripts/config/g1_grasp_policy_inspire_dataset.yaml)

The dataset YAML sets `rheo_26d_state_action: true`, which routes conversion
through `convert_g1_state_action_to_lerobot_26d()`. That function:

1. Reads `robot_joint_state[:, 15:29]` to get 14 arm joints.
2. Reads `robot_inspire_joint_state[:, 0:12]` to get 12 actuated hand joints.
3. Concatenates into a canonical 26D state vector
   (`left_arm | right_arm | left_hand | right_hand`) — order enforced by
   `STATE_26_NAMES_ENV_ORDER`.
4. Extracts the matching 26D slice from `processed_actions` as the action
   label.

The YAML also maps sim camera names to ACT feature keys:

```yaml
rheo_camera_mappings_obs:
  front_camera: observation.images.cam_room
```

Output layout:

```
lerobot/
  data/chunk-000/
    episode_000000.parquet     26D state + 26D action + timestamps
    ...
  videos/chunk-000/
    observation.images.cam_room/
      episode_000000.mp4       480x640 RGB, H.264
  meta/
    info.json                  feature shapes/dtypes/fps
    episodes.jsonl
    tasks.jsonl
    episodes_stats.jsonl       (auto-generated by the training launcher if missing)
```

### 3.3 IL training

> **Code:** [`train_act_grasp_policy_inspire.sh`](../../scripts/policy/train_act_grasp_policy_inspire.sh),
> [`act_config_inspire_ftp.yaml`](../../scripts/policy/act_config_inspire_ftp.yaml)

`train_act_grasp_policy_inspire.sh` is a thin bash wrapper around
`python -m lerobot.scripts.train`. It does four things LeRobot would not do
itself:

1. **Parses `--dataset_path` and `--resume_path`**, forwards everything else
   as extra args to LeRobot.
2. **Strips the `experiment:` section** from `act_config_inspire_ftp.yaml`
   into a temp file (`$FILTERED_CONFIG`). LeRobot's `TrainPipelineConfig` is
   a draccus dataclass that rejects unknown top-level keys, so the rheo-
   specific `experiment:` block has to be hidden from it.
3. **Exports `INSPIRE_FTP_EXPERIMENT_CONFIG=$CONFIG_PATH`** (the unfiltered
   path) so downstream code (eval, RL converters, policy wrapper) can reload
   the `experiment:` block via `InspireFTPExperimentConfig.from_env_or_default()`.
4. **Auto-generates `episodes_stats.jsonl`** if missing — LeRobot v2.1 needs
   per-episode normalization stats to build the dataloader.

Then it invokes:

```bash
python -m lerobot.scripts.train \
    --config_path "$FILTERED_CONFIG" \
    --dataset.repo_id grasp_policy_inspire \
    --dataset.root "$DATASET_PATH" \
    --dataset.video_backend pyav \
    --output_dir "$OUTPUT_DIR" \
    [--resume $RESUME_PATH] \
    "${EXTRA_ARGS[@]}"
```

`--dataset.video_backend pyav` is required because the default `torchcodec`
backend needs `libnvrtc.so.13`, which is not installed in the grasp Docker
image.

**Output directory:**

```
scripts/simulation/rl/results/act_grasp_policy_inspire/train_<timestamp>/
    checkpoints/
        <step>/
            pretrained_model/        <- this is what --model_path points at
            training_state.pt
    config.yaml
    train.log
```

### 3.4 IL evaluation — the plumbing

> **Code:** [`eval_act_inspire.py`](../../scripts/simulation/examples/eval_act_inspire.py),
> [`act_closedloop_policy.py`](../../scripts/simulation/act_closedloop_policy.py),
> [`inspire_ftp_experiment_config.py`](../../scripts/utils/inspire_ftp_experiment_config.py)

This is the section you want if you are trying to understand how an ACT
checkpoint actually drives the robot. The flow is:

```
env.step(prev_action)
        |
        v
obs dict {
    "policy": {
        "robot_joint_state":         (1, 87)
        "robot_inspire_joint_state": (1, 12)
    },
    "camera_images": {
        "front_camera":              (1, 480, 640, 3)  uint8 on GPU
    }
}
        |
        v   [every chunk_size=50 steps, otherwise pop from buffer]
        |
policy.get_action_from_raw(obs)
        |
        +--> _extract_observations_from_raw(obs):
        |       body_87  = obs["policy"]["robot_joint_state"]
        |       hand_12  = obs["policy"]["robot_inspire_joint_state"]
        |       state_26 = InspireFTPExperimentConfig.extract_state(body_87, hand_12)
        |                  = cat([body[:, 15:22], body[:, 22:29],
        |                         hand[:, 0:6],   hand[:, 6:12]], dim=-1)
        |       img      = obs["camera_images"]["front_camera"]
        |                  .float() / 255.0
        |                  .permute(0, 3, 1, 2)                  -> (1, 3, 480, 640)
        |       act_obs  = {
        |           "observation.state":            state_26,
        |           "observation.images.cam_room":  img,
        |       }
        |
        +--> _forward_action_chunk(act_obs):
        |       for i in range(num_envs):
        |           single_obs = {k: v[i:i+1] for k, v in act_obs.items()}
        |           chunk_i    = self.policy.select_action(single_obs)   # LeRobot ACT
        |       policy_actions = stack(chunks)                            # (num_envs, 50, 26)
        |       sim_actions    = exp_config.scatter_to_sim(policy_actions)
        |                        # (num_envs, 50, 41) — zeros at non-scatter indices
        |       -> truncate / pad to action_chunk_length=50
        |
        v
numpy array (50, 41)
        |
        v   [pop one per env.step()]
        v
env.step(action_41)
        |
        v
InspireFTPJointPositionAction.apply_actions()
        |
        +--> super().apply_actions()                  <-- sets 41 actuated joint targets
        +--> compute 12 mimic targets from actuated
        +--> set_joint_position_target(mimic_vals, joint_ids=mimic_art_ids)
```

Four details that are easy to miss and matter when debugging:

1. **`get_action_from_raw()` vs `get_action()`.** The old evaluator
   (`evaluate_episode()`) calls `process_observation()`, which splits the
   observation into separate keys like `state.left_arm` and `video.room_view`
   and then the policy wrapper reassembles them. `get_action_from_raw()` skips
   that entirely — it reads `obs["policy"]` and `obs["camera_images"]`
   directly. This is why `eval_act_inspire.py` can be ~230 lines instead of
   threading through the GR00T-era infrastructure, and why its action stream
   is easier to reason about under `--log_actions`.

2. **Non-contiguous scatter indices.** `scatter_to_sim()` does not write the
   26D policy output into positions 0..25 of a 41-long vector — it scatters
   to specific indices that match the USD tree-traversal order with mimic
   joints removed. See [Section 4](#4-joint--action-dimension-flow) for the
   full table.

3. **Camera key renaming.** The simulator calls the camera `front_camera`
   (that is its key in `obs["camera_images"]`), but the ACT feature dict has
   to use `observation.images.cam_room` because that is the key the
   checkpoint was trained with. The mapping lives in
   `InspireFTPExperimentConfig.cameras` (default: `{"front_camera":
   "observation.images.cam_room"}`), sourced from the `experiment:` block of
   `act_config_inspire_ftp.yaml`.

4. **Experiment config load path at eval time.** `eval_act_inspire.py` builds
   a small temporary policy YAML for `ACTClosedloopPolicy`, and writes
   `experiment_config_path: <absolute path to act_config_inspire_ftp.yaml>`
   into it. `ACTClosedloopPolicy.__init__` loads that YAML via
   `InspireFTPExperimentConfig.from_yaml(...)`, which is how eval-time
   joint-group and camera selection stays consistent with training-time
   selection. If the path is missing, `from_env_or_default()` is used
   instead, which reads `INSPIRE_FTP_EXPERIMENT_CONFIG` from the environment
   or falls back to the defaults hard-coded in `inspire_ftp_experiment_config.py`.

**CLI surface** of `eval_act_inspire.py`:

| Flag | Default | Purpose |
|-|-|-|
| `--task` | `Isaac-Grasp-Policy-G129-InspireFTP-Joint` | Gym ID |
| `--model_path` | (required unless `--test`) | Path to `pretrained_model` folder |
| `--num_episodes` | 10 | |
| `--max_steps` | 5000 | Per-episode step cap |
| `--seed` | 4 | Env + policy seed |
| `--success_stage` | 3 | Stage at which `check_success()` returns True |
| `--object` | `tool_0` | Choose surgical tool USD (`tool_0`..`tool_4`) |
| `--slot` | 4 | Tray slot index (0..5) — overrides scene + reset event |
| `--action_chunk_size` | 50 | Actions executed per chunk before re-observing |
| `--clamp_actions` | 0.0 | If > 0, `np.clip(chunk, -val, val)` |
| `--log_actions` | off | Per-chunk min/max/per-dim stats printed to stdout |
| `--save_video` | off | Writes a single-view MP4 via `_MultiViewConcatWriter` |
| `--video_dir` | `./eval_videos` | |
| `--test` | off | Use a dummy zero-action policy (no checkpoint needed) |

**Results file:** `./eval_results/results_<timestamp>_act_inspire.txt`
(relative to the cwd inside the container — which maps to
`~/repos/i4h-workflows/workflows/rheo/eval_results/` on the host because
`/workspaces/workflows/rheo` is the bind-mounted repo).

### 3.5 RL post-training (scaffolded for Inspire)

> **Code:** [`rlinf_ext/__init__.py`](../../scripts/simulation/rl/rlinf_ext/__init__.py)
> (lines 51, 78–81, 560–705),
> [`rlinf_ext/act_policy.py`](../../scripts/simulation/rl/rlinf_ext/act_policy.py),
> [`config/isaaclab_ppo_act_grasp_policy_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml),
> [`config/model/act_inspire_ftp.yaml`](../../scripts/simulation/rl/rlinf_ext/config/model/act_inspire_ftp.yaml)

RL post-training uses RLinf PPO to refine the IL-trained ACT policy with
environment rewards. For the Inspire FTP task, the RL scaffolding is
**registered and wired up but not yet validated end-to-end** — the active
day-to-day workflow is IL-only eval through `eval_act_inspire.py`.

What the registration does (in `rlinf_ext/__init__.py:register()`):

- Registers the Inspire FTP gym IDs with RLinf's env map:
  - `Isaac-Grasp-Policy-G129-InspireFTP-Joint` → `IsaaclabGraspPolicyInspireEnv`
  - `Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval` → same class
- Registers ACT obs/action converters keyed `"act_inspire_ftp"`:
  - `_convert_inspire_obs_to_act_format` — uses
    `InspireFTPExperimentConfig.from_env_or_default()` to pick camera keys
    and normalize `(B, H, W, C) uint8 → (B, C, H, W) float`
  - `_convert_inspire_act_action_to_sim` — truncates the chunk and calls
    `exp_config.scatter_to_sim_numpy()` to build the 41D sim action

`IsaaclabGraspPolicyInspireEnv._wrap_obs()` (lines 593–607) is the piece
that defines the 26D critic state used by RL: it concatenates
`obs["policy"]["robot_joint_state"][:, 15:29]` (14D arm slice) with
`obs["policy"]["robot_inspire_joint_state"]` (12D hand) to produce a
`(B, 26)` tensor. This ordering matches what
`InspireFTPExperimentConfig.extract_state()` produces for training, so the
critic sees the same 26D state vector as the actor.

The model wrapper (`ACTForRLActionPrediction` + `ValueHead`) is shared
between the Dex3 and Inspire variants — it lives in `act_policy.py` and
attaches a 3-layer MLP value head to the ACT transformer encoder features
for critic estimation.

Everything RL-side ultimately reads
`INSPIRE_FTP_EXPERIMENT_CONFIG` via `from_env_or_default()`, so the same
env-var trick used at training time keeps joint-group and camera selection
consistent across all three stages (IL train → IL eval → RL).

---

## 4. Joint & Action Dimension Flow

The G1 + Inspire FTP articulation has **53 joints** total:

| Group | Count | Notes |
|-|-|-|
| Body | 29 | legs + waist + arms (no hand) |
| Hand, actuated | 12 | 6 per hand: thumb_yaw, thumb_pitch, index, middle, ring, pinky |
| Hand, mimic | 12 | 6 per hand: finger _2 joints + thumb _3, thumb _4 |

The **41D action space** is the 29 body joints + 12 actuated hand joints;
the 12 mimic joints are driven separately inside
`InspireFTPJointPositionAction.apply_actions()` (see [Section 5](#5-mimic-joint-enforcement)).

The **26D policy space** is a strict subset of the 41D action space:

```
                 IsaacLab env observations
             robot_joint_state          (B, 87)   29 body joints x [pos|vel|torque]
             robot_inspire_joint_state  (B, 12)   12 actuated hand joints
                           |
                           |  InspireFTPExperimentConfig.extract_state(body_87, hand_12)
                           v
                 Policy input state       (B, 26)
                 [ left_arm(7) | right_arm(7) | left_hand(6) | right_hand(6) ]
                           |
                           |  ACT policy forward pass (CVAE, 50-step chunk)
                           v
                 Policy action chunk      (B, 50, 26)
                           |
                           |  InspireFTPExperimentConfig.scatter_to_sim(policy_action)
                           v
                 Sim action tensor        (B, 50, 41)
                 [ zeros at non-scatter indices ]
                           |
                           |  env.step(action_41) -> InspireFTPJointPositionAction
                           v
                 Articulation target      (B, 53)
                 [ 41 actuated + 12 mimic computed from actuated ]
```

### Scatter table — `GROUP_SIM_INDICES`

From [`inspire_ftp_experiment_config.py:68–73`](../../scripts/utils/inspire_ftp_experiment_config.py#L68-L73):

| Group | Obs source | Canonical order | 41D sim indices |
|-|-|-|-|
| `left_arm` | `body[15:22]` | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw | `[11, 15, 19, 21, 23, 25, 27]` |
| `right_arm` | `body[22:29]` | (same, right) | `[12, 16, 20, 22, 24, 26, 28]` |
| `left_hand` | `hand[0:6]` | thumb_yaw, thumb_pitch, index, middle, ring, pinky | `[33, 39, 29, 30, 32, 31]` |
| `right_hand` | `hand[6:12]` | (same, right) | `[38, 40, 34, 35, 37, 36]` |

### Why the indices are non-contiguous

The 41D action layout comes from the env's `actuated_joint_names`, which
follows USD tree-traversal order on the full 53-joint articulation with the
12 mimic joints removed. That interleaves body and arm joints (arms share
shoulder pitch/roll/yaw chains with the rest of the body) and leaves gaps in
the hand section where mimic joints used to be. The `GROUP_SIM_INDICES`
table records the result of walking that ordering and writing down where
each policy-dim joint actually lands.

`scatter_to_sim()` itself is dead simple:

```python
def scatter_to_sim(self, policy_action):
    sim = torch.zeros(*policy_action.shape[:-1], 41, ...)
    sim[..., self.sim_scatter_indices] = policy_action
    return sim
```

All non-scatter positions (legs, waist, the 29-body joints the policy does
not control) are left as 0, which for a joint position controller means
"target angle = 0 rad". In practice those joints are held near zero by the
robot's default pose, so this matches the training data.

---

## 5. Mimic Joint Enforcement

> **Code:** [`mimic_action.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py)

The Inspire FTP hand has 6 actuated joints per side, plus 6 mimic joints
that in hardware are mechanically coupled to them via tendons. The sim
recreates that coupling in software via a custom
`InspireFTPJointPositionAction` that subclasses IsaacLab's
`JointPositionAction`.

### Mimic rules

```python
_MIMIC_RULES_PER_SIDE = [
    ("{side}_index_2_joint",   "{side}_index_1_joint",  1.0843),
    ("{side}_middle_2_joint",  "{side}_middle_1_joint", 1.0843),
    ("{side}_ring_2_joint",    "{side}_ring_1_joint",   1.0843),
    ("{side}_little_2_joint",  "{side}_little_1_joint", 1.0843),
    # Thumb is a chain: _3 mimics _2, then _4 mimics _3.
    ("{side}_thumb_3_joint",   "{side}_thumb_2_joint",  0.8024),
    ("{side}_thumb_4_joint",   "{side}_thumb_3_joint",  0.9487),
]
```

- The four fingers each have a single `_2` mimic joint that tracks its `_1`
  parent at 1.0843x — the DIP follows the PIP with a slight amplification.
- The thumb is a two-link chain: `thumb_3` tracks `thumb_2` at 0.8024x, and
  `thumb_4` tracks `thumb_3` at 0.9487x. **Order matters** because the
  second rule reads a value the first rule just computed.

Both rules are replicated for `left` and `right` → 12 mimic joints total.

### Apply order

`apply_actions()`:

1. Calls `super().apply_actions()` → sets position targets for all 41
   joints in the action space.
2. Walks `_mimic_parent_info` (built once in `__init__`) to compute each
   mimic joint's target. Each entry tags the parent as either `"action"`
   (pull from `self.processed_actions[:, parent_idx]`) or `"mimic"` (pull
   from the already-computed mimic buffer, enabling the thumb chain).
3. Calls `self._asset.set_joint_position_target(mimic_vals,
   joint_ids=self._mimic_art_ids)` to write the mimic targets onto the
   articulation in one go.

### What a policy author needs to know

- **The policy never sees mimic joints.** Both training data and inference
  actions are 26D (or 41D in sim), covering only the actuated joints.
- **The scatter does not write to mimic positions either.** `scatter_to_sim`
  targets the 41D actuated action space; mimic joints are written separately
  inside `apply_actions()`, not as part of the policy output.
- **Tendon behavior is baked into the rewards.** Because mimic joints
  respond deterministically to their parents, the IL policy implicitly
  learns "good grip shapes" just by commanding the 6 actuated joints — it
  does not need to reason about DIP or distal thumb joints directly.

---

## 6. Experiment Config Propagation

A recurring pattern in the Inspire pipeline is that the `experiment:` block
of `act_config_inspire_ftp.yaml` — specifically the `cameras:` and
`joint_groups:` lists — is the **single source of truth** for which parts
of the robot and scene the ACT policy consumes. That block has to reach four
different consumers, and it does so through a combination of an env var and
an explicit config path:

```
act_config_inspire_ftp.yaml
    |
    |--- experiment.cameras
    |--- experiment.joint_groups
    |
    +---> (1) IL training (train_act_grasp_policy_inspire.sh)
    |         - Strips experiment: into /tmp/act_config_inspire_XXXX.yaml
    |         - Exports INSPIRE_FTP_EXPERIMENT_CONFIG=<original path>
    |
    +---> (2) IL eval (eval_act_inspire.py)
    |         - Writes experiment_config_path: <original path>
    |           into a temp policy YAML
    |         - ACTClosedloopPolicy.__init__ loads via
    |           InspireFTPExperimentConfig.from_yaml(exp_cfg_path)
    |
    +---> (3) RL obs/action converters (rlinf_ext)
    |         - InspireFTPExperimentConfig.from_env_or_default()
    |         - Reads INSPIRE_FTP_EXPERIMENT_CONFIG
    |
    +---> (4) RL policy wrapper (act_policy.py)
              - Same from_env_or_default() call
```

The load path is:

1. **If `experiment_config_path` is set in the policy YAML** (path 2):
   `InspireFTPExperimentConfig.from_yaml(path)` is called directly.
2. **Else** fall back to `from_env_or_default()` which checks
   `INSPIRE_FTP_EXPERIMENT_CONFIG`; if that is unset, use the hard-coded
   defaults in [`inspire_ftp_experiment_config.py:93–97`](../../scripts/utils/inspire_ftp_experiment_config.py#L93-L97)
   (all 4 joint groups, front camera by default → `observation.images.cam_room`).

`from_env_or_default()` caches its result in a module-level variable, so
the config is read once per process.

This is the reason that, if you train with a custom subset of
`joint_groups` (e.g. right-side only → 13D), you must make sure both
(a) `INSPIRE_FTP_EXPERIMENT_CONFIG` and (b) the eval script's
`experiment_config_path` still point at the same YAML you trained with —
otherwise the state dimensions and scatter indices will not match the
checkpoint.

---

## 7. RLinf Integration

`RLINF_EXT_MODULE=rlinf_ext` tells RLinf at import time to load
[`rlinf_ext/__init__.py`](../../scripts/simulation/rl/rlinf_ext/__init__.py)
and call `register()`. For the Inspire FTP variant the registration
installs three things:

```
register()
   |
   +-- Gym env class
   |     Isaac-Grasp-Policy-G129-InspireFTP-Joint       -> IsaaclabGraspPolicyInspireEnv
   |     Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval  -> IsaaclabGraspPolicyInspireEnv
   |
   +-- Obs converter
   |     "act_inspire_ftp" -> _convert_inspire_obs_to_act_format
   |
   +-- Action converter
         "act_inspire_ftp" -> _convert_inspire_act_action_to_sim
```

| Component | File | Role |
|-|-|-|
| Env class | `rlinf_ext/__init__.py:565-636` | Wraps `IsaaclabBaseEnv`; `_wrap_obs()` builds the 26D critic state and the front-camera image dict |
| Obs converter | `rlinf_ext/__init__.py:653-681` | Takes the wrapped env obs, loads `InspireFTPExperimentConfig`, emits `{observation.state: (B,26), observation.images.cam_room: (B,3,H,W)}` |
| Action converter | `rlinf_ext/__init__.py:684-705` | Takes the `(B, chunk, 26)` policy output, truncates to `chunk_size`, scatters to `(B, chunk, 41)` via `scatter_to_sim_numpy()` |
| RL actor wrapper | `rlinf_ext/act_policy.py` | `ACTForRLActionPrediction` + `ValueHead` — shared between Dex3 and Inspire; RL-specific head on top of the ACT transformer encoder features |
| PPO YAML | `rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml` | Algorithm, env, and actor/critic hyperparams |
| Model YAML | `rlinf_ext/config/model/act_inspire_ftp.yaml` | Selects `obs_converter_type: act_inspire_ftp`, `add_value_head: true` |

The 26D state in `_wrap_obs()` is built as
`cat(body[:, 15:29], inspire_hand, dim=-1)`, giving the same layout the
training-time `extract_state()` produces. This is what keeps IL-trained
weights compatible with the RL actor/critic without any remapping.

As of this writing the Inspire RL path is scaffolded but not yet validated
end to end. Use `eval_act_inspire.py` for day-to-day IL evaluation; come
back to the RL stack once the IL checkpoint is performing as expected.

---

## 8. Task Definition

> **Code:** [`tasks/grasp_policy_inspire/`](../../scripts/simulation/tasks/grasp_policy_inspire/)

### Gym IDs

| Gym ID | Purpose | Env cfg |
|-|-|-|
| `Isaac-Grasp-Policy-G129-InspireFTP-Joint` | IL + RL training, random block placement | `g1_grasp_policy_inspire_env_cfg.py` |
| `Isaac-Grasp-Policy-G129-InspireFTP-Joint-Eval` | Deterministic per-env placement | same file |
| `Isaac-Grasp-Policy-G129-InspireFTP-Teleop` | VR demo recording, 53-joint PinkIK+DexPilot retarget | `g1_grasp_policy_inspire_teleop_env_cfg.py` |

### Scene layout

- **Robot:** G1 29-DoF with Inspire FTP hands, base at `(-1.849, 1.94, 0.812)`
- **Surgical tray:** static prop at `TRAY_POS = (-1.499, 2.034, 0.846)`,
  rotated 90° CW via `TRAY_ROT = (0.707, 0, 0, -0.707)`
- **Tools:** `SINUS_TOOL_USD_PATHS` maps `tool_0..tool_4` to their USD files;
  `--object` on `eval_act_inspire.py` selects which one is the physics-
  enabled grasp target. The selected tool is named `block` in the scene so
  that shared reward/termination code (inherited from the Dex3 task via
  `simulation.tasks.grasp_policy.mdp.*`) works unchanged.
- **Tray slots:** 6 slots (`TRAY_SLOT_POSITIONS[0..5]`) defined by local
  offsets from Xform markers baked into the tray USD.
  `ACTIVE_SLOT_IDX = 4` by default; `--slot N` on the eval CLI overrides
  both the initial block spawn and the reset event's slot position.
- **Tool orientation:** `TOOL_ROT = (0.707, 0, 0, -0.707)` (matches the
  tray rotation).
- **Bin / target pad:** 15cm x 15cm x 0.5cm pad at `(-1.55, 1.61, 0.835)`
- **Cameras:** `front_camera` + `left_wrist_camera` + `right_wrist_camera` at 480x640 RGB

### Observations

```python
class PolicyCfg(ObsGroup):
    robot_joint_state         = ObsTerm(func=mdp.get_robot_body_joint_states)       # (B, 87)
    robot_inspire_joint_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)    # (B, 12)

class CameraImagesCfg(ObsGroup):
    front_camera = ObsTerm(func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb", "normalize": False})
```

`concatenate_terms = False` on both groups, so each term is accessible by
its own key in the observation dict.

### Actions

```python
joint_pos = mdp.InspireFTPJointPositionActionCfg(
    asset_name="robot",
    joint_names=actuated_joint_names,   # 41 entries
    scale=1.0,
    use_default_offset=False,
    offset=offset_dict,                 # {-0.3 on left/right_elbow_joint}
    preserve_order=True,
)
```

- `scale=1.0` means the action dimension is "target joint angle in rad",
  no rescaling.
- `offset` only biases the two elbows.
- **There is no clamping anywhere in the action pipeline.** If a policy
  outputs a value outside the robot's joint limits, the PD controller on
  that joint will apply large corrective torques. This is why
  `eval_act_inspire.py` exposes `--log_actions` (to inspect per-chunk action
  stats) and `--clamp_actions VAL` (to `np.clip(chunk, -VAL, VAL)` as a
  diagnostic).

### Rewards and success

The Inspire FTP task reuses the reward/termination MDP from the Dex3 grasp
task (`scripts/simulation/tasks/grasp_policy/mdp/`). The success signal is
a 3-stage state machine:

| Stage | Transition | Condition | Reward |
|-|-|-|-|
| 0 → 1 | Lift | Tool z > table + 5cm | +1.0 |
| 1 → 2 | Transport | Tool x,y inside bin bounds | +1.0 |
| 2 → 3 | Place | Tool x,y inside bin AND below rim AND above floor | +1.0 |

`check_success(env, success_stage=3)` in `examples/utils.py` reads
`env._task_stage` and returns True when it reaches 3. That is the condition
`eval_act_inspire.py` uses to end an episode as "SUCCESS".

### Terminations

[`grasp_policy_inspire/mdp/terminations.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/terminations.py)
is a thin re-export of the Dex3 versions:

- `time_out` — episode length exceeded (episode_length_s = 200.0 s →
  ~10000 sim steps at 50 Hz)
- `task_success_termination(success_stage=3)`
- `object_drop_termination(drop_height_threshold=0.5)`

---

## 9. Docker

The Inspire FTP pipeline uses the same container as the Dex3 grasp
workflow. There is no `-inspire` build flag — variant selection is entirely
runtime, via Python configs.

| | Value |
|-|-|
| Dockerfile | [`docker/Dockerfile.grasp`](../../docker/Dockerfile.grasp) |
| Run script | [`docker/run_docker_grasp.sh`](../../docker/run_docker_grasp.sh) |
| LeRobot | Always installed (pinned commit) |
| GR00T | Not installed |
| Prompt | `[GRASP]` |

### Mounts

| Host | Container | Purpose |
|-|-|-|
| `~/datasets` | `/datasets` | Recorded HDF5 + converted LeRobot datasets |
| `~/models` | `/models` | Trained checkpoints |
| `~/eval` | `/eval` | Evaluation output (if you write there) |
| `i4h-workflows/` | `/workspaces` | Live source mount — edits on the host apply immediately |

### Video decoding

The training launcher passes `--dataset.video_backend pyav` because the
default `torchcodec` backend needs `libnvrtc.so.13`, which is not present
in the grasp image. `pyav` uses CPU FFmpeg decoding and works without
additional CUDA libs.

### Typical one-liners

```bash
# Smoke test eval with dummy zero-action policy (no checkpoint needed)
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --test --max_steps 200 --enable_cameras

# Real eval with an IL-trained Inspire checkpoint + action diagnostics
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/\
act_grasp_policy_inspire/train_<timestamp>/checkpoints/<step>/pretrained_model \
    --num_episodes 3 --max_steps 5000 --slot 4 --enable_cameras \
    --log_actions --save_video

# Train ACT on an existing Inspire LeRobot dataset
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy_inspire.sh \
        --dataset_path /datasets/grasp_policy_inspire_lerobot
```

---

## 10. Key Files Reference

### Task

| File | Description |
|-|-|
| [`scripts/simulation/tasks/grasp_policy_inspire/__init__.py`](../../scripts/simulation/tasks/grasp_policy_inspire/__init__.py) | Gym ID registration (Joint, Joint-Eval, Teleop) |
| [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py) | Scene, robot, tray, tool, bin, observation and action config |
| [`scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py) | 53-joint teleop env with PinkIK wrist + DexPilot hand retarget |
| [`scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py) | `get_robot_body_joint_states` (87D), `get_robot_inspire_joint_states` (12D) |
| [`scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py) | `InspireFTPJointPositionAction` + mimic rules |
| [`scripts/simulation/tasks/grasp_policy_inspire/mdp/terminations.py`](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/terminations.py) | Re-exports `object_drop_termination` and `task_success_termination` |

### Data

| File | Description |
|-|-|
| [`scripts/utils/convert_hdf5_to_lerobot.py`](../../scripts/utils/convert_hdf5_to_lerobot.py) | HDF5 → LeRobot v2.1; the `rheo_26d_state_action` flag triggers the Inspire branch |
| [`scripts/config/g1_grasp_policy_inspire_dataset.yaml`](../../scripts/config/g1_grasp_policy_inspire_dataset.yaml) | Inspire-specific dataset conversion config |

### Experiment + training

| File | Description |
|-|-|
| [`scripts/utils/inspire_ftp_experiment_config.py`](../../scripts/utils/inspire_ftp_experiment_config.py) | `InspireFTPExperimentConfig` dataclass — joint groups, cameras, `extract_state()`, `scatter_to_sim()`, `from_env_or_default()` |
| [`scripts/policy/act_config_inspire_ftp.yaml`](../../scripts/policy/act_config_inspire_ftp.yaml) | Single source of truth: experiment block + ACT architecture + training hyperparams |
| [`scripts/policy/train_act_grasp_policy_inspire.sh`](../../scripts/policy/train_act_grasp_policy_inspire.sh) | Training launcher (strips experiment block, exports env var, calls LeRobot) |

### Eval

| File | Description |
|-|-|
| [`scripts/simulation/examples/eval_act_inspire.py`](../../scripts/simulation/examples/eval_act_inspire.py) | Primary ACT-on-Inspire evaluation script |
| [`scripts/simulation/act_closedloop_policy.py`](../../scripts/simulation/act_closedloop_policy.py) | `ACTClosedloopPolicy` — loads the checkpoint, dispatches on `hand_type="inspire_ftp"`, exposes `get_action_from_raw()` |
| [`scripts/simulation/examples/eval_grasp_policy_inspire.py`](../../scripts/simulation/examples/eval_grasp_policy_inspire.py) | Legacy unified evaluator (still works, harder to debug) |
| [`scripts/simulation/examples/utils.py`](../../scripts/simulation/examples/utils.py) | `check_success()`, `set_viewport_camera()`, `_MultiViewConcatWriter` (reused by `eval_act_inspire.py`) |

### RL

| File | Description |
|-|-|
| [`scripts/simulation/rl/rlinf_ext/__init__.py`](../../scripts/simulation/rl/rlinf_ext/__init__.py) | `register()` — Inspire gym IDs, env class, obs/action converters |
| [`scripts/simulation/rl/rlinf_ext/act_policy.py`](../../scripts/simulation/rl/rlinf_ext/act_policy.py) | `ACTForRLActionPrediction` + `ValueHead` (shared with Dex3) |
| [`scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml`](../../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml) | PPO + env + model YAML |
| [`scripts/simulation/rl/rlinf_ext/config/model/act_inspire_ftp.yaml`](../../scripts/simulation/rl/rlinf_ext/config/model/act_inspire_ftp.yaml) | Selects `obs_converter_type: act_inspire_ftp` |

### Docker

| File | Description |
|-|-|
| [`docker/Dockerfile.grasp`](../../docker/Dockerfile.grasp) | Grasp image (LeRobot, no GR00T) |
| [`docker/run_docker_grasp.sh`](../../docker/run_docker_grasp.sh) | Launcher; same image for Dex3 and Inspire |
