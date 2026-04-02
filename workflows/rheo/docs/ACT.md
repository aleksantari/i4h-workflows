<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ACT Pipeline Architecture

Technical reference for the ACT (Action Chunking Transformer) imitation learning
pipeline in the rheo workflow. For step-by-step usage instructions, see
[`grasp_policy_guide.md`](grasp_policy_guide.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [ACT Model Architecture](#2-act-model-architecture)
3. [Pipeline Stages](#3-pipeline-stages)
4. [Shared Infrastructure with GR00T](#4-shared-infrastructure-with-groot)
5. [Joint Dimension Flow](#5-joint-dimension-flow)
6. [Docker Setup](#6-docker-setup)
7. [RLinf Integration Pattern](#7-rlinf-integration-pattern)
8. [Configuration Reference](#8-configuration-reference)
9. [Task Definition](#9-task-definition)
10. [Key Files Reference](#10-key-files-reference)

---

## 1. Overview

ACT is a CVAE-based imitation learning policy that predicts multi-step action
chunks from visual and proprioceptive observations. In the rheo pipeline it
serves as an alternative to GR00T for precision manipulation tasks, specifically
the **grasp_policy** task (pick up a block and place it in a bin) using the G1
robot with Dex3 hands.

### End-to-End Pipeline

```
Teleoperate (AVP)
    |
    v
record_demos.py -----> HDF5 (87D body + 14D hands + 3 cameras + 43D WBC actions)
    |
    v
convert_hdf5_to_lerobot.py -----> LeRobot v2.1 (parquet + MP4 video, 28D state/action)
    |
    v
train_act_grasp_policy.sh -----> lerobot.scripts.train (ACT IL)
    |
    v
eval_grasp_policy.py --policy_type act -----> ACTClosedloopPolicy (IL evaluation)
    |
    v
rl/train_act_grasp_policy.sh -----> RLinf PPO (RL post-training with ValueHead)
    |
    v
eval_grasp_policy.py --policy_type act -----> Final evaluation
```

Both ACT and GR00T share the same data collection pipeline, task environments,
evaluation loop, and RLinf RL framework. The key difference is the model
architecture and training toolchain (LeRobot for ACT vs GR00T SFT for GR00T).

---

## 2. ACT Model Architecture

ACT (Action Chunking with Transformers) is a Conditional Variational Autoencoder
(CVAE) that predicts a sequence of future actions from a single observation.

### Input Modalities

| Input | Shape | Source |
|-------|-------|--------|
| Joint state | `(B, 28)` | `left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7)` |
| Front camera | `(B, 3, 480, 640)` | Room-view RGB (ResNet18 backbone) |
| Left wrist camera | `(B, 3, 480, 640)` | Left wrist-mounted RGB |
| Right wrist camera | `(B, 3, 480, 640)` | Right wrist-mounted RGB |

### Output

| Output | Shape | Description |
|--------|-------|-------------|
| Action chunk | `(B, 100, 28)` | 100 future joint position targets (arms + hands) |

### Architecture Components

```
                     Observation
                    /     |     \
          ResNet18   ResNet18   ResNet18       Joint state (28D)
          (front)   (left)     (right)             |
              \        |         /                 |
               Visual tokens (3 x 512D)      Linear projection
                        \                        /
                         Transformer Encoder (4 layers)
                                  |
                           Latent z ~ N(mu, sigma)    [VAE: 32-dim]
                                  |
                         Transformer Decoder (1 layer)
                                  |
                         Action sequence (100 x 28D)
```

### Model Parameters

| Component | Parameters | Details |
|-----------|-----------|---------|
| 3x ResNet18 backbones | ~33M | Pretrained ImageNet weights, shared architecture |
| Transformer encoder | ~8M | 4 layers, 512-dim, 8 heads, 3200 FFN |
| VAE encoder | ~4M | 4 layers, 32-dim latent |
| Transformer decoder | ~4M | 1 layer |
| Linear projections | ~3M | Input/output embeddings |
| **Total** | **~52M** | |

### Training Loss

```
L = L_reconstruction + kl_weight * D_KL(q(z|obs,action) || p(z|obs))
```

- `L_reconstruction`: L1 loss between predicted and ground-truth action chunks
- `kl_weight`: 10.0 (strong regularization, encourages diverse behaviors)
- At inference, z is sampled from the prior `p(z|obs)` (no teacher forcing)

---

## 3. Pipeline Stages

### 3.1 Data Collection

> **Code:** [`scripts/simulation/record_demos.py`](../scripts/simulation/record_demos.py)

Records teleoperated demonstrations in HDF5 format. The operator controls the
robot via AVP hand tracking (or keyboard), while IsaacLab records observations
and the Whole-Body Controller's output actions.

**HDF5 structure per episode:**

```
episode_NNNNNN/
  obs/
    robot_joint_state         (T, 87)    Full-body joint positions
    robot_dex3_joint_state    (T, 14)    Dex3 hand joint positions
    front_camera              (T, H, W, 4)  RGBA images
    left_wrist_camera         (T, H, W, 4)  RGBA images
    right_wrist_camera        (T, H, W, 4)  RGBA images
  processed_actions           (T, 43)    WBC + PINK IK output
  metadata/
    success                   bool
```

Camera observations are recorded at simulation framerate (30 Hz) and stored as
raw uint8 arrays. The `processed_actions` capture the full 43D WBC output
(legs + waist + arms + hands).

Auto-success detection saves episodes after N consecutive frames where the
block reaches the target stage, reducing manual annotation.

### 3.2 Dataset Conversion

> **Code:** [`scripts/utils/convert_hdf5_to_lerobot.py`](../scripts/utils/convert_hdf5_to_lerobot.py)

Converts HDF5 demos to LeRobot v2.1 format, which the LeRobot training pipeline
expects.

**Key transformations:**

| From (HDF5) | To (LeRobot) | Operation |
|-------------|-------------|-----------|
| `robot_joint_state[15:29]` (14D) + `robot_dex3_joint_state` (14D) | `observation.state` (28D) | Slice + concatenate |
| `processed_actions` (43D) | `action` (28D) | Extract arm+hand joints only |
| Camera RGBA arrays | MP4 video files | Multiprocess video encoding |
| Episode arrays | Parquet files | One file per episode in `data/chunk-000/` |

**Output structure:**

```
lerobot/
  data/chunk-000/
    episode_000000.parquet    28D state, 28D action, timestamps
    episode_000001.parquet
    ...
  videos/chunk-000/
    observation.images.cam_room/
      episode_000000.mp4
    observation.images.cam_left_wrist/
      episode_000000.mp4
    observation.images.cam_right_wrist/
      episode_000000.mp4
  meta/
    info.json                 Feature metadata (shapes, dtypes, FPS)
    episodes.jsonl            Episode lengths
    tasks.jsonl               Task descriptions
    episodes_stats.jsonl      Per-episode statistics (for normalization)
```

### 3.3 IL Training

> **Code:** [`scripts/policy/train_act_grasp_policy.sh`](../scripts/policy/train_act_grasp_policy.sh),
> [`scripts/policy/act_config.yaml`](../scripts/policy/act_config.yaml)

A bash wrapper around LeRobot's native training pipeline
(`python -m lerobot.scripts.train`). The wrapper handles:

1. CLI argument parsing (`--dataset_path`, `--resume_path`)
2. Timestamped output directory creation
3. Auto-generation of `episodes_stats.jsonl` if missing (required by LeRobot v2.1)
4. PYTHONPATH setup for the rheo workspace

The training loop is entirely LeRobot's: it loads the dataset, constructs the
ACT model from config, runs AdamW optimization, and saves checkpoints.

**Config system:** LeRobot uses **draccus** (not Hydra). CLI overrides use
`--key value` format (e.g., `--steps 50000`), and config values map directly to
dataclass fields (`TrainPipelineConfig`, `ACTConfig`, `DatasetConfig`).

### 3.4 IL Evaluation

> **Code:** [`scripts/simulation/examples/eval_grasp_policy.py`](../scripts/simulation/examples/eval_grasp_policy.py),
> [`scripts/simulation/act_closedloop_policy.py`](../scripts/simulation/act_closedloop_policy.py)

The unified evaluator (`eval_grasp_policy.py`) supports `--policy_type act` to
load an ACT checkpoint via `ACTClosedloopPolicy`. The evaluation loop:

1. Creates the IsaacLab gym environment (`Isaac-Grasp-Policy-G129-Dex3-Joint`)
2. Loads the ACT checkpoint via `ACTPolicy.from_pretrained()`
3. Runs episodes: extracts observations, feeds to ACT, pads 28D->43D, steps env
4. Tracks success rate across episodes (3-stage reward completion)

`ACTClosedloopPolicy` implements `PolicyBase` (from `isaaclab_arena`) and manages
the action chunk lifecycle: it requests a new 100-step chunk when the current one
is exhausted, returning one action per simulation step.

### 3.5 RL Post-Training

> **Code:** [`scripts/simulation/rl/rlinf_ext/act_policy.py`](../scripts/simulation/rl/rlinf_ext/act_policy.py),
> [`scripts/simulation/rl/train_act_grasp_policy.sh`](../scripts/simulation/rl/train_act_grasp_policy.sh)

RL post-training uses PPO (via RLinf) to refine the IL-trained ACT policy with
environment rewards. The key adaptation:

- **`ACTForRLActionPrediction`** wraps the LeRobot ACTPolicy into RLinf's
  `BasePolicy` interface
- A **`ValueHead`** (3-layer MLP, 256-dim hidden) is attached for critic
  estimation, using features from ACT's transformer encoder
- The ACT backbone can be frozen or fine-tuned at a lower learning rate (5e-6)
  while the value head trains at a higher rate (1e-4)

**RL training setup:**

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO with GAE |
| Environments | 64 parallel |
| Epochs | 100 |
| Sequence length | 4096 |
| Actor LR | 5e-6 |
| Value head LR | 1e-4 |
| Discount | 0.99 |
| GAE lambda | 0.95 |

---

## 4. Shared Infrastructure with GR00T

The ACT pipeline reuses most of the rheo infrastructure originally built for
GR00T. The table below shows what is shared and what is specialized.

| Component | Shared? | Notes |
|-----------|---------|-------|
| **Demo recording** | Fully shared | `record_demos.py` is policy-agnostic; same HDF5 structure feeds both pipelines |
| **Teleop devices** | Fully shared | AVP hand tracking outputs 16D gripper+wrist commands, independent of downstream policy |
| **Dataset conversion** | Fully shared | `convert_hdf5_to_lerobot.py` converts to 28D state/action for both |
| **Task environments** | Fully shared | Same gym IDs (`Isaac-Grasp-Policy-G129-Dex3-*`), rewards, terminations |
| **Observation schema** | Same raw obs | Both extract from `robot_joint_state` (87D) + `robot_dex3_joint_state` (14D) + 3 cameras |
| **43D padding** | Identical logic | Both prepend 15 zeros (legs/waist) to 28D arm+hand actions |
| **RLinf registration** | Same pattern | Both use plugin architecture: obs converter + action converter + model factory |
| **Eval loop** | Shared | `eval_grasp_policy.py` dispatches on `--policy_type` |
| **Action chunking** | Pattern duplicated | Both implement identical chunk state management, but `ACTClosedloopPolicy` does not inherit from `BaseClosedloopPolicy` |
| **ObsProcessor** | Available, unused | `obs_processor.py` provides a model-agnostic `ProcessedObservation` dataclass; both policies implement their own extraction |
| **Joint remapping** | GR00T only | GR00T uses config-based `remap_policy_joints_to_sim_joints()`; ACT uses direct index slicing |
| **Docker image** | Separate | ACT uses `Dockerfile.grasp` (LeRobot, no GR00T); GR00T uses `Dockerfile.x86` |
| **Config system** | Different | ACT: LeRobot draccus YAML; GR00T: custom YAML configs in `scripts/config/` |
| **Model loading** | Different | ACT: `ACTPolicy.from_pretrained()`; GR00T: `Gr00tPolicy` + optional TensorRT |

### Design Decision: Why ACT Duplicates BaseClosedloopPolicy

`BaseClosedloopPolicy` defines the abstract interface (`_load_model()`,
`_get_action_chunk()`) and shared action chunking logic. However,
`ACTClosedloopPolicy` reimplements the chunking state rather than inheriting
from it. This is because:

1. ACT's `_load_policy()` signature differs from `_load_model()` (takes config
   path instead of returning model)
2. ACT needs per-env sequential inference (`select_action` processes one env at
   a time), while the base class assumes batched forward passes
3. The chunk truncation/padding logic (matching `action_chunk_length` to ACT's
   `chunk_size`) is ACT-specific

The GR00T wrapper (`CustomGr00tClosedloopPolicy`) also does not inherit from
`BaseClosedloopPolicy` for similar reasons: it manages its own joint remapping
pipeline that doesn't fit the base class's return type assumptions.

---

## 5. Joint Dimension Flow

The G1 robot has 43 controllable joints. ACT operates on a 28D subset
(arms + hands), with legs and waist held at zero.

```
IsaacLab environment
    robot_joint_state: 87D (full body including fingers)
    robot_dex3_joint_state: 14D (Dex3 hand joints)
            |
    ACTExperimentConfig.extract_state(body_87d, dex3_14d)
    Selects configured joint groups (default: all 4)
            |
    ACT input/output: policy_dim (default 28D, configurable)
    [left_arm(7) | right_arm(7) | left_hand(7) | right_hand(7)]
            |
    ACTExperimentConfig.scatter_to_sim(policy_action)
    Places joints at correct 43D positions
            |
    Simulator action: 43D
    [legs(10) | waist(5) | left_arm(7) | right_arm(7) | left_hand(7) | right_hand(7)]
```

### Joint Group → Sim Position Mapping

| Joint Group | Body State Indices | Dex3 Indices | 43D Sim Positions |
|-------------|-------------------|--------------|-------------------|
| `left_arm` | 15–21 | — | 15–21 |
| `right_arm` | 22–28 | — | 22–28 |
| `left_hand` | — | 0–6 | 29–35 |
| `right_hand` | — | 7–13 | 36–42 |

When using a subset (e.g., `[right_arm, right_hand]`), `extract_state()` selects
only those groups from the raw observations (producing 14D), and
`scatter_to_sim()` places the 14D output at positions [22–28, 36–42] in the 43D
action tensor, with zeros elsewhere.

### Why ACT Does Not Need Joint Remapping

GR00T's internal joint ordering differs from the simulator's 43D ordering, so it
requires config-driven remapping via `remap_policy_joints_to_sim_joints()` in
[`joint_conversion.py`](../scripts/utils/joint_conversion.py).

ACT's output is trained on data that was extracted in the same fixed order
(`body_state[15:29]` + `dex3_state`), so the mapping is handled by
`ACTExperimentConfig.scatter_to_sim()` which places each group at its known
sim positions. No external remapping configuration is needed.

---

## 6. Docker Setup

The ACT pipeline uses a dedicated Docker image to avoid dependency conflicts
between LeRobot and GR00T.

### Image Comparison

| | `grasp-policy` (ACT) | `rheo` (GR00T) |
|-|---------------------|----------------|
| **Dockerfile** | [`Dockerfile.grasp`](../docker/Dockerfile.grasp) | [`Dockerfile.x86`](../docker/Dockerfile.x86) |
| **Run script** | [`run_docker_grasp.sh`](../docker/run_docker_grasp.sh) | `run_docker.sh -g1.5` |
| **LeRobot** | Always installed (pinned commit) | Optional (`--build-arg INSTALL_LEROBOT=true`) |
| **GR00T** | Not installed | Optional (`--build-arg INSTALL_GROOT=true`) |
| **Prompt** | `[GRASP]` | `[RHEO]` |

### Container Mounts

Both images share the same host mounts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `~/datasets` | `/datasets` | Recorded HDF5 demos + converted datasets |
| `~/models` | `/models` | Trained model checkpoints |
| `~/eval` | `/eval` | Evaluation results and videos |
| `i4h-workflows/` | `/workspaces` | Live code editing (source volume mount) |

### Video Decoding

The default video decoder (`torchcodec`) requires `libnvrtc.so.13` which is not
present in the grasp-policy image. The training script overrides this with
`--dataset.video_backend pyav`, which uses CPU-based FFmpeg decoding instead.

---

## 7. RLinf Integration Pattern

Both ACT and GR00T integrate with RLinf through a plugin system defined in
[`rlinf_ext/__init__.py`](../scripts/simulation/rl/rlinf_ext/__init__.py).
The `RLINF_EXT_MODULE=rlinf_ext` environment variable triggers the `register()`
function at import time.

### Registration Architecture

```
RLINF_EXT_MODULE=rlinf_ext
        |
    rlinf_ext/__init__.py:register()
        |
        +-- Register gym IDs
        |     Isaac-Grasp-Policy-G129-Dex3-Joint
        |     Isaac-Grasp-Policy-G129-Dex3-Joint-Eval
        |
        +-- Register obs converters
        |     "dex3" --> _convert_dex3_obs_to_gr00t_format()  [GR00T]
        |     "act"  --> _convert_dex3_obs_to_act_format()    [ACT]
        |
        +-- Register action converters
        |     "dex3" --> _convert_to_dex3_action()            [GR00T]
        |     "act"  --> _convert_act_action_to_sim()         [ACT]
        |
        +-- Register model factories
              "new_embodiment" --> patched get_model()         [GR00T]
              "act"            --> act_policy.get_model()      [ACT]
```

### Converter Comparison

**Observation conversion** — both converters receive the same RLinf observation
format and produce model-specific input dicts:

| Step | GR00T (`"dex3"`) | ACT (`"act"`) |
|------|-----------------|---------------|
| Images | Map to `video.room_view`, `video.left_wrist_view`, etc. | Map to `observation.images.cam_room`, `observation.images.cam_left_wrist`, etc. |
| Format | Keep (B,H,W,C) uint8 | Convert to (B,C,H,W) float32 [0,1] |
| State | Split into `state.left_arm`, `state.right_arm`, etc. | Flatten to `observation.state` (28D) |

**Action conversion** — both converters pad 28D policy output to 43D sim
actions by prepending 15 zeros for legs/waist. The logic is identical.

### Model Config

RLinf selects model type via YAML config:

- **ACT:** [`config/model/act_dex3.yaml`](../scripts/simulation/rl/rlinf_ext/config/model/act_dex3.yaml) — `model_type: "act"`, `obs_converter_type: "act"`, `add_value_head: true`
- **GR00T:** [`config/model/gr00t_dex3.yaml`](../scripts/simulation/rl/rlinf_ext/config/model/gr00t_dex3.yaml) — `model_type: "gr00t"`, `obs_converter_type: "dex3"`, `embodiment_tag: "new_embodiment"`

---

## 8. Configuration Reference

### Experiment Config (Camera & Joint Selection)

> **Code:** [`scripts/utils/act_experiment_config.py`](../scripts/utils/act_experiment_config.py)

The `experiment:` section in `act_config.yaml` is the single source of truth for
which cameras and joint groups the ACT pipeline uses. All downstream consumers
(IL eval, RL obs/action converters, RL policy wrapper) read from this config via
the `ACTExperimentConfig` dataclass.

```yaml
experiment:
  cameras:
    front_camera: "observation.images.cam_room"
    # left_wrist_camera: "observation.images.cam_left_wrist"  # comment out to exclude
    right_wrist_camera: "observation.images.cam_right_wrist"
  joint_groups:
    # - left_arm
    - right_arm
    # - left_hand
    - right_hand
```

When changing the experiment config, also update the `input_features` and
`output_features` shapes in the same YAML to match:

| Experiment | `observation.state` shape | `action` shape | Camera entries |
|------------|--------------------------|----------------|----------------|
| Default (all) | `[28]` | `[28]` | 3 cameras |
| Right-side only | `[14]` | `[14]` | 2 cameras (remove left wrist) |
| Arms only | `[14]` | `[14]` | 3 cameras |

**How the config propagates:**

| Stage | Consumer | Mechanism |
|-------|----------|-----------|
| IL training | `train_act_grasp_policy.sh` | Exports `ACT_EXPERIMENT_CONFIG` env var |
| IL eval | `ACTClosedloopPolicy` | Reads `experiment_config_path` from eval YAML |
| RL training | `rlinf_ext` converters | `ACTExperimentConfig.from_env_or_default()` |
| RL policy | `act_policy.py` | `ACTExperimentConfig.from_env_or_default()` |

**Key methods on `ACTExperimentConfig`:**

- `extract_state(body_87d, dex3_14d)` — selects configured joint groups from raw env observations
- `scatter_to_sim(policy_action)` — places policy-dim actions at correct 43D sim positions
- `rlinf_state_keys()` — returns RLinf state key names for configured groups
- `rlinf_video_keys()` — returns RLinf video key to ACT feature key mapping

### act_config.yaml

> **Code:** [`scripts/policy/act_config.yaml`](../scripts/policy/act_config.yaml)

Top-level training config passed to `lerobot.scripts.train`. All values can be
overridden via CLI (`--key value`).

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 100,000 | Total gradient updates |
| `batch_size` | 64 | Samples per step |
| `log_freq` | 250 | Steps between log entries |
| `save_freq` | 25,000 | Steps between checkpoint saves |
| `eval_freq` | 10,000 | Steps between eval runs |
| `num_workers` | 4 | DataLoader workers |
| `seed` | 1000 | Random seed |

#### Policy Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 100 | Actions predicted per forward pass |
| `n_obs_steps` | 1 | Observation history length |
| `dim_model` | 512 | Transformer hidden dimension |
| `n_heads` | 8 | Attention heads |
| `dim_feedforward` | 3200 | FFN intermediate dimension |
| `n_encoder_layers` | 4 | Transformer encoder depth |
| `n_decoder_layers` | 1 | Transformer decoder depth |
| `use_vae` | true | Enable CVAE (disable for deterministic ACT) |
| `latent_dim` | 32 | VAE latent space dimension |
| `n_vae_encoder_layers` | 4 | VAE encoder depth |
| `kl_weight` | 10.0 | KL divergence loss weight |
| `vision_backbone` | resnet18 | CNN backbone for image features |
| `pretrained_backbone_weights` | ResNet18_Weights.IMAGENET1K_V1 | Backbone initialization |
| `dropout` | 0.1 | Transformer dropout (default, not in config) |

#### Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer_lr` | 1e-5 | Learning rate for transformer heads |
| `optimizer_lr_backbone` | 1e-5 | Learning rate for ResNet backbones |
| `optimizer_weight_decay` | 1e-4 | AdamW weight decay |

### CLI Override Examples

```bash
# Quick sanity check (1000 steps, small batch)
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /workspaces/workflows/rheo/datasets/grasp_policy/demo/lerobot \
    --steps 1000 --batch_size 8 --log_freq 50

# Full training with more frequent checkpoints
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /workspaces/workflows/rheo/datasets/grasp_policy/demo/lerobot \
    --save_freq 10000

# Shorter action chunks (for short-horizon tasks)
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /workspaces/workflows/rheo/datasets/grasp_policy/demo/lerobot \
    --policy.chunk_size 50

# Resume from checkpoint
./docker/run_docker_grasp.sh \
    bash scripts/policy/train_act_grasp_policy.sh \
    --dataset_path /workspaces/workflows/rheo/datasets/grasp_policy/demo/lerobot \
    --resume_path /workspaces/.../checkpoint_050000
```

---

## 9. Task Definition

> **Code:** [`scripts/simulation/tasks/grasp_policy/`](../scripts/simulation/tasks/grasp_policy/)

### Gym Variants

| Gym ID | Purpose | Block Placement |
|--------|---------|----------------|
| `Isaac-Grasp-Policy-G129-Dex3-Joint` | RL training | Random within +/-3cm |
| `Isaac-Grasp-Policy-G129-Dex3-Joint-Eval` | RL evaluation | Deterministic per-env |
| `Isaac-Grasp-Policy-G129-Dex3-Teleop` | VR demo recording | Fixed, extended episode |

### Scene Layout

- **Robot:** G1 with Dex3 hands, standing position at (-1.85, 1.94, 0.81)
- **Block:** 5cm cube, ~100g, initial position (-1.55, 1.90, 0.885)
- **Bin/Target:** 15x15cm pad on table at (-1.55, 1.61, 0.835)
- **Cameras:** Front (room view), left wrist, right wrist — all 480x640 RGB

### 3-Stage Sparse Rewards

> **Code:** [`scripts/simulation/tasks/grasp_policy/mdp/rewards.py`](../scripts/simulation/tasks/grasp_policy/mdp/rewards.py)

The reward function uses a state machine with 3 stages. Each stage must be
completed before the next one triggers.

| Stage | Transition | Condition | Reward |
|-------|-----------|-----------|--------|
| 0 -> 1 | **Grasp** | Block lifted >5cm above table | +1.0 |
| 1 -> 2 | **Transport** | Block positioned over bin bounds (x/y) | +1.0 |
| 2 -> 3 | **Place** | Block inside bin (z < rim, z > floor, x/y in bounds) | +1.0 |

Total maximum reward per episode: **3.0**

### Termination Conditions

> **Code:** [`scripts/simulation/tasks/grasp_policy/mdp/terminations.py`](../scripts/simulation/tasks/grasp_policy/mdp/terminations.py)

| Condition | Trigger |
|-----------|---------|
| `time_out` | Episode length exceeded |
| `success` | Block reaches stage 3 |
| `object_drop` | Block falls below 0.5m threshold |

---

## 10. Key Files Reference

| Component | File | Description |
|-----------|------|-------------|
| **Task registration** | [`scripts/simulation/tasks/grasp_policy/__init__.py`](../scripts/simulation/tasks/grasp_policy/__init__.py) | Gym ID registration (Joint, Joint-Eval, Teleop) |
| **Task env config** | [`scripts/simulation/tasks/grasp_policy/g1_grasp_policy_env_cfg.py`](../scripts/simulation/tasks/grasp_policy/g1_grasp_policy_env_cfg.py) | Scene, robot, block, bin configuration |
| **MDP: observations** | [`scripts/simulation/tasks/grasp_policy/mdp/observations.py`](../scripts/simulation/tasks/grasp_policy/mdp/observations.py) | 87D body + 14D hands + 3 cameras |
| **MDP: rewards** | [`scripts/simulation/tasks/grasp_policy/mdp/rewards.py`](../scripts/simulation/tasks/grasp_policy/mdp/rewards.py) | 3-stage sparse rewards |
| **MDP: terminations** | [`scripts/simulation/tasks/grasp_policy/mdp/terminations.py`](../scripts/simulation/tasks/grasp_policy/mdp/terminations.py) | Success, timeout, object drop |
| **Demo recording** | [`scripts/simulation/record_demos.py`](../scripts/simulation/record_demos.py) | Policy-agnostic HDF5 recording |
| **Demo recording wrapper** | [`scripts/simulation/record_demos_grasp_policy.sh`](../scripts/simulation/record_demos_grasp_policy.sh) | Convenience wrapper with grasp_policy defaults |
| **Data conversion** | [`scripts/utils/convert_hdf5_to_lerobot.py`](../scripts/utils/convert_hdf5_to_lerobot.py) | HDF5 to LeRobot v2.1 (parquet + MP4) |
| **Dataset config** | [`scripts/config/g1_grasp_policy_dataset.yaml`](../scripts/config/g1_grasp_policy_dataset.yaml) | HDF5-to-LeRobot camera mappings |
| **Field mapping** | [`scripts/utils/assemble_trocar_lerobot_fields.py`](../scripts/utils/assemble_trocar_lerobot_fields.py) | 28D joint extraction logic |
| **Experiment config** | [`scripts/utils/act_experiment_config.py`](../scripts/utils/act_experiment_config.py) | Config-driven camera + joint group selection |
| **IL training config** | [`scripts/policy/act_config.yaml`](../scripts/policy/act_config.yaml) | ACT architecture + training hyperparams + experiment config |
| **IL training launcher** | [`scripts/policy/train_act_grasp_policy.sh`](../scripts/policy/train_act_grasp_policy.sh) | Bash wrapper around `lerobot.scripts.train` |
| **IL eval entry point** | [`scripts/simulation/examples/eval_grasp_policy.py`](../scripts/simulation/examples/eval_grasp_policy.py) | Unified evaluator (`--policy_type gr00t\|act\|test`) |
| **ACT inference wrapper** | [`scripts/simulation/act_closedloop_policy.py`](../scripts/simulation/act_closedloop_policy.py) | ACTClosedloopPolicy (28D->43D, action chunking) |
| **ACT eval config** | [`scripts/config/g1_act_closedloop_grasp_policy.yaml`](../scripts/config/g1_act_closedloop_grasp_policy.yaml) | ACT inference YAML (model path, chunk length) |
| **Base policy** | [`scripts/simulation/base_closedloop_policy.py`](../scripts/simulation/base_closedloop_policy.py) | Action chunking base class (not currently inherited) |
| **Obs processor** | [`scripts/simulation/obs_processor.py`](../scripts/simulation/obs_processor.py) | Model-agnostic observation extraction |
| **Joint conversion** | [`scripts/utils/joint_conversion.py`](../scripts/utils/joint_conversion.py) | 43-DOF remapping (GR00T only) |
| **RL ACT wrapper** | [`scripts/simulation/rl/rlinf_ext/act_policy.py`](../scripts/simulation/rl/rlinf_ext/act_policy.py) | ACTForRLActionPrediction + ValueHead |
| **RL registration** | [`scripts/simulation/rl/rlinf_ext/__init__.py`](../scripts/simulation/rl/rlinf_ext/__init__.py) | Obs/action converter + model factory registration |
| **RL training launcher** | [`scripts/simulation/rl/train_act_grasp_policy.sh`](../scripts/simulation/rl/train_act_grasp_policy.sh) | RLinf PPO training script |
| **RL PPO config** | [`scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml`](../scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml) | PPO + env + model config |
| **RL model config** | [`scripts/simulation/rl/rlinf_ext/config/model/act_dex3.yaml`](../scripts/simulation/rl/rlinf_ext/config/model/act_dex3.yaml) | ACT model config for RLinf |
| **Docker: ACT image** | [`docker/Dockerfile.grasp`](../docker/Dockerfile.grasp) | Grasp-policy Docker image (LeRobot, no GR00T) |
| **Docker: run script** | [`docker/run_docker_grasp.sh`](../docker/run_docker_grasp.sh) | ACT Docker launcher |
| **Teleop: hand tracking** | [`scripts/teleop_devices/handtracking.py`](../scripts/teleop_devices/handtracking.py) | AVP OpenXR hand tracking retargeting |
