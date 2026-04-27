# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See the [parent CLAUDE.md](../../CLAUDE.md) for repo-wide conventions (linting, license headers, DCO sign-off, testing framework).

## Primary Track: G1 + Inspire FTP Grasp Policy

The active focus of this workflow is the **G1 + Inspire FTP 5-finger hand grasp policy** — pick a surgical tool from a tray and place it on a target pad, trained via ACT imitation learning and (optionally) RLinf PPO post-training. When a request says "the policy" / "the task" / "the pipeline" without qualifying, default to this track.

Authoritative docs (re-read them when in doubt — the rest of `docs/` is older and may have drifted):

- [`docs/inspire/grasp_policy_guide.md`](docs/inspire/grasp_policy_guide.md) — end-to-end tutorial (teleop → record → convert → IL → RL → eval).
- [`docs/inspire/task_reference.md`](docs/inspire/task_reference.md) — gym IDs, action/obs spaces, scene, mimic rules, comparison vs Dex3.
- [`docs/inspire/pipeline_contracts.md`](docs/inspire/pipeline_contracts.md) — pipeline invariants, elbow offset chain, conversion details, debug playbook. Treat its "Pipeline invariants" list as load-bearing facts.

Adjacent tracks (live in the same workflow, but not the active work): Trocar assembly (GR00T N1.5, RL) and Arena locomanipulation (GR00T N1.6). The earlier Dex3-hand variant of this grasp task has been removed.

## Execution Model

Everything runs inside Docker. Never run simulation scripts on the host. There are two Docker scripts:

- **`docker/run_docker.sh`** — General-purpose (GR00T N1.5/N1.6, trocar, locomanip)
- **`docker/run_docker_grasp.sh`** — Inspire FTP grasp tasks (LeRobot always installed, no GR00T flags)

```bash
# Inspire FTP grasp — smoketest (41D dummy policy)
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py --test

# Inspire FTP grasp — ACT eval
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py --model_path /models/...

# Inspire FTP grasp — interactive shell
./docker/run_docker_grasp.sh

# GR00T N1.6 locomanip tasks
./docker/run_docker.sh -g1.6 python scripts/simulation/examples/policy_runner.py ...

# GR00T N1.5 trocar tasks
./docker/run_docker.sh -g1.5 python -u scripts/simulation/examples/eval_assemble_trocar.py ...
```

`run_docker.sh` key flags: `-g1.5` / `-g1.6` (GR00T version), `-u <gpu>` (select GPU), `-N` (new container), `-r` (rebuild), `-R` (rebuild no cache), `-d`/`-m`/`-e` (override dataset/model/eval mount dirs).

`run_docker_grasp.sh` key flags: `-u <gpu>`, `-N`, `-r`, `-R`, `-d`/`-m`/`-e` (same as above, no GR00T version flags).

Inside the container: `python` is aliased to `/isaac-sim/python.sh`. Working directory is `/workspaces/workflows/rheo`.

Host mounts: `$HOME/datasets` → `/datasets`, `$HOME/models` → `/models`, `$HOME/eval` → `/eval`. The entire `i4h-workflows` repo mounts to `/workspaces` for live code editing.

## Simulation Tracks

| Track | Use case | GR00T | Entry point | Env definition |
|-------|----------|-------|-------------|----------------|
| **IsaacLab-Arena** | Locomanipulation (tray pick-and-place, cart push) | N1.6 (`-g1.6`) | `policy_runner.py` | `scripts/simulation/environments/` |
| **IsaacLab** | Precision manipulation (trocar assembly, grasp policy) | N1.5 (`-g1.5`) | `eval_assemble_trocar.py`, `eval_act_inspire.py` | `scripts/simulation/tasks/` |

Arena environments are registered via `register_and_patch.py` into an `ExampleEnvironments` dict before the sim starts. IsaacLab-track environments use standard `gymnasium.register()` with gym IDs like `Isaac-Grasp-Policy-G129-InspireFTP-Joint` or `Isaac-Assemble-Trocar-G129-Dex3-Joint`.

## Policy Types

| Policy | Model | Action dims | Chunk size | Wrapper / Config |
|--------|-------|-------------|------------|------------------|
| **GR00T** | DiT (TensorRT) | 43 DOF | 16 | `gr00t_closedloop_policy.py` (trocar) |
| **ACT (Inspire FTP)** | CVAE (LeRobot) | 26 DOF dual-arm or 13 DOF single-arm (scattered to 41D sim) | 50 | `act_closedloop_policy.py` + `act_config_inspire.yaml` (dim_model=256) |

Both extend `BaseClosedloopPolicy`, which manages per-env action chunk state (current chunk, index, exhaustion tracking). Subclasses implement `_load_model()` and `_get_action_chunk()`.

The Inspire FTP grasp policy eval entry point is `eval_act_inspire.py`.

> **Inspire FTP eval gotcha:** Use `predict_action_chunk()`, NOT `select_action()`, in `ACTClosedloopPolicy`. `select_action()` returns one action and the wrapper pads to chunk length by repeating, producing 50 identical actions per chunk (chunk-collapse bug). Symptom: `std=0` per active dim under `--log_actions`.

## Code Organization

```
scripts/
├── config/                        # Policy YAML configs (model path, joint mappings, camera, action horizon)
│   └── inspire/                       # Inspire-FTP dataset conversion YAMLs
│       ├── g1_grasp_policy_inspire_dataset.yaml          # 26D dual-arm
│       ├── g1_grasp_policy_inspire_dataset_right_arm.yaml # 13D right arm
│       └── g1_grasp_policy_inspire_dataset_left_arm.yaml  # 13D left arm
├── policy/
│   ├── gr00t_config.py                  # UnitreeG1SimDataConfig — modality definitions for GR00T
│   ├── gr00t_locomanip_config.py        # Locomanip-specific modality config
│   ├── apply_gr00t_rl_patch.py          # Context manager: git-apply RL patch during eval
│   ├── act_config_inspire.yaml          # LeRobot ACT training config — Inspire FTP grasp (chunk=50, dim_model=256)
│   └── train_act_grasp_policy_inspire.sh # LeRobot ACT training launcher (Inspire FTP)
├── simulation/
│   ├── base_closedloop_policy.py  # Abstract base: shared action chunking for GR00T + ACT
│   ├── gr00t_closedloop_policy.py # GR00T policy wrapper (43 DOF, action_horizon=16) — trocar
│   ├── act_closedloop_policy.py   # ACT policy wrapper (26D/13D → 41D scatter, action_horizon=50) — Inspire FTP
│   ├── obs_processor.py           # Model-agnostic observation extraction (ProcessedObservation dataclass)
│   ├── register_and_patch.py      # Registers Arena envs + assets before sim starts
│   ├── record_demos.py            # Generic IsaacLab demo recording (auto-success, VR gestures)
│   ├── replay_demos_isaaclab.py   # Generic demo replay with success rate validation
│   ├── environments/              # Arena-track env definitions (ExampleEnvironmentBase subclasses)
│   ├── tasks/
│   │   ├── assemble_trocar/       # Trocar assembly task (gym registration, env cfg, mdp/, config/)
│   │   └── grasp_policy_inspire/  # Inspire FTP block-grasp task (gym ids, mimic action, mdp, robot/camera config)
│   ├── examples/                  # Runnable entry points
│   │   ├── policy_runner.py             # GR00T N1.6 Arena evaluation
│   │   ├── eval_assemble_trocar.py      # GR00T N1.5 trocar evaluation (--rl_ckpt flag)
│   │   ├── eval_act_inspire.py          # Inspire FTP ACT evaluator (the active eval entry point)
│   │   └── triggered_policy_runner.py   # HTTP-triggered for VLM agents
│   ├── assets/                    # USD path constants + Arena asset/background library registration
│   ├── embodiments/               # Patched G1 robot embodiment
│   └── rl/
│       ├── rlinf_ext/             # RLinf extension (env registration, obs/action converters, model patches)
│       │   ├── act_policy.py      # ACT wrapper for RLinf RL post-training (ValueHead for critic)
│       │   └── config/            # RLinf YAML configs (env, PPO hyperparams, model architecture)
│       └── train_gr00t_assemble_trocar.sh  # GR00T RL training launcher (trocar)
├── utils/
│   ├── joint_conversion.py        # Policy-to-sim joint remapping (43 DOF, GR00T trocar)
│   ├── inspire_lerobot_fields.py # Inspire FTP 26D + 13D state/action conversion (handles 38D teleop, 41D, 53D)
│   ├── inspire_experiment_config.py # 26D joint groups + scatter_to_sim (26D→41D)
│   ├── policy_tasks.py            # TensorRT DiT wrapper, success-hold wrapper
│   ├── webrtc_cam.py              # WebRTC video streaming (aiortc)
│   ├── trigger_server.py          # HTTP trigger server for remote policy activation
│   └── convert_hdf5_to_lerobot.py # Dataset format conversion
└── teleop_devices/
    ├── keyboard_23d_adapter.py    # 23-DOF keyboard control
    ├── motion_controllers.py      # Meta Quest controller support
    └── handtracking.py            # AVP OpenXR hand tracking → gripper+wrist retarget (PINK IK)
agents/
├── agents/                        # VLM agent implementations (chat, monitoring, robot control)
└── configs/                       # Agent YAML configs (global + per-agent)
docs/                              # End-to-end guides (inspire/grasp_policy_guide, inspire/task_reference, inspire/pipeline_contracts, utils/avp_teleoperation, utils/cloudxr)
tests/
├── helpers.py                     # Test decorators + subprocess runner
└── test_sim/                      # Unit and integration tests
```

## PYTHONPATH

Set by the Dockerfile and training launcher scripts:
- `workflows/rheo/scripts` — enables `from simulation.X` and `from policy.X` imports
- `workflows/rheo/scripts/simulation/rl` — makes `rlinf_ext` importable as top-level module (RL training only)

## Key Patterns

### Action Chunking (BaseClosedloopPolicy)
`BaseClosedloopPolicy` is the abstract base for all policy wrappers. It maintains `current_action_chunk` and `current_action_index` per env, requesting a new forward pass only when the chunk is exhausted. Reset via `policy.reset(env_ids)`. GR00T uses 16-step chunks; ACT (Inspire FTP) uses 50-step chunks.

### Observation Processing
`obs_processor.py` provides model-agnostic observation extraction via the `ProcessedObservation` dataclass. It extracts camera images (front, left_wrist, right_wrist) and arm+hand joint state from IsaacLab environments. The Inspire FTP path uses `inspire_experiment_config.py` to select 26D / 13D state slicing.

### Joint Remapping
For the GR00T trocar path, policy output (43 DOF, body-group order) is remapped to simulator joint order via `joint_conversion.py`, with mapping YAMLs from `third_party/IsaacLab-Arena/isaaclab_arena_gr00t/config/g1/`; 15 zeros are padded at the front for leg/waist joints. For the Inspire FTP ACT path, scattering from 26D/13D policy output to the 41D sim action lives in `inspire_experiment_config.py:GROUP_SIM_INDICES`.

### RL Checkpoint Evaluation (--rl_ckpt)
When evaluating an RL-trained GR00T checkpoint, you **must** pass `--rl_ckpt` to `eval_assemble_trocar.py`. This applies `tools/env_setup/patches/gr00t_policy_padding_dropout.patch` via `git apply` (eagle input padding to 850 tokens + dropout→Identity replacement). The patch is auto-reverted via context manager. Without this flag on an RL checkpoint, inference silently produces wrong results.

### Arena Environment Registration
For Arena-track tasks, `register_and_patch.py` must run before the simulation app starts. It registers environment classes into `ExampleEnvironments` and registers asset libraries (objects, backgrounds, embodiments) via side-effect imports. The `policy_runner.py` entry point handles this automatically.

### RLinf Extension Module
RL training sets `RLINF_EXT_MODULE=rlinf_ext` to load `scripts/simulation/rl/rlinf_ext/__init__.py:register()`. This registers gym IDs (`Isaac-Assemble-Trocar-G129-Dex3-*` for GR00T trocar and `Isaac-Grasp-Policy-G129-InspireFTP-*` for ACT Inspire FTP grasp) into RLinf's env map, registers obs/action converters (GR00T `dex3` for trocar, ACT `act_inspire_ftp` for Inspire FTP grasp), monkeypatches `get_model` for the `new_embodiment` tag, and imports policy configs. `act_policy.py` wraps ACT with a `ValueHead` for RL critic estimation.

### Demo Recording
`record_demos.py` is the generic IsaacLab demo recording script (replaces task-specific scripts). Features: auto-success detection (saves after N consecutive success frames), VR gesture controls (OpenXR hand-tracking START/STOP/RESET), and XR UI overlays. `replay_demos_isaaclab.py` replays and validates recorded demos with optional success rate checking.

### AVP Hand Tracking
`handtracking.py` retargets Apple Vision Pro OpenXR hand poses to G1 gripper+wrist commands. Pinch gripper uses hysteresis (0.03m close / 0.05m open thresholds). PINK IK solves wrist retargeting. Outputs 16D: [left_gripper, right_gripper, left_wrist(7), right_wrist(7)].

> **Note:** `handtracking.py` is the Arena/trocar fallback (binary gripper). The Inspire FTP Teleop env registers its own `UnitreeG1RetargeterCfg` directly in `g1_grasp_policy_inspire_teleop_env_cfg.py` for full 24-joint DexPilot retargeting — `handtracking.py` is bypassed entirely on that path.

### Inspire FTP Mimic Joints
The Inspire FTP hand has 24 joints — 12 actuated + 12 mimic. `InspireFTPJointPositionAction.apply_actions()` enforces the 12 mimic rules from the policy's 12 actuated targets (multipliers 1.0843 for finger PIPs, 0.8024 / 0.9487 for the chained thumb segments). **Never bypass this action class** with direct `set_joint_position_target` writes against mimic joints — grasps collapse silently. Rule processing order is also load-bearing for the thumb chain (`thumb_3` must be computed before `thumb_4`).

### Inspire FTP Elbow Offset Chain
A `−0.3` rad offset is applied to `{left,right}_elbow_joint` in `InspireFTPJointPositionActionCfg.offset`. To compensate, the HDF5→LeRobot converter adds `+0.3` to the elbow column of every parquet `action`. **Exactly one comp on each side** — never both, never neither. The converter add must be out-of-place (in-place `+=` aliases the state buffer through shared NumPy views). Symptom of a broken chain: arm drifts monotonically during eval. Full details in [`docs/inspire/pipeline_contracts.md`](docs/inspire/pipeline_contracts.md).

## Testing

Tests run inside Docker. Conditional decorators in `tests/helpers.py`:
- `@requires_isaac_sim` — skips if `omni` module unavailable
- `@requires_groot` — skips if `gr00t.policy` unavailable
- `@requires_gpu_memory(min_gib=20)` — GPU VRAM gate

`run_with_monitoring_capture(command, timeout, target_lines)` runs subprocess-based integration tests with output matching.

## Adding a New Task

### Arena-track (locomanipulation)
1. Create env class in `scripts/simulation/environments/` extending `ExampleEnvironmentBase`
2. Register in `register_and_patch.py` → `register_workflow_cli()`
3. Add policy config YAML in `scripts/config/`

### IsaacLab-track (precision manipulation)
1. Create task package in `scripts/simulation/tasks/<name>/`
2. Add `__init__.py` with `gymnasium.register()` calls (Joint, Joint-Eval, Teleop variants)
3. Define `env_cfg.py` with `ManagerBasedRLEnvCfg` subclass
4. Add `mdp/` subdirectory (observations, rewards, terminations, events)
5. Add `config/` subdirectory (robot presets, camera presets)
6. See `scripts/simulation/tasks/grasp_policy_inspire/` and `scripts/simulation/tasks/assemble_trocar/` as reference implementations.

## Pipeline Stages

**GR00T path:** Data collection (`record_demos.py`) → Annotation (`annotate_demos.py`) → Synthetic generation (`generate_dataset.py` / Cosmos Transfer 2.5) → HDF5→LeRobot conversion (`convert_hdf5_to_lerobot.py`) → Fine-tuning (GR00T SFT) → RL post-training (`train_gr00t_assemble_trocar.sh`) → Evaluation (`eval_assemble_trocar.py` / `policy_runner.py`) → Deployment (WebRTC + VLM agents via `triggered_policy_runner.py`)

**ACT path (Inspire FTP) — primary track:** VR teleoperation (AVP DexPilot, 38D PinkIK, optional `--arm {left,right,both}` masking) → Record HDF5 (`record_demos.py`, full 87D + 12D obs always recorded) → HDF5→LeRobot conversion to **26D dual-arm** or **13D single-arm** (three YAMLs in `scripts/config/`, observation-derived actions with elbow `+0.3` baked in) → ACT IL training (`train_act_grasp_policy_inspire.sh`) → Optional RL post-training (`rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml` via RLinf) → Evaluation (`eval_act_inspire.py`)

See [`docs/inspire/grasp_policy_guide.md`](docs/inspire/grasp_policy_guide.md) for the complete walkthrough.
