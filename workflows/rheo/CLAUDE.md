# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See the [parent CLAUDE.md](../../CLAUDE.md) for repo-wide conventions (linting, license headers, DCO sign-off, testing framework).

## Execution Model

Everything runs inside Docker. Never run simulation scripts on the host. There are two Docker scripts:

- **`docker/run_docker.sh`** — General-purpose (GR00T N1.5/N1.6, trocar, locomanip)
- **`docker/run_docker_grasp.sh`** — Inspire FTP grasp tasks (LeRobot always installed, no GR00T flags)

```bash
# Inspire FTP grasp — smoketest (41D dummy policy)
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_grasp_policy_inspire.py --test

# Inspire FTP grasp — ACT eval
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_grasp_policy_inspire.py --policy_type act --model_path /models/...

# Inspire FTP grasp — interactive shell
./docker/run_docker_grasp.sh

# GR00T N1.6 locomanip tasks
./docker/run_docker.sh -g1.6 python scripts/simulation/examples/policy_runner.py ...

# GR00T N1.5 trocar tasks
./docker/run_docker.sh -g1.5 python -u scripts/simulation/examples/eval_assemble_trocar.py ...

# GR00T N1.5 Dex3 grasp (original hands, not Inspire)
./docker/run_docker.sh -g1.5 python scripts/simulation/examples/eval_grasp_policy.py --policy_type act ...
```

`run_docker.sh` key flags: `-g1.5` / `-g1.6` (GR00T version), `-u <gpu>` (select GPU), `-N` (new container), `-r` (rebuild), `-R` (rebuild no cache), `-d`/`-m`/`-e` (override dataset/model/eval mount dirs).

`run_docker_grasp.sh` key flags: `-u <gpu>`, `-N`, `-r`, `-R`, `-d`/`-m`/`-e` (same as above, no GR00T version flags).

Inside the container: `python` is aliased to `/isaac-sim/python.sh`. Working directory is `/workspaces/workflows/rheo`.

Host mounts: `$HOME/datasets` → `/datasets`, `$HOME/models` → `/models`, `$HOME/eval` → `/eval`. The entire `i4h-workflows` repo mounts to `/workspaces` for live code editing.

## Simulation Tracks

| Track | Use case | GR00T | Entry point | Env definition |
|-------|----------|-------|-------------|----------------|
| **IsaacLab-Arena** | Locomanipulation (tray pick-and-place, cart push) | N1.6 (`-g1.6`) | `policy_runner.py` | `scripts/simulation/environments/` |
| **IsaacLab** | Precision manipulation (trocar assembly, grasp policy) | N1.5 (`-g1.5`) | `eval_assemble_trocar.py`, `eval_grasp_policy.py` | `scripts/simulation/tasks/` |

Arena environments are registered via `register_and_patch.py` into an `ExampleEnvironments` dict before the sim starts. IsaacLab-track environments use standard `gymnasium.register()` with gym IDs like `Isaac-Assemble-Trocar-G129-Dex3-Joint` or `Isaac-Grasp-Policy-G129-Dex3-Joint`.

## Policy Types

| Policy | Model | Action dims | Chunk size | Wrapper |
|--------|-------|-------------|------------|---------|
| **GR00T** | DiT (TensorRT) | 43 DOF | 16 | `gr00t_closedloop_policy.py` |
| **ACT** | CVAE (LeRobot) | 28 DOF (padded to 43) | 100 | `act_closedloop_policy.py` |

Both extend `BaseClosedloopPolicy`, which manages per-env action chunk state (current chunk, index, exhaustion tracking). Subclasses implement `_load_model()` and `_get_action_chunk()`.

`eval_grasp_policy.py` supports both via `--policy_type` (gr00t, act, test).

## Code Organization

```
scripts/
├── config/                        # Policy YAML configs (model path, joint mappings, camera, action horizon)
├── policy/
│   ├── gr00t_config.py            # UnitreeG1SimDataConfig — modality definitions for GR00T
│   ├── gr00t_locomanip_config.py  # Locomanip-specific modality config
│   ├── apply_gr00t_rl_patch.py    # Context manager: git-apply RL patch during eval
│   ├── act_config.yaml            # LeRobot ACT training config (chunk_size=100, CVAE, kl_weight=10)
│   └── train_act_grasp_policy.sh  # LeRobot ACT training launcher (supports resume, Hydra overrides)
├── simulation/
│   ├── base_closedloop_policy.py  # Abstract base: shared action chunking for GR00T + ACT
│   ├── gr00t_closedloop_policy.py # GR00T policy wrapper (43 DOF, action_horizon=16)
│   ├── act_closedloop_policy.py   # ACT policy wrapper (28D→43D padding, action_horizon=100)
│   ├── obs_processor.py           # Model-agnostic observation extraction (ProcessedObservation dataclass)
│   ├── register_and_patch.py      # Registers Arena envs + assets before sim starts
│   ├── record_demos.py            # Generic IsaacLab demo recording (auto-success, VR gestures)
│   ├── replay_demos_isaaclab.py   # Generic demo replay with success rate validation
│   ├── environments/              # Arena-track env definitions (ExampleEnvironmentBase subclasses)
│   ├── tasks/
│   │   ├── assemble_trocar/       # Trocar assembly task (gym registration, env cfg, mdp/, config/)
│   │   └── grasp_policy/          # Block grasp & place task (3 gym variants: Joint, Joint-Eval, Teleop)
│   ├── examples/                  # Runnable entry points
│   │   ├── policy_runner.py       # GR00T N1.6 Arena evaluation
│   │   ├── eval_assemble_trocar.py # GR00T N1.5 trocar evaluation (--rl_ckpt flag)
│   │   ├── eval_grasp_policy.py   # Unified grasp evaluator (--policy_type gr00t|act|test)
│   │   └── triggered_policy_runner.py # HTTP-triggered for VLM agents
│   ├── assets/                    # USD path constants + Arena asset/background library registration
│   ├── embodiments/               # Patched G1 robot embodiment
│   └── rl/
│       ├── rlinf_ext/             # RLinf extension (env registration, obs/action converters, model patches)
│       │   ├── act_policy.py      # ACT wrapper for RLinf RL post-training (ValueHead for critic)
│       │   └── config/            # RLinf YAML configs (env, PPO hyperparams, model architecture)
│       ├── train_gr00t_assemble_trocar.sh  # GR00T RL training launcher
│       └── train_act_grasp_policy.sh       # ACT RL training launcher
├── utils/
│   ├── joint_conversion.py        # Policy-to-sim joint remapping (43 DOF)
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
docs/                              # End-to-end guides (grasp_policy_guide, avp_teleoperation, cloudxr)
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
`BaseClosedloopPolicy` is the abstract base for all policy wrappers. It maintains `current_action_chunk` and `current_action_index` per env, requesting a new forward pass only when the chunk is exhausted. Reset via `policy.reset(env_ids)`. GR00T uses 16-step chunks; ACT uses 100-step chunks.

### Observation Processing
`obs_processor.py` provides model-agnostic observation extraction via the `ProcessedObservation` dataclass. It extracts camera images (front, left_wrist, right_wrist) and 28D arm+hand joint state from IsaacLab environments. Used by both GR00T and ACT policy wrappers.

### Joint Remapping
Policy output (43 DOF, body-group order) is remapped to simulator joint order. Mapping defined by YAML files in `third_party/IsaacLab-Arena/isaaclab_arena_gr00t/config/g1/` and applied via `joint_conversion.py`. For 28 DOF policies (trocar, grasp), 15 zeros are padded at the front for leg/waist joints.

### RL Checkpoint Evaluation (--rl_ckpt)
When evaluating an RL-trained GR00T checkpoint, you **must** pass `--rl_ckpt` to `eval_assemble_trocar.py`. This applies `tools/env_setup/patches/gr00t_policy_padding_dropout.patch` via `git apply` (eagle input padding to 850 tokens + dropout→Identity replacement). The patch is auto-reverted via context manager. Without this flag on an RL checkpoint, inference silently produces wrong results.

### Arena Environment Registration
For Arena-track tasks, `register_and_patch.py` must run before the simulation app starts. It registers environment classes into `ExampleEnvironments` and registers asset libraries (objects, backgrounds, embodiments) via side-effect imports. The `policy_runner.py` entry point handles this automatically.

### RLinf Extension Module
RL training sets `RLINF_EXT_MODULE=rlinf_ext` to load `scripts/simulation/rl/rlinf_ext/__init__.py:register()`. This registers gym IDs (both `Isaac-Assemble-Trocar-*` and `Isaac-Grasp-Policy-*`) into RLinf's env map, registers obs/action converters for GR00T and ACT, monkeypatches `get_model` for the `new_embodiment` tag, and imports policy configs. `act_policy.py` wraps ACT with a `ValueHead` for RL critic estimation.

### Demo Recording
`record_demos.py` is the generic IsaacLab demo recording script (replaces task-specific scripts). Features: auto-success detection (saves after N consecutive success frames), VR gesture controls (OpenXR hand-tracking START/STOP/RESET), and XR UI overlays. `replay_demos_isaaclab.py` replays and validates recorded demos with optional success rate checking.

### AVP Hand Tracking
`handtracking.py` retargets Apple Vision Pro OpenXR hand poses to G1 gripper+wrist commands. Pinch gripper uses hysteresis (0.03m close / 0.05m open thresholds). PINK IK solves wrist retargeting. Outputs 16D: [left_gripper, right_gripper, left_wrist(7), right_wrist(7)].

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
6. See `docs/custom_manipulation_task_guide.md` for the full walkthrough

## Pipeline Stages

**GR00T path:** Data collection (`record_demos.py`) → Annotation (`annotate_demos.py`) → Synthetic generation (`generate_dataset.py` / Cosmos Transfer 2.5) → HDF5→LeRobot conversion (`convert_hdf5_to_lerobot.py`) → Fine-tuning (GR00T SFT) → RL post-training (`train_gr00t_assemble_trocar.sh`) → Evaluation (`eval_assemble_trocar.py` / `policy_runner.py`) → Deployment (WebRTC + VLM agents via `triggered_policy_runner.py`)

**ACT path:** VR teleoperation (AVP) → Record HDF5 (`record_demos.py`) → HDF5→LeRobot conversion → ACT IL training (`train_act_grasp_policy.sh`) → RL post-training (`train_act_grasp_policy.sh` via RLinf) → Evaluation (`eval_grasp_policy.py`)

See `docs/grasp_policy_guide.md` for the complete ACT pipeline walkthrough.
