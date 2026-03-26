# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See the [parent CLAUDE.md](../../CLAUDE.md) for repo-wide conventions (linting, license headers, DCO sign-off, testing framework).

## Execution Model

Everything runs inside Docker via `docker/run_docker.sh`. Never run simulation scripts on the host.

```bash
# Run a command (GR00T N1.6 for locomanip tasks)
./docker/run_docker.sh -g1.6 python scripts/simulation/examples/policy_runner.py ...

# Run a command (GR00T N1.5 for trocar tasks)
./docker/run_docker.sh -g1.5 python -u scripts/simulation/examples/eval_assemble_trocar.py ...

# Interactive shell
./docker/run_docker.sh -g1.6
```

Key flags: `-g1.5` / `-g1.6` (GR00T version), `-u <gpu>` (select GPU), `-N` (new container), `-r` (rebuild), `-R` (rebuild no cache), `-d`/`-m`/`-e` (override dataset/model/eval mount dirs).

Inside the container: `python` is aliased to `/isaac-sim/python.sh`. Working directory is `/workspaces/workflows/rheo`.

Host mounts: `$HOME/datasets` → `/datasets`, `$HOME/models` → `/models`, `$HOME/eval` → `/eval`. The entire `i4h-workflows` repo mounts to `/workspaces` for live code editing.

## Two Simulation Tracks

| Track | Use case | GR00T | Entry point | Env definition |
|-------|----------|-------|-------------|----------------|
| **IsaacLab-Arena** | Locomanipulation (tray pick-and-place, cart push) | N1.6 (`-g1.6`) | `policy_runner.py` | `scripts/simulation/environments/` |
| **IsaacLab** | Precision manipulation (trocar assembly) | N1.5 (`-g1.5`) | `eval_assemble_trocar.py` | `scripts/simulation/tasks/assemble_trocar/` |

Arena environments are registered via `register_and_patch.py` into an `ExampleEnvironments` dict before the sim starts. Trocar environments use standard `gymnasium.register()` with gym IDs like `Isaac-Assemble-Trocar-G129-Dex3-Joint`.

## Code Organization

```
scripts/
├── config/                        # Policy YAML configs (model path, joint mappings, camera, action horizon)
├── policy/
│   ├── gr00t_config.py            # UnitreeG1SimDataConfig — modality definitions for GR00T
│   ├── gr00t_locomanip_config.py  # Locomanip-specific modality config
│   └── apply_gr00t_rl_patch.py    # Context manager: git-apply RL patch during eval
├── simulation/
│   ├── gr00t_closedloop_policy.py # CustomGr00tClosedloopPolicy — action chunking + joint remapping
│   ├── register_and_patch.py      # Registers Arena envs + assets before sim starts
│   ├── environments/              # Arena-track env definitions (ExampleEnvironmentBase subclasses)
│   ├── tasks/
│   │   └── assemble_trocar/       # IsaacLab-track task package (gym registration in __init__.py)
│   │       ├── g1_assemble_trocar_env_cfg.py  # ManagerBasedRLEnvCfg subclass
│   │       ├── mdp/               # Observations, rewards, terminations, events
│   │       └── config/            # Robot + camera presets
│   ├── examples/                  # Runnable entry points (policy_runner, eval_assemble_trocar, etc.)
│   ├── assets/                    # USD path constants + Arena asset/background library registration
│   ├── embodiments/               # Patched G1 robot embodiment
│   └── rl/
│       ├── rlinf_ext/             # RLinf extension (env registration, obs/action converters, model patches)
│       └── train_gr00t_assemble_trocar.sh  # RL training/eval launcher
├── utils/
│   ├── joint_conversion.py        # Policy-to-sim joint remapping (43 DOF)
│   ├── policy_tasks.py            # TensorRT DiT wrapper, success-hold wrapper
│   ├── webrtc_cam.py              # WebRTC video streaming (aiortc)
│   ├── trigger_server.py          # HTTP trigger server for remote policy activation
│   └── convert_hdf5_to_lerobot.py # Dataset format conversion
└── teleop_devices/                # Keyboard adapter (23-DOF), Meta Quest motion controllers
agents/
├── agents/                        # VLM agent implementations (chat, monitoring, robot control)
└── configs/                       # Agent YAML configs (global + per-agent)
tests/
├── helpers.py                     # Test decorators + subprocess runner
└── test_sim/                      # Unit and integration tests
```

## PYTHONPATH

Set by the Dockerfile and `train_gr00t_assemble_trocar.sh`:
- `workflows/rheo/scripts` — enables `from simulation.X` and `from policy.X` imports
- `workflows/rheo/scripts/simulation/rl` — makes `rlinf_ext` importable as top-level module (RL training only)

## Key Patterns

### Action Chunking
`CustomGr00tClosedloopPolicy` predicts 16 actions per observation (`action_horizon=16`). It maintains `current_action_chunk` and `current_action_index` per env, requesting a new GR00T forward pass only when the chunk is exhausted. Reset via `policy.reset(env_ids)`.

### Joint Remapping
Policy output (43 DOF, body-group order) is remapped to simulator joint order. Mapping defined by YAML files in `third_party/IsaacLab-Arena/isaaclab_arena_gr00t/config/g1/` and applied via `joint_conversion.py`. For trocar (28 DOF subset), the `rlinf_ext` converter pads 15 zeros at the front.

### RL Checkpoint Evaluation (--rl_ckpt)
When evaluating an RL-trained checkpoint, you **must** pass `--rl_ckpt` to `eval_assemble_trocar.py`. This applies `tools/env_setup/patches/gr00t_policy_padding_dropout.patch` via `git apply` (eagle input padding to 850 tokens + dropout→Identity replacement). The patch is auto-reverted via context manager. Without this flag on an RL checkpoint, inference silently produces wrong results.

### Arena Environment Registration
For Arena-track tasks, `register_and_patch.py` must run before the simulation app starts. It registers environment classes into `ExampleEnvironments` and registers asset libraries (objects, backgrounds, embodiments) via side-effect imports. The `policy_runner.py` entry point handles this automatically.

### RLinf Extension Module
RL training sets `RLINF_EXT_MODULE=rlinf_ext` to load `scripts/simulation/rl/rlinf_ext/__init__.py:register()`. This registers `Isaac-Assemble-Trocar-*` gym IDs into RLinf's env map, registers obs/action converters, monkeypatches `get_model` for the `new_embodiment` tag, and imports `policy.gr00t_config`.

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
2. Add `__init__.py` with `gymnasium.register()` calls
3. Define `env_cfg.py` with `ManagerBasedRLEnvCfg` subclass
4. Add `mdp/` subdirectory (observations, rewards, terminations, events)
5. Add `config/` subdirectory (robot presets, camera presets)

## Pipeline Stages

Data collection (`record_demos_*.py`) → Annotation (`annotate_demos.py`) → Synthetic generation (`generate_dataset.py` / Cosmos Transfer 2.5) → HDF5→LeRobot conversion (`convert_hdf5_to_lerobot.py`) → Fine-tuning (GR00T SFT) → RL post-training (`train_gr00t_assemble_trocar.sh`) → Evaluation (`eval_assemble_trocar.py` / `policy_runner.py`) → Deployment (WebRTC + VLM agents via `triggered_policy_runner.py`)
