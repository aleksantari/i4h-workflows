# Rheo Project Assessment Report

**Date**: 2026-04-07
**Branch**: `grasp-policy`
**Scope**: Audit of `/workflows/rheo/` — what's original, what we added, how it's organized, what could be improved.

---

## 1. Inventory: Original vs. Our Additions

### Original Rheo (shipped with repo, v0.5.0)

| Category | Files/Dirs |
|----------|-----------|
| **Tasks** | `tasks/assemble_trocar/` (trocar assembly — IsaacLab track) |
| **Locomanip tasks** | `tasks/g1_observe_object_task.py`, `g1_push_cart_task.py`, `g1_tray_pick_and_place_task.py` |
| **Arena envs** | `environments/g1_locomanip_{observe_object,push_cart,tray_pick_and_place}_environment.py` |
| **GR00T infra** | `gr00t_closedloop_policy.py`, `policy/gr00t_config.py`, `policy/gr00t_locomanip_config.py` |
| **Entry points** | `eval_assemble_trocar.py`, `policy_runner.py`, `policy_runner_cli.py`, `triggered_policy_runner.py`, `observe_runner.py`, `webrtc_runner_cli.py` |
| **Recording** | `record_demos_assemble_trocar.py`, `record_demos_locomanip.py`, `replay_demos.py` |
| **Agents** | Entire `agents/` directory (VLM agents + configs) |
| **Docker** | `Dockerfile.x86`, `run_docker.sh` |
| **Assets/Utils** | `assets/`, `embodiments/`, `register_and_patch.py`, `joint_conversion.py`, `policy_tasks.py`, `keyboard_env_reseter.py`, `webrtc_cam.py`, `trigger_server.py` |
| **Config** | `config/g1_assemble_trocar_dataset.yaml`, `config/g1_gr00t_closedloop_*.yaml`, `config/g1_locomanip_dataset_config.yaml` |
| **RL** | `rl/train_gr00t_assemble_trocar.sh`, `rl/rlinf_ext/` (original trocar registrations) |
| **Docs** | `docs/rheo/` (trocar + locomanip finetuning guides) |
| **Tests** | `tests/helpers.py`, `tests/test_sim/test_locomanip.py`, `tests/test_sim/test_integration_eval_assemble_trocar.py` |
| **Other** | `annotate_demos.py`, `generate_dataset.py`, `merge_demos.py`, `teleop_devices/keyboard_23d_adapter.py`, `teleop_devices/motion_controllers.py` |

### Our Additions (March 28 - April 7, 2026)

| Category | Files/Dirs | Added |
|----------|-----------|-------|
| **Dex3 grasp task** | `tasks/grasp_policy/` (full MDP: env_cfg, teleop_cfg, observations, rewards, events, terminations) | Mar 28 |
| **Inspire FTP task** | `tasks/grasp_policy_inspire/` (full MDP + `mimic_action.py` + `config/robot_config.py`) | Apr 2 |
| **Eval scripts** | `examples/eval_grasp_policy.py`, `examples/eval_grasp_policy_inspire.py` | Mar 28, Apr 2 |
| **Policy infra** | `base_closedloop_policy.py`, `act_closedloop_policy.py`, `obs_processor.py` | Mar 28 |
| **ACT training** | `policy/act_config.yaml`, `policy/act_config_inspire_ftp.yaml`, `policy/train_act_grasp_policy.sh` | Mar 28-Apr 3 |
| **Recording** | `record_demos.py` (unified), `replay_demos_isaaclab.py` | Mar 31 |
| **Hand tracking** | `teleop_devices/handtracking.py` (AVP + DexPilot IK) | Mar 27 |
| **Docker grasp** | `Dockerfile.grasp`, `run_docker_grasp.sh` | Apr 2 |
| **Inspire assets** | `assets/robots/g1-29dof-inspire-ftp-urdf/`, `assets/robots/g1-29dof-inspire-ftp-usd/` | Apr 2 |
| **Surgical tools** | `assets/sinus_toolkit_v1/` (5 tools with USD + OBJ) | Apr 4 |
| **Inspire utils** | `utils/inspire_ftp_lerobot_fields.py`, `utils/inspire_ftp_experiment_config.py`, `utils/inspect_inspire_ftp_joints.py`, `utils/convert_inspire_ftp_urdf_to_usd.py` | Apr 2 |
| **Dex3 utils** | `utils/act_experiment_config.py`, `utils/assemble_trocar_lerobot_fields.py`, `utils/extended_dataset_config.py` | Mar 28 |
| **Dataset conversion** | `utils/convert_hdf5_to_lerobot.py` | Mar 31 |
| **Inspire configs** | `config/g1_grasp_policy_inspire_dataset.yaml` | Apr 2 |
| **Dex3 configs** | `config/g1_act_closedloop_grasp_policy.yaml`, `config/g1_grasp_policy_dataset.yaml` | Mar 28 |
| **RLinf Inspire** | `rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy_inspire.yaml`, `rl/rlinf_ext/config/env/isaaclab_grasp_policy_inspire.yaml`, `rl/rlinf_ext/config/model/act_inspire_ftp.yaml` | Apr 4 |
| **RLinf Dex3** | `rl/rlinf_ext/config/isaaclab_ppo_act_grasp_policy.yaml`, `rl/rlinf_ext/config/env/isaaclab_grasp_policy.yaml`, `rl/rlinf_ext/config/model/act_dex3.yaml` | Mar 28 |
| **RLinf ACT wrapper** | `rl/rlinf_ext/act_policy.py` (ACT + ValueHead for RL critic) | Apr 4 |
| **RL training** | `rl/train_act_grasp_policy.sh` | Apr 4 |
| **Datasets** | `datasets/grasp_policy/`, `datasets/inspire_ftp/`, `datasets/inspire_ftp_new/` | Mar 31-Apr 5 |
| **Docs** | `docs/grasp_policy/` (5 guides), `docs/utils/avp_teleoperation.md`, `docs/utils/hand.md` | Apr 2-5 |

---

## 2. Dependency Map

```
assemble_trocar/config/
  |-- CameraBaseCfg, CameraPresets, G1RobotPresets
  |
  |-->  grasp_policy/config/__init__.py        (re-exports ALL three)
  \-->  grasp_policy_inspire/config/__init__.py (re-exports CameraBaseCfg + CameraPresets only)

assemble_trocar/mdp/observations.py
  |-- get_robot_body_joint_states, get_robot_dex3_joint_states
  |
  \-->  grasp_policy/mdp/observations.py       (re-exports both)

grasp_policy/mdp/
  |-- rewards.py, events.py, terminations.py  (own implementations)
  |
  |-->  grasp_policy_inspire/mdp/rewards.py       (re-exports)
  |-->  grasp_policy_inspire/mdp/events.py        (re-exports)
  \-->  grasp_policy_inspire/mdp/terminations.py  (re-exports)

grasp_policy_inspire/ owns independently:
  |-- mdp/observations.py      (Inspire-specific: get_robot_inspire_joint_states)
  |-- mdp/mimic_action.py      (Inspire-specific: mimic joint enforcement)
  \-- config/robot_config.py   (Inspire-specific: G1InspireRobotPresets)
```

**Risk**: If NVIDIA updates `assemble_trocar/config/` or `assemble_trocar/mdp/observations.py` internals, both grasp tasks break. The chain is 3 tasks deep.

---

## 3. Code Health Assessment

### Things That Are Good

- **Task structure**: Both grasp tasks follow the exact IsaacLab pattern (gym registration, env_cfg, teleop_cfg, MDP modules). Consistent with trocar.
- **SPDX headers**: All Python files have proper Apache-2.0 license headers.
- **Shared MDP logic**: Rewards, events, and terminations are shared via imports (no copy-paste duplication). This is the right pattern.
- **Documentation**: Comprehensive guides for both tasks, plus AVP teleoperation and hand hardware docs.
- **RLinf integration**: Clean separation of env/model/PPO configs. Follows existing trocar pattern.
- **Mimic action**: `InspireFTPJointPositionAction` correctly enforces mechanical coupling at the action level. Well-implemented.
- **Asset organization**: URDF/USD robot models and surgical tools are properly structured with config files.

### Issues Found

#### HIGH - Fragile Import Chain

- **What**: `grasp_policy` -> `assemble_trocar` for config + observations; `grasp_policy_inspire` -> `grasp_policy` for rewards/events/terminations + `assemble_trocar` for camera config.
- **Risk**: Upstream trocar changes silently break our tasks.
- **Recommendation**: When ready, copy the shared definitions (camera config, body joint obs function) into our own modules. Approximately 50 lines of code total. The rewards/events/terminations chain (inspire -> dex3) is fine since we own both.

#### MEDIUM - Eval Script Duplication

- **What**: `eval_grasp_policy.py` (284 lines) and `eval_grasp_policy_inspire.py` (283 lines) share approximately 90% identical code.
- **Risk**: Fixing a bug in eval logic requires updating both files.
- **Recommendation**: Extract the shared eval loop, video recording, and results reporting into `examples/utils.py` (which already exists). Each eval script becomes a thin wrapper (approximately 50 lines) with task-specific argument defaults and policy loading.

#### LOW - Stale Data Files

- **What**: `datasets/inspire_ftp_new/gripper_test.hdf5` (875MB) from deleted gripper approach. `datasets/dataset.hdf5` (96 bytes, empty). `datasets/inspire_ftp/tool_2_demos.hdf5` (96 bytes, empty).
- **Risk**: Wastes disk space; `inspire_ftp_new/` name is confusing.
- **Recommendation**: Delete when convenient. These are local-only (gitignored).

#### LOW - Shared terminations.py is Misleading

- **What**: `tasks/terminations.py` at the shared tasks/ level contains `object_at_destination()` -- a cart-specific helper for locomanip tasks.
- **Risk**: Misleading name suggests it's used by grasp tasks. It's not -- grasp tasks have their own termination logic.
- **Recommendation**: This is original code, leave it alone. Just be aware it's not relevant to our tasks.

#### INFO - Legacy Recording Scripts Still Present

- **What**: `record_demos_assemble_trocar.py` (303 lines) and `record_demos_locomanip.py` still exist alongside our unified `record_demos.py` (498 lines).
- **Risk**: None -- these are original code and still work. Our unified script supersedes them for our tasks.
- **Recommendation**: Don't delete (they're original). For our tasks, always use `record_demos.py`.

#### INFO - Utils Directory is Flat

- **What**: `scripts/utils/` mixes general utilities (joint_conversion, policy_tasks) with task-specific converters (inspire_ftp_lerobot_fields, act_experiment_config) and one-off tools (inspect_inspire_ftp_joints).
- **Risk**: As we add more tasks, this gets crowded.
- **Recommendation**: Fine for now. If it grows past approximately 20 files, consider subdirectories (`utils/data/`, `utils/tasks/`).

---

## 4. File Count Summary

| Category | Original | Our Additions | Total |
|----------|----------|---------------|-------|
| Task definition files | 17 | 21 | 38 |
| Entry points (examples/) | 7 | 2 | 9 |
| Policy/training scripts | 5 | 5 | 10 |
| Config YAMLs | 4 | 3 | 7 |
| RLinf configs | 3 | 6 | 9 |
| Utility scripts | 5 | 7 | 12 |
| Docker files | 2 | 2 | 4 |
| Documentation | 4 | 7 | 11 |
| Test files | 3 | 0 | 3 |
| Simulation infra | 7 | 4 | 11 |

**Notable**: We have 0 test files. The original rheo has integration tests for trocar and locomanip but we have none for grasp tasks.

---

## 5. Actionable Recommendations (Priority Order)

1. **Add tests for grasp tasks** -- A `test_eval_grasp_policy.py` that runs `--test` mode (dummy policy smoketest) would catch regressions early. Follows the existing `test_integration_eval_assemble_trocar.py` pattern. Both Dex3 and Inspire should have one.

2. **Break the trocar import dependency** -- Copy `CameraBaseCfg`, `CameraPresets`, `G1RobotPresets`, and `get_robot_body_joint_states` into `grasp_policy/` modules (approximately 50 lines). Our tasks become fully independent of trocar internals.

3. **DRY up eval scripts** -- Extract the shared eval loop into `examples/utils.py`. Each eval script becomes a thin approximately 50-line wrapper.

4. **Clean up stale datasets** -- Delete `inspire_ftp_new/` directory and empty placeholder files.

5. **Consider naming consistency** -- Our Inspire utils use `inspire_ftp_` prefix but the task dir is `grasp_policy_inspire`. Minor, but could standardize if we ever contribute.

---

## 6. What NOT to Change

- Any file in `assemble_trocar/` (original)
- Any file in `environments/` (original)
- Any locomanip task files (original)
- `agents/` directory (original)
- `handtracking.py` Dex3 gripper retargeter (original, in IsaacLab third_party)
- Legacy recording scripts (original)
- `tasks/terminations.py` (original, locomanip-specific)
