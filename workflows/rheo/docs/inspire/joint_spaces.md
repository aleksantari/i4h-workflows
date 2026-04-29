<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Inspire FTP Joint-Space Audit — Anchors and Derivations

> **Status:** Audit only. No code changes implied by this doc.
> **Last validated against:** the post-reorg `grasp-policy` branch (April 2026).

This is a contract reference, scoped to **joint orderings and joint-name conventions** in the Inspire FTP pipeline. Read it before changing the USD asset, the mimic set, the 26D / 13D / 41D layouts, or any of the scatter / index tables. It is the companion to [`pipeline_contracts.md`](pipeline_contracts.md) — that doc covers the pipeline layer-by-layer; this one covers the *joint-space plumbing* that runs through every layer.

The pipeline carries **15+ separate joint-name and joint-order constants** spread across 6 files in **3 naming conventions**, authored at different times during the Dex3 → Inspire FTP embodiment switch. Several are *re-authored copies* that should be derivations of a single anchor; nothing in the codebase enforces they stay synced today. The goal here is to declare the anchor, enumerate every other ordering as a derivation from it, and flag the duplicates that are drift hotspots.

---

## 1. The anchor

The ground truth is the **physical asset**, not any Python list. The chain is:

```
URDF  ──► USD  ──► loaded articulation  ──► env_cfg.joint_names (hand-authored mirror)
```

Concretely:

| Step | File | Role |
|---|---|---|
| **URDF (truth)** | [`assets/robots/g1-29dof-inspire-ftp-urdf-wrist_cam/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf`](../../assets/robots/g1-29dof-inspire-ftp-urdf-wrist_cam/) | Authoritative kinematic + mimic spec. URDF naming throughout (`left_index_1_joint`). |
| **URDF → USD converter** | [`scripts/utils/inspire/convert_inspire_urdf_to_usd.py`](../../scripts/utils/inspire/convert_inspire_urdf_to_usd.py) | Wraps `isaaclab.sim.converters.UrdfConverter`. Three load-bearing flags: `merge_fixed_joints=False` (keeps `d435_link` / `mid360_link` / `imu_in_pelvis` as separate prims so camera transforms survive), `convert_mimic_joints_to_normal_joints=False` (mimic enforcement is our action class's job), `joint_drive` gains zero (overridden by `robot_config.py` actuators at sim startup). |
| **USD (loaded asset)** | [`assets/robots/g1-29dof-inspire-ftp-usd-wrist_cam/g1_29dof_inspire_ftp.usd`](../../assets/robots/g1-29dof-inspire-ftp-usd-wrist_cam/) | Generated from the URDF; what the simulator actually reads. |
| **Sim binding** | [`config/robot_config.py:33-35`](../../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py#L33-L35) (`UNITREE_G1_29DOF_INSPIRE_FTP_USD`) → [`robot_config.py:80-222`](../../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py#L80-L222) (`G129_CFG_WITH_INSPIRE_BASE_FIX`) | Points the `ArticulationCfg` at the USD path, sets actuator gains, default joint positions, gravity-off, fixed root. |
| **Articulation order** | `articulation.data.joint_names` at runtime | The simulator's authoritative list, in USD tree-traversal order. Inaccessible without booting IsaacLab. |
| **Hand-authored mirror** | [`g1_grasp_policy_inspire_env_cfg.py:joint_names` (53 entries)](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L53-L110) | A static Python list that *must match* the articulation order. URDF naming. Used everywhere downstream as the practical anchor. **Verify after any USD swap** (see §4.7). |

The 53-entry layout is 29 body + 24 hand (10 actuated `_1` joints, then 4 mimic `_2`, then 2 actuated `_2` thumbs, then 4 more mimic `_2`, then `_3`/`_4` thumb mimics — interleaved L/R per USD traversal). The order is **load-bearing** for many downstream constants — slicing `joint_names[29:]` to get the 24 hand joints, deriving `actuated_joint_names` by removing mimic entries, computing the `_LEFT_HAND_38D_IDX` / `_RIGHT_HAND_38D_IDX` partition, and indexing into the 41-D `actuated_joint_names` for the policy scatter.

**Caveat:** there is a *second* hand-authored copy of the same 53-joint list in [`inspire_lerobot_fields.py:RECORDED_ACTION_53_JOINT_NAMES`](../../scripts/utils/inspire/inspire_lerobot_fields.py#L94-L150) using Nucleus naming for the hand half. This is a drift hotspot; see §4.1.

---

## 1.5 Grounding status (last updated 2026-04-29)

The L0 + L1 anchor layer + the URDF↔Nucleus / PinkIK teleop layer + the canonical observation layer + the 38-D teleop hand partition + the runtime action / obs behavior are pinned by [`tests/test_sim/test_inspire_urdf_grounding.py`](../../tests/test_sim/test_inspire_urdf_grounding.py). **21 tests, ~45s wall time** (~20s Kit boot, ~3s asserts, ~20s for the runtime mimic settle + obs resets), all green. Coverage on every invocation:

**Layer 1 — URDF spec ↔ code (14 checks):**

- URDF has 53 articulated joints; env_cfg `joint_names` matches in count and name-set.
- URDF's mimic children == `_MIMIC_JOINT_NAMES`.
- URDF's `(child, parent, multiplier)` triples match `MIMIC_RULES`; offset is asserted zero.
- Every articulated joint matches exactly one actuator regex group in `G129_CFG_WITH_INSPIRE_BASE_FIX`; no dead patterns.
- `_URDF_TO_NUCLEUS` covers the full `HAND_JOINT_NAMES` domain, is bijective, and maintains side / finger consistency (catches the *finger-mix-up* bug class — the historical motivator).
- PinkIK `pink_controlled_joint_names` matches exactly the 14 arm joints.
- `_BODY_JOINT_NAMES_CANONICAL` set equals URDF body joints; length 29; no dups.
- `_BODY_JOINT_NAMES_CANONICAL[15:22]` is left arm in canonical order; `[22:29]` is right arm — *the slice contract* downstream consumers depend on.
- `_INSPIRE_ACTUATED_NAMES` set equals URDF actuated hand joints; length 12; 6/6 left-right balance.
- `_INSPIRE_ACTUATED_NAMES[0:6]` is left hand in canonical order; `[6:12]` is right hand.
- `LEFT_HAND_38D_IDX` / `RIGHT_HAND_38D_IDX` partition `[14, 38)` exactly once; per-entry side prefix matches `HAND_JOINT_NAMES`.

**Layer 2 — USD ↔ code (4 checks):**

- env_cfg `joint_names` matches the loaded USD's articulation list-wise — *the* order check Layer 1 explicitly cannot do.
- `_resolve_indices` produces correct articulation indices for canonical body names.
- `_resolve_indices` produces correct articulation indices for canonical hand names.
- Wrist FK link names (`left_wrist_yaw_link` / `right_wrist_yaw_link`) referenced by `record_demos.py` exist on the articulation.

**Layer 3 — runtime behavior (3 checks):**

- `InspireJointPositionAction.apply_actions()` drives mimic joints to `multiplier × parent` after a real `env.step`. Catches refactor regressions in the action class (super()/mimic ordering), future PhysX changes that silently enable native mimic constraint enforcement, and bypass-write scenarios.
- `get_robot_body_joint_states` output values at canonical slots match `DEFAULT_JOINT_POS` (end-to-end body obs chain).
- `get_robot_inspire_joint_states` output is all zeros at default pose (end-to-end hand obs chain).

Run with:

```bash
./docker/run_docker_grasp.sh python -m unittest tests.test_sim.test_inspire_urdf_grounding -v
```

Entries in §3 marked **🛡️** are inside this safety net. Drift hotspots in §4 are individually annotated **Monitored** / **Partially Monitored** / **Open** depending on whether the test catches them today.

Incidentally fixed during the grounding pass: deprecated `effort_limit` / `velocity_limit` fields on `ImplicitActuatorCfg` in [`robot_config.py`](../../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py) (waist + hands groups). `effort_limit` renamed to `effort_limit_sim`; `velocity_limit` deleted as it was silently ignored on implicit actuators. No behavior change.

---

## 2. Naming conventions

Three naming conventions coexist. Each is fixed by an external interface and **cannot be unified**.

| Convention | Example body | Example hand | Used by | Forced by |
|---|---|---|---|---|
| **URDF** | `left_elbow_joint` | `left_index_1_joint` | env_cfg, mimic_action, observations, record_demos, Inspire FTP USD | the URDF/USD asset itself |
| **Nucleus** | (n/a — bodies use URDF) | `L_index_proximal_joint` | DexPilot retargeter (hand-only URDF), `inspire_lerobot_fields.py` for hands | IsaacLab's hand-only URDF that DexPilot loads |
| **LeRobot canonical** | `left_elbow_joint` (URDF) | `L_index_proximal_joint` (Nucleus) | `STATE_26_NAMES_ENV_ORDER`, parquet `info.json`, modality JSON | LeRobot ecosystem convention (datasets must round-trip) |

The bridge between URDF and Nucleus for the 24 hand joints is the 24-key dictionary [`_URDF_TO_NUCLEUS`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py#L64-L91) in `g1_grasp_policy_inspire_teleop_env_cfg.py`. Notable per-joint correspondences:

- `*_index_1_joint` ↔ `*_index_proximal_joint`, `*_index_2_joint` ↔ `*_index_intermediate_joint`
- `*_little_1_joint` ↔ `*_pinky_proximal_joint` (note the rename: `little` ↔ `pinky`)
- `*_thumb_1_joint` ↔ `*_thumb_proximal_yaw_joint`
- `*_thumb_2_joint` ↔ `*_thumb_proximal_pitch_joint`
- `*_thumb_3_joint` ↔ `*_thumb_intermediate_joint`
- `*_thumb_4_joint` ↔ `*_thumb_distal_joint`

Side-prefix bridge: `left_*` ↔ `L_*`, `right_*` ↔ `R_*`. The thumb segment numbering is load-bearing — `_1` is yaw, `_2` is pitch, `_3` is intermediate (mimic), `_4` is distal (mimic chained on `_3`).

---

## 3. Inventory of orderings

Health legend: ✅ = derived in code from the anchor or another canonical · ⚠️ = re-authored (could drift; should later become a derivation) · 🔁 = duplicates another constant in the codebase · **🛡️ = test-pinned** by `test_inspire_urdf_grounding.py` (regression-net coverage; see §1.5).

### L0 — Anchor and immediate derivations

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `joint_names` | [env_cfg.py:53](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L53) | 53 (URDF) | The anchor — hand-authored, must match USD tree-traversal order. | (anchor) 🛡️ |
| `_MIMIC_JOINT_NAMES` | [env_cfg.py:124](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L124) | 12 (URDF) | Set of 12 mimic joint names; could be derived as `set(joint_names) ∩ {*_2_joint excluding thumb_2, *_thumb_3_joint, *_thumb_4_joint}`. | ⚠️ 🛡️ |
| `actuated_joint_names` | [env_cfg.py:139](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L139) | 41 (URDF) | `[n for n in joint_names if n not in _MIMIC_JOINT_NAMES]`. | ✅ 🛡️ (transitively, via the two above) |
| `offset_dict` | [env_cfg.py:112](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L112) | 2 entries | Hand-authored: both elbows at `−0.3`. Cancels the `+0.3` baked into parquet `action`. See [`pipeline_contracts.md`](pipeline_contracts.md) elbow-chain section. | (load-bearing constant) |
| Actuator regex patterns | [robot_config.py:109-221](../../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py#L109-L221) | 5 groups × 1-7 patterns each | `joint_names_expr` lists inside `G129_CFG_WITH_INSPIRE_BASE_FIX.actuators`. Every articulated joint must match exactly one (group, pattern); no dead patterns. | ⚠️ 🛡️ |

### L1 — Mimic chain

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `_MIMIC_RULES_PER_SIDE` | [mimic_action.py:42](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py#L42-L51) | 6 (parent → mimic, mult) | Hand-authored. **Order is load-bearing**: `thumb_3` must precede `thumb_4` because `thumb_4`'s parent is itself a mimic. | (load-bearing constant) 🛡️ (via `MIMIC_RULES`) |
| `MIMIC_RULES` | [mimic_action.py:54](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/mimic_action.py#L54-L57) | 12 | Expansion of `_MIMIC_RULES_PER_SIDE` over `("left", "right")`. | ✅ 🛡️ |
| Multipliers (`1.0843`, `0.8024`, `0.9487`) | mimic_action.py:42-50 | inline | Per the Inspire FTP URDF `<mimic>` tags. | (URDF-fixed constant) 🛡️ |

### L1 — Observation canonical

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `_BODY_JOINT_NAMES_CANONICAL` | [observations.py:39](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py#L39-L69) | 29 (URDF) | Hand-authored body order **intentionally different from USD**. Interleaved L/R groups by body part (legs together, waist together, then arms), so arm joints land at fixed `[15:22]` (left) and `[22:29]` (right). | ⚠️ 🛡️ (set parity + slice contract + bridge + obs layout) |
| `_INSPIRE_ACTUATED_NAMES` | [observations.py:72](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py#L72-L85) | 12 (URDF) | Hand-authored 12-actuated-hand-joint order: `[L thumb_1, L thumb_2, L index_1, L middle_1, L ring_1, L little_1, R thumb_1 …]`. Same canonical hand order encoded in `STATE_26_NAMES_ENV_ORDER[14:26]` under Nucleus naming. | ⚠️ 🛡️ (set parity + slice contract + bridge + obs layout) |

### L1 — Teleop retargeter

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `HAND_JOINT_NAMES` | [teleop_env_cfg.py:58](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py#L58) | 24 (URDF) | `joint_names[29:]`. | ✅ |
| `_URDF_TO_NUCLEUS` | [teleop_env_cfg.py:64](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py#L64-L91) | 24 keys | Hand-authored bridge dict between URDF and Nucleus naming for the 24 hand joints. | (bridge — not a duplicate) |
| `RETARGETER_HAND_JOINT_NAMES` | [teleop_env_cfg.py:93](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py#L93) | 24 (Nucleus) | `[_URDF_TO_NUCLEUS[n] for n in HAND_JOINT_NAMES]`. | ✅ |

### L1 — Single-arm masking (38D action)

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `LEFT_HAND_38D_IDX` | [teleop_env_cfg.py](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py) | 12 ints | ✅ Derived: `[14 + i for i, n in enumerate(HAND_JOINT_NAMES) if n.startswith("left_")]`. `record_demos.py` imports this. | ✅ 🛡️ |
| `RIGHT_HAND_38D_IDX` | [teleop_env_cfg.py](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py) | 12 ints | ✅ Derived: symmetric for `"right_"`. | ✅ 🛡️ |

### L2 — LeRobot 26D dual-arm

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `STATE_26_GROUP_ORDER` | [inspire_lerobot_fields.py:26](../../scripts/utils/inspire/inspire_lerobot_fields.py#L26) | 4-tuple | `("left_arm", "right_arm", "left_hand", "right_hand")`. | (axiom of the 26D layout) |
| `STATE_26_NAMES_ENV_ORDER` | [inspire_lerobot_fields.py:30](../../scripts/utils/inspire/inspire_lerobot_fields.py#L30-L61) | 26 (LeRobot) | Hand-authored: `[14 arm (URDF) + 12 hand (Nucleus)]`. Same 12-hand block as `_INSPIRE_ACTUATED_NAMES` after URDF↔Nucleus rename. | ⚠️ |
| `INSPIRE_ACTUATED_JOINT_INDICES` | [inspire_lerobot_fields.py:65](../../scripts/utils/inspire/inspire_lerobot_fields.py#L65-L68) | 12 ints | Indices of the 12 actuated hand joints inside the 53-D action — derivable from `RECORDED_ACTION_53_JOINT_NAMES`. | ⚠️ |
| `ARM_JOINT_INDICES` | [inspire_lerobot_fields.py:71](../../scripts/utils/inspire/inspire_lerobot_fields.py#L71-L74) | 14 ints | Indices of the 14 arm joints inside the 53-D action — derivable from `RECORDED_ACTION_53_JOINT_NAMES`. | ⚠️ |
| `STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA` | [inspire_lerobot_fields.py:77](../../scripts/utils/inspire/inspire_lerobot_fields.py#L77-L79) | 26 (np) | Zeros except `+0.3` at indices 3 (left elbow) and 10 (right elbow). The `+0.3` side of the elbow-offset chain. | (load-bearing constant) |
| `STATE_26_BODY_COL_LEFT_ARM` / `_RIGHT_ARM` | [inspire_lerobot_fields.py:84-85](../../scripts/utils/inspire/inspire_lerobot_fields.py#L84-L85) | `range(15, 22)` / `range(22, 29)` | Column slices into the 87-D body observation that yield each arm. Pinned by `_BODY_JOINT_NAMES_CANONICAL` design. | ✅ (slice constants) |
| `STATE_26_INSPIRE_COL_LEFT_HAND` / `_RIGHT_HAND` | [inspire_lerobot_fields.py:88-89](../../scripts/utils/inspire/inspire_lerobot_fields.py#L88-L89) | `range(0, 6)` / `range(6, 12)` | Column slices into the 12-D hand observation. Pinned by `_INSPIRE_ACTUATED_NAMES` design. | ✅ (slice constants) |

### L2 — LeRobot 53D recorded action

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `RECORDED_ACTION_53_JOINT_NAMES` | [inspire_lerobot_fields.py:94](../../scripts/utils/inspire/inspire_lerobot_fields.py#L94-L150) | 53 (LeRobot — URDF body, Nucleus hand) | Hand-authored — second copy of the USD ordering, with the hand half renamed via `_URDF_TO_NUCLEUS`. | 🔁 (drift hotspot — see §4) |
| `_MIMIC_JOINT_NAMES_NUCLEUS` | [inspire_lerobot_fields.py:164](../../scripts/utils/inspire/inspire_lerobot_fields.py#L164-L177) | 12 (Nucleus) | Hand-authored — second copy of `_MIMIC_JOINT_NAMES`, renamed via `_URDF_TO_NUCLEUS`. | 🔁 (drift hotspot — see §4) |
| `RECORDED_ACTION_41_JOINT_NAMES` | [inspire_lerobot_fields.py:179](../../scripts/utils/inspire/inspire_lerobot_fields.py#L179-L181) | 41 (LeRobot) | `tuple(n for n in RECORDED_ACTION_53_JOINT_NAMES if n not in _MIMIC_JOINT_NAMES_NUCLEUS)`. | ✅ |
| `ACTION_HDF5_TO_ENV_26` | [inspire_lerobot_fields.py:156](../../scripts/utils/inspire/inspire_lerobot_fields.py#L156-L158) | 26 ints | `[name_to_idx_53[n] for n in STATE_26_NAMES_ENV_ORDER]`. | ✅ |
| `ACTION_HDF5_TO_ENV_26_FROM_41` | [inspire_lerobot_fields.py:186](../../scripts/utils/inspire/inspire_lerobot_fields.py#L186-L188) | 26 ints | Same as above but using the 41-D recorded order. | ✅ |

### L2 — LeRobot 13D single-arm

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `STATE_13_NAMES_ENV_ORDER` | [inspire_lerobot_fields.py:195](../../scripts/utils/inspire/inspire_lerobot_fields.py#L195-L211) | 13 (LeRobot) | Right-arm subset of `STATE_26_NAMES_ENV_ORDER`. | ⚠️ (could derive — it's `STATE_26_NAMES_ENV_ORDER[7:14] + [20:26]`) |
| `STATE_13_LEFT_NAMES_ENV_ORDER` | inspire_lerobot_fields.py:264 | 13 (LeRobot) | Left-arm subset. Mirror image. | ⚠️ |
| `STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA` | [inspire_lerobot_fields.py:214](../../scripts/utils/inspire/inspire_lerobot_fields.py#L214-L215) | 13 (np) | `+0.3` at index 3 only (the single arm's elbow). | (load-bearing constant) |
| `STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA` | inspire_lerobot_fields.py | 13 (np) | Same as above; left elbow at index 3. | (load-bearing constant) |
| `ACTION_HDF5_TO_ENV_13` / `_FROM_41` (right & left) | inspire_lerobot_fields.py:218-222 + left mirror | 13 ints each | Index maps from 53-D and 41-D recorded actions into the 13-D layouts. | ✅ |

### L3 — Policy / scatter

| Constant | File:line | Shape | Source / derivation | Health |
|---|---|---|---|---|
| `GROUP_SIZES` | [inspire_experiment_config.py:38](../../scripts/utils/inspire/inspire_experiment_config.py#L38-L43) | 4 entries | `{left_arm: 7, right_arm: 7, left_hand: 6, right_hand: 6}`. | (axiom of the 26D layout) |
| `ARM_BODY_RANGES` | [inspire_experiment_config.py:46](../../scripts/utils/inspire/inspire_experiment_config.py#L46-L49) | 2 entries | `{left_arm: (15, 22), right_arm: (22, 29)}`. Mirrors `STATE_26_BODY_COL_*`. | ⚠️ (duplicates the slice constants) |
| `HAND_INSPIRE_RANGES` | [inspire_experiment_config.py:52](../../scripts/utils/inspire/inspire_experiment_config.py#L52-L55) | 2 entries | `{left_hand: (0, 6), right_hand: (6, 12)}`. Mirrors `STATE_26_INSPIRE_COL_*`. | ⚠️ (duplicates the slice constants) |
| `GROUP_SIM_INDICES` | [inspire_experiment_config.py:67](../../scripts/utils/inspire/inspire_experiment_config.py#L67-L76) | 4 lists, 26 ints total | Per-group scatter into the 41-D action space. **Runtime-verified** by `verify_scatter_indices.py`; see [`scatter_indices.md`](scatter_indices.md) for the audit trail (pinky/middle bug history). | ⚠️ — derivable from `actuated_joint_names.index(joint_name)` for each canonical group order |
| `DEFAULT_JOINT_GROUPS` / `DEFAULT_CAMERAS` | inspire_experiment_config.py:100-104 | small lists/dicts | Defaults for `InspireExperimentConfig`. | (defaults) |
| `SIM_ACTION_DIM` | inspire_experiment_config.py:78 | `41` | Constant. | ✅ (= `len(actuated_joint_names)`) |

---

## 4. Drift hotspots — duplicates that should later become derivations

These are the constants that *encode the same fact* as another constant in the codebase. They are the highest-leverage refactor targets, because a desync silently corrupts downstream pipelines. Each callout names the truth, the duplicate, and what would break.

### 4.1 `RECORDED_ACTION_53_JOINT_NAMES` ↔ `joint_names`

**Truth:** `joint_names` (env_cfg.py, URDF) is the anchor — it's what the simulator actually sees.
**Duplicate:** `RECORDED_ACTION_53_JOINT_NAMES` (inspire_lerobot_fields.py, body in URDF + hand in Nucleus) is a parallel hand-authored copy.
**Derivation rule:** `tuple(n if i < 29 else _URDF_TO_NUCLEUS[n] for i, n in enumerate(joint_names))`.
**Failure mode if desynced:** `ACTION_HDF5_TO_ENV_26` (and friends) compute index lookups against the duplicate; if its 53-entry order drifts from the env's, the converter routes recorded actions to the wrong joints. Symptom: the policy trains on cleanly-converted parquet, but the converted parquet is wrong. Hard to detect without a roundtrip test (parquet → 41-D scatter → compare to recorded).
**Status:** **Open.** Foundation (`joint_names` + `_URDF_TO_NUCLEUS`) is now test-pinned, so a derivation-equality check is straightforward to add.

### 4.2 `_MIMIC_JOINT_NAMES_NUCLEUS` ↔ `_MIMIC_JOINT_NAMES`

**Truth:** `_MIMIC_JOINT_NAMES` (env_cfg.py, URDF) — used by `actuated_joint_names` filtering and by `InspireJointPositionAction`.
**Duplicate:** `_MIMIC_JOINT_NAMES_NUCLEUS` (inspire_lerobot_fields.py) — used to build `RECORDED_ACTION_41_JOINT_NAMES`.
**Derivation rule:** `{_URDF_TO_NUCLEUS[n] for n in _MIMIC_JOINT_NAMES}`.
**Failure mode if desynced:** the 53-D → 41-D filter on the data side excludes a different mimic set than the env action manager uses, so the recorded-action conversion writes to (or skips) the wrong joints. Same silent corruption mode as 4.1.
**Status:** **Open.** `_MIMIC_JOINT_NAMES` is now test-pinned; one extra assertion against `_URDF_TO_NUCLEUS` closes this.

### 4.3 `_BODY_JOINT_NAMES_CANONICAL` ↔ `joint_names[:29]`

**Truth:** `joint_names[:29]` is the USD body order; `_BODY_JOINT_NAMES_CANONICAL` is *intentionally* a different order (interleaved by body part — legs, waist, arms — so arms land at slices `[15:22]` / `[22:29]`).
**Relationship:** `_BODY_JOINT_NAMES_CANONICAL` is a **deliberate reorder spec**, not a duplicate. It must be the exact same *set* as `joint_names[:29]`, just permuted.
**Derivation rule:** `set(_BODY_JOINT_NAMES_CANONICAL) == set(joint_names[:29])` is the invariant; the order is a hand-authored design choice (and load-bearing for downstream slicing).
**Failure mode if desynced:** if a body joint is added/removed in the USD but the canonical-order list isn't updated, observation extraction silently produces zeros (or raises — the `_resolve_indices` lookup falls through to `KeyError`). Less silent than 4.1/4.2 but worth a set-equality test.
**Status:** **Monitored** (2026-04-29). `test_body_canonical_set_matches` pins set parity + length 29 + no duplicates. `test_body_canonical_arm_slice_contract` additionally pins the slice positions (`[15:22]` left arm, `[22:29]` right arm) so a deliberate-reorder regression fails loudly. `test_body_canonical_resolves_to_articulation` and `test_body_obs_layout_at_default_pose` cover the bridge end-to-end.

### 4.4 `_INSPIRE_ACTUATED_NAMES` ↔ `STATE_26_NAMES_ENV_ORDER[14:26]`

**Truth:** these are the same 12 actuated hand joints, in the same canonical order. `_INSPIRE_ACTUATED_NAMES` uses URDF; `STATE_26_NAMES_ENV_ORDER[14:26]` uses Nucleus.
**Derivation rule:** `[_URDF_TO_NUCLEUS[n] for n in _INSPIRE_ACTUATED_NAMES] == STATE_26_NAMES_ENV_ORDER[14:26]`.
**Failure mode if desynced:** the 12-D hand observation would slice to a different per-finger order than the LeRobot 26-D state, so observations and actions disagree about which dim is "left index" vs "left middle." Symptom at training: nontrivially-mistrained policy that fails on the same fingers it was trained on. This is exactly the class of bug the user has flagged historically.
**Status:** **Partially Monitored** (2026-04-29). The URDF-side anchor `_INSPIRE_ACTUATED_NAMES` is now test-pinned: set equality with URDF actuated hand joints, hand slice contract (`[0:6]` left, `[6:12]` right), bridge correctness, and runtime obs layout — see `test_inspire_actuated_*` and `test_inspire_obs_layout_at_default_pose`. The cross-file derivation against `STATE_26_NAMES_ENV_ORDER[14:26]` (via `_URDF_TO_NUCLEUS`) is **not yet tested**; defer to the `inspire_lerobot_fields.py` audit pass when that file is reviewed.

### 4.5 `_LEFT_HAND_38D_IDX` / `_RIGHT_HAND_38D_IDX` ↔ side-prefix partition of `HAND_JOINT_NAMES`

**Truth:** `HAND_JOINT_NAMES` (= `joint_names[29:]`) is the source-of-truth ordering of the 24 hand joints inside the 38-D teleop action's hand block (which starts at index 14).
**Derivation:** [`teleop_env_cfg.py`](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_teleop_env_cfg.py) now defines `LEFT_HAND_38D_IDX` and `RIGHT_HAND_38D_IDX` as comprehensions: `[14 + i for i, n in enumerate(HAND_JOINT_NAMES) if n.startswith("left_")]` (and symmetric for right). `record_demos.py` imports these — no longer hardcoded.
**Failure mode if desynced:** single-arm teleop masking writes to the wrong hand or skips fingers — the bug class the user explicitly mentioned. Adding a hand joint to the USD without updating the partition would silently misroute teleop commands.
**Status:** **Monitored** (2026-04-29). The hardcoded duplicates in `record_demos.py` were replaced with imports of the comprehensions in `teleop_env_cfg.py`. `test_hand_38d_partition_by_side` pins the partition: lengths (12 each), no overlap, exact coverage of `[14, 38)`, and per-entry side-prefix correctness against `HAND_JOINT_NAMES`.

### 4.6 `GROUP_SIM_INDICES` ↔ `actuated_joint_names` lookup per canonical group order

**Truth:** the 41-D `actuated_joint_names` ordering is the simulator's truth. Each canonical group (left_arm, right_arm, left_hand, right_hand) has a canonical per-joint order (shoulder_pitch / shoulder_roll / shoulder_yaw / elbow / wrist_roll / wrist_pitch / wrist_yaw for arms; thumb_yaw / thumb_pitch / index / middle / ring / pinky for hands).
**Duplicate:** `GROUP_SIM_INDICES` is a hand-authored 4-list table giving the 41-D action index for each canonical-group entry.
**Derivation rule:** for each group, `[actuated_joint_names.index(j) for j in canonical_group_order(group)]`.
**Failure mode if desynced:** the policy's 26-D output scatters to the wrong 41-D positions — exactly the pinky/middle bug captured in [`scatter_indices.md`](scatter_indices.md) (April 2026). The current `GROUP_SIM_INDICES` values are runtime-verified; the hazard is future drift, e.g. someone reorders `actuated_joint_names` (via a USD or filter change) without re-verifying the scatter.
**Status:** **Open.** `actuated_joint_names` is now test-pinned, so the index-lookup derivation is well-defined and a regression test would be tractable.

### 4.7 Asset-side hazards — converter / inspector point at the wrong USD

Two findings about the URDF→USD→sim chain itself (the chain in §1) that can silently desync the anchor (`joint_names`) from the actually-loaded articulation.

**Hazard A — `inspect_inspire_joints.py` inspects a different USD than the one the env loads.** *(Resolved 2026-04-28.)*

Previously [`inspect_inspire_joints.py`](../../scripts/utils/inspire/inspect_inspire_joints.py) hard-coded the upstream Nucleus USD path (`{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof_inspire_hand.usd`) — the one that's missing `d435_link` and the wrist-cam mounts. The fix imports `UNITREE_G1_29DOF_INSPIRE_FTP_USD` from `robot_config.py` so the inspector now loads the same local USD the env uses. The classification keywords were also corrected to URDF naming (`_thumb_1_joint` etc.) so the actuated / mimic split is meaningful for the local asset. Steady-state coverage of the same contract is now provided by [`tests/test_sim/test_inspire_urdf_grounding.py`](../../tests/test_sim/test_inspire_urdf_grounding.py); the inspector remains for one-off introspection (new embodiments, deep debug).

**Hazard B — the URDF→USD converter targets the non-wrist-cam variant; the wrist-cam USD has no script of record.**

There are *four* asset directories on disk:

| Directory | Contents |
|---|---|
| `assets/robots/g1-29dof-inspire-ftp-urdf/` | Source URDF, no wrist-cam mounts |
| `assets/robots/g1-29dof-inspire-ftp-urdf-wrist_cam/` | Source URDF *with* wrist-cam mounts |
| `assets/robots/g1-29dof-inspire-ftp-usd/` | USD generated from the no-wrist-cam URDF |
| `assets/robots/g1-29dof-inspire-ftp-usd-wrist_cam/` | USD with wrist-cam — **what `robot_config.py` actually loads** |

[`convert_inspire_urdf_to_usd.py:48-52`](../../scripts/utils/inspire/convert_inspire_urdf_to_usd.py#L48-L52) hard-codes the URDF input as `g1-29dof-inspire-ftp-urdf/...urdf` and the USD output as `g1-29dof-inspire-ftp-usd/`. **Neither is the variant `robot_config.py` loads.** The active wrist-cam USD must therefore have been produced either by a one-off edit to this script or by a manual USD modification — there's no committed path that regenerates it. After a URDF change, re-running the converter as written produces the *non-wrist-cam* USD, which is then ignored.

**Fix options (deferred):**
1. Parameterize the converter to take URDF + output paths as CLI flags; document the wrist-cam invocation.
2. Emit *both* variants in one run (loop over the two URDFs).
3. Delete the non-wrist-cam URDF/USD pair if it's not used anywhere — quick check: `grep -r "g1-29dof-inspire-ftp-usd[^-]" scripts/ tests/ docs/`.

**Failure mode for both hazards:** silent. A USD swap that touches articulation order will pass through the inspector (because it inspects the wrong USD), the converter (because it produces the wrong variant), and `joint_names` (because the mirror is hand-authored and won't be re-verified) — and only manifest as misrouted joints at eval time, exactly the bug class the user has hit historically.

### 4.8 `_URDF_TO_NUCLEUS` values not anchored against IsaacLab's hand URDF

**Truth:** the values in `_URDF_TO_NUCLEUS` (Nucleus-named hand joints like `L_index_proximal_joint`) must exist as joint names in IsaacLab's internal Nucleus hand URDF — the one DexPilot loads inside `UnitreeG1RetargeterCfg`.
**Current grounding:** the URDF *side* of the bridge is test-pinned (every key in `_URDF_TO_NUCLEUS` is a real `HAND_JOINT_NAMES` entry; bijective; per-finger consistent). The Nucleus *side* is **not** statically anchored — we don't import or parse the IsaacLab hand URDF anywhere in the test.
**Failure mode if desynced:** a typo in a Nucleus value (e.g. `L_index_proximate_joint`) would cause `UnitreeG1RetargeterCfg` to raise at *retargeter init* when it can't find the joint in its internal URDF. That fires only when teleop actually starts — so the gap is "stale config sits silently until the next teleop session." Less dangerous than silent misrouting, but later in the cycle than the static tests catch.
**Status:** **Open.** Optional — IsaacLab's runtime check already catches typos at init time, so the value is incremental belt-and-suspenders coverage. Would require finding IsaacLab's Nucleus hand URDF (somewhere under `isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire`), parsing its joint names, and asserting `set(_URDF_TO_NUCLEUS.values()) ⊆ {names in IsaacLab hand URDF}`. Defer until a Nucleus typo bites in practice.

---

## 5. What NOT to consolidate

Three things look like duplication but are deliberate. Leave them alone.

- **The three naming conventions** (URDF / Nucleus / LeRobot canonical). Each is fixed by an external interface (see §2). Unifying them would force a fork of either IsaacLab's hand URDF or the LeRobot ecosystem convention.
- **`_BODY_JOINT_NAMES_CANONICAL` is a deliberate reorder of `joint_names[:29]`** — *not* a redundant copy. The interleaved L/R-by-body-part order makes arm joints land at fixed `[15:22]` / `[22:29]` slices, which keeps every downstream consumer simple. The set-equality invariant in §4.3 is what to enforce; the order itself is design intent.
- **`_URDF_TO_NUCLEUS` is a bridge dict**, not a duplicate. It's the *only* place that encodes the URDF↔Nucleus mapping; every other Nucleus-named constant should derive *through* it.

---

## 6. Implications for tests

**Landed (2026-04-28):** [`tests/test_sim/test_inspire_urdf_grounding.py`](../../tests/test_sim/test_inspire_urdf_grounding.py) — eleven tests across three layers. **Layer 1 (URDF → code, 9 checks)** pins the joint anchor and immediate derivations including the URDF↔Nucleus bridge and PinkIK arm partition. **Layer 2 (USD → code, 1 check)** pins the articulation order via gym.make + reset. **Layer 3 (runtime behavior, 1 check)** pins `InspireJointPositionAction.apply_actions()` by sending a known target, settling, and asserting the mimic ratio is met within tolerance.

**Next — L2/L3 derivation checks for the §4 hotspots.** They follow the same pattern: given the now-test-pinned foundation (`joint_names`, `_MIMIC_JOINT_NAMES`, `MIMIC_RULES`, `_URDF_TO_NUCLEUS`), do the higher-layer constants encode the same fact? Examples:

- `set(_MIMIC_JOINT_NAMES_NUCLEUS) == {_URDF_TO_NUCLEUS[n] for n in _MIMIC_JOINT_NAMES}` (4.2)
- `_LEFT_HAND_38D_IDX == [14 + i for i, n in enumerate(HAND_JOINT_NAMES) if n.startswith("left_")]` (4.5)
- `set(_BODY_JOINT_NAMES_CANONICAL) == set(joint_names[:29])` and `len == 29` (4.3)

Each is a few lines of test code. **They are now safe to author** because the foundation they'd build on top of is already pinned — any regression in that foundation will fail loudly via the existing test before propagating to the higher-layer checks. Without that, every higher-level test would have to re-verify the foundation, making the test surface fragile.

Hotspot 4.4 (`_INSPIRE_ACTUATED_NAMES` ↔ `STATE_26_NAMES_ENV_ORDER[14:26]`) is the highest-priority next target — directly maps to the historical "fingers crossed" bug class.

---

## 7. Implications for refactor

The follow-up refactor (separate plan, separate PR) replaces every ⚠️ / 🔁 entry in §3 with a code-derivation, computed at module load. **Behavior must not change** — the eval smoketest (`./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py --test`), the **grounding test** (`test_inspire_urdf_grounding`), and a parquet round-trip on a known-good HDF5 are the canaries. The grounding test is the most informative of the three: any L1 regression introduced by the refactor fails loudly with a localized message instead of surfacing as silent data corruption later in the pipeline. The Layer 3 runtime check additionally guards the action class itself — any reorder of `super().apply_actions()` vs the mimic write loop in `InspireJointPositionAction` fails the runtime assertion before the refactor PR can land. Hand-authored constants that *should* survive: `joint_names` (anchor), `_BODY_JOINT_NAMES_CANONICAL` (deliberate reorder, see §5), `_URDF_TO_NUCLEUS` (bridge), `_MIMIC_RULES_PER_SIDE` (semantic content), `offset_dict` (load-bearing magnitudes), the canonical group-order specs, multipliers from the URDF.

Concretely, the refactor would:

1. Define `_BODY_JOINT_NAMES_CANONICAL` as the explicit reorder spec (kept hand-authored, but checked against `set(joint_names[:29])` at module load).
2. Compute `_INSPIRE_ACTUATED_NAMES` as a slice of `STATE_26_NAMES_ENV_ORDER` after URDF↔Nucleus translation (or vice versa — whichever side is least-touched by external consumers).
3. Compute `_LEFT_HAND_38D_IDX` / `_RIGHT_HAND_38D_IDX` from `HAND_JOINT_NAMES` side-prefix.
4. Compute `RECORDED_ACTION_53_JOINT_NAMES` and `_MIMIC_JOINT_NAMES_NUCLEUS` from `joint_names` and `_MIMIC_JOINT_NAMES` via `_URDF_TO_NUCLEUS`. (This requires `inspire_lerobot_fields.py` to import from `g1_grasp_policy_inspire_env_cfg.py` — fine, since the converter is run inside Docker where IsaacLab is available; alternatively, hoist the small set of needed constants into a host-importable module.)
5. Compute `GROUP_SIM_INDICES` from `actuated_joint_names` lookups against the canonical group orders.

After the refactor, the §6 derivation tests become trivial (most asserting computed equality against a frozen golden) and a USD swap requires editing only `joint_names` plus running `inspect_inspire_joints.py`.

---

## 8. Cross-links

- [`tests/test_sim/test_inspire_urdf_grounding.py`](../../tests/test_sim/test_inspire_urdf_grounding.py) — the regression net for L0/L1 (this doc's §1.5).
- [`pipeline_contracts.md`](pipeline_contracts.md) — pipeline by layer; this doc deepens its Layer 1 + Layer 2 coverage of joint plumbing.
- [`scatter_indices.md`](scatter_indices.md) — runtime audit of `GROUP_SIM_INDICES` (the pinky/middle bug history).
- [`task_reference.md`](task_reference.md) — gym IDs, action / obs spaces.
- [`elbow_offset.md`](elbow_offset.md) — the `+0.3` / `−0.3` offset chain.
- [`debug_policy.md`](debug_policy.md) — eval-time diagnostic playbook.
- [`scripts/utils/inspire/inspect_inspire_joints.py`](../../scripts/utils/inspire/inspect_inspire_joints.py) — one-off introspection of the active USD (bootstrap / deep debug; complements the test, not a replacement).
- [`scripts/utils/inspire/verify_scatter_indices.py`](../../scripts/utils/inspire/verify_scatter_indices.py) — verify `GROUP_SIM_INDICES` against the running env.
