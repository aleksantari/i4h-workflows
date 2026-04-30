# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for G1 + Inspire FTP joint identity constants.

Pure Python (stdlib types only) — no IsaacLab, no torch, no numpy. This makes
the module importable from any context, including:

- ``simulation/tasks/grasp_policy_inspire/*`` (env, mimic, observations) — IsaacLab boots
- ``utils/inspire/inspire_lerobot_fields.py`` (HDF5→LeRobot converter) — non-Kit Python
- ``tests/test_sim/test_inspire_urdf_grounding.py`` (Layer 1 grounding) — inside Docker

Hoisted from previously-scattered hand-authored copies in env_cfg.py,
teleop_env_cfg.py, observations.py, mimic_action.py, and inspire_lerobot_fields.py.
After this hoist, those files import + alias from here so existing public names
(``joint_names``, ``_MIMIC_JOINT_NAMES``, etc.) keep working.

**Why this lives at the top of ``scripts/`` rather than in ``scripts/utils/inspire/``:**
when AppLauncher boots Kit, OpenCV (``cv2``) gets imported and its initializer
prepends ``cv2/utils/`` to sys.path, shadowing any subsequent ``utils.X``
import. Putting the module at the top level under ``scripts/`` makes it
importable as ``from inspire_joint_constants import ...`` — no ``utils.``
prefix, no shadow.

The URDF asset (`assets/robots/g1-29dof-inspire-ftp-urdf-wrist_cam/`) is the
ultimate truth. ``JOINT_NAMES`` below is a hand-authored mirror of the USD's
articulation order and **must be re-verified after any USD swap** via:

    ./docker/run_docker_grasp.sh python scripts/utils/inspire/inspect_inspire_joints.py

The grounding test (`tests/test_sim/test_inspire_urdf_grounding.py`) pins the
mirror to the running articulation list-wise on every invocation.

See `docs/inspire/joint_spaces.md` for the architectural rationale.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hand-authored sources of truth
# ---------------------------------------------------------------------------

# 53-joint USD articulation order: 29 body + 24 hand (10 actuated _1, then
# 4 mimic _2, 2 actuated thumb _2, 4 more mimic _2, then thumb _3/_4 mimics —
# interleaved L/R per USD tree traversal). URDF naming throughout.
JOINT_NAMES: list[str] = [
    # --- Body (29): USD articulation tree traversal order ---
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8
    "left_knee_joint",            # 9
    "right_knee_joint",           # 10
    "left_shoulder_pitch_joint",  # 11
    "right_shoulder_pitch_joint", # 12
    "left_ankle_pitch_joint",     # 13
    "right_ankle_pitch_joint",    # 14
    "left_shoulder_roll_joint",   # 15
    "right_shoulder_roll_joint",  # 16
    "left_ankle_roll_joint",      # 17
    "right_ankle_roll_joint",     # 18
    "left_shoulder_yaw_joint",    # 19
    "right_shoulder_yaw_joint",   # 20
    "left_elbow_joint",           # 21
    "right_elbow_joint",          # 22
    "left_wrist_roll_joint",      # 23
    "right_wrist_roll_joint",     # 24
    "left_wrist_pitch_joint",     # 25
    "right_wrist_pitch_joint",    # 26
    "left_wrist_yaw_joint",       # 27
    "right_wrist_yaw_joint",      # 28
    # --- Hands (24): left/right interleaved, actuated then mimic ---
    "left_index_1_joint",         # 29 [actuated]
    "left_little_1_joint",        # 30 [actuated]
    "left_middle_1_joint",        # 31 [actuated]
    "left_ring_1_joint",          # 32 [actuated]
    "left_thumb_1_joint",         # 33 [actuated]
    "right_index_1_joint",        # 34 [actuated]
    "right_little_1_joint",       # 35 [actuated]
    "right_middle_1_joint",       # 36 [actuated]
    "right_ring_1_joint",         # 37 [actuated]
    "right_thumb_1_joint",        # 38 [actuated]
    "left_index_2_joint",         # 39 [mimic]
    "left_little_2_joint",        # 40 [mimic]
    "left_middle_2_joint",        # 41 [mimic]
    "left_ring_2_joint",          # 42 [mimic]
    "left_thumb_2_joint",         # 43 [actuated]
    "right_index_2_joint",        # 44 [mimic]
    "right_little_2_joint",       # 45 [mimic]
    "right_middle_2_joint",       # 46 [mimic]
    "right_ring_2_joint",         # 47 [mimic]
    "right_thumb_2_joint",        # 48 [actuated]
    "left_thumb_3_joint",         # 49 [mimic]
    "right_thumb_3_joint",        # 50 [mimic]
    "left_thumb_4_joint",         # 51 [mimic]
    "right_thumb_4_joint",        # 52 [mimic]
]

# 12 mimic joint names (URDF naming). These joints are passive — driven by
# the custom InspireJointPositionAction.apply_actions() from their parents,
# never directly commanded by the 41-D action manager.
MIMIC_JOINT_NAMES: set[str] = {
    "left_index_2_joint",
    "left_little_2_joint",
    "left_middle_2_joint",
    "left_ring_2_joint",
    "right_index_2_joint",
    "right_little_2_joint",
    "right_middle_2_joint",
    "right_ring_2_joint",
    "left_thumb_3_joint",
    "right_thumb_3_joint",
    "left_thumb_4_joint",
    "right_thumb_4_joint",
}

# URDF → Nucleus rename for the 24 hand joints. The Nucleus naming convention
# is what IsaacLab's DexPilot retargeter and the LeRobot canonical 26-D state
# layout use for the hand half. URDF uses {left,right}_<finger>_<seg>_joint;
# Nucleus uses {L,R}_<finger>_<segment>_joint with a few renames (note the
# only finger-token rename: URDF "little" ↔ Nucleus "pinky").
URDF_TO_NUCLEUS: dict[str, str] = {
    # Left hand
    "left_index_1_joint": "L_index_proximal_joint",
    "left_index_2_joint": "L_index_intermediate_joint",
    "left_little_1_joint": "L_pinky_proximal_joint",
    "left_little_2_joint": "L_pinky_intermediate_joint",
    "left_middle_1_joint": "L_middle_proximal_joint",
    "left_middle_2_joint": "L_middle_intermediate_joint",
    "left_ring_1_joint": "L_ring_proximal_joint",
    "left_ring_2_joint": "L_ring_intermediate_joint",
    "left_thumb_1_joint": "L_thumb_proximal_yaw_joint",
    "left_thumb_2_joint": "L_thumb_proximal_pitch_joint",
    "left_thumb_3_joint": "L_thumb_intermediate_joint",
    "left_thumb_4_joint": "L_thumb_distal_joint",
    # Right hand
    "right_index_1_joint": "R_index_proximal_joint",
    "right_index_2_joint": "R_index_intermediate_joint",
    "right_little_1_joint": "R_pinky_proximal_joint",
    "right_little_2_joint": "R_pinky_intermediate_joint",
    "right_middle_1_joint": "R_middle_proximal_joint",
    "right_middle_2_joint": "R_middle_intermediate_joint",
    "right_ring_1_joint": "R_ring_proximal_joint",
    "right_ring_2_joint": "R_ring_intermediate_joint",
    "right_thumb_1_joint": "R_thumb_proximal_yaw_joint",
    "right_thumb_2_joint": "R_thumb_proximal_pitch_joint",
    "right_thumb_3_joint": "R_thumb_intermediate_joint",
    "right_thumb_4_joint": "R_thumb_distal_joint",
}

# Canonical 29-joint body order — *intentionally different* from
# JOINT_NAMES[:29]. Interleaved by body part (legs, then waist, then arms),
# so that arm joints land at fixed contiguous slices [15:22] (left) and
# [22:29] (right). Downstream consumers slice this layout in observations
# and LeRobot conversion. Set parity with JOINT_NAMES[:29] is the invariant;
# the order itself is a design choice.
BODY_JOINT_NAMES_CANONICAL: list[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",   # 15 — start of left arm slice
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",  # 22 — start of right arm slice
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Canonical 12-joint hand order (URDF naming): thumb_yaw (=_1), thumb_pitch
# (=_2), index_1, middle_1, ring_1, little_1 — left first, then right.
# Slice [0:6] = left hand; [6:12] = right hand. Pinned by the slice contract
# test against this layout.
INSPIRE_ACTUATED_NAMES: list[str] = [
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_index_1_joint",
    "left_middle_1_joint",
    "left_ring_1_joint",
    "left_little_1_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
]

# Per-side mimic rule template. Order is **load-bearing** for the thumb chain:
# thumb_3 must be computed before thumb_4 because thumb_4's parent is itself
# a mimic. Reorder alphabetically and the distal thumb segment silently zeros.
# Multipliers come from the URDF <mimic multiplier="..."> tags; the grounding
# test pins these against the URDF on every invocation.
_MIMIC_RULES_TEMPLATE: list[tuple[str, str, float]] = [
    # Finger _2 mimics _1 (proximal) at 1.0843×
    ("{side}_index_2_joint", "{side}_index_1_joint", 1.0843),
    ("{side}_middle_2_joint", "{side}_middle_1_joint", 1.0843),
    ("{side}_ring_2_joint", "{side}_ring_1_joint", 1.0843),
    ("{side}_little_2_joint", "{side}_little_1_joint", 1.0843),
    # Thumb chain: _3 mimics _2 (proximal pitch); _4 mimics _3 (intermediate)
    ("{side}_thumb_3_joint", "{side}_thumb_2_joint", 0.8024),
    ("{side}_thumb_4_joint", "{side}_thumb_3_joint", 0.9487),
]

# 12 mimic relationships expanded for both hands. (child, parent, multiplier).
MIMIC_RULES: list[tuple[str, str, float]] = [
    (mimic_tmpl.format(side=side), parent_tmpl.format(side=side), mult)
    for side in ("left", "right")
    for mimic_tmpl, parent_tmpl, mult in _MIMIC_RULES_TEMPLATE
]


# ---------------------------------------------------------------------------
# Derived constants — computed at module load from the sources above.
# ---------------------------------------------------------------------------

# 41 actuated joints (29 body + 12 actuated hand) in USD articulation order.
# This is the layout of the Joint / Joint-Eval env's 41-D action vector.
ACTUATED_JOINT_NAMES: list[str] = [n for n in JOINT_NAMES if n not in MIMIC_JOINT_NAMES]

# 24 hand joints in USD articulation order (= JOINT_NAMES[29:]). Defines the
# layout of the 38-D teleop action's hand block (positions 14:38).
HAND_JOINT_NAMES: list[str] = JOINT_NAMES[29:]

# 24 hand joint names in Nucleus naming, in URDF-A order. Passed to
# UnitreeG1RetargeterCfg as ``hand_joint_names`` so DexPilot output lands at
# the same positions as our 38-D teleop action's hand block.
RETARGETER_HAND_JOINT_NAMES: list[str] = [URDF_TO_NUCLEUS[n] for n in HAND_JOINT_NAMES]

# 12 mimic joints in Nucleus naming.
MIMIC_JOINT_NAMES_NUCLEUS: set[str] = {URDF_TO_NUCLEUS[n] for n in MIMIC_JOINT_NAMES}

# 53-joint USD order with hand joints renamed to Nucleus. Used by the
# HDF5→LeRobot converter for legacy 53-D recorded actions.
JOINT_NAMES_NUCLEUS_HAND: tuple[str, ...] = tuple(
    n if i < 29 else URDF_TO_NUCLEUS[n] for i, n in enumerate(JOINT_NAMES)
)

# 38-D teleop action hand-block partition by side. The hand block at indices
# 14:38 follows HAND_JOINT_NAMES (URDF-A) order, which interleaves L/R per
# the USD traversal — so per-side masking can NOT use a contiguous slice.
LEFT_HAND_38D_IDX: list[int] = [
    14 + i for i, n in enumerate(HAND_JOINT_NAMES) if n.startswith("left_")
]
RIGHT_HAND_38D_IDX: list[int] = [
    14 + i for i, n in enumerate(HAND_JOINT_NAMES) if n.startswith("right_")
]

# Joint group → ordered name list (URDF naming). Anchors the policy-state
# joint groups consumed by Inspire FTP ACT experiments. Slices of the
# canonical body / actuated-hand layouts above; arm slices are contiguous in
# BODY_JOINT_NAMES_CANONICAL by design (positions 15:22 and 22:29) and hand
# slices are contiguous in INSPIRE_ACTUATED_NAMES (positions 0:6 and 6:12).
# Consumers (e.g., inspire_experiment_config.GROUP_SIM_INDICES) derive scatter
# indices via [ACTUATED_JOINT_NAMES.index(n) for n in GROUP_NAMES[g]].
GROUP_NAMES: dict[str, list[str]] = {
    "left_arm": BODY_JOINT_NAMES_CANONICAL[15:22],
    "right_arm": BODY_JOINT_NAMES_CANONICAL[22:29],
    "left_hand": INSPIRE_ACTUATED_NAMES[0:6],
    "right_hand": INSPIRE_ACTUATED_NAMES[6:12],
}
