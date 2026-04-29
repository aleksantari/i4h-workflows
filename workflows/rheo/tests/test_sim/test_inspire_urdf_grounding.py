# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint-space grounding test — Layer 1 (URDF spec ↔ code), Layer 2 (USD ↔ code), Layer 3 (runtime behavior).

Layer 1 (13 checks): parses the active Inspire FTP URDF as XML and asserts that
the joint-space constants in env_cfg, mimic_action, robot_config, the teleop
env_cfg, and the canonical observation lists agree with it. Catches drift in:
  - Joint count, name set, mimic set
  - Mimic relationship triples (parent / multiplier / offset==0)
  - Actuator regex coverage in robot_config
  - URDF ↔ Nucleus bridge (`_URDF_TO_NUCLEUS`): domain, bijection, finger / side consistency
  - PinkIK ``pink_controlled_joint_names`` regex coverage of the 14 arm joints
  - ``_BODY_JOINT_NAMES_CANONICAL`` set + arm slice contract ([15:22] left, [22:29] right)
  - ``_INSPIRE_ACTUATED_NAMES`` set + hand slice contract ([0:6] left, [6:12] right)

Layer 2 (3 checks): order-sensitive checks against the loaded USD articulation:
  - env_cfg ``joint_names`` matches articulation list-wise.
  - ``_resolve_indices`` produces correct articulation indices for body canonical names.
  - ``_resolve_indices`` produces correct articulation indices for hand canonical names.

Layer 3 (3 checks): runtime behavioral checks via ``env.step`` / obs functions:
  - ``InspireJointPositionAction.apply_actions()`` drives mimic to multiplier × parent.
  - ``get_robot_body_joint_states`` output matches ``DEFAULT_JOINT_POS`` at canonical slots.
  - ``get_robot_inspire_joint_states`` output is all zeros at default pose.

Boots IsaacLab Kit at module load (~10s) and constructs the Joint-Eval gym
env (~10s). Run inside Docker:

    ./docker/run_docker_grasp.sh python -m unittest tests.test_sim.test_inspire_urdf_grounding -v

See docs/inspire/joint_spaces.md for the full audit this test pins.
"""

# AppLauncher MUST be called before importing most isaaclab.* modules.
# enable_cameras=True is required because the Inspire FTP env's observation
# manager has camera obs terms; without it, IsaacLab strips cameras from the
# scene but leaves the obs term, leading to "front_camera does not exist".
from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True, enable_cameras=True)
_simulation_app = _app_launcher.app

# Standard library — safe before or after Kit boot.
import re  # noqa: E402
import unittest  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

# Third-party — must come AFTER AppLauncher.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# isaaclab_tasks.utils.parse_cfg.parse_env_cfg resolves the env_cfg_entry_point
# registered with the gym ID into an actual cfg instance.
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# Project imports — must come AFTER AppLauncher.
# Importing the package triggers gym.register(...) for the Inspire-* gym IDs,
# which the Layer 2 + Layer 3 tests need via gym.make.
import simulation.tasks.grasp_policy_inspire  # noqa: E402, F401

from simulation.tasks.grasp_policy_inspire.config.robot_config import (  # noqa: E402
    G129_CFG_WITH_INSPIRE_BASE_FIX,
)
from simulation.tasks.grasp_policy_inspire.g1_grasp_policy_inspire_env_cfg import (  # noqa: E402
    _MIMIC_JOINT_NAMES,
    actuated_joint_names as ACTUATED_JOINT_NAMES,
    joint_names as ENV_CFG_JOINT_NAMES,
)
from simulation.tasks.grasp_policy_inspire.g1_grasp_policy_inspire_teleop_env_cfg import (  # noqa: E402
    HAND_JOINT_NAMES,
    TeleopActionsCfg,
    _URDF_TO_NUCLEUS,
)
from simulation.tasks.grasp_policy_inspire.mdp.mimic_action import MIMIC_RULES  # noqa: E402
from simulation.tasks.grasp_policy_inspire.mdp.observations import (  # noqa: E402
    _BODY_JOINT_NAMES_CANONICAL,
    _INSPIRE_ACTUATED_NAMES,
    _resolve_indices,
    get_robot_body_joint_states,
    get_robot_inspire_joint_states,
)

# The active URDF — the wrist_cam variant, matching what robot_config.py loads.
_URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets" / "robots" / "g1-29dof-inspire-ftp-urdf-wrist_cam"
    / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
)


def _parse_urdf():
    """Parse the URDF into (articulated, mimic_triples, all_joints).

    - articulated: list[str] of joints with non-zero DOF (revolute / continuous /
      prismatic). The 53 to compare against env_cfg.joint_names *as a set*.
    - mimic_triples: list of (child, parent, multiplier, offset) tuples, one per
      <mimic> tag. Offset is asserted to be 0 (the action class ignores it).
    - all_joints: list[(name, type)] including fixed joints — useful for
      debugging the converter and mount points; not used by the assertions.
    """
    tree = ET.parse(_URDF_PATH)
    root = tree.getroot()

    articulated: list[str] = []
    all_joints: list[tuple[str, str]] = []
    mimic_triples: list[tuple[str, str, float, float]] = []

    for joint in root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get("type", "")
        all_joints.append((name, joint_type))

        if joint_type in ("revolute", "continuous", "prismatic"):
            articulated.append(name)
        mimic_el = joint.find("mimic")
        if mimic_el is not None:
            mimic_triples.append((
                name,
                mimic_el.get("joint"),
                float(mimic_el.get("multiplier", "1.0")),
                float(mimic_el.get("offset", "0.0")),
            ))

    return articulated, mimic_triples, all_joints


# Parse once at module load — every Layer 1 method reads from these.
_ARTICULATED, _MIMIC_TRIPLES, _ALL_JOINTS = _parse_urdf()

# PinkIK arm-joint regex patterns, sourced from the active TeleopActionsCfg
# default. Reading them at module load means a future refactor of the patterns
# is automatically picked up by the partition test.
_PINK_CONTROLLED_PATTERNS: list[str] = list(
    TeleopActionsCfg().pink_ik_cfg.pink_controlled_joint_names
)


# ---------------------------------------------------------------------------
# Layer 2 + Layer 3 scaffolding — gym.make the Joint-Eval env so we can both
# read the loaded articulation's joint_names (Layer 2) and call env.step to
# exercise InspireJointPositionAction.apply_actions (Layer 3).
# ---------------------------------------------------------------------------

# Settling: number of env.step() calls after sending a target before reading
# joint_pos. Each step is one decimation block (4 physics ticks at 200Hz =
# 50Hz control). 20 steps ≈ 0.4s of simulated time, plenty for a finger
# joint with our PD gains to converge to within a few percent of target.
_MIMIC_SETTLE_STEPS = 20

# Tolerance on the |observed - expected| absolute joint position error
# at steady state. Generous enough to absorb finite-stiffness PD lag for both
# the parent and the mimic; tight enough to catch a missing or wrong-sign
# multiplier (the historical bug class this test targets).
_MIMIC_TOL_RAD = 0.05


class InspireGroundingTests(unittest.TestCase):
    """Thirteen Layer-1 checks (URDF → code) + three Layer-2 + three Layer-3."""

    # Set in setUpClass after the env is built.
    _env: gym.Env = None
    _robot = None
    _ARTICULATION_JOINT_NAMES: list[str] = []

    @classmethod
    def setUpClass(cls):
        """gym.make the Joint-Eval env once and capture the articulation."""
        # Joint-Eval is the deterministic variant — reset noise is zeroed,
        # so the steady-state pose under a given action is reproducible.
        # Pattern mirrors eval_act_inspire.py: parse_env_cfg → gym.make(cfg=...)
        # because IsaacLab's ManagerBasedRLEnv expects ``cfg`` positionally,
        # not the registered ``env_cfg_entry_point`` class.
        task_id = "Isaac-Grasp-Policy-G129-Inspire-Joint-Eval"
        env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=1)
        cls._env = gym.make(task_id, cfg=env_cfg)
        cls._env.reset()
        cls._robot = cls._env.unwrapped.scene["robot"]
        cls._ARTICULATION_JOINT_NAMES = list(cls._robot.data.joint_names)

    @classmethod
    def tearDownClass(cls):
        """Cleanly close the env so Kit's stage teardown doesn't warn."""
        if cls._env is not None:
            cls._env.close()
            cls._env = None

    # -- Layer 1: 1. Joint count parity ---------------------------------------

    def test_joint_count_parity(self):
        """env_cfg.joint_names mirrors the URDF's articulated joint count (53)."""
        EXPECTED = 53
        self.assertEqual(
            len(_ARTICULATED), EXPECTED,
            f"URDF has {len(_ARTICULATED)} articulated joints, expected {EXPECTED}. "
            f"Total <joint> elements: {len(_ALL_JOINTS)} (53 revolute + 45 fixed expected). "
            f"If this fails, the URDF was edited; downstream tests are unreliable."
        )
        self.assertEqual(
            len(ENV_CFG_JOINT_NAMES), EXPECTED,
            f"env_cfg.joint_names has {len(ENV_CFG_JOINT_NAMES)} entries, expected {EXPECTED}."
        )

    # -- Layer 1: 2. Joint name set matches -----------------------------------

    def test_joint_name_set_matches(self):
        """env_cfg.joint_names contains exactly the URDF's articulated names."""
        urdf_set = set(_ARTICULATED)
        env_set = set(ENV_CFG_JOINT_NAMES)
        only_in_urdf = urdf_set - env_set
        only_in_env = env_set - urdf_set
        self.assertEqual(
            urdf_set, env_set,
            f"Joint name sets disagree.\n"
            f"  In URDF, missing from env_cfg: {sorted(only_in_urdf)}\n"
            f"  In env_cfg, missing from URDF: {sorted(only_in_env)}\n"
            f"Note: this checks the SET only; ordering is Layer 2's job."
        )

    # -- Layer 1: 3. Mimic set matches ----------------------------------------

    def test_mimic_set_matches(self):
        """URDF's mimic children == env_cfg._MIMIC_JOINT_NAMES (12 each)."""
        urdf_mimic_children = {child for child, *_ in _MIMIC_TRIPLES}
        only_in_urdf = urdf_mimic_children - _MIMIC_JOINT_NAMES
        only_in_env = _MIMIC_JOINT_NAMES - urdf_mimic_children
        self.assertEqual(
            urdf_mimic_children, _MIMIC_JOINT_NAMES,
            f"Mimic-joint sets disagree.\n"
            f"  Mimic in URDF, not in _MIMIC_JOINT_NAMES: {sorted(only_in_urdf)}\n"
            f"  In _MIMIC_JOINT_NAMES, not mimic in URDF: {sorted(only_in_env)}\n"
            f"A drift here corrupts actuated_joint_names (URDF→sim mismatch on "
            f"which joints the 41-D action commands)."
        )

    # -- Layer 1: 4. Mimic triples match --------------------------------------

    def test_mimic_triples_match(self):
        """Each URDF <mimic> tuple matches mimic_action.MIMIC_RULES.

        Defends against:
          - Multiplier drift (URDF rev to 1.10 with code still at 1.0843)
          - Parent reassignment (URDF flips a mimic's parent without code update)
          - Nonzero offset in URDF (the action class doesn't read offset; nonzero
            would silently corrupt sim until support is added)
        """
        code_by_child: dict[str, tuple[str, float]] = {
            child: (parent, mult) for child, parent, mult in MIMIC_RULES
        }
        for urdf_child, urdf_parent, urdf_mult, urdf_offset in _MIMIC_TRIPLES:
            with self.subTest(child=urdf_child):
                self.assertIn(
                    urdf_child, code_by_child,
                    f"URDF mimic {urdf_child!r} has no entry in MIMIC_RULES."
                )
                code_parent, code_mult = code_by_child[urdf_child]
                self.assertEqual(
                    urdf_parent, code_parent,
                    f"Mimic {urdf_child}: URDF parent={urdf_parent!r}, "
                    f"code parent={code_parent!r}."
                )
                self.assertAlmostEqual(
                    urdf_mult, code_mult, places=4,
                    msg=f"Mimic {urdf_child}: URDF mult={urdf_mult}, code mult={code_mult}.",
                )
                self.assertEqual(
                    urdf_offset, 0.0,
                    f"Mimic {urdf_child}: URDF offset={urdf_offset}, expected 0.0. "
                    f"InspireJointPositionAction.apply_actions() does NOT read offset — "
                    f"a nonzero offset would silently corrupt the sim. Either revert the "
                    f"URDF change or extend the action class to honor offset.",
                )

    # -- Layer 1: 5. Actuator regex partition ---------------------------------

    def test_actuator_regex_partition(self):
        """Every URDF articulated joint matches exactly one actuator group's regex.

        Defends against:
          - Joint with no actuator → floppy at runtime (no PD control)
          - Joint matching multiple groups → ambiguous gains, IsaacLab picks one
          - Dead pattern (matches zero joints) → typo or stale config
        """
        flat_patterns: list[tuple[str, str]] = [
            (group, pat)
            for group, actuator_cfg in G129_CFG_WITH_INSPIRE_BASE_FIX.actuators.items()
            for pat in actuator_cfg.joint_names_expr
        ]
        pattern_hits: dict[tuple[str, str], int] = {key: 0 for key in flat_patterns}

        for joint in _ARTICULATED:
            matches = [(g, p) for g, p in flat_patterns if re.fullmatch(p, joint)]
            for hit in matches:
                pattern_hits[hit] += 1
            with self.subTest(joint=joint):
                self.assertEqual(
                    len(matches), 1,
                    f"Joint {joint!r} matched {len(matches)} actuator patterns "
                    f"(expected exactly 1): {matches}"
                )

        dead = [key for key, count in pattern_hits.items() if count == 0]
        self.assertEqual(
            dead, [],
            f"Actuator patterns matched zero joints (dead patterns): {dead}",
        )

    # -- Layer 1: 6. URDF → Nucleus domain coverage --------------------------

    def test_urdf_to_nucleus_domain_coverage(self):
        """Every URDF hand joint has a Nucleus mapping; no orphan keys.

        ``_URDF_TO_NUCLEUS`` is the bridge dict that every Nucleus-named
        constant downstream is supposed to derive *through*. If its key set
        drifts from ``HAND_JOINT_NAMES`` (e.g. a hand joint is added or
        renamed), downstream Nucleus translations silently produce
        wrong / missing entries.
        """
        urdf_keys = set(_URDF_TO_NUCLEUS.keys())
        hand_set = set(HAND_JOINT_NAMES)
        only_in_hand = hand_set - urdf_keys
        only_in_map = urdf_keys - hand_set
        self.assertEqual(
            urdf_keys, hand_set,
            f"_URDF_TO_NUCLEUS domain disagrees with HAND_JOINT_NAMES.\n"
            f"  Hand joints missing from mapping: {sorted(only_in_hand)}\n"
            f"  Mapping keys not a hand joint: {sorted(only_in_map)}"
        )

    # -- Layer 1: 7. URDF → Nucleus bijection --------------------------------

    def test_urdf_to_nucleus_bijective(self):
        """No two URDF hand joints map to the same Nucleus name."""
        values = list(_URDF_TO_NUCLEUS.values())
        seen: dict[str, str] = {}
        duplicates: list[tuple[str, str]] = []
        for urdf_name, nucleus_name in _URDF_TO_NUCLEUS.items():
            if nucleus_name in seen:
                duplicates.append((seen[nucleus_name], urdf_name))
            else:
                seen[nucleus_name] = urdf_name
        self.assertEqual(
            len(set(values)), len(values),
            f"_URDF_TO_NUCLEUS is not bijective; duplicate Nucleus values:\n"
            f"  {duplicates}\n"
            f"Two URDF joints mapping to the same Nucleus name corrupts the "
            f"retargeter output."
        )

    # -- Layer 1: 8. URDF → Nucleus per-finger consistency -------------------

    def test_urdf_to_nucleus_finger_consistency(self):
        """Side prefix and finger token agree between URDF and Nucleus naming.

        Catches the ``finger-mix-up`` bug class — an authoring error that pairs
        e.g. ``left_index_1_joint`` with ``L_pinky_proximal_joint``. The dict
        is the single source of truth for the bridge, so this self-consistency
        check is the only way to catch a typo or a copy-paste shift.
        """
        # URDF prefix → Nucleus prefix.
        side_map = {"left_": "L_", "right_": "R_"}
        # URDF finger token → Nucleus finger token. Note the only rename:
        # URDF's ``little`` is Nucleus's ``pinky``.
        finger_map = {
            "_index_": "_index_",
            "_middle_": "_middle_",
            "_ring_": "_ring_",
            "_little_": "_pinky_",
            "_thumb_": "_thumb_",
        }
        for urdf_name, nucleus_name in _URDF_TO_NUCLEUS.items():
            with self.subTest(urdf=urdf_name):
                # Side prefix must match.
                u_side = next(
                    (u_pref for u_pref in side_map if urdf_name.startswith(u_pref)),
                    None,
                )
                self.assertIsNotNone(
                    u_side,
                    f"URDF name {urdf_name!r} has unrecognized side prefix "
                    f"(expected one of {sorted(side_map.keys())})."
                )
                self.assertTrue(
                    nucleus_name.startswith(side_map[u_side]),
                    f"{urdf_name!r} starts with {u_side!r} but "
                    f"Nucleus name {nucleus_name!r} does not start with "
                    f"{side_map[u_side]!r}."
                )
                # Finger token must match (URDF's ``little`` ↔ Nucleus's ``pinky``).
                u_finger = next(
                    (u_tok for u_tok in finger_map if u_tok in urdf_name),
                    None,
                )
                self.assertIsNotNone(
                    u_finger,
                    f"URDF name {urdf_name!r} has unrecognized finger token "
                    f"(expected one of {sorted(finger_map.keys())})."
                )
                self.assertIn(
                    finger_map[u_finger], nucleus_name,
                    f"URDF {urdf_name!r} contains {u_finger!r} but Nucleus "
                    f"{nucleus_name!r} does not contain {finger_map[u_finger]!r}. "
                    f"This is the finger-mix-up bug class."
                )

    # -- Layer 1: 9. PinkIK arm-joint regex partition ------------------------

    def test_pink_ik_arm_joint_partition(self):
        """PinkIK ``pink_controlled_joint_names`` matches exactly the 14 arm joints.

        Defends against:
          - Pattern misses an arm joint → PinkIK doesn't solve for it → DOF freezes
          - Pattern matches a non-arm joint → PinkIK tries to solve over an unrelated joint
          - Dead pattern (matches zero joints) → typo or stale config
        """
        expected_arm_joints = {
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        }
        # Sanity: the expected 14 arm joints exist in the URDF.
        missing_from_urdf = expected_arm_joints - set(_ARTICULATED)
        self.assertEqual(
            missing_from_urdf, set(),
            f"Expected arm joints not present in URDF: {sorted(missing_from_urdf)}"
        )

        pattern_hits: dict[str, int] = {p: 0 for p in _PINK_CONTROLLED_PATTERNS}
        for joint in _ARTICULATED:
            matches = [p for p in _PINK_CONTROLLED_PATTERNS if re.fullmatch(p, joint)]
            for p in matches:
                pattern_hits[p] += 1
            is_arm = joint in expected_arm_joints
            with self.subTest(joint=joint):
                if is_arm:
                    self.assertEqual(
                        len(matches), 1,
                        f"Arm joint {joint!r} matched {len(matches)} pink patterns "
                        f"(expected exactly 1): {matches}"
                    )
                else:
                    self.assertEqual(
                        len(matches), 0,
                        f"Non-arm joint {joint!r} unexpectedly matched pink "
                        f"patterns: {matches}"
                    )

        dead = [p for p, c in pattern_hits.items() if c == 0]
        self.assertEqual(
            dead, [],
            f"PinkIK patterns matched zero joints (dead patterns): {dead}",
        )

    # -- Layer 1: 10. Body canonical set matches URDF body joints ------------

    def test_body_canonical_set_matches(self):
        """``_BODY_JOINT_NAMES_CANONICAL`` covers exactly the 29 URDF body joints.

        Set equality only — order is the slice contract test's job.
        ``_BODY_JOINT_NAMES_CANONICAL`` is intentionally a *different order*
        from ``joint_names[:29]`` (interleaved by body part for slice
        stability) but must contain the same SET.
        """
        body_urdf = set(ENV_CFG_JOINT_NAMES[:29])
        body_canonical = set(_BODY_JOINT_NAMES_CANONICAL)
        only_in_urdf = body_urdf - body_canonical
        only_in_canonical = body_canonical - body_urdf
        self.assertEqual(
            body_canonical, body_urdf,
            f"_BODY_JOINT_NAMES_CANONICAL set disagrees with URDF body joints.\n"
            f"  In URDF body, missing from canonical: {sorted(only_in_urdf)}\n"
            f"  In canonical, missing from URDF body: {sorted(only_in_canonical)}"
        )
        self.assertEqual(
            len(_BODY_JOINT_NAMES_CANONICAL), 29,
            f"_BODY_JOINT_NAMES_CANONICAL has {len(_BODY_JOINT_NAMES_CANONICAL)} entries, "
            f"expected 29."
        )
        self.assertEqual(
            len(set(_BODY_JOINT_NAMES_CANONICAL)), len(_BODY_JOINT_NAMES_CANONICAL),
            "_BODY_JOINT_NAMES_CANONICAL contains duplicate names."
        )

    # -- Layer 1: 11. Body canonical arm slice contract ----------------------

    def test_body_canonical_arm_slice_contract(self):
        """Slice [15:22] is left arm in canonical order; [22:29] is right arm.

        Downstream consumers (``STATE_26_BODY_COL_LEFT_ARM = range(15, 22)``
        in ``inspire_lerobot_fields.py``) hardcode these slice positions.
        Reordering ``_BODY_JOINT_NAMES_CANONICAL`` silently breaks every
        LeRobot conversion that follows — set equality alone won't catch it.
        """
        expected_left_arm = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        expected_right_arm = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.assertEqual(
            _BODY_JOINT_NAMES_CANONICAL[15:22], expected_left_arm,
            "Canonical[15:22] should be left arm in canonical order."
        )
        self.assertEqual(
            _BODY_JOINT_NAMES_CANONICAL[22:29], expected_right_arm,
            "Canonical[22:29] should be right arm in canonical order."
        )

    # -- Layer 1: 12. Inspire actuated set matches URDF actuated hand --------

    def test_inspire_actuated_set_matches(self):
        """``_INSPIRE_ACTUATED_NAMES`` covers exactly the 12 URDF actuated hand joints.

        Set equality + length + 6/6 left-right balance.
        """
        urdf_actuated_hand = set(ENV_CFG_JOINT_NAMES[29:]) - _MIMIC_JOINT_NAMES
        canonical_actuated = set(_INSPIRE_ACTUATED_NAMES)
        only_in_urdf = urdf_actuated_hand - canonical_actuated
        only_in_canonical = canonical_actuated - urdf_actuated_hand
        self.assertEqual(
            canonical_actuated, urdf_actuated_hand,
            f"_INSPIRE_ACTUATED_NAMES set disagrees with URDF actuated hand joints.\n"
            f"  In URDF actuated, missing from canonical: {sorted(only_in_urdf)}\n"
            f"  In canonical, missing from URDF actuated: {sorted(only_in_canonical)}"
        )
        self.assertEqual(
            len(_INSPIRE_ACTUATED_NAMES), 12,
            f"_INSPIRE_ACTUATED_NAMES has {len(_INSPIRE_ACTUATED_NAMES)} entries, expected 12."
        )
        left_count = sum(1 for n in _INSPIRE_ACTUATED_NAMES if n.startswith("left_"))
        right_count = sum(1 for n in _INSPIRE_ACTUATED_NAMES if n.startswith("right_"))
        self.assertEqual(left_count, 6, f"Expected 6 left-hand entries, got {left_count}.")
        self.assertEqual(right_count, 6, f"Expected 6 right-hand entries, got {right_count}.")

    # -- Layer 1: 13. Inspire actuated hand slice contract -------------------

    def test_inspire_actuated_hand_slice_contract(self):
        """Slice [0:6] is left hand in canonical order; [6:12] is right hand.

        Downstream consumers (``STATE_26_INSPIRE_COL_LEFT_HAND = range(0, 6)``
        in ``inspire_lerobot_fields.py``) hardcode these slice positions.
        Reordering ``_INSPIRE_ACTUATED_NAMES`` silently scrambles the
        per-finger LeRobot mapping — exactly the historical "fingers crossed"
        bug class.

        Canonical hand order: thumb_yaw (=_1), thumb_pitch (=_2),
        index_1, middle_1, ring_1, little_1.
        """
        expected_left_hand = [
            "left_thumb_1_joint",
            "left_thumb_2_joint",
            "left_index_1_joint",
            "left_middle_1_joint",
            "left_ring_1_joint",
            "left_little_1_joint",
        ]
        expected_right_hand = [
            "right_thumb_1_joint",
            "right_thumb_2_joint",
            "right_index_1_joint",
            "right_middle_1_joint",
            "right_ring_1_joint",
            "right_little_1_joint",
        ]
        self.assertEqual(
            _INSPIRE_ACTUATED_NAMES[0:6], expected_left_hand,
            "_INSPIRE_ACTUATED_NAMES[0:6] should be left hand in canonical order."
        )
        self.assertEqual(
            _INSPIRE_ACTUATED_NAMES[6:12], expected_right_hand,
            "_INSPIRE_ACTUATED_NAMES[6:12] should be right hand in canonical order."
        )

    # -- Layer 2: 12. Body canonical bridges to articulation correctly -------

    def test_body_canonical_resolves_to_articulation(self):
        """``_resolve_indices`` produces correct articulation indices for canonical body names.

        Pins the bridge between the canonical (Ordering B) names and the
        loaded USD's articulation positions (Ordering A). What the spec-only
        checks (set equality, slice contract) cannot detect is the bridge
        itself producing wrong indices for any reason — this test closes that.
        """
        art_names = self._ARTICULATION_JOINT_NAMES
        expected = [art_names.index(n) for n in _BODY_JOINT_NAMES_CANONICAL]
        actual = _resolve_indices(
            art_names, _BODY_JOINT_NAMES_CANONICAL, torch.device("cpu")
        ).tolist()
        self.assertEqual(
            actual, expected,
            f"_resolve_indices produced wrong articulation indices for canonical body names.\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )

    # -- Layer 2: 13. Hand canonical bridges to articulation correctly -------

    def test_inspire_actuated_resolves_to_articulation(self):
        """``_resolve_indices`` produces correct articulation indices for canonical hand names."""
        art_names = self._ARTICULATION_JOINT_NAMES
        expected = [art_names.index(n) for n in _INSPIRE_ACTUATED_NAMES]
        actual = _resolve_indices(
            art_names, _INSPIRE_ACTUATED_NAMES, torch.device("cpu")
        ).tolist()
        self.assertEqual(
            actual, expected,
            f"_resolve_indices produced wrong articulation indices for canonical hand names.\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )

    # -- Layer 3: 11. Body obs layout at default pose ------------------------

    def test_body_obs_layout_at_default_pose(self):
        """``get_robot_body_joint_states`` output values at canonical slots
        match ``DEFAULT_JOINT_POS``.

        End-to-end test of the full chain: articulation → ``_resolve_indices``
        → ``torch.gather`` → output. Catches bridge-level bugs that the
        spec-only and bridge-only checks cannot detect (e.g. cache staleness,
        wrong gather dim, batch handling regressions).
        """
        # Reset to ensure default joint positions — defensive against test ordering.
        self._env.reset()
        obs = get_robot_body_joint_states(self._env.unwrapped)
        self.assertEqual(
            tuple(obs.shape), (1, 87),
            f"Body obs shape {tuple(obs.shape)} != expected (1, 87)."
        )
        pos_29 = obs[0, :29].cpu()

        # Per DEFAULT_JOINT_POS in robot_config.py:
        #   left/right shoulder_pitch = -0.5
        #   left/right elbow         = -0.3
        #   everything else (legs, waist, other arm joints) = 0.0
        expected_at_canonical_idx = {
            15: -0.5,  # left_shoulder_pitch_joint (start of left arm slice)
            18: -0.3,  # left_elbow_joint
            22: -0.5,  # right_shoulder_pitch_joint (start of right arm slice)
            25: -0.3,  # right_elbow_joint
            0:  0.0,   # left_hip_pitch_joint (a leg, default 0)
            12: 0.0,   # waist_yaw_joint
        }
        for idx, expected_val in expected_at_canonical_idx.items():
            with self.subTest(canonical_idx=idx, name=_BODY_JOINT_NAMES_CANONICAL[idx]):
                self.assertAlmostEqual(
                    pos_29[idx].item(), expected_val, places=3,
                    msg=(
                        f"Canonical pos[{idx}] ({_BODY_JOINT_NAMES_CANONICAL[idx]!r}) "
                        f"= {pos_29[idx].item():.4f}, expected {expected_val:.4f}. "
                        f"Likely cause: bridge reading from wrong articulation "
                        f"position, or DEFAULT_JOINT_POS changed."
                    )
                )

    # -- Layer 3: 12. Inspire hand obs layout at default pose ----------------

    def test_inspire_obs_layout_at_default_pose(self):
        """``get_robot_inspire_joint_states`` output is all zeros at default pose.

        ``DEFAULT_JOINT_POS`` sets all hand joints to 0.0. End-to-end test
        of the full chain for the 12-D hand obs.
        """
        # Reset to ensure default joint positions — defensive against test ordering.
        self._env.reset()
        obs = get_robot_inspire_joint_states(self._env.unwrapped)
        self.assertEqual(
            tuple(obs.shape), (1, 12),
            f"Inspire obs shape {tuple(obs.shape)} != expected (1, 12)."
        )
        pos_12 = obs[0].cpu()
        for idx in range(12):
            with self.subTest(canonical_idx=idx, name=_INSPIRE_ACTUATED_NAMES[idx]):
                self.assertAlmostEqual(
                    pos_12[idx].item(), 0.0, places=3,
                    msg=(
                        f"Hand obs[{idx}] ({_INSPIRE_ACTUATED_NAMES[idx]!r}) "
                        f"= {pos_12[idx].item():.4f}, expected 0.0 (default pose)."
                    )
                )

    # -- Layer 3: 10. Runtime mimic enforcement -----------------------------

    def test_mimic_enforcement_at_runtime(self):
        """``apply_actions()`` drives mimic joints to ``multiplier × parent``.

        Sends a 41-D action with one specific actuated hand joint at a known
        nonzero target, settles for a few env.step() calls, then reads the
        actual joint positions and asserts the URDF mimic ratio is met
        within tolerance.

        Catches:
          - Refactor regressions in apply_actions ordering (super before mimic
            computation; reverse the order and the mimic reads stale data).
          - Future PhysX behavior changes that silently enable native mimic
            constraint enforcement — would compete with our manual writes.
          - Bypass: if anything writes to the actuated parent joint without
            triggering apply_actions, the mimic stays at its current value
            and this test fires.

        Tolerance is moderate (a few hundredths of a radian) to absorb PD
        finite-stiffness lag, which affects both the parent and the mimic.
        Tight enough to catch a wrong / missing multiplier (the historical
        bug class).
        """
        # Pick a single non-thumb finger so the chain is one-deep
        # (left_index_2_joint mimics left_index_1_joint at 1.0843×).
        parent_joint_name = "left_index_1_joint"
        mimic_joint_name = "left_index_2_joint"
        expected_multiplier = 1.0843

        # Target value chosen to be well within joint limits but big enough
        # that PD lag is small relative to the value (avoids divide-by-near-
        # zero amplification in the assertion's relative-error interpretation).
        target_value = 0.5

        # Build the 41-D action (the env's action space size).
        device = self._env.unwrapped.device
        action = torch.zeros(1, 41, device=device)
        parent_action_idx = ACTUATED_JOINT_NAMES.index(parent_joint_name)
        action[0, parent_action_idx] = target_value

        # Step several times to settle.
        for _ in range(_MIMIC_SETTLE_STEPS):
            self._env.step(action)

        # Read achieved joint positions and verify the URDF ratio.
        pos = self._robot.data.joint_pos[0].cpu().numpy()
        parent_art_idx = self._ARTICULATION_JOINT_NAMES.index(parent_joint_name)
        mimic_art_idx = self._ARTICULATION_JOINT_NAMES.index(mimic_joint_name)
        parent_pos = float(pos[parent_art_idx])
        mimic_pos = float(pos[mimic_art_idx])
        expected_mimic = expected_multiplier * parent_pos

        self.assertAlmostEqual(
            mimic_pos, expected_mimic, delta=_MIMIC_TOL_RAD,
            msg=(
                f"\nMimic enforcement failed at runtime.\n"
                f"  parent ({parent_joint_name}) pos: {parent_pos:.4f} rad\n"
                f"  mimic ({mimic_joint_name}) pos:  {mimic_pos:.4f} rad\n"
                f"  expected mimic = {expected_multiplier} × parent = {expected_mimic:.4f} rad\n"
                f"  abs error: {abs(mimic_pos - expected_mimic):.4f} rad "
                f"(tolerance: {_MIMIC_TOL_RAD})\n"
                f"Likely causes: (a) apply_actions ordering broken — super() before "
                f"mimic write, (b) mimic write skipped (bypass), (c) PhysX is "
                f"competing with our writes via native mimic constraint, "
                f"(d) MIMIC_RULES has a wrong multiplier or parent."
            ),
        )

    # -- Layer 2: 11. Articulation order matches env_cfg.joint_names list-wise

    def test_articulation_order_matches_env_cfg(self):
        """USD articulation order matches env_cfg.joint_names list-wise.

        This is what Layer 1 explicitly cannot check — URDF XML order is an
        arbitrary authoring artifact, but the USD's articulation order
        (determined by the converter's tree traversal) is what every downstream
        observation slice and action index assumes.

        Catches:
          - env_cfg.joint_names re-ordered without re-verifying against the USD
          - Converter behavior change shifts the traversal order
          - Wrong USD loaded (e.g. the §4.7 hazard about wrist_cam vs not)
        """
        art_names = self._ARTICULATION_JOINT_NAMES
        env_names = list(ENV_CFG_JOINT_NAMES)
        self.assertEqual(
            len(art_names), len(env_names),
            f"Articulation has {len(art_names)} joints; env_cfg has {len(env_names)}."
        )
        if art_names != env_names:
            # Find the first mismatch for an actionable error message.
            mismatches: list[str] = []
            for i, (a, e) in enumerate(zip(art_names, env_names)):
                if a != e:
                    mismatches.append(f"  index {i}: articulation={a!r}, env_cfg={e!r}")
                    if len(mismatches) >= 5:
                        mismatches.append(f"  ... and possibly more.")
                        break
            self.fail(
                "USD articulation order disagrees with env_cfg.joint_names.\n"
                + "\n".join(mismatches)
                + "\nFix env_cfg.joint_names to match the articulation order, "
                + "or run scripts/utils/inspire/inspect_inspire_joints.py "
                + "(after pointing it at the active USD per audit doc §4.7)."
            )


if __name__ == "__main__":
    unittest.main()
