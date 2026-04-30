# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding tests for the Inspire FTP experiment-config / scatter contract (§4.6).

Pure Python — **no Kit boot required**. Pins the joint-identity contract
between the policy state space (26-D dual-arm or 13-D single-arm) and the
41-D sim action space. The historical pinky/middle bug (left/right hand
indices swapping `little_1` ↔ `middle_1`) lived in this exact mapping; after
the joint_constants refactor every dict in ``inspire_experiment_config``
is derived from the joint-identity anchor, but we still pin the parity
invariants here as a regression canary.

Catches:
  - Drift between ``GROUP_SIM_INDICES`` and the actuated articulation order
  - Drift between ``ARM_BODY_RANGES`` / ``HAND_INSPIRE_RANGES`` and their
    respective canonical layouts
  - Wrong group sizes / wrong sim action dim
  - Default / single-arm dataclass scatter shape and content

Run from ``workflows/rheo/`` (no Docker required for this test):

    python -m unittest tests.test_sim.test_inspire_experiment_config -v
"""

import sys
import unittest
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from inspire_joint_constants import (  # noqa: E402
    ACTUATED_JOINT_NAMES,
    BODY_JOINT_NAMES_CANONICAL,
    GROUP_NAMES,
    INSPIRE_ACTUATED_NAMES,
    JOINT_NAMES,
    MIMIC_JOINT_NAMES,
)
from utils.inspire.inspire_experiment_config import (  # noqa: E402
    ARM_BODY_RANGES,
    GROUP_SIM_INDICES,
    GROUP_SIZES,
    HAND_INSPIRE_RANGES,
    SIM_ACTION_DIM,
    VALID_GROUPS,
    InspireExperimentConfig,
)


class GroupSimIndicesTests(unittest.TestCase):
    """§4.6 — the policy → 41-D sim scatter contract."""

    def test_group_sim_indices_match_actuated_lookup(self):
        """Each group's scatter equals [ACTUATED_JOINT_NAMES.index(n) for n in GROUP_NAMES[g]].

        Pre-refactor this was hand-authored and could silently drift; post-refactor
        it is derived. The test pins the structural identity so any future
        re-introduction of hand-authored values is caught immediately.
        """
        for g, names in GROUP_NAMES.items():
            expected = [ACTUATED_JOINT_NAMES.index(n) for n in names]
            self.assertEqual(GROUP_SIM_INDICES[g], expected, f"group={g}")

    def test_group_sim_indices_partition_actuated_arms_and_hands(self):
        """Concatenated arm + hand indices cover every actuated arm/hand joint exactly once.

        ACTUATED_JOINT_NAMES = 29 body + 12 actuated hand. The 14 arm joints
        (positions 11/12/15/16/19/20/21/22/23/24/25/26/27/28) plus the 12
        actuated hand joints (positions 29-40) form the 26 entries the policy
        controls — a strict subset of the 41-D actuated layout.
        """
        all_indices = (
            GROUP_SIM_INDICES["left_arm"]
            + GROUP_SIM_INDICES["right_arm"]
            + GROUP_SIM_INDICES["left_hand"]
            + GROUP_SIM_INDICES["right_hand"]
        )
        self.assertEqual(len(all_indices), 26, "policy dim should be 26 (14 arm + 12 hand)")
        self.assertEqual(len(set(all_indices)), 26, "scatter targets must be unique")
        self.assertTrue(all(0 <= i < SIM_ACTION_DIM for i in all_indices))

    def test_group_sim_indices_resolve_to_correct_names(self):
        """Reverse-lookup each scatter index back to a joint name and confirm it
        equals the GROUP_NAMES entry. This is the structural twin of the test
        above — same invariant, viewed from the action-space side.
        """
        for g, names in GROUP_NAMES.items():
            resolved = [ACTUATED_JOINT_NAMES[i] for i in GROUP_SIM_INDICES[g]]
            self.assertEqual(resolved, names, f"group={g}")

    def test_hand_groups_avoid_mimic_joints(self):
        """Hand scatter indices must not point at mimic joints — those are
        driven by InspireJointPositionAction.apply_actions(), not the action
        manager. ACTUATED_JOINT_NAMES has mimic joints removed by construction,
        but pin it explicitly here.
        """
        for g in ("left_hand", "right_hand"):
            for i in GROUP_SIM_INDICES[g]:
                self.assertNotIn(
                    ACTUATED_JOINT_NAMES[i],
                    MIMIC_JOINT_NAMES,
                    f"{g} scatter target {i} is a mimic joint",
                )


class GroupSizesTests(unittest.TestCase):
    def test_group_sizes_match_name_list_lengths(self):
        self.assertEqual(GROUP_SIZES, {g: len(names) for g, names in GROUP_NAMES.items()})

    def test_group_sizes_known_values(self):
        self.assertEqual(GROUP_SIZES["left_arm"], 7)
        self.assertEqual(GROUP_SIZES["right_arm"], 7)
        self.assertEqual(GROUP_SIZES["left_hand"], 6)
        self.assertEqual(GROUP_SIZES["right_hand"], 6)

    def test_sim_action_dim(self):
        self.assertEqual(SIM_ACTION_DIM, 41)
        self.assertEqual(SIM_ACTION_DIM, len(ACTUATED_JOINT_NAMES))
        self.assertEqual(SIM_ACTION_DIM, len(JOINT_NAMES) - len(MIMIC_JOINT_NAMES))


class ArmBodyRangesTests(unittest.TestCase):
    def test_arm_ranges_slice_canonical_to_group_names(self):
        """ARM_BODY_RANGES[g] is the contiguous slice of BODY_JOINT_NAMES_CANONICAL
        that equals GROUP_NAMES[g]. Pinning this means observations.py's
        canonical body-order assumption is honored end-to-end.
        """
        for g in ("left_arm", "right_arm"):
            s, e = ARM_BODY_RANGES[g]
            self.assertEqual(BODY_JOINT_NAMES_CANONICAL[s:e], GROUP_NAMES[g], f"group={g}")

    def test_arm_ranges_known_values(self):
        # The canonical body layout was hand-authored to make arm slices contiguous.
        # If these change, observations.py and inspire_lerobot_fields.py both need updating.
        self.assertEqual(ARM_BODY_RANGES["left_arm"], (15, 22))
        self.assertEqual(ARM_BODY_RANGES["right_arm"], (22, 29))


class HandInspireRangesTests(unittest.TestCase):
    def test_hand_ranges_slice_actuated_to_group_names(self):
        for g in ("left_hand", "right_hand"):
            s, e = HAND_INSPIRE_RANGES[g]
            self.assertEqual(INSPIRE_ACTUATED_NAMES[s:e], GROUP_NAMES[g], f"group={g}")

    def test_hand_ranges_known_values(self):
        self.assertEqual(HAND_INSPIRE_RANGES["left_hand"], (0, 6))
        self.assertEqual(HAND_INSPIRE_RANGES["right_hand"], (6, 12))


class ValidGroupsTests(unittest.TestCase):
    def test_valid_groups_match_group_names_keys(self):
        self.assertEqual(set(VALID_GROUPS), set(GROUP_NAMES.keys()))


class InspireExperimentConfigTests(unittest.TestCase):
    """End-to-end: the dataclass projects GROUP_NAMES into the right tensor indices."""

    def test_default_dual_arm_26d(self):
        cfg = InspireExperimentConfig()
        self.assertEqual(cfg.joint_groups, ["left_arm", "right_arm", "left_hand", "right_hand"])
        self.assertEqual(cfg.policy_dim, 26)
        # Body indices: arms concatenated in joint_groups order = [15..22) + [22..29)
        self.assertEqual(cfg.body_state_indices, list(range(15, 29)))
        # Inspire (12-D) indices: hands in order = [0..6) + [6..12)
        self.assertEqual(cfg.inspire_state_indices, list(range(0, 12)))
        # Sim scatter: groups concatenated in declaration order
        expected_scatter = (
            GROUP_SIM_INDICES["left_arm"]
            + GROUP_SIM_INDICES["right_arm"]
            + GROUP_SIM_INDICES["left_hand"]
            + GROUP_SIM_INDICES["right_hand"]
        )
        self.assertEqual(cfg.sim_scatter_indices, expected_scatter)

    def test_right_arm_only_13d(self):
        cfg = InspireExperimentConfig(joint_groups=["right_arm", "right_hand"])
        self.assertEqual(cfg.policy_dim, 13)
        self.assertEqual(cfg.body_state_indices, list(range(22, 29)))
        self.assertEqual(cfg.inspire_state_indices, list(range(6, 12)))
        self.assertEqual(
            cfg.sim_scatter_indices,
            GROUP_SIM_INDICES["right_arm"] + GROUP_SIM_INDICES["right_hand"],
        )

    def test_left_arm_only_13d(self):
        cfg = InspireExperimentConfig(joint_groups=["left_arm", "left_hand"])
        self.assertEqual(cfg.policy_dim, 13)
        self.assertEqual(cfg.body_state_indices, list(range(15, 22)))
        self.assertEqual(cfg.inspire_state_indices, list(range(0, 6)))
        self.assertEqual(
            cfg.sim_scatter_indices,
            GROUP_SIM_INDICES["left_arm"] + GROUP_SIM_INDICES["left_hand"],
        )

    def test_unknown_group_raises(self):
        with self.assertRaises(ValueError):
            InspireExperimentConfig(joint_groups=["left_arm", "third_arm"])


if __name__ == "__main__":
    unittest.main()