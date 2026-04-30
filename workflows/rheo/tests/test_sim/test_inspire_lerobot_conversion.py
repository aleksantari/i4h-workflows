# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HDF5 → LeRobot conversion tests for the Inspire FTP pipeline.

Pure Python (numpy + the Kit-free joint-identity anchor + the conversion
helpers themselves). **No Kit boot required** — runs in milliseconds because
all the constants and conversion math are stdlib + numpy. Pairs with the
joint-space grounding test (which pins joint identity) by additionally
covering the actual conversion logic that turns HDF5 obs into LeRobot rows.

Catches:
  - Elbow-offset chain magnitudes (`+0.3` at the correct indices)
  - Aliasing-safety: the `+delta` add must be out-of-place so the state
    buffer doesn't get mutated through shared NumPy views (April 2026 bug)
  - Slice extraction: `_extract_26d` / `_extract_13d` / `_extract_13d_left`
    pull the correct columns from body / hand obs
  - Branch consistency: 53-D and 41-D recorded-action paths produce results
    consistent with the index tables and elbow delta
  - 13-D single-arm conversions are byte-equivalent slices of the 26-D path

Run from `workflows/rheo/` (no Docker required for this test):

    python -m unittest tests.test_sim.test_inspire_lerobot_conversion -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure scripts/ is on sys.path so `inspire_joint_constants` and
# `utils.inspire.inspire_lerobot_fields` resolve. Done before the project
# imports below.
_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from utils.inspire.inspire_lerobot_fields import (  # noqa: E402
    ACTION_HDF5_TO_ENV_26,
    ACTION_HDF5_TO_ENV_26_FROM_41,
    STATE_13_LEFT_NAMES_ENV_ORDER,
    STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA,
    STATE_13_NAMES_ENV_ORDER,
    STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA,
    STATE_26_BODY_COL_LEFT_ARM,
    STATE_26_BODY_COL_RIGHT_ARM,
    STATE_26_INSPIRE_COL_LEFT_HAND,
    STATE_26_INSPIRE_COL_RIGHT_HAND,
    STATE_26_NAMES_ENV_ORDER,
    STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA,
    _extract_13d,
    _extract_13d_left,
    _extract_26d,
    convert_g1_state_action_to_lerobot_13d,
    convert_g1_state_action_to_lerobot_13d_left,
    convert_g1_state_action_to_lerobot_26d,
)


def _make_marker_state(T: int, body_dim: int = 87, hand_dim: int = 12):
    """Build synthetic body (T, body_dim) + hand (T, hand_dim) with marker values.

    Each cell is uniquely identifiable:
        body[t, j]    = 1000*t + j               (uniquely identifies (body, t, j))
        hand[t, j]    = 100000 + 1000*t + j      (uniquely identifies (hand, t, j))

    This lets assertions verify exactly which cells were copied by the
    extraction logic.
    """
    body = np.fromfunction(
        lambda t, j: 1000 * t + j, (T, body_dim), dtype=np.float64
    )
    hand = np.fromfunction(
        lambda t, j: 100000 + 1000 * t + j, (T, hand_dim), dtype=np.float64
    )
    return body, hand


# ---------------------------------------------------------------------------
# Group 1 — elbow-offset magnitudes
# ---------------------------------------------------------------------------

class ElbowOffsetMagnitudeTests(unittest.TestCase):
    """Pin the `+0.3` elbow-offset magnitude at correct indices, zero elsewhere.

    The elbow-offset chain (`+0.3` in parquet, `−0.3` in the env's offset_dict)
    is load-bearing for the entire IL training pipeline. A typo in the literal
    (e.g. `0.03`) would silently corrupt every recorded dataset.
    """

    def test_state_26_elbow_delta_magnitude(self):
        """26-D delta has +0.3 at left/right elbow indices and 0 everywhere else."""
        left_elbow_idx = STATE_26_NAMES_ENV_ORDER.index("left_elbow_joint")
        right_elbow_idx = STATE_26_NAMES_ENV_ORDER.index("right_elbow_joint")

        self.assertAlmostEqual(
            STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[left_elbow_idx], 0.3,
            msg=f"Left elbow delta at index {left_elbow_idx} is "
                f"{STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[left_elbow_idx]}, expected 0.3."
        )
        self.assertAlmostEqual(
            STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[right_elbow_idx], 0.3,
            msg=f"Right elbow delta at index {right_elbow_idx} is "
                f"{STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA[right_elbow_idx]}, expected 0.3."
        )

        nonzero = [
            (i, float(v))
            for i, v in enumerate(STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA)
            if v != 0.0
        ]
        self.assertEqual(
            sorted(nonzero),
            sorted([(left_elbow_idx, 0.3), (right_elbow_idx, 0.3)]),
            f"Unexpected nonzero entries in 26-D elbow delta: {nonzero}"
        )

    def test_state_13_right_elbow_delta_magnitude(self):
        """13-D right delta has +0.3 at right_elbow only."""
        right_elbow_idx = STATE_13_NAMES_ENV_ORDER.index("right_elbow_joint")
        self.assertAlmostEqual(
            STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA[right_elbow_idx], 0.3
        )
        nonzero = [
            (i, float(v))
            for i, v in enumerate(STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA)
            if v != 0.0
        ]
        self.assertEqual(nonzero, [(right_elbow_idx, 0.3)])

    def test_state_13_left_elbow_delta_magnitude(self):
        """13-D left delta has +0.3 at left_elbow only."""
        left_elbow_idx = STATE_13_LEFT_NAMES_ENV_ORDER.index("left_elbow_joint")
        self.assertAlmostEqual(
            STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA[left_elbow_idx], 0.3
        )
        nonzero = [
            (i, float(v))
            for i, v in enumerate(STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA)
            if v != 0.0
        ]
        self.assertEqual(nonzero, [(left_elbow_idx, 0.3)])


# ---------------------------------------------------------------------------
# Group 2 — slice-extraction logic
# ---------------------------------------------------------------------------

class ExtractSliceTests(unittest.TestCase):
    """Verify _extract_26d / _extract_13d / _extract_13d_left route columns correctly."""

    def test_extract_26d_routes_correct_columns(self):
        """26-D extract = [body[:, 15:22] | body[:, 22:29] | hand[:, 0:6] | hand[:, 6:12]]."""
        T = 3
        body, hand = _make_marker_state(T)
        out = _extract_26d(body, hand)

        self.assertEqual(out.shape, (T, 26))

        np.testing.assert_array_equal(
            out[:, 0:7], body[:, list(STATE_26_BODY_COL_LEFT_ARM)],
            err_msg="26-D[0:7] should be body[:, 15:22] (left arm)."
        )
        np.testing.assert_array_equal(
            out[:, 7:14], body[:, list(STATE_26_BODY_COL_RIGHT_ARM)],
            err_msg="26-D[7:14] should be body[:, 22:29] (right arm)."
        )
        np.testing.assert_array_equal(
            out[:, 14:20], hand[:, list(STATE_26_INSPIRE_COL_LEFT_HAND)],
            err_msg="26-D[14:20] should be hand[:, 0:6] (left hand)."
        )
        np.testing.assert_array_equal(
            out[:, 20:26], hand[:, list(STATE_26_INSPIRE_COL_RIGHT_HAND)],
            err_msg="26-D[20:26] should be hand[:, 6:12] (right hand)."
        )

    def test_extract_26d_dtype_promoted_to_float64(self):
        """_extract_26d output is float64 even when input is float32."""
        body, hand = _make_marker_state(2)
        out = _extract_26d(body.astype(np.float32), hand.astype(np.float32))
        self.assertEqual(out.dtype, np.float64)

    def test_extract_13d_right_matches_26d_right_slice(self):
        """13-D right extract == [26-D arm[7:14] | 26-D hand[20:26]]."""
        T = 4
        body, hand = _make_marker_state(T)
        full_26d = _extract_26d(body, hand)
        full_13d_right = _extract_13d(body, hand)

        expected = np.concatenate([full_26d[:, 7:14], full_26d[:, 20:26]], axis=1)
        np.testing.assert_array_equal(full_13d_right, expected)

    def test_extract_13d_left_matches_26d_left_slice(self):
        """13-D left extract == [26-D arm[0:7] | 26-D hand[14:20]]."""
        T = 4
        body, hand = _make_marker_state(T)
        full_26d = _extract_26d(body, hand)
        full_13d_left = _extract_13d_left(body, hand)

        expected = np.concatenate([full_26d[:, 0:7], full_26d[:, 14:20]], axis=1)
        np.testing.assert_array_equal(full_13d_left, expected)


# ---------------------------------------------------------------------------
# Group 3 — teleop path (action = state[t+1] + elbow_delta)
# ---------------------------------------------------------------------------

class TeleopActionDerivationTests(unittest.TestCase):
    """The teleop path is what every AVP-recorded HDF5 hits today.

    Contract:
        state[t]  = full_26d[t]
        action[t] = full_26d[t+1] + elbow_delta   (out-of-place add)

    The historical bug (April 2026) was an in-place += that aliased state[t]
    via shared NumPy views. These tests pin the magnitudes AND the aliasing
    safety so a "small refactor" can't reintroduce the bug.
    """

    def test_26d_teleop_action_equals_next_state_plus_delta(self):
        """26-D teleop: action[t] = state[t+1] + elbow_delta."""
        T = 5
        body, hand = _make_marker_state(T)
        state, action = convert_g1_state_action_to_lerobot_26d(
            body, hand, action_full=None
        )

        self.assertEqual(state.shape, (T - 1, 26))
        self.assertEqual(action.shape, (T - 1, 26))

        full_26d = _extract_26d(body, hand)
        np.testing.assert_array_equal(state, full_26d[:-1])
        np.testing.assert_array_almost_equal(
            action, full_26d[1:] + STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA
        )

    def test_26d_teleop_does_not_mutate_state_via_aliasing(self):
        """The +delta add must be out-of-place: state buffer must NOT change.

        Regression test for the April 2026 in-place += aliasing bug. ``state``
        and ``full_26d[1:]`` share underlying storage (state is a view of
        ``full_26d[:-1]``, full_26d[1:] is the same buffer offset by one row),
        so an in-place modification of ``action_pre`` would leak into state[0]
        on the elbow columns.
        """
        T = 5
        body, hand = _make_marker_state(T)
        state, action = convert_g1_state_action_to_lerobot_26d(
            body, hand, action_full=None
        )

        # Re-extract a fresh reference for what state SHOULD be.
        expected_state = _extract_26d(body, hand)[:-1]
        np.testing.assert_array_equal(
            state, expected_state,
            err_msg=(
                "State buffer was mutated by the +delta add. Aliasing "
                "regression: ensure the implementation uses "
                "`action = full_26d[1:] + delta` (out-of-place), NOT "
                "`action = full_26d[1:]; action += delta`."
            ),
        )

        # Independence sanity check: mutating action must not affect state.
        action[0, 0] = 99999.0
        self.assertNotEqual(
            state[0, 0], 99999.0,
            "State and action share storage — mutating action affected state. "
            "This is the aliasing bug."
        )

    def test_13d_right_teleop_action_equals_next_state_plus_delta(self):
        """13-D right teleop: action[t] = state[t+1] + right_elbow_delta."""
        T = 4
        body, hand = _make_marker_state(T)
        state, action = convert_g1_state_action_to_lerobot_13d(
            body, hand, action_full=None
        )

        full_13d = _extract_13d(body, hand)
        np.testing.assert_array_equal(state, full_13d[:-1])
        np.testing.assert_array_almost_equal(
            action, full_13d[1:] + STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA
        )

    def test_13d_left_teleop_action_equals_next_state_plus_delta(self):
        """13-D left teleop: action[t] = state[t+1] + left_elbow_delta."""
        T = 4
        body, hand = _make_marker_state(T)
        state, action = convert_g1_state_action_to_lerobot_13d_left(
            body, hand, action_full=None
        )

        full_13d_left = _extract_13d_left(body, hand)
        np.testing.assert_array_equal(state, full_13d_left[:-1])
        np.testing.assert_array_almost_equal(
            action, full_13d_left[1:] + STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA
        )


# ---------------------------------------------------------------------------
# Group 4 — recorded-action paths (53-D, 41-D)
# ---------------------------------------------------------------------------

class RecordedActionPathTests(unittest.TestCase):
    """The 53-D and 41-D recorded-action paths.

    These paths use the index lookup tables (ACTION_HDF5_TO_ENV_*) instead of
    the `state[t+1]` derivation. Tests verify the conversion picks the right
    columns and applies the elbow delta correctly.
    """

    def test_action_53d_path_uses_index_table_and_delta(self):
        """53-D recorded action: action = action_full[:-1, idx_table] + delta."""
        T = 4
        body, hand = _make_marker_state(T)

        # Synthetic 53-D action with marker values per cell.
        action_full = np.fromfunction(
            lambda t, j: 200000 + 1000 * t + j, (T, 53), dtype=np.float64
        )

        _, action = convert_g1_state_action_to_lerobot_26d(body, hand, action_full)

        self.assertEqual(action.shape, (T - 1, 26))

        expected = (
            action_full[:-1, ACTION_HDF5_TO_ENV_26]
            + STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA
        )
        np.testing.assert_array_almost_equal(action, expected)

        # Concrete spot-check at left elbow (canonical idx 3): the value must
        # be the corresponding 53-D position offset by +0.3.
        left_elbow_idx_53 = ACTION_HDF5_TO_ENV_26[3]
        for t in range(T - 1):
            self.assertAlmostEqual(
                action[t, 3], action_full[t, left_elbow_idx_53] + 0.3,
                msg=f"Left elbow at t={t} mismatched the 53-D index lookup."
            )

    def test_action_41d_path_uses_index_table_and_delta(self):
        """41-D recorded action: action = action_full[:-1, idx_table_from_41] + delta."""
        T = 4
        body, hand = _make_marker_state(T)

        action_full = np.fromfunction(
            lambda t, j: 300000 + 1000 * t + j, (T, 41), dtype=np.float64
        )

        _, action = convert_g1_state_action_to_lerobot_26d(body, hand, action_full)

        expected = (
            action_full[:-1, ACTION_HDF5_TO_ENV_26_FROM_41]
            + STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA
        )
        np.testing.assert_array_almost_equal(action, expected)

    def test_recorded_path_does_not_mutate_input(self):
        """Recorded-action conversion must not mutate the caller's action_full."""
        T = 3
        body, hand = _make_marker_state(T)
        action_full = np.fromfunction(
            lambda t, j: 200000 + 1000 * t + j, (T, 53), dtype=np.float64
        )
        before = action_full.copy()

        _, _ = convert_g1_state_action_to_lerobot_26d(body, hand, action_full)

        np.testing.assert_array_equal(
            action_full, before,
            err_msg="convert_g1_state_action_to_lerobot_26d mutated its action_full input."
        )


if __name__ == "__main__":
    unittest.main()
