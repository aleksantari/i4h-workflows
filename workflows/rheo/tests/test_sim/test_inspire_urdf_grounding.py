# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint-space grounding test — Layer 1 (URDF spec ↔ code) + Layer 2 (USD ↔ code).

Layer 1 (5 checks): parses the active Inspire FTP URDF as XML and asserts that
the joint-space constants in env_cfg, mimic_action, and robot_config agree with
it. Catches drift in joint names, mimic relationships, multipliers, and
actuator regex coverage. Pure Python apart from the constants imports.

Layer 2 (1 check): spawns the articulation via `G129_CFG_WITH_INSPIRE_BASE_FIX`
and asserts `articulation.data.joint_names` matches `env_cfg.joint_names`
*list-wise* (order matters). This is the order-sensitive check that Layer 1
explicitly cannot do — URDF XML order is an arbitrary authoring artifact;
USD articulation order is determined by the converter's tree traversal.

Boots IsaacLab Kit at module load (~5-10s) so the imported modules and the
articulation spawning work. Run inside Docker:

    ./docker/run_docker_grasp.sh python -m unittest tests.test_sim.test_inspire_urdf_grounding -v

See docs/inspire/joint_spaces.md for the full audit this test pins.
"""

# AppLauncher MUST be called before importing most isaaclab.* modules.
from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True)
_simulation_app = _app_launcher.app

# Standard library — safe before or after Kit boot.
import re  # noqa: E402
import unittest  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

# IsaacLab + project imports — must come AFTER AppLauncher.
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from simulation.tasks.grasp_policy_inspire.config.robot_config import (  # noqa: E402
    G129_CFG_WITH_INSPIRE_BASE_FIX,
)
from simulation.tasks.grasp_policy_inspire.g1_grasp_policy_inspire_env_cfg import (  # noqa: E402
    _MIMIC_JOINT_NAMES,
    joint_names as ENV_CFG_JOINT_NAMES,
)
from simulation.tasks.grasp_policy_inspire.mdp.mimic_action import MIMIC_RULES  # noqa: E402

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


# ---------------------------------------------------------------------------
# Layer 2 scaffolding — minimal scene that spawns just the robot articulation.
# ---------------------------------------------------------------------------

@configclass
class _GroundingSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + dome light + the Inspire FTP G1."""

    num_envs: int = 1
    env_spacing: float = 2.0

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )
    robot = G129_CFG_WITH_INSPIRE_BASE_FIX.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class InspireGroundingTests(unittest.TestCase):
    """Five Layer-1 checks (URDF → code) + one Layer-2 check (USD → code)."""

    # Set in setUpClass after the articulation has been spawned + initialized.
    _ARTICULATION_JOINT_NAMES: list[str] = []

    @classmethod
    def setUpClass(cls):
        """Spawn the articulation once and capture its joint_names list."""
        sim_cfg = sim_utils.SimulationCfg(dt=0.005)
        cls._sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = _GroundingSceneCfg(num_envs=1, env_spacing=2.0)
        cls._scene = InteractiveScene(scene_cfg)
        cls._sim.reset()
        cls._ARTICULATION_JOINT_NAMES = list(cls._scene["robot"].data.joint_names)

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

    # -- Layer 2: 6. Articulation order matches env_cfg.joint_names list-wise -

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
