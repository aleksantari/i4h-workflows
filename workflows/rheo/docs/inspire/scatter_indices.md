# Inspire FTP 41-D Scatter Index Audit

> **Status (2026-04-19):** Middle/pinky slots in `GROUP_SIM_INDICES` are swapped on both hands. Training-side labels are correct, so the policy learns valid input→output pairs, but at eval the scatter routes the "middle" command to the physical pinky joint (and vice versa). Single-episode memorization cannot reproduce a recorded trajectory until this is fixed.
>
> **See also:** [`joint_spaces.md`](joint_spaces.md) §4.6 covers `GROUP_SIM_INDICES` as a drift hotspot with the recommended derivation rule from `actuated_joint_names`.

## 1. Runtime evidence

Inside `./docker/run_docker_grasp.sh`, `python scripts/utils/inspire/verify_scatter_indices.py --headless --enable_cameras` boots the `Isaac-Grasp-Policy-G129-Inspire-Joint` env and prints `actuated_joint_names` in the order the env actually uses (JointPositionActionCfg has `preserve_order=True`, so this is authoritative):

```
 29: left_index_1_joint
 30: left_little_1_joint     ← physical PINKY, not middle
 31: left_middle_1_joint     ← physical MIDDLE, not pinky
 32: left_ring_1_joint
 33: left_thumb_1_joint
 34: right_index_1_joint
 35: right_little_1_joint    ← physical PINKY
 36: right_middle_1_joint    ← physical MIDDLE
 37: right_ring_1_joint
 38: right_thumb_1_joint
 39: left_thumb_2_joint
 40: right_thumb_2_joint
```

Current `GROUP_SIM_INDICES` at [scripts/utils/inspire/inspire_experiment_config.py:68-73](../../scripts/utils/inspire/inspire_experiment_config.py#L68-L73):

```python
GROUP_SIM_INDICES = {
    "left_arm":   [11, 15, 19, 21, 23, 25, 27],
    "right_arm":  [12, 16, 20, 22, 24, 26, 28],
    "left_hand":  [33, 39, 29, 30, 32, 31],   # wrong: 30 is pinky, 31 is middle
    "right_hand": [38, 40, 34, 35, 37, 36],   # wrong: 35 is pinky, 36 is middle
}
```

Mapped to physical joints:

| Canonical slot (STATE_13) | `GROUP_SIM_INDICES["right_hand"]` | env runtime joint | Correct? |
|---|---|---|---|
| R_thumb_yaw | 38 | right_thumb_1_joint | ✅ (under unverified naming assumption — see §4) |
| R_thumb_pitch | 40 | right_thumb_2_joint | ✅ (under unverified naming assumption — see §4) |
| R_index | 34 | right_index_1_joint | ✅ |
| **R_middle** | **35** | **right_little_1_joint (pinky)** | **❌ SWAPPED** |
| R_ring | 37 | right_ring_1_joint | ✅ |
| **R_pinky** | **36** | **right_middle_1_joint (middle)** | **❌ SWAPPED** |

Symmetric bug on the left hand (positions 30 ↔ 31). Arms and thumbs are unaffected.

## 2. Why training still looks fine

The training-time state/action labeling is self-consistent. `_extract_13d` in [scripts/utils/inspire/inspire_lerobot_fields.py:226-232](../../scripts/utils/inspire/inspire_lerobot_fields.py#L226-L232) reads `state_inspire[:, 6:12]`, and `state_inspire` is produced by `get_robot_inspire_joint_states` in [scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py:72-85](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py#L72-L85). The observation list `_INSPIRE_ACTUATED_NAMES` is written in canonical order:

```python
_INSPIRE_ACTUATED_NAMES = [
    "left_thumb_1_joint",  "left_thumb_2_joint",
    "left_index_1_joint",  "left_middle_1_joint", "left_ring_1_joint", "left_little_1_joint",
    "right_thumb_1_joint", "right_thumb_2_joint",
    "right_index_1_joint", "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint",
]
```

and `_resolve_indices` looks each name up in the articulation at runtime. So the 12-D inspire observation is always canonical: position 9 is the physical middle joint, position 11 is the physical pinky. The LeRobot column labeled `R_middle_proximal_joint` is filled from inspire[9], which is the physical middle. Training is correct.

## 3. Why eval is broken

`policies/act.scatter_to_sim` uses `GROUP_SIM_INDICES` to place the 13-D policy output into the 41-D env action. With the current (wrong) `right_hand` list:

- Policy outputs `action[10]` intended for **middle**; scatter writes it to env slot 35 → driven into `right_little_1_joint` (pinky).
- Policy outputs `action[12]` intended for **pinky**; scatter writes it to env slot 36 → driven into `right_middle_1_joint` (middle).

For a typical grasp trajectory both fingers close together, so the failure looks like "fingers aren't quite wrapping the block" rather than something obviously scrambled. For single-episode memorization of a carefully recorded trajectory, the mismatch is enough to prevent reproduction.

The corresponding symmetric swap on left hand matters for the dual-arm (26-D) recipe; right-arm smoketests (13-D) are only affected by the right-hand swap.

## 4. Remaining unverified assumption: thumb_1 vs thumb_2

`_INSPIRE_ACTUATED_NAMES` labels `right_thumb_1_joint` as **thumb_yaw** and `right_thumb_2_joint` as **thumb_pitch** (positions 6 and 7 in the 12-D obs). The scatter indices 38 and 40 then route the policy's thumb_yaw/thumb_pitch commands into those same joints.

If the URDF-to-USD conversion flipped those DOFs (i.e., if `right_thumb_1_joint` is physically pitch and `right_thumb_2_joint` is physically yaw), **it would not show up as an eval-time bug** — training would have learned `state(physical_pitch)→action(physical_pitch)` under the "yaw" label, and eval routes that action back to the same physical joint. Self-consistent, no behavioral failure. So this assumption is invisible to the memorization test and can be ignored for now.

To verify later, inspect the URDF joint axes directly (or lock one thumb joint at a known angle and observe which axis moves).

## 5. Fix

Swap the last two entries in each hand's list:

```python
GROUP_SIM_INDICES = {
    "left_arm":   [11, 15, 19, 21, 23, 25, 27],
    "right_arm":  [12, 16, 20, 22, 24, 26, 28],
    "left_hand":  [33, 39, 29, 31, 32, 30],   # thumb_yaw, thumb_pitch, idx, MID, ring, PINKY
    "right_hand": [38, 40, 34, 36, 37, 35],   # thumb_yaw, thumb_pitch, idx, MID, ring, PINKY
}
```

Also update the inline comments on those two lines (they currently assert `mid@30,31,35,36`; after the swap, the comments should read `mid@31,32,...` style or be rewritten to match reality).

**No retraining is required.** The training-time LeRobot dataset is correct; only the eval-time scatter routing needs fixing. Existing checkpoints are salvageable. Retrain the smoketest only if you want to confirm memorization works end-to-end.

**Other checkpoints affected:** any Inspire FTP ACT checkpoint ever evaluated with this scatter (all of them). RL rollouts using the same scatter were also computing gradients under the wrong physical-joint assignment, so any RL-finetuned Inspire checkpoint needs re-evaluation at minimum, possibly retraining depending on how much the middle/pinky swap biased reward.

## 6. Files touched by the investigation

- [scripts/utils/inspire/verify_scatter_indices.py](../../scripts/utils/inspire/verify_scatter_indices.py) — the runtime D1 check. Keep this; rerun any time `GROUP_SIM_INDICES`, `actuated_joint_names`, or the env cfg joint order changes. Invoke as:
  ```bash
  ./docker/run_docker_grasp.sh python scripts/utils/inspire/verify_scatter_indices.py --headless --enable_cameras
  ```
- [scripts/utils/inspire/inspire_experiment_config.py](../../scripts/utils/inspire/inspire_experiment_config.py) — holds `GROUP_SIM_INDICES`. The file to edit.
- [scripts/utils/inspire/inspire_lerobot_fields.py](../../scripts/utils/inspire/inspire_lerobot_fields.py) — HDF5→LeRobot conversion. Verified correct for this bug (training-side state labeling is fine). Still carries the elbow offset compensation (see `elbow_offset.md`).
- [scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py) — `joint_names` and `actuated_joint_names` in this file define the env's 41-D action layout.
- [scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py](../../scripts/simulation/tasks/grasp_policy_inspire/mdp/observations.py) — `_INSPIRE_ACTUATED_NAMES` defines the 12-D inspire obs ordering. Canonical. Do not change without a coordinated converter update.

## 7. Root cause

`RECORDED_ACTION_53_JOINT_NAMES` in `inspire_lerobot_fields.py` places joints in the order `(index, middle, pinky, ring, thumb_yaw)` within each hand (lines 126-135 of that file). The env cfg's `joint_names` places them in `(index, little, middle, ring, thumb)` order. Someone building `GROUP_SIM_INDICES` cross-referenced the 53-D name list to get "middle at 35, pinky at 36" without noticing the env's actual 41-D order reverses those two. The indices work perfectly against the 53-D RECORDED list — they just don't describe the runtime env, which is what matters at eval.

This is the same class of bug family as the elbow offset: a legacy index list captured from one source of truth (the recorded-action spec) drifted out of alignment with another source of truth (the running env) and the difference only surfaces at eval time because training round-trips through neither list — it uses the observation-term API which resolves by name.
