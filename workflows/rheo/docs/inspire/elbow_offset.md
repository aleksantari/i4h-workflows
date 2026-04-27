# The Elbow Offset: Why `-0.3` Keeps Haunting Us

> **Status:** Currently load-bearing. Do not remove without a coordinated refactor (converter + env cfg + eval scatter + retrain all checkpoints).

## TL;DR

The G1 humanoid's two elbow joints carry a `-0.3 rad` constant shift between the **policy's raw action space** and the **physical joint target** fed to the simulator. It exists so that an untrained policy outputting zeros lands the robot in a sensible resting pose instead of a straight-armed T-pose. This shift leaks into three unrelated code paths (HDF5→LeRobot conversion, the scatter to the 41D sim action at eval, and any analysis that compares recorded `action[t]` against `state[t+1]`). Every time we refactor one of them, we re-learn that the other two exist. This doc is the single source of truth for what it is, why it's there, where it's handled, and what would be involved in removing it.

## 1. Where the `-0.3` physically comes from

The G1's default posture for all our tabletop tasks has both elbows bent at `-0.3 rad` (~17°). This is set in two different places, each with a different role:

### 1a. `default_joint_pos` (spawn pose)

The `ArticulationCfg.InitialStateCfg.joint_pos` dict in each task's `robot_config.py` puts elbows at `-0.3` at reset:

- [scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py:60,67](../../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py#L60)
- [scripts/simulation/tasks/assemble_trocar/config/robot_config.py:131,138](../../scripts/simulation/tasks/assemble_trocar/config/robot_config.py#L131)

This is the **physics state** at `env.reset()`. It is independent of any action mapping — the robot simply starts with its elbows bent.

### 1b. `offset_dict` on the action term (coordinate shift)

Each task also wires an `offset=` dict into its `JointPositionActionCfg`:

```python
offset_dict = {
    "left_elbow_joint": -0.3,
    "right_elbow_joint": -0.3,
}

joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=actuated_joint_names,
    scale=1.0,
    use_default_offset=False,
    offset=offset_dict,
    preserve_order=True,
)
```

- [scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py:112-115,264](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L112-L115)
- [scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py:84-87,170](../../scripts/simulation/tasks/assemble_trocar/g1_assemble_trocar_env_cfg.py#L84-L87)

IsaacLab's `JointPositionActionCfg` computes the per-step joint target as

```
joint_target = scale · raw_action + offset
```

With `scale=1.0` and `offset[elbow] = -0.3`, the policy's "raw action" space is **delta from rest pose** for the elbow (and identity for every other joint, since their offset is 0).

### 1c. The design intent

- **Zero action ≈ hold rest pose.** A freshly initialized RL policy (or a collapsed one) outputs something near zero. With the offset, that means "elbows stay bent, arms stay down." Without it, zero action commands elbow = 0, which stretches the arm straight and likely makes the robot tip forward or drag the tray edge.
- **Matches the inductive bias used by trocar.** The Inspire FTP task originally inherited this convention from the (now-removed) Dex3 grasp task, which itself adopted it from trocar — hence the same pattern in both surviving task packages.

So: `default_joint_pos` says "start here in physics space"; `offset_dict` says "policy outputs are centered on this pose in action space." They happen to use the same numerical value because they describe the same physical pose, but they are conceptually independent — you could keep one and drop the other.

## 2. Why it leaks into data conversion

The teleop recordings (`record_demos.py`) store PinkIK's output, which is **absolute joint targets** (next-step joint positions), not the raw action the training-time policy will consume. Specifically, for the single-arm Inspire FTP recorder the HDF5 action has shape `(T, 38)` and contains target joint positions directly.

At ACT training time, the policy must predict something that the env can execute verbatim: the **raw action**, because the env unconditionally computes `joint_target = raw + offset` on top. If we feed target-joint-positions straight into training, the model learns `raw := target` and at rollout the env double-applies the offset: the elbow ends up at `target − 0.3`, which bends the arm further every step. Trajectory collapses.

The fix in the HDF5→LeRobot converter is to invert the env's mapping once, at data-prep time:

```
raw = target − offset
    = target − (−0.3)   (at the elbow dim)
    = target + 0.3
```

This is exactly what the three `STATE_*_RAW_ACTION_FROM_PROCESSED_DELTA` vectors encode in [scripts/utils/inspire_lerobot_fields.py](../../scripts/utils/inspire_lerobot_fields.py):

| Variant | Vector | Non-zero entries |
|---|---|---|
| Dual-arm 26-D | `STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA` | `[3] = 0.3` (left elbow), `[10] = 0.3` (right elbow) |
| Right-arm 13-D | `STATE_13_RAW_ACTION_FROM_PROCESSED_DELTA` | `[3] = 0.3` (right elbow) |
| Left-arm 13-D | `STATE_13_LEFT_RAW_ACTION_FROM_PROCESSED_DELTA` | `[3] = 0.3` (left elbow) |

Each variant of `convert_g1_state_action_to_lerobot_*D` adds this vector to `full_*D[1:]` in the 38-D PinkIK teleop branch. **Forgetting to add it in any one branch silently shifts the elbow dimension by 0.3 rad every frame at eval time — which is how this whole investigation started.**

Analogous logic lives in the trocar converter ([scripts/utils/assemble_trocar_lerobot_fields.py:57-61](../../scripts/utils/assemble_trocar_lerobot_fields.py#L57-L61)) where `STATE_28_RAW_ACTION_FROM_PROCESSED_DELTA` carries `0.3` at indices 3 and 10 for the same reason.

### 2a. The 53-D recorded branch is *already correct*

For episodes recorded against the 53-D action space (full joint_pos action term, mimic joints included), the HDF5 `actions` tensor is already in raw-action space, because `JointPositionActionCfg` applies its offset when *executing* actions, not when recording them. The converter simply indexes into the 53-D slice and no elbow compensation is needed. The compensation only applies to the 38-D PinkIK-teleop branch where the recorded array is observed joint positions, not raw actions.

## 3. Why it leaks into eval-time scatter

The eval-time flow for single-arm ACT is:

1. Policy predicts a 13-D raw action (one arm + one hand).
2. [scripts/simulation/policies/act.py](../../scripts/simulation/policies/act.py) scatters those 13 values into a 41-D vector at the canonical `GROUP_SIM_INDICES` slots.
3. The remaining 28 entries of the 41-D vector must hold the non-controlled joints at their rest state — otherwise the passive arm, torso, and opposite hand droop under gravity.

That "hold rest state" value is **also in raw action space**, not joint-target space, because the env will again apply `+ offset` on top. So the dummy/hold action for the elbow of the *non-commanded* arm has to be `−offset_dict[elbow] = +0.3`, not 0 and not the physical rest pose `-0.3`.

This is handled explicitly in [scripts/simulation/examples/eval_act_inspire.py:220-231](../../scripts/simulation/examples/eval_act_inspire.py#L220-L231):

```python
# hold_41d is computed from default_joint_pos (= physical rest pose),
# so hold_41d must be in RAW action space: subtract the term's offset
# (and divide by scale) so that once the term re-applies them, each
# joint lands at its default_joint_pos. Elbows carry -0.3 in the
# env's offset_dict — without this subtraction they'd be double-
# offset and the non-controlled forearm would droop past the init.
```

If this subtraction is ever removed, the non-commanded arm ends up at `-0.6 rad` at the elbow on the first step, which manifests as "the other arm sags visibly during eval." The symptom is easy to spot visually but obnoxious to diagnose from first principles.

## 4. Inventory: every file that has to agree

When reasoning about any elbow-angle value in the Inspire FTP grasp or trocar pipeline, you must know which space the value lives in. The table below lists every code path that touches the offset.

| Layer | File | What it does with -0.3 / +0.3 |
|---|---|---|
| Env spawn | [grasp_policy_inspire/config/robot_config.py:60,67](../../scripts/simulation/tasks/grasp_policy_inspire/config/robot_config.py#L60) | Sets physical elbow = -0.3 at reset. Independent of action mapping. |
| Env action term | [grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py:112-115,264](../../scripts/simulation/tasks/grasp_policy_inspire/g1_grasp_policy_inspire_env_cfg.py#L112-L115) | Passes `offset=offset_dict` to `JointPositionActionCfg`. Defines the raw-action coordinate system. |
| Teleop converter (26-D dual) | [scripts/utils/inspire_lerobot_fields.py:77-79](../../scripts/utils/inspire_lerobot_fields.py#L77-L79), used in `convert_g1_state_action_to_lerobot_26d` | Adds +0.3 at elbow dims when converting 38-D PinkIK → raw-action LeRobot. |
| Teleop converter (13-D right) | [scripts/utils/inspire_lerobot_fields.py:213-215](../../scripts/utils/inspire_lerobot_fields.py#L213-L215), used in `convert_g1_state_action_to_lerobot_13d` | Same, right elbow only. (Bug fix: was missing in the `else` branch for the 38-D case; fixed 2026-04-19.) |
| Teleop converter (13-D left) | [scripts/utils/inspire_lerobot_fields.py:282-284](../../scripts/utils/inspire_lerobot_fields.py#L282-L284), used in `convert_g1_state_action_to_lerobot_13d_left` | Same, left elbow only. |
| Trocar converter | [scripts/utils/assemble_trocar_lerobot_fields.py:57-61](../../scripts/utils/assemble_trocar_lerobot_fields.py#L57-L61) | Same pattern for the trocar task. |
| Eval scatter (hold) | [scripts/simulation/examples/eval_act_inspire.py:220-231](../../scripts/simulation/examples/eval_act_inspire.py#L220-L231) | Inverts offset when computing the 41-D hold vector fed to `policy.set_default_action`. |

Any new trainer, new recorder, new scatter path, or new debugging notebook that **moves data between "joint position" and "raw action" spaces must account for this shift at the elbow dims.**

## 5. Debugging signatures

If you suspect the offset has been mishandled:

- **Missed in converter:** `action[t] − state[t+1]` at the elbow dim equals `0` (should be `+0.3`). At eval the elbow drifts negative by ~0.3 rad after the first frame and never recovers. The recorded trajectory looks "correct" in replay-mode but learned policies never reproduce it. This is the bug that prompted this doc. Spot it with:
  ```python
  import pyarrow.parquet as pq
  t = pq.read_table("<ep_parquet>")
  a0 = t.column("action")[0].as_py()
  s1 = t.column("observation.state")[1].as_py()
  print("elbow a0 - s1 =", a0[3] - s1[3])   # should be ≈ 0.3
  ```

- **Missed in eval hold:** the non-commanded arm's elbow sags to ~`-0.6 rad` on the first step of rollout and stays there. Visual only — nothing crashes.

- **Double-applied in converter:** elbow raw action in the LeRobot dataset is ~`+0.6`. At rollout the robot pushes its forearm up into the shoulder. Also visible in dataset stats as an unusually large mean on the elbow dim.

- **Stats sanity:** `datasets/.../meta/stats.json`, `action.mean[3]` for a right-arm dataset should be ≈ `rest_elbow_target + 0.3`, i.e., a small positive number. A near-zero or negative mean is suspicious.

## 6. The case for eventually removing it

The offset is a **convenience**, not a requirement:

- **Pro-remove:** Three conversion code paths collapse to the 53-D branch. The eval-time hold inversion becomes a no-op. Future task forks won't inherit the trap. One fewer source of silent bugs per data pipeline.
- **Pro-keep (short-term):** All existing checkpoints (trocar GR00T + RL, Inspire FTP ACT) were trained with the offset present. Removing it invalidates every one of them — they would command elbow = 0 and immediately drive the arm into an extended posture.
- **Pro-keep (RL prior):** "Zero action = rest pose" is a mildly useful inductive bias for PPO-from-scratch. Without it, exploration starts from a random-arms posture. In practice this is small compared to other RL problems on this task, but it's non-zero.

### What a removal would actually touch

If/when we decide to remove it, the changes are coordinated but mechanical:

1. Set `offset_dict = {}` (or drop `offset=` entirely) in both task env cfgs (`grasp_policy_inspire`, `assemble_trocar`). `default_joint_pos` stays — the physical rest pose is unchanged.
2. Delete the three `STATE_*_RAW_ACTION_FROM_PROCESSED_DELTA` vectors in `inspire_lerobot_fields.py` and their additions in the `convert_*` functions. Same for `assemble_trocar_lerobot_fields.py`.
3. Remove the offset subtraction in `eval_act_inspire.py`'s hold-41D computation — the raw-action rest pose then equals `default_joint_pos` directly.
4. Rerun **all** HDF5 → LeRobot conversions for all tasks; prior datasets become stale.
5. Retrain **all** IL checkpoints against the new datasets; prior checkpoints become stale.
6. Retrain **all** RL checkpoints (or at minimum verify that existing ones are still in-distribution — they will not be at the elbow dim).

Because the blast radius spans both the IL and RL pipelines for the surviving tasks, removing the offset is best done as a single dedicated refactor, not piggybacked onto another change.

## 7. Rule of thumb while the offset remains

**Every time you touch any code that moves data between "joint position target" and "raw policy action," ask:** *am I on the side of the equation with the `+offset`, or the side with the `−offset`?* For the G1 elbow, both sides of that equation differ by `0.3 rad`, and the sign flips depending on whether you're preparing data (→ subtract env offset → `+0.3`) or interpreting policy output at eval (→ env will add offset → no action needed from you; the env handles it). The hold-41D path in eval is an additional subtlety because you're constructing a raw-action vector by hand from the physical rest pose.

When in doubt:
- **Physical space (`env.scene["robot"].data.joint_pos`, `default_joint_pos`):** elbow sits at `-0.3`.
- **Raw action space (policy output, LeRobot `action` column, `hold_41d`):** elbow sits at `0`.
- **Difference between the two at the elbow dim:** `+0.3` (physical − raw).

Everything in this repo is one of those two coordinate systems. If your code crosses the boundary, check the sign.
