# Debug log — Inspire FTP ACT policy rollout failure

A running narrative of hypotheses, findings, and decisions while diagnosing why the
trained ACT model fails to reach the surgical tool during evaluation. Newest entries
at the bottom; earlier context is kept so the thought process stays traceable.

---

## Session 1 — 2026-04-09

### What we observed

- Trained ACT on `datasets/inspire_tool0_slot1/single_arm.hdf5`
  - 30 teleop demos, episode lengths in the 260–425 step range
  - Task: pick up `tool_0` at slot 1 of the surgical tray and drop into the bin
- Running `scripts/simulation/examples/eval_act_inspire.py --slot 1` (matching training)
- Rolling out for 5000 sim steps, the arm never even gets to the tool — let alone
  attempts a grasp. No obvious trajectory toward the tray.

### Sanity-check arithmetic (user's framing)

- Sim runs at 50 Hz. `chunk_size = 50` means one inference commits ~1 s of motion.
- Training episodes are ~300 steps, so a successful rollout should need ≈6 inferences.
- 5000 steps is ~16× the length of a training episode — well beyond training distribution.
- Conclusion: if the arm is still flailing at step 5000 on the same slot it was trained
  on, something is mechanically wrong in the train → eval handshake, not just poor
  generalization.

### Hypotheses considered

1. **Slot mismatch.** Training was on slot 1; `eval_act_inspire.py` defaults to `--slot 4`.
   Slot 1 → slot 4 is a `0.2365 m` Y-shift, which would put the tool outside the trained
   distribution. **User confirmed** they were running with `--slot 1`, so this is not
   the active cause. Noted for future runs: the script default (`4`) is misleading.
2. **Action coordinate mismatch** — the most load-bearing hypothesis. Investigated below.
3. **Data quantity.** 30 demos is borderline for ACT (paper uses ≈50 per task).
   Filed as secondary; only revisit after the primary bug is fixed.
4. **Initial reset state drift.** Verified `obs[0]` of training demos matches the env's
   `DEFAULT_JOINT_POS` (elbows at `-0.3`). Not a divergence.
5. **Normalization.** LeRobot ACT handles `MEAN_STD` normalization internally via
   `ACTPolicy.select_action()` using dataset stats baked into the checkpoint. Eval
   does not bypass this. Not a divergence.
6. **Camera preprocessing.** Same format training and eval (uint8 → float/255 → CHW).
   Not a divergence.
7. **26D → 41D scatter indices.** Verified against `actuated_joint_names` order in the
   env cfg. Correct.

### Primary finding: elbow offset asymmetry (`inspire_ftp_lerobot_fields.py:294`)

Walking the HDF5 → LeRobot → train → eval pipeline end-to-end turned up a broken
contract between how the training data stores actions and how the eval env applies
actions.

**HDF5 action shape is 38, not 41 or 53.** The teleop env
(`g1_grasp_policy_inspire_teleop_env_cfg.py:27-29`) uses `PinkInverseKinematicsActionCfg`,
whose action format is `[l_wrist_pos(3) | l_wrist_quat(4) | r_wrist_pos(3) | r_wrist_quat(4) | hand_joints(24)] = 38`.
`record_demos.py` stores exactly that 38-D PinkIK input as both `actions` and
`processed_actions`. Confirmed by reading `demo_0`:

```
data/demo_0/
  actions:           (425, 38) float32
  processed_actions: (425, 38) float32
  obs/robot_joint_state:         (425, 87)
  obs/robot_inspire_joint_state: (425, 12)
```

**The conversion's `else` branch fires for every episode.** In
`scripts/utils/inspire_ftp_lerobot_fields.py:270-300`:

```python
if action_full.shape[1] == 53:
    action = action_full[:-1, ACTION_HDF5_TO_ENV_26]
    action += STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA   # +0.3 on elbows
elif action_full.shape[1] == 41:
    action = action_full[:-1, ACTION_HDF5_TO_ENV_26_FROM_41]
    action += STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA   # +0.3 on elbows
else:
    # Teleop (38D PinkIK) or unknown:
    # "Elbow offset is already baked into the observed positions."
    action = full_26d[1:]                                # NO +0.3 applied
```

Because the data is always 38-D, the third branch is always taken. The parquet column
`action[t]` is literally `observed_state[t+1]` in canonical 26-D order — true joint
positions, unshifted.

**The eval env applies a -0.3 elbow offset that was never compensated.**
`g1_grasp_policy_inspire_env_cfg.py:257-264`:

```python
joint_pos = mdp.InspireFTPJointPositionActionCfg(
    scale=1.0,
    use_default_offset=False,
    offset={"left_elbow_joint": -0.3, "right_elbow_joint": -0.3},
)
```

`JointPositionAction.apply_actions()` computes `target = action * scale + offset`,
so for elbows specifically: `sim_target = policy_output - 0.3`.

The 53-D and 41-D branches account for this by baking `+0.3` into the parquet action
column, so `policy_output = true_target + 0.3` and the env subtracts it back out. The
38-D branch was missed — the parquet stores `true_target` directly, so at eval the env
subtracts `0.3` from an already-correct target, driving both elbows `0.3 rad (~17°)`
more negative than the policy intended on every single step.

**Sample confirmation.** First observation of `demo_0`:
```
body[0, 15:22] (left_arm):  [-0.5, 0, 0, -0.3, 0, 0, 0]
body[0, 22:29] (right_arm): [-0.5, 0, 0, -0.3, 0, 0, 0]
```
Both elbows at `-0.3`. The policy will learn to output ≈`-0.3` near the start of an
episode. At eval, the env commands `-0.6`.

**Closed-loop amplification.** The bug is self-reinforcing:
1. `t=0`: obs elbow `-0.3`, policy predicts next ≈`-0.3`, env commands `-0.6`.
2. PD controller drives elbow toward `-0.6`; several steps later the observed elbow
   is around `-0.5`.
3. Policy sees `-0.5` (already off the edge of training distribution), predicts next
   ≈`-0.5`, env commands `-0.8`.
4. Camera view drifts, state distribution drifts further, predictions become unreliable,
   the arm never extends.

This is consistent with "arm flailing, never reaches the tool" as the dominant symptom.

### Recommended fix (primary)

Add the elbow compensation to the 38-D else branch so the training contract is
symmetric across all three HDF5 widths. One line in
`scripts/utils/inspire_ftp_lerobot_fields.py` around line 294:

```python
else:
    # Teleop recording (38D PinkIK) or unknown width:
    # action = next-step observed joint positions.
    action = full_26d[1:]                                    # (T-1, 26)
    action += STATE_26_RAW_ACTION_FROM_PROCESSED_DELTA       # +0.3 elbows
                                                             # to cancel env offset
```

After the edit:
1. Re-run the HDF5 → LeRobot conversion to regenerate the parquet under
   `datasets/inspire_tool0_slot1/single_arm/lerobot/`.
2. Retrain (fresh 50 k-step run) or fine-tune the current checkpoint off the new
   dataset. Fine-tune is usually enough to surface whether the fix is helping.
3. Rerun eval with `--log_actions --slot 1` and verify elbow columns (sim indices
   21, 22) stay in the `-0.5..-0.1 rad` band rather than drifting below `-0.6`.

**Why not remove the env offset instead.** The `-0.3` elbow offset on
`InspireFTPJointPositionActionCfg` exists so that the policy's "zero action" resolves
to a safe bent-elbow pose matching `DEFAULT_JOINT_POS`. Removing it would change the
zero-action resolution and ripple through reset defaults, success-check pose, and
the matching 53-D/41-D branches. A one-line conversion fix is localized and safe.

### What to check after the primary fix lands

- Spot-check the regenerated parquet: load `episode_0`, confirm `action[0, 3]` and
  `action[0, 10]` (the 26-D left/right elbow columns) are `~0.0` rather than `~-0.3`.
- Diagnose eval with `--log_actions`: watch the per-chunk elbow ranges.
- Reach check at step ~150: arm should be approaching the tray. By step ~300 it
  should be attempting a grasp.
- If the arm now reaches but grasps are unreliable, revisit the data-quantity and
  state-as-action hypotheses (a teleop action term that records joint targets
  directly would give a stronger training signal than next-state prediction).

### Open questions / things to keep tracking

- The 38-D teleop representation means the parquet "action" is effectively a one-step
  state predictor. That is a legitimate training signal at 50 Hz but is weaker than
  recording joint targets. Consider whether a future data collection pass should
  switch the teleop env to expose the post-IK joint targets instead.
- The `eval_act_inspire.py` default `--slot 4` is a footgun for anyone training on a
  different slot. Worth changing the default to `1` or making it explicit.
- `TRAY_SLOT_POSITIONS` differ by only `±0.055 m` in X and `0.2365 m` in Y between the
  back-row and front-row tool positions — enough to matter for generalization if we
  ever try to eval on an untrained slot.
