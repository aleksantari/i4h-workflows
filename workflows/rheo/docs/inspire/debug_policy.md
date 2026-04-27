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

---

## Session 2 — 2026-04-19

### Where things stand

Since Session 1 we have landed two fixes and confirmed the smoketest is still not
reproducing the recorded trajectory — though the fingers are now visibly closing
correctly after the second fix. Documenting remaining hypotheses so none are lost
if context rolls over again.

**Fixes already applied:**

1. **Elbow offset compensation** — 38-D branch of
   [scripts/utils/inspire_ftp_lerobot_fields.py](../../scripts/utils/inspire_ftp_lerobot_fields.py)
   now adds `+0.3` to elbow columns (matching the 53-D/41-D branches). Context and
   rationale: [elbow_offset.md](elbow_offset.md). Required re-converting the
   dataset and rebuilding `demo_ep28`, then retraining the smoketest.
2. **Middle/pinky scatter swap** — [scripts/utils/inspire_ftp_experiment_config.py](../../scripts/utils/inspire_ftp_experiment_config.py)
   `GROUP_SIM_INDICES` for both hands now respects the env's `actuated_joint_names`
   order (`little_1` before `middle_1`). No retraining needed — training labels were
   correct, only eval-time scatter was wrong. Full writeup:
   [scatter_indices.md](scatter_indices.md).

After both fixes + re-eval of the `005000` smoketest checkpoint, fingers look
better but the policy still does not reproduce the recorded motion. Remaining
hypotheses below.

### Remaining hypotheses (unverified)

**From the original D1–D5 diagnostic plan:**

- ~~**D2 — Runtime obs state layout.**~~ **Checked 2026-04-19. Clean.**
  Extended `verify_scatter_indices.py` to dump canonical name order plus post-reset
  obs values. Confirmed: `body[22:29]` = 7 right-arm joints in canonical order;
  `inspire[6:12]` = `[R_thumb_1, R_thumb_2, R_index_1, R_middle_1, R_ring_1, R_little_1]`
  (= `thumb_yaw, thumb_pitch, index, middle, ring, pinky` per the canonical label
  convention); runtime `body[22]=-0.5`, `body[25]=-0.3`, all others 0 — identical
  to parquet row 0 of `demo_ep28`. Obs state path is not the bug.
- ~~**D5 — Training loss floor.**~~ **Checked 2026-04-19. Clean.**
  `train/l1_loss` reached 0.034 at step 5000 (still decreasing slowly). To verify
  memorization quality in real units, wrote
  [scripts/utils/offline_replay_mae.py](../../scripts/utils/offline_replay_mae.py)
  and replayed ep28 through the checkpoint. Results: overall raw MAE
  **0.0138 rad (~0.79°/joint/step)**, uniform across arm and finger dims. The
  model memorized the trajectory tightly. This means the failure must be at
  eval time — the model is not seeing the observations it expects, not failing
  to reproduce the actions given good observations.

**Environmental / distribution-shift hypotheses (mentioned, never checked):**

- **Reset-pose mismatch (investigating next).** Does `obs[0]` at eval start
  match the first frame of the recorded episode? The env resets to
  `default_joint_pos` (elbows at -0.3, arms at 0) + randomized block position.
  The recorded demo starts from whatever pose the teleop operator happened to
  be in. For a memorized single-episode policy, any drift in the first frame
  means the first predicted action is wrong, and the model has no mechanism to
  recover — every subsequent frame is further off-distribution.
- **Control cadence / dt mismatch.** Record-time step dt vs eval-time step dt.
  If eval ticks at a different rate than the 50 Hz the dataset was recorded at,
  the 100-step chunk covers a different wall-clock window than trained. Check
  `sim.dt`, `decimation`, and any render-gating delays.
- ~~**Camera pose drift.**~~ **Checked 2026-04-19. Minor contributor only.**
  Camera prim paths, pos/rot offsets, focal lengths, resolution, lighting, and
  materials are all identical between teleop and eval envs (teleop inherits from
  base env and only overrides actions / XR / episode length). One real
  discrepancy: teleop env sets `sim.render_interval = 2` while the base (eval)
  env uses `render_interval = decimation = 4`. Both envs render at the same
  policy-step boundaries but teleop gets 2× more DLAA temporal history samples.
  Pixel-diff of reset frames after 10 static-arm warmup steps:
  front camera MAE=1.05/255 (0.41%), wrist cams MAE≈0.54/255 (0.21%). 99% of
  pixels differ by ≤8/255. Post-ResNet18 normalization this is ~1.6% input
  shift — small, not the primary bug. In-motion divergence could be larger
  (DLAA less converged during movement) but untested. Cleanup target: set
  `self.sim.render_interval = self.decimation` in the teleop env to match the
  production cadence.
- ~~**Normalization stats scope.**~~ **Checked 2026-04-19. Clean.**
  `demo_ep28/lerobot/meta/episodes_stats.jsonl` (count=211, ep28 only) was written
  at 16:39 after the parquet at 16:33 and before the 20:39 training run — fresh
  and single-episode-scoped. Action[3] (right elbow) stats shift +0.3 relative to
  observation.state[3], confirming the elbow-offset fix is baked in correctly and
  no stale pre-fix stats are in play. Image stats are the LeRobot default
  placeholder mean=0.5/std=0.25 (not ImageNet, not real ep28 pixel stats) — same
  both at train and eval so self-consistent for memorization, worth knowing but
  not a bug.

### Check order (cheapest first)

1. **Reset-pose mismatch** — compare env `obs[0]` against parquet row 0 in
   `demo_ep28/lerobot`. No sim boot needed beyond one env.reset(). *(Next.)*
2. **Normalization stats** — inspect `meta/stats.json` timestamps and values.
   Trivial JSON read.
3. **D2 obs layout** — add state check to `verify_scatter_indices.py` and rerun
   inside docker.
4. **D5 loss floor** — open wandb.
5. **Cadence / cameras** — only if the above pass.

### Elbow-offset fix re-validation — 2026-04-20. Confirmed correct.

User reported the pre-fix smoketest (`right_arm_smoketest_20260419-152551/checkpoints/005000`,
trained on parquet before the `+0.3` shift was added) visually reaches closer to
the tool at eval than the post-fix checkpoint (`right_arm_smoketest_20260419-203926/checkpoints/005000`)
under the same `eval_act_inspire.py` invocation with a deterministic block pose.
This is counter-intuitive — the fix should make the model command the recorded
trajectory exactly, while the pre-fix should systematically under-reach.

**Diagnostic E1:** Ran
[scripts/utils/offline_replay_mae.py](../../scripts/utils/offline_replay_mae.py)
against both checkpoints using the current (post-fix) `demo_ep28` parquet.

| Dim | Pre-fix raw MAE | Post-fix raw MAE |
|-----|----------------:|-----------------:|
| 0 shoulder_pitch | 0.010 | 0.010 |
| 1 shoulder_roll  | 0.007 | 0.008 |
| 2 shoulder_yaw   | 0.018 | 0.019 |
| **3 elbow**      | **0.298** | **0.008** |
| 4 wrist_roll     | 0.016 | 0.016 |
| 5 wrist_pitch    | 0.008 | 0.009 |
| 6 wrist_yaw      | 0.008 | 0.008 |
| 7–12 hand joints | 0.004–0.021 | 0.005–0.022 |
| **Overall**      | 0.0358 | 0.0138 |

The entire 0.022 overall-MAE gap is concentrated in dim 3 (right_elbow); in std
units the elbow jumps from 0.05 (post-fix, comparable to every other dim) to
**1.79 (pre-fix)**. The pre-fix model memorized `action[3] ≈ observed_state[3]`
(no shift); the post-fix model memorized `action[3] ≈ observed_state[3] + 0.3`.
**The fix is mechanically doing exactly what was designed.**

After the env's `-0.3` elbow offset:
- **Post-fix commanded elbow target = recorded `observation.state[3]`** — matches
  the teleop trajectory on every step, including the peak-reach extension at
  ≈+0.14 rad.
- **Pre-fix commanded elbow target = recorded `observation.state[3]` − 0.3** —
  0.3 rad more bent than the recording everywhere (peak at ≈−0.16 rad).

**Implication for the "pre-fix reaches closer" observation.** The pre-fix model
is *not* reaching closer because the fix is wrong — it reaches closer *despite*
commanding an over-bent elbow. Two candidate explanations (not yet tested;
filed for the main debug thread):

1. The teleop recording itself didn't successfully reach/grasp in the first
   place, so a checkpoint that memorizes it perfectly cannot succeed either.
2. Eval-world geometry differs from teleop-world (block pose, PD tracking lag,
   dome lighting -> perception, etc.), so the "correct" elbow angle to reach
   the tool in eval is not the same as the recorded observation.

**Decision: keep the fix.** The post-fix training contract is correct, the
model memorized it to 0.008 rad MAE, and the remaining regression lives
elsewhere in the train→eval handshake — most likely in the observation-side
distribution shift items still on the open list above (reset-pose mismatch,
control cadence, or recording quality).

### Inference pattern — chunk-exhaustion vs temporal ensembling — 2026-04-20

**Current pattern (chunk-exhaustion).** Our wrapper
[scripts/simulation/act_closedloop_policy.py](../../scripts/simulation/act_closedloop_policy.py)
calls `self.policy.predict_action_chunk(obs)` once per inference and returns a
`(1, 100, 13)` chunk. The eval loop in
[scripts/simulation/examples/eval_act_inspire.py](../../scripts/simulation/examples/eval_act_inspire.py)
buffers the first `--action_chunk_size` (default 50) actions from that chunk,
pops one per env step, and re-queries only when the buffer is empty. Net
behavior: **one inference per 50 env steps (~1 s at 50 Hz), fully open-loop
within each chunk, no blending across chunks.** `select_action` is explicitly
bypassed (see comment at
[scripts/simulation/act_closedloop_policy.py:280-282](../../scripts/simulation/act_closedloop_policy.py#L280-L282))
because with the default LeRobot config (`temporal_ensemble_coeff=None`) it
would just pop from an `n_action_steps`-deep queue and collapse each chunk to
a single repeated action.

**What the docker-bundled LeRobot offers.** `ACTPolicy`
(`/isaac-sim/kit/python/lib/python3.11/site-packages/lerobot/common/policies/act/modeling_act.py`)
ships `ACTTemporalEnsembler` at lines 180-268 — the ACT paper's
exponential-weighted overlap blender. Activation mechanism at line 77: if
`config.temporal_ensemble_coeff is not None`, `ACTPolicy.__init__`
instantiates `self.temporal_ensembler = ACTTemporalEnsembler(coeff, chunk_size)`,
and `ACTPolicy.select_action` (lines 120-123) routes every call through
`predict_action_chunk → ensembler.update → single blended action`. Weights
are `w_i = exp(-coeff * i)` with the paper default `coeff=0.01` giving
**older actions more weight** (rationale: aggressive newer-weighting
diminishes the smoothing benefit of chunking — see
https://github.com/huggingface/lerobot/pull/319). Validation constraint at
`configuration_act.py` line 148: `n_action_steps` must be 1 when ensembling
is enabled (policy is re-queried every env step).

**Why this is the top remaining inference-time hypothesis.** Our open-loop
window is 50 env steps on a single observation. At raw MAE 0.008 rad on the
elbow (post-fix offline replay) the model is essentially perfect at every
single step in training, but closed-loop rollout amplifies any observation
that lands slightly off the training manifold — once the arm deviates, the
next chunk gets a novel observation and the error compounds. Temporal
ensembling injects fresh visual feedback every step *and* averages single-query
errors across ≤100 overlapping predictions, which is exactly the failure mode
the ACT paper designed the ensembler to address.

**Implementation landed 2026-04-20 (pre-A/B).** Added `temporal_ensemble_coeff`
plumbing:
- Wrapper
  [scripts/simulation/act_closedloop_policy.py](../../scripts/simulation/act_closedloop_policy.py):
  when the YAML config sets `temporal_ensemble_coeff`, after loading the
  checkpoint we mutate `self.policy.config.temporal_ensemble_coeff`,
  `n_action_steps=1`, attach a fresh `ACTTemporalEnsembler(coeff, chunk_size)`,
  and call `self.policy.reset()`. `_forward_action_chunk` then routes through
  `select_action` (returning `(1, action_dim)`) instead of
  `predict_action_chunk` (returning the full chunk). `action_chunk_length`
  drops to 1 so the eval buffer triggers a fresh inference every env step.
- CLI flag
  [scripts/simulation/examples/eval_act_inspire.py](../../scripts/simulation/examples/eval_act_inspire.py):
  `--temporal_ensemble_coeff FLOAT` (default `None`). Passed through the
  temp YAML into the wrapper. Mode is logged at run start.
- The weights are identical across paths — ensembling is purely inference-time,
  no retraining needed.

**A/B to run on the post-fix smoketest.** Same deterministic block pose
(`--pin_block_from_hdf5 demo.hdf5 --pin_demo_key demo_28`):

```bash
# A. Chunk-exhaustion (current default)
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
  --model_path /workspaces/.../right_arm_smoketest_20260419-203926/checkpoints/005000/pretrained_model \
  --pin_block_from_hdf5 /workspaces/workflows/rheo/datasets/inspire_right_arm/demo.hdf5 \
  --pin_demo_key demo_28 --enable_cameras --save_video

# B. Temporal ensembling (ACT paper default coefficient)
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
  --model_path ... --pin_block_from_hdf5 ... --pin_demo_key demo_28 \
  --enable_cameras --save_video --temporal_ensemble_coeff 0.01
```

**A/B result — 2026-04-20.** Both runs completed, 300 steps each, same checkpoint
(`right_arm_smoketest_20260419-203926/checkpoints/005000`), same pinned block
pose (demo_28).

| Mode | Chunks | Success | Reward | Stage at step 250 |
|------|-------:|--------:|-------:|------------------:|
| A — chunk-exhaustion (default) | 6 | 0/1 | 0.00 | 0 |
| B — temporal ensemble (coeff=0.01) | 300 | 0/1 | 0.00 | 0 |

The chunk count confirms the mechanical switch landed correctly (A = 300/50 = 6
inferences; B = 1 inference per env step). **Neither run succeeded.** Both
remained at stage 0 throughout — the arm never advanced out of reach-toward-tool
into grasp.

Videos (for visual inspection of trajectory differences):
- A: `eval_videos/20260420_132538_act_pretrained_model_front.mp4`
- B: `eval_videos/20260420_132819_act_pretrained_model_front.mp4`

**Interpretation.** Open-loop amplification is not the primary eval-time bug.
If the 100-step open-loop window were driving the regression, ensembling with
per-step observation refresh and exponential blending should have produced a
qualitatively different trajectory, at least visually; it didn't improve
reward or stage. Either (a) the input distribution to the model at eval is
sufficiently off-manifold that per-step re-query doesn't help (the model
keeps predicting something coherent but wrong), or (b) the memorization
itself has a gap we haven't seen yet in the static-frame offline MAE.

**CLI flag retained** as a low-cost experimental lever. Next pass should
focus on the observation-side candidates that ensembling does not address:
reset-pose drift between teleop recording and eval reset, PD controller
tracking lag during the initial peak-reach, or the dome-lighting / DLAA
render-interval discrepancy between teleop (render_interval=2) and eval
(render_interval=4) that was flagged in the camera-pose check above.
