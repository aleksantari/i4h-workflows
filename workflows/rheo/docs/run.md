# Running ACT Inspire FTP Evaluations

Quick-reference commands for evaluating the trained checkpoints. All commands
run from `workflows/rheo/` on the host and launch inside the grasp Docker
container. Logs go to `eval_logs/`, result summaries to `eval_results/`, and
videos to `eval_videos/`.

## Standard eval flags

As of 2026-04-20 the default eval setup is:

- `--temporal_ensemble_coeff 0.01` — switch inference from chunk-exhaustion
  (1 forward pass per 50–100 env steps) to per-step temporal ensembling
  (1 forward pass per env step, exponentially weighted overlap blending).
  See `docs/debug_policy.md` Session 2 for the rationale.
- `--pin_block_from_hdf5 … --pin_demo_key demo_28` — deterministic block
  pose matching the ep28 teleop demo, so rollout is reproducible and directly
  comparable to the teleop ground-truth.
- `--pin_block_frame_idx 10` — pin from the per-timestep `states/...` trajectory
  at the given frame instead of the pre-settle `initial_state` pose. Default
  `0` keeps legacy behavior. Find a good index by running
  `scripts/utils/inspect_block_settle.py --hdf5 ... --demo_key demo_28`.
- `--log_actions` — dump per-step applied 41-D action to
  `eval_results/actions_<timestamp>_ep<NN>.npy` for offline rollout-vs-GT
  plotting via `scripts/utils/plot_rollout_vs_gt.py`.

## Right-arm smoketest (5K steps, kl_weight=0, 1 demo)

Run: `right_arm_smoketest_20260419-203926`. Post-elbow-offset-fix smoketest.
Offline-replay MAE on ep28 = 0.014 rad; closed-loop MAE ≈ 0.16 rad. Useful as
the fastest signal while iterating on the wrapper / reset / inference path.

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/right_arm_smoketest_20260419-203926/checkpoints/005000/pretrained_model \
    --pin_block_from_hdf5 /workspaces/workflows/rheo/datasets/inspire_right_arm/demo.hdf5 \
    --pin_demo_key demo_28 \
    --pin_block_frame_idx 10 \
    --temporal_ensemble_coeff 0.01 \
    --log_actions \
    --num_episodes 1 --max_steps 300 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/right_arm_smoketest_5k_$(date +%Y%m%d_%H%M%S).log
```

Compare the logged actions against ep28 ground truth:

```bash
bash -ic 'use_conda grasp && python scripts/utils/plot_rollout_vs_gt.py \
    --rollout eval_results/actions_<TIMESTAMP>_ep00.npy \
    --gt_parquet datasets/inspire_right_arm/demo_ep28/lerobot/data/chunk-000/episode_000000.parquet \
    --out eval_results/rollout_vs_ep28_<TIMESTAMP>.png'
```

## Right-arm full model (15K steps, kl_weight=1, 30 demos)

Run: `right_arm_20260417-035155`. Sweep checkpoints 5K, 10K, 15K to find the
best generalization point — IL policies typically peak in the middle of
training and degrade slightly near the end.

### Checkpoint 5K

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/right_arm_20260417-035155/checkpoints/005000/pretrained_model \
    --pin_block_from_hdf5 /workspaces/workflows/rheo/datasets/inspire_right_arm/demo.hdf5 \
    --pin_demo_key demo_28 \
    --pin_block_frame_idx 10 \
    --temporal_ensemble_coeff 0.01 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/right_arm_ckpt5k_$(date +%Y%m%d_%H%M%S).log
```

### Checkpoint 10K

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/right_arm_20260417-035155/checkpoints/010000/pretrained_model \
    --pin_block_from_hdf5 /workspaces/workflows/rheo/datasets/inspire_right_arm/demo.hdf5 \
    --pin_demo_key demo_28 \
    --pin_block_frame_idx 10 \
    --temporal_ensemble_coeff 0.01 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/right_arm_ckpt10k_$(date +%Y%m%d_%H%M%S).log
```

### Checkpoint 15K

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/right_arm_20260417-035155/checkpoints/015000/pretrained_model \
    --pin_block_from_hdf5 /workspaces/workflows/rheo/datasets/inspire_right_arm/demo.hdf5 \
    --pin_demo_key demo_28 \
    --pin_block_frame_idx 10 \
    --temporal_ensemble_coeff 0.01 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/right_arm_ckpt15k_$(date +%Y%m%d_%H%M%S).log
```

## Legacy: dual-arm overfit sanity-check (pre-elbow-fix)

Retained for reference only; the elbow-offset fix broke compatibility with
these checkpoints.

```bash
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/train_20260410-044535/checkpoints/last/pretrained_model \
    --slot 1 --object tool_0 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/overfit_1ep_$(date +%Y%m%d_%H%M%S).log
```

## What to look for

- **Success rate** (printed at end of run and saved to `eval_results/`) — the primary metric.
- **Video spot-check** — open one clip from `eval_videos/` per checkpoint; watch the approach angle, grasp close timing, and whether the hand stays locked on the tool during transport.
- **Rollout-vs-GT plot** (smoketest) — overall MAE < 0.05 rad and per-step MAE flat near t=0 means the policy is on-manifold at reset. A jump at t=0 signals a reset / initial-obs mismatch, not a drift problem.
- **Expected trend across checkpoints**: 5K should be the cleanest for motion quality; 10K often matches or slightly beats it on success rate; 15K may plateau or regress if the policy is starting to overfit.

If 5K ≈ 15K, the full 15K run was longer than necessary — the next retrain can stop earlier. If 15K is clearly worse, that's direct evidence of overfitting and the config should cap steps lower.
