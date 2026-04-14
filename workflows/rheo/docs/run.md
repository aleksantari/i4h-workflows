# Running ACT Inspire FTP Evaluations

Quick-reference commands for evaluating the trained checkpoints. All commands
run from `workflows/rheo/` on the host and launch inside the grasp Docker
container. Logs go to `eval_logs/`, result summaries to `eval_results/`, and
videos to `eval_videos/`.

## Overfit sanity-check model

Trained on a single episode (`episode_0`, slot 1, tool_0) with `kl_weight=0`,
`lr=1e-4`, 8000 steps. Expected behavior: near-perfect tracking on a starting
pose that matches the demo, failure on randomized poses.

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/train_20260410-044535/checkpoints/last/pretrained_model \
    --slot 1 --object tool_0 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/overfit_1ep_$(date +%Y%m%d_%H%M%S).log
```

## Full model (30 demos, 50K steps, kl_weight=1)

Run: `train_20260412-222923`. Sweep checkpoints 20K, 30K, 40K to find the
best generalization point — IL policies typically peak in the middle of
training and degrade slightly near the end.

### Checkpoint 20K

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/train_20260412-222923/checkpoints/020000/pretrained_model \
    --slot 1 --object tool_0 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/ckpt20k_$(date +%Y%m%d_%H%M%S).log
```

### Checkpoint 30K

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/train_20260412-222923/checkpoints/030000/pretrained_model \
    --slot 1 --object tool_0 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/ckpt30k_$(date +%Y%m%d_%H%M%S).log
```

### Checkpoint 40K

```bash
mkdir -p eval_logs
./docker/run_docker_grasp.sh python scripts/simulation/examples/eval_act_inspire.py \
    --model_path /workspaces/workflows/rheo/scripts/simulation/rl/results/act_grasp_policy_inspire/train_20260412-222923/checkpoints/040000/pretrained_model \
    --slot 1 --object tool_0 \
    --num_episodes 1 --max_steps 500 \
    --save_video --enable_cameras \
    2>&1 | tee eval_logs/ckpt40k_$(date +%Y%m%d_%H%M%S).log
```

## What to look for

- **Success rate** (printed at end of run and saved to `eval_results/`) — the primary metric.
- **Video spot-check** — open one clip from `eval_videos/` per checkpoint; watch the approach angle, grasp close timing, and whether the hand stays locked on the tool during transport.
- **Expected trend**: 20K should be the cleanest for motion quality; 30K often matches or slightly beats it on success rate; 40K may plateau or regress if the policy is starting to overfit.

If 20K ≈ 40K, the full 50K run was longer than necessary — the next retrain can stop earlier. If 40K is clearly worse, that's direct evidence of overfitting and the config should cap steps lower.
