# CloudXR & Isaac Lab Teleoperation Reference

Comprehensive reference for NVIDIA CloudXR streaming and Isaac Lab teleoperation
with the Unitree G1 humanoid robot.

---

## Our Local Setup

**IMPORTANT:** Always use the versions inside the `grasp` conda env. There may be
other Isaac Lab / Isaac Sim installations on the system — ignore them.

| Component | Version | Location |
|-----------|---------|----------|
| Isaac Sim | 5.1.0.0 | `grasp` conda env (pip package `isaacsim`) |
| Isaac Lab | 2.3.2 | `grasp` conda env + source at `~/repos/IsaacLab_v23` |
| GPU | RTX 5090 | Driver 580.119.02, CUDA 12.8 |
| OS | Pop!_OS 22.04 | Kernel 6.17.4 |
| AVP Client | Isaac XR Teleop Sample Client v2.3.0 | Sideloaded via Xcode |

### Two CloudXR Runtime Modes

Isaac Sim 5.1 ships with **CloudXR Runtime built-in** as the extension
`omni.kit.xr.system.openxr`. This means there are two ways to run the XR pipeline:

| Mode | AR Panel Setting | How It Works | When to Use |
|------|-----------------|--------------|-------------|
| **Built-in** | "CloudXR Runtime (5.0)" | Runtime runs inside the Isaac Sim process. No external container, no env vars, no IPC socket. | **Docker workflows** (rheo `run_docker.sh`), simplest setup |
| **External Docker** | "System OpenXR Runtime" | Runtime runs in a separate Docker container. Requires OpenXR env vars + IPC socket mount. | **Local conda** workflows, Isaac Lab `container.py` |

**Built-in runtime files** (shipped with isaacsim pip package):
```
isaacsim/extscache/omni.kit.xr.system.openxr-.../bin/
├── libcloudxr.so              # CloudXR native library
├── libopenxr_cloudxr.so       # OpenXR runtime plugin
└── openxr_cloudxr.json        # OpenXR runtime manifest
```

The extension also includes `cloudxr_wrapper.py` which exposes
`get_cloudxr_runtime_version()` for querying the bundled version.

**Key insight:** The built-in mode is why rheo's `run_docker.sh` does NOT need to
mount OpenXR paths or set `XDG_RUNTIME_DIR` / `XR_RUNTIME_JSON`. The runtime starts
automatically inside the same process when `--xr` is passed.

### Teleop Device Compatibility

| Device | `--teleop_device` | Input Type | Works With |
|--------|-------------------|------------|------------|
| AVP hand tracking | `handtracking` | Hand joint positions via OpenXR | Isaac Lab generic tasks |
| Quest controllers | `motion_controllers` | Controller pose + buttons/triggers | Rheo tasks (trocar, locomanip) |

**Current limitation:** Rheo's trocar and locomanip tasks only register
`motion_controllers` as a teleop device (designed for Quest controllers). To use AVP
hand tracking with these tasks, a `handtracking` device config with appropriate
retargeters would need to be added to the env configs.

### Quick Start: Built-in Mode (Docker / rheo workflows)

```bash
# No external CloudXR container needed — built-in runtime handles everything.
# Make sure the external cloudxr-runtime container is STOPPED to avoid port conflicts.
docker stop cloudxr-runtime 2>/dev/null

# Run rheo trocar task with XR (Quest controllers)
./workflows/rheo/docker/run_docker.sh -g1.5 \
  python scripts/simulation/record_demos_assemble_trocar.py \
  --task Isaac-Assemble-Trocar-G129-Dex3-Teleop \
  --teleop_device motion_controllers \
  --enable_pinocchio \
  --enable_cameras \
  --num_demos 1 \
  --xr

# In Isaac Sim UI: AR Panel → OpenXR → "CloudXR Runtime (5.0)" → Start AR
# On headset: connect to workstation IP → Play
```

### Quick Start: External Docker Mode (local conda)

```bash
# Terminal 1: Start external CloudXR Runtime container
docker start cloudxr-runtime 2>/dev/null || \
docker run -d --name cloudxr-runtime \
    --user $(id -u):$(id -g) --gpus=all \
    -e "ACCEPT_EULA=Y" -e "NV_GPU_INDEX=0" \
    --mount type=bind,src=$HOME/repos/IsaacLab_v23/openxr,dst=/openxr \
    -p 48010:48010/tcp \
    -p 47998:47998/udp -p 47999:47999/udp -p 48000:48000/udp \
    -p 48005:48005/udp -p 48008:48008/udp -p 48012:48012/udp \
    nvcr.io/nvidia/cloudxr-runtime:5.0.1

# Terminal 2: Launch Isaac Lab teleop (AVP hand tracking)
use_conda grasp
cd ~/repos/IsaacLab_v23
export XDG_RUNTIME_DIR=$HOME/repos/IsaacLab_v23/openxr/run
export XR_RUNTIME_JSON=$HOME/repos/IsaacLab_v23/openxr/share/openxr/1/openxr_cloudxr.json

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --teleop_device handtracking \
    --enable_pinocchio

# In Isaac Sim UI: AR Panel → OpenXR → "System OpenXR Runtime" → Start AR
# On AVP: open Isaac XR Teleop Sample Client → enter server IP → Connect → Play
```

### Previous Issues & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| AR panel shows `OpenXR Active Runtime: None` | Env vars not set before Isaac Sim launch (external mode) | Export `XDG_RUNTIME_DIR` and `XR_RUNTIME_JSON` in the same shell BEFORE running `./isaaclab.sh` |
| Tracking works but no video (`Avg Game FPS: 0.00`) | External CloudXR container didn't know which GPU to encode from | Add `-e "NV_GPU_INDEX=0"` to the docker run command |
| IPC socket permission denied (external mode) | Container running as root | Add `--user $(id -u):$(id -g)` to docker run |
| AVP video streams but hand tracking doesn't control robot | Rheo tasks use `motion_controllers` (Quest), not `handtracking` (AVP) | Need to add `handtracking` device config to rheo env configs |
| Port 48010 already in use | External CloudXR container still running while using built-in mode | `docker stop cloudxr-runtime` before using built-in mode |

---

## CloudXR SDK 6.x Overview

CloudXR SDK 6.x has three components:

| Component | Role | Platform |
|-----------|------|----------|
| **CloudXR Runtime** | Server-side — renders, encodes, streams stereoscopic frames | Linux (Ubuntu 22.04+), Windows |
| **CloudXR Framework** | Native Apple client — Swift package for visionOS | Apple Vision Pro (visionOS 2.4+) |
| **CloudXR.js** | Web client — JavaScript SDK for browser-based XR | Meta Quest 3, Pico 4 Ultra (via WebXR) |

The pipeline: Application renders stereo views via OpenXR --> CloudXR Runtime encodes
with NVENC (H.264/H.265) --> streams over WebRTC --> client decodes and displays in headset.
Tracking data flows back from client to server in real-time.

**Docs:** <https://docs.nvidia.com/cloudxr-sdk/latest/index.html>

---

## Server Setup (Linux)

### System Requirements

- **GPU:** NVIDIA Blackwell (recommended) or Ada generation with NVENC encoder
  - Compute-only GPUs (A100, H100) are **not supported** (no NVENC)
- **Driver:** 573.42+
- **OS:** Ubuntu 22.04 or 24.04
- **CPU:** 16+ cores recommended
- **RAM:** 64 GB+

### Runtime Installation

CloudXR Runtime runs as a systemd service or Docker container. It registers as an
OpenXR runtime so applications (Isaac Sim, Isaac Lab) render through it.

```bash
# Set OpenXR runtime to CloudXR
export XR_RUNTIME_JSON=/path/to/openxr_cloudxr.json
export XDG_RUNTIME_DIR=/path/to/openxr/run
```

### Runtime Management API

The Runtime exposes a REST/RPC API for configuration. Key properties:

| Property | Default | Description |
|----------|---------|-------------|
| `endpoint-ip` | `0.0.0.0` | Server listen IP |
| `server-port` | `48010` | Signaling port (TCP) |
| `media-port` | `47998` | Base media port (UDP) |
| `enable-ice` | `false` | Enable ICE/TURN for NAT traversal |
| `client-token` | (none) | Auth token clients must provide |
| `certificate-pem` | (none) | TLS cert for secure connections |
| `key-pem` | (none) | TLS private key |
| `audio-streaming` | `false` | Enable audio (Windows only currently) |
| `disable-alpha` | `false` | Disable alpha channel streaming |
| `runtime-foveation` | `true` | Enable foveated rendering |
| `device-profile` | (auto) | Override client device profile |

**Docs:** <https://docs.nvidia.com/cloudxr-sdk/latest/contents/runtime_management.html>

---

## Apple Vision Pro Client

### CloudXR Framework (Native visionOS)

- **Swift package** available from NVIDIA's GitHub
- Requires **Xcode 16.3+**, **visionOS 2.4+**
- Uses RealityKit for rendering decoded frames in an `ImmersiveSpace`
- Core class: `CloudXRSession` — manages connection, frame decode, and display

Connection modes:

| Mode | Protocol | Use Case |
|------|----------|----------|
| `local` | HTTP + UDP | Same LAN, no encryption |
| `localSecure` | HTTPS + UDP | Same LAN, TLS |
| `remote` | HTTP + UDP + ICE | NAT traversal |
| `remoteSecure` | HTTPS + UDP + ICE | NAT traversal + TLS |

### Isaac XR Teleop Sample Client

NVIDIA provides a pre-built visionOS app specifically for Isaac Lab teleoperation.
It streams hand tracking + head pose back to the server for robot control.

- Available as part of the Isaac Lab CloudXR teleoperation workflow
- Requires visionOS 2.6+

### Foveated Streaming (visionOS 26.4+)

Apple added a first-party **Foveated Streaming API** in visionOS 26.4 that wraps
CloudXR technology directly into the OS. This is the future direction — native OS-level
XR streaming without needing a separate client app.

**Docs:** <https://docs.nvidia.com/cloudxr-sdk/latest/contents/cxr_framework.html>

---

## Network & Ports

### Required Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 48010 | TCP | Signaling (WebSocket) |
| 47998 | UDP | Media stream (base) |
| 47999 | UDP | Media stream |
| 48000 | UDP | Media stream |
| 48005 | UDP | Media stream |
| 48008 | UDP | Media stream |
| 48012 | UDP | Media stream |

### Linux Firewall Rules

```bash
# TCP signaling
sudo ufw allow 48010/tcp

# UDP media range
sudo ufw allow 47998:48012/udp
```

### Bandwidth & Latency

- **Bandwidth:** 200 Mbps recommended, 100 Mbps minimum
- **Latency:** 20-30 ms recommended, 100 ms maximum
- **Default bitrate:** 50 Mbps per eye
- **Max resolution:** 4K per eye
- **Network:** WiFi 6 (802.11ax) recommended

### Known Limitations

- **Single stream session** per OS instance (one headset per server)
- **No video/audio encryption** — use VPN for security on untrusted networks
- **IPv4 only** — no IPv6 support
- **Audio streaming** — Windows only (Linux support planned)
- **Max resolution:** 4K

**Docs:** <https://docs.nvidia.com/cloudxr-sdk/latest/contents/system_requirements.html>

---

## Isaac Lab CloudXR Teleoperation

### Architecture

```
Isaac Lab (renders stereo views via OpenXR)
    |
    v
CloudXR Runtime (Docker container, NVENC encode + WebRTC stream)
    |
    v
XR Headset (decode + display + hand/head tracking)
    |
    v  (tracking data sent back)
CloudXR Runtime --> Isaac Lab (OpenXR device input --> retargeters --> robot commands)
```

### Docker Setup (Recommended)

```bash
# Start CloudXR Runtime alongside Isaac Lab
./docker/container.py start \
    --files docker-compose.cloudxr-runtime.patch.yaml \
    --env-file .env.cloudxr-runtime

# If running Isaac Lab locally (outside Docker), set OpenXR env vars:
export XDG_RUNTIME_DIR=$(pwd)/openxr/run
export XR_RUNTIME_JSON=$(pwd)/openxr/share/openxr/1/openxr_cloudxr.json
```

Docker Compose exposes ports 48010 (TCP) and 47998-48012 (UDP), mounts the OpenXR
IPC socket, and allocates the GPU to the runtime container.

Relevant files in this repo's IsaacLab third_party:
- `third_party/IsaacLab/docker/.env.cloudxr-runtime`
- `third_party/IsaacLab/docker/docker-compose.cloudxr-runtime.patch.yaml`

### Running Teleoperation

```bash
# Generic teleoperation with hand tracking (any humanoid)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --teleop_device handtracking \
    --enable_pinocchio

# Then in Isaac Sim UI: AR Panel --> "OpenXR" output plugin --> "Start AR"
```

For the rheo workflow's G1 tasks, use the workflow-specific teleop scripts:

```bash
# Loco-manipulation data collection with XR
python workflows/rheo/scripts/simulation/record_demos_locomanip.py \
    --teleop_device motion_controllers --xr

# Trocar assembly data collection (XR only, default device: motion_controllers)
python workflows/rheo/scripts/simulation/record_demos_assemble_trocar.py --xr
```

**Docs:**
- <https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html>
- <https://isaac-sim.github.io/IsaacLab/main/source/deployment/cloudxr_teleoperation_cluster.html>

---

## G1 Teleoperation Specifics

### Supported Teleop Devices

| Device | Class | Output |
|--------|-------|--------|
| Keyboard | `Se2Keyboard`, `Se3Keyboard` | Discrete velocity/pose |
| SpaceMouse | `Se2SpaceMouse`, `Se3SpaceMouse` | Continuous 6-DOF |
| Gamepad | `Se2Gamepad`, `Se3Gamepad` | Analog velocity/pose |
| OpenXR (CloudXR) | `OpenXRDevice` | Hand joints + head pose |
| Manus + Vive | `ManusVive` | Full finger + body tracking |
| Quest Controllers | `MotionControllersTeleopDevice` | Button/joystick + pose |

### G1 Retargeters (in this repo)

Located in `workflows/rheo/scripts/teleop_devices/motion_controllers.py`:

- **`G1TriHandUpperBodyMotionControllerGripperRetargeter`** — Maps Quest controller
  pose + triggers to G1 TriHand wrist position and finger joints
- **`G1LowerBodyStandingMotionControllerRetargeter`** — Maps joystick inputs to
  G1 locomotion (x/y velocity, yaw rate, hip height)

Additional retargeters in Isaac Lab:
- **`UnitreeG1Retargeter`** — Hand tracking to G1 hand commands
- **`Se3RelRetargeter`** / **`Se3AbsRetargeter`** — Generic hand-to-EE mapping
- **`GripperRetargeter`** — Thumb-index distance with hysteresis

### G1 Isaac Lab Environments

| Environment | Task |
|-------------|------|
| `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` | Pick & place with locomotion |
| `Isaac-Locomanipulation-G1-Abs-Mimic-v0` | Mimic-compatible variant |
| `Isaac-G1-SteeringWheel-Locomanipulation` | Navigation SDG |
| `Isaac-PickPlace-G1-InspireFTP-Abs-v0` | With Inspire FTP hand |

**Known issue:** G1 environments may be blacklisted by default in Isaac Lab's
`__init__.py`. Remove from blacklist to use. See:
<https://github.com/isaac-sim/IsaacLab/issues/3939>

### G1 USD Assets (Isaac Sim 5.1)

| Asset | Path |
|-------|------|
| G1 23-DOF | `Unitree/G1_23dof/g1.usd` |
| G1 23-DOF Minimal | `Unitree/G1_23dof/g1_minimal.usd` |
| G1 Full (45 joints) | `Unitree/G1/g1.usd` |

**Note:** Isaac Lab USD files have different waist configurations compared to Unitree's
official USD releases.

---

## Imitation Learning Pipeline

### End-to-End Workflow

```
1. Collect demos (teleop)  -->  2. Annotate subtasks  -->  3. Generate synthetic demos
                                                                      |
                                                                      v
                           5. Deploy / evaluate  <--  4. Train policy (BC)
```

### Step 1: Collect Demonstrations

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
    --device cpu --teleop_device handtracking \
    --dataset_file ./datasets/dataset_g1.hdf5 \
    --num_demos 10 --enable_pinocchio
```

### Step 2: Annotate Subtasks (for Mimic)

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --device cpu \
    --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
    --input_file ./datasets/dataset_g1.hdf5 \
    --output_file ./datasets/annotated_g1.hdf5 \
    --enable_pinocchio
```

### Step 3: Generate Synthetic Demonstrations

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --device cpu --headless --num_envs 20 \
    --generation_num_trials 1000 \
    --input_file ./datasets/annotated_g1.hdf5 \
    --output_file ./datasets/generated_g1.hdf5 \
    --enable_pinocchio
```

### Step 4: Train Policy

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
    --algo bc --normalize_training_actions \
    --dataset ./datasets/generated_g1.hdf5
```

### Data Formats

- **Isaac Lab / Robomimic:** HDF5 with episodes containing actions (T, action_dim),
  observations dict, end-effector poses (T, 4, 4), object poses, subtask flags
- **GR00T N1:** LeRobot / Hugging Face format (video, state, action triplets)

### Performance Benchmarks

- GR1T2 pick-and-place: ~75-86% success with BC from 1000 generated demos
- Training time: ~29 min on RTX ADA 6000
- Generation: 65-80% success rate, 18-40 min on GPU
- GR00T Blueprint: 780,000 synthetic trajectories in 11 hours

---

## Alternative: Meta Quest 3 via ALVR (No CloudXR EA Required)

For Meta Quest 3 without CloudXR Early Access enrollment:

1. Install **ALVR v20.14.1** on the Quest
2. Use **SteamVR** as the OpenXR runtime on the server
3. The `G1TriHandUpperBodyMotionControllerRetargeter` handles Quest controller-to-G1 mapping
4. Verified working by community for G1 upper-body teleoperation

Full-body tracking is also possible via ALVR's OSC-based body tracking (UDP port 9000).

**Sources:**
- <https://forums.developer.nvidia.com/t/meta-quest-for-humanoid-g1-teleoperation-in-isaac-sim/360846>
- <https://simintel.co/2025/12/26/meta-quest-3-teleop-body-tracking.html>

---

## GR00T Integration

### GR00T N1 Foundation Model

- Dual-system: Vision-Language reasoning + Diffusion Transformer action generation
- 2B parameters, cross-embodiment (GR-1, 1X Neo, Unitree G1)

### GR00T Blueprints Pipeline

```
Teleop (AVP / SpaceMouse / Quest)
    --> Raw Demos (20-40 needed)
    --> GR00T-Mimic (annotation + synthetic generation)
    --> NVIDIA Cosmos Transfer (photorealism augmentation)
    --> Training Dataset
    --> GR00T N1 Post-Training
```

Synthetic + real data combined yields ~40% performance boost over real data alone.

### GR00T-Teleop

Uses Apple Vision Pro via CloudXR for demonstration collection. This is the
NVIDIA-blessed path used in official GR00T demos.

**Sources:**
- <https://developer.nvidia.com/isaac/gr00t>
- <https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/>
- <https://developer.nvidia.com/blog/building-a-synthetic-motion-generation-pipeline-for-humanoid-robot-learning/>

---

## Reference Links

### CloudXR SDK
- Main docs: <https://docs.nvidia.com/cloudxr-sdk/latest/index.html>
- Runtime Management API: <https://docs.nvidia.com/cloudxr-sdk/latest/contents/runtime_management.html>
- CloudXR Framework (Apple): <https://docs.nvidia.com/cloudxr-sdk/latest/contents/cxr_framework.html>
- System requirements: <https://docs.nvidia.com/cloudxr-sdk/latest/contents/system_requirements.html>

### Isaac Lab Teleoperation
- CloudXR teleop setup: <https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html>
- Teleop + imitation learning overview: <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>
- Kubernetes deployment: <https://isaac-sim.github.io/IsaacLab/main/source/deployment/cloudxr_teleoperation_cluster.html>
- Docker deployment: <https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html>
- Isaac Lab devices API: <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html>
- Isaac Lab Arena teleop: <https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/static_manipulation/step_2_teleoperation.html>
- Isaac Lab 2.3 blog: <https://developer.nvidia.com/blog/streamline-robot-learning-with-whole-body-control-and-enhanced-teleoperation-in-nvidia-isaac-lab-2-3/>

### Unitree G1
- unitree_sim_isaaclab: <https://github.com/unitreerobotics/unitree_sim_isaaclab>
- xr_teleoperate: <https://github.com/unitreerobotics/xr_teleoperate>
- Isaac Sim robot assets: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html>
- G1 USD discussion: <https://github.com/isaac-sim/IsaacLab/discussions/4064>
- G1 env blacklist issue: <https://github.com/isaac-sim/IsaacLab/issues/3939>
- Quest retargeters PR: <https://github.com/isaac-sim/IsaacLab/pull/3950>

### GR00T
- Developer page: <https://developer.nvidia.com/isaac/gr00t>
- GR00T N1 blog: <https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/>
- Synthetic motion pipeline: <https://developer.nvidia.com/blog/building-a-synthetic-motion-generation-pipeline-for-humanoid-robot-learning/>

### Community / Forums
- Meta Quest G1 teleop thread: <https://forums.developer.nvidia.com/t/meta-quest-for-humanoid-g1-teleoperation-in-isaac-sim/360846>
- Quest 3 body tracking blog: <https://simintel.co/2025/12/26/meta-quest-3-teleop-body-tracking.html>
- NVIDIA + Apple Vision Pro article: <https://9to5mac.com/2024/08/05/nvidia-uses-apple-vision-pro-to-capture-teleoperated-demonstrations-and-control-humanoid-robots/>

### Local Repo Files
- CloudXR Docker config: `third_party/IsaacLab/docker/.env.cloudxr-runtime`
- CloudXR Docker Compose: `third_party/IsaacLab/docker/docker-compose.cloudxr-runtime.patch.yaml`
- CloudXR teleop docs: `third_party/IsaacLab/docs/source/how-to/cloudxr_teleoperation.rst`
- Rheo motion controllers: `workflows/rheo/scripts/teleop_devices/motion_controllers.py`
- Rheo locomanip recording: `workflows/rheo/scripts/simulation/record_demos_locomanip.py`
- Rheo trocar recording: `workflows/rheo/scripts/simulation/record_demos_assemble_trocar.py`
- WebRTC streaming: `workflows/rheo/scripts/simulation/examples/webrtc_runner_cli.py`
