# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA Isaac for Healthcare Workflows — healthcare robotics workflows built on the NVIDIA Isaac platform. Contains five independent workflows under `workflows/`: **so_arm_starter**, **robotic_ultrasound**, **robotic_surgery**, **telesurgery**, and **rheo**. Each workflow is self-contained with its own scripts, tests, and CMake config.

## Build & Run

```bash
# CMake build (C++/Holoscan components)
cmake -B build && cmake --build build

# Run a workflow via the i4h CLI wrapper (downloads HoloHub utilities on first run)
./i4h run <workflow> <mode> --local    # local execution
./i4h run <workflow> <mode>            # Docker execution
./i4h list                             # list available workflows
./i4h modes <workflow>                 # list modes for a workflow

# Install workflow-specific Python dependencies
python tools/install_deps.py --workflow <workflow_name>
```

## Linting

Pre-commit runs all linters. Settings live in `pyproject.toml` and `.pre-commit-config.yaml`.

```bash
pre-commit run --all-files
```

Tools: **ruff** (format + lint, 120-char lines, Python 3.11), **isort** (import sorting), **codespell**, **markdownlint**. The `workflows/telesurgery/cmake/pybind11/` directory is excluded from ruff.

## Testing

Tests use Python `unittest` with `coverage`. Only three workflows have test runner support: `robotic_ultrasound`, `robotic_surgery`, `so_arm_starter`.

```bash
# Unit tests for a workflow (20-min timeout per test)
python tools/run_all_tests.py --workflow <workflow_name>

# Integration tests
python tools/run_all_tests.py --workflow <workflow_name> --integration

# Single test file directly
python -m unittest workflows/<workflow>/tests/test_foo.py

# RTI DDS license required for most workflows (not robotic_surgery)
export RTI_LICENSE_FILE=<path>
```

Test files live under `workflows/<workflow>/tests/` and follow the `test_*.py` naming pattern. Integration tests use the `test_integration_*.py` pattern.

## License Headers (CI-Enforced)

All source files (`.py`, `.sh`, `.ipynb`, `.slurm`, `.h`, `.hpp`, `.cu`, `.cpp`, `Dockerfile*`) must include SPDX headers:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

The CI copyright check (`bash .github/check_missing_license.sh`) will fail PRs with missing headers.

## Commits

All commits require DCO sign-off: `git commit -s -m "message"`.

## Architecture

- **`workflows/<name>/`** — each workflow is independent with `scripts/` (simulation, training, policy, DDS, holoscan_apps), `tests/`, and its own `CMakeLists.txt` and `metadata.json`
- **`third_party/`** — git submodules: Isaac-GR00T, IsaacLab, IsaacLab-Arena, RLinf
- **`tools/`** — shared utilities: `install_deps.py`, `run_all_tests.py`, CMake modules
- **`./i4h`** — CLI entry point that wraps HoloHub for workflow execution (supports Docker and local modes)
- Large assets (USD files, 3D models, TRT engines) are not in the repo; download via `i4h-asset-retrieve`
