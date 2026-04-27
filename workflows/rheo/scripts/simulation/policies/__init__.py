# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Closed-loop policy wrappers for IsaacLab evaluation.

Each module wraps a trained model into a `PolicyBase` interface used by the
rheo evaluation pipeline:

- `base.py` — abstract `BaseClosedloopPolicy` (action-chunking scaffolding for
  future wrappers; not currently inherited).
- `act.py` — `ACTClosedloopPolicy` (LeRobot ACT, Inspire FTP grasp policy).
- `vla.py` — placeholder for a forthcoming VLA wrapper.
"""
