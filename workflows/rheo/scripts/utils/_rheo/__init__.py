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

"""Borrowed-rheo utility helpers.

Files in this package are only used by borrowed-rheo entry points
(`policy_runner.py`, `triggered_policy_runner.py`, `observe_runner.py`,
`_rheo/*` scripts) and the GR00T trocar pipeline. They are kept available for
those tracks but are not part of the active G1 + Inspire FTP grasp policy
pipeline. Treat with care; refactor only when broken.
"""
