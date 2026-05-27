#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class SO110LeaderConfig:
    """Configuration for the SO-110 Leader (8 single motors, no pairing).

    Joint names match the SO110Follower so action keys flow through teleop and
    datasets without remapping.
    """

    port: str

    # Kept False to match the follower (the bus only honors drive_mode in
    # RANGE_M100_100 / RANGE_0_100 modes — not DEGREES). Action keys flow
    # leader→follower in the same normalized space, no remapping.
    use_degrees: bool = False

    # Joints that physically rotate continuously (no end-stops).
    full_turn_motors: list[str] = field(
        default_factory=lambda: ["wrist_roll", "shoulder_yaw"]
    )


@TeleoperatorConfig.register_subclass("so110_leader")
@dataclass
class SO110LeaderTeleopConfig(TeleoperatorConfig, SO110LeaderConfig):
    pass
