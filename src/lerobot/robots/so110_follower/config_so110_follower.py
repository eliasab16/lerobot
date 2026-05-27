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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SO110FollowerConfig:
    """Configuration for the SO-110 Follower (8 logical DOF, paired motors on
    elbow_lift / shoulder_swing / shoulder_lift for added torque).

    The bus carries 11 motors. Joints with a `_secondary` companion on the bus
    are mirrored on writes and read from the primary; secondaries are never
    exposed in observation/action features.
    """

    port: str

    disable_torque_on_disconnect: bool = True

    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Must be False for paired motors to work: the bus only honors drive_mode
    # in RANGE_M100_100 / RANGE_0_100 modes, not DEGREES. With use_degrees=True,
    # a mirrored secondary's drive_mode=1 is silently ignored and the pair fights.
    use_degrees: bool = False

    # Logical joint names that physically rotate continuously (skip range sweep,
    # use full 0-4095 raw range during calibration). `_secondary` companions of
    # listed joints inherit the same behavior automatically.
    full_turn_motors: list[str] = field(
        default_factory=lambda: ["wrist_roll", "shoulder_yaw"]
    )

    # Per-motor PID coefficients. Each accepts either a single int (applied to
    # every motor) or a dict {motor_name: value} of overrides. Motors not
    # listed in a dict fall back to the scalar default below. Secondaries
    # inherit their primary's override if the secondary itself isn't listed,
    # so paired motors stay matched and don't fight.
    p_coefficient: int | dict[str, int] = 16
    i_coefficient: int | dict[str, int] = 0
    d_coefficient: int | dict[str, int] = 32

    # STS3215 Acceleration register (41). Caps how fast the servo can change
    # velocity, applied symmetrically to both ramp-up and ramp-down. Lower =
    # smoother starts/stops, less wobble on motion end; higher = snappier but
    # more overshoot. Accepts a scalar (every motor) or a {motor_name: value}
    # dict (overrides; secondaries inherit their primary). STS3215 has no
    # separate deceleration register, so this is the only knob for stop-rate.
    acceleration: int | dict[str, int] = 254

    # Periodic motor temperature sampling. After every `temperature_sample_interval_s`
    # seconds of send_action() calls, the follower reads Present_Temperature
    # (STS3215 reg 63, °C, all motors in one batched sync_read) and logs the
    # result at INFO. Each motor emits a single WARNING the first time it
    # crosses `temperature_warning_c`; the warning re-arms once it drops back
    # below threshold − 3°C (hysteresis). Set the interval to None to disable.
    # STS3215 firmware thermal-protects around 70°C by default, so the warning
    # threshold should leave clear headroom.
    temperature_sample_interval_s: float | None = 5.0
    temperature_warning_c: int = 55


@RobotConfig.register_subclass("so110_follower")
@dataclass
class SO110FollowerRobotConfig(RobotConfig, SO110FollowerConfig):
    pass
