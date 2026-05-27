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

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so110_follower import SO110FollowerRobotConfig

logger = logging.getLogger(__name__)


SECONDARY_SUFFIX = "_secondary"

_BUS_RETRY_MAX_ATTEMPTS = 3


def _retry_bus(label, fn, *args, **kwargs):
    """Retry a bus operation on transient ConnectionError (e.g. malformed packet
    on a Feetech daisy chain). Bus-side sync_read has num_retry=0 by default,
    so a single corrupt response aborts long calibration loops; this is the
    caller-side safety net.
    """
    last_exc = None
    for attempt in range(1, _BUS_RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except ConnectionError as e:
            last_exc = e
            if attempt == _BUS_RETRY_MAX_ATTEMPTS:
                raise
            logger.warning(
                f"Bus glitch during {label} (attempt {attempt}/{_BUS_RETRY_MAX_ATTEMPTS}): {e}. Retrying."
            )
            print(f"\nBus communication error during {label}. Retrying — keep moving the joint.")
    raise last_exc  # unreachable; for type-checker


class SO110Follower(Robot):
    """SO-110 follower (8 logical DOF). Paired motors on elbow_lift,
    shoulder_swing, shoulder_lift add torque while remaining invisible to the
    outside world: every external surface (features, observation, action,
    dataset columns) speaks only the 8 primary joint names. Writes to a primary
    are mirrored to its `_secondary` on the bus deterministically.
    """

    config_class = SO110FollowerRobotConfig
    name = "so110_follower"

    def __init__(self, config: SO110FollowerRobotConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        motors = {
            "gripper": Motor(1, "sts3215", MotorNormMode.RANGE_0_100),
            "wrist_tilt": Motor(2, "sts3215", norm_mode_body),
            "wrist_yaw": Motor(3, "sts3215", norm_mode_body),
            "wrist_roll": Motor(4, "sts3215", norm_mode_body),
            "elbow_lift": Motor(5, "sts3215", norm_mode_body),
            "elbow_lift_secondary": Motor(6, "sts3215", norm_mode_body),
            "shoulder_swing": Motor(7, "sts3215", norm_mode_body),
            "shoulder_swing_secondary": Motor(8, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(9, "sts3215", norm_mode_body),
            "shoulder_lift_secondary": Motor(10, "sts3215", norm_mode_body),
            "shoulder_yaw": Motor(11, "sts3215", norm_mode_body),
        }

        self._primary_motors = [m for m in motors if not m.endswith(SECONDARY_SUFFIX)]
        self._secondary_of = {
            m: f"{m}{SECONDARY_SUFFIX}"
            for m in self._primary_motors
            if f"{m}{SECONDARY_SUFFIX}" in motors
        }

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        self._last_temp_sample_t: float | None = None
        self._temp_over_warn: set[str] = set()  # motors currently above warning threshold
        self._last_temps: dict[str, int] = {}  # last accepted reading per motor (debounce)
        if config.temperature_sample_interval_s is not None:
            if config.temperature_sample_interval_s <= 0:
                raise ValueError(
                    f"temperature_sample_interval_s must be > 0 or None, got "
                    f"{config.temperature_sample_interval_s}"
                )
            logger.info(
                f"{self.name}: temperature sampling every "
                f"{config.temperature_sample_interval_s}s, warn ≥ {config.temperature_warning_c}°C"
            )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self._primary_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        out = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            h, w = cfg.height, cfg.width
            if getattr(cfg, "crop_to_square", False):
                h = w = min(h, w)
            out[cam] = (h, w, 3)
        return out

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def _calibrate_joints(self, primaries: list[str]) -> dict[str, MotorCalibration]:
        """Run center + sweep on each primary (and its paired `_secondary` if any).

        Returns the new `MotorCalibration` entries for every motor touched.
        Secondaries are mounted mirrored to their primary, so the encoder
        direction is inverted; `drive_mode=1` makes the bus invert reads/writes
        so a single normalized goal moves the pair coherently.
        """
        full_turn = set(self.config.full_turn_motors or [])
        new_calibration: dict[str, MotorCalibration] = {}

        self.bus.disable_torque()
        for primary in primaries:
            pair = [primary]
            if primary in self._secondary_of:
                pair.append(self._secondary_of[primary])

            for m in pair:
                self.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)

            pair_str = " + ".join(f"'{m}'" for m in pair) if len(pair) > 1 else f"'{primary}'"
            input(f"\nPosition joint '{primary}' ({pair_str}) at the middle of its range and press ENTER...")
            offsets = _retry_bus("set_half_turn_homings", self.bus.set_half_turn_homings, pair)

            if primary in full_turn:
                mins = {m: 0 for m in pair}
                maxes = {m: 4095 for m in pair}
                print(f"'{primary}' configured as full-turn (using full 0-4095 range).")
            else:
                print(f"Sweep joint '{primary}' through its full range. Press ENTER when done.")
                mins, maxes = self._record_ranges_safe(pair)

            for m in pair:
                drive_mode = 1 if m.endswith(SECONDARY_SUFFIX) else 0
                new_calibration[m] = MotorCalibration(
                    id=self.bus.motors[m].id,
                    drive_mode=drive_mode,
                    homing_offset=int(offsets[m]),
                    range_min=int(mins[m]),
                    range_max=int(maxes[m]),
                )

        return new_calibration

    def _record_ranges_safe(
        self, motors: list[str], retries: int = 3, sleep_ms: int = 5
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Drop-in replacement for FeetechMotorsBus.record_ranges_of_motion that
        uses sync_read(num_retry=N) and a small inter-iteration sleep. The bus
        method runs as fast as possible with no retries — fine for multi-motor
        batched reads, brittle for tight single-motor loops on this hardware.
        """
        start = self.bus.sync_read("Present_Position", motors, normalize=False, num_retry=retries)
        mins = dict(start)
        maxes = dict(start)
        pos = start

        while True:
            try:
                pos = self.bus.sync_read("Present_Position", motors, normalize=False, num_retry=retries)
            except ConnectionError as e:
                logger.warning(f"sync_read failed after {retries} retries: {e}. Continuing.")
                time.sleep(sleep_ms / 1000.0)
                continue

            for m in motors:
                if pos[m] < mins[m]:
                    mins[m] = pos[m]
                if pos[m] > maxes[m]:
                    maxes[m] = pos[m]

            print("\n-------------------------------------------")
            print(f"{'NAME':<25} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
            for m in motors:
                print(f"{m:<25} | {mins[m]:>6} | {pos[m]:>6} | {maxes[m]:>6}")

            if enter_pressed():
                break

            move_cursor_up(len(motors) + 3)
            time.sleep(sleep_ms / 1000.0)

        same = [m for m in motors if mins[m] == maxes[m]]
        if same:
            raise ValueError(f"No motion detected on motors: {same}")

        return mins, maxes

    def _validate_partial_motors(self, motors: list[str]) -> None:
        for m in motors:
            if m.endswith(SECONDARY_SUFFIX):
                raise ValueError(
                    f"'{m}' is a secondary motor; partial calibration takes logical "
                    f"joint names. Use '{m.removesuffix(SECONDARY_SUFFIX)}' to "
                    "recalibrate the pair together."
                )
            if m not in self._primary_motors:
                raise ValueError(
                    f"Joint '{m}' not found. Valid joints: {self._primary_motors}"
                )

    def calibrate(self, motors: list[str] | None = None) -> None:
        if motors is not None:
            self._validate_partial_motors(motors)
            if not self.calibration:
                raise ValueError(
                    "No existing calibration found. Run full calibration first before "
                    "calibrating specific joints."
                )
            logger.info(f"\nRunning partial calibration of {self} for joints: {set(motors)}")
            try:
                updates = self._calibrate_joints(motors)
                merged = dict(self.calibration)
                merged.update(updates)
                self.calibration = merged
                self.bus.write_calibration(self.calibration)
                self._save_calibration()
                print(f"Partial calibration of joints {set(motors)} saved to {self.calibration_fpath}")
                return
            except Exception:
                logger.exception(
                    "Error during partial calibration. Restoring previous calibration."
                )
                # Restore hardware-level calibration registers from the unchanged
                # in-memory dict; in-memory and on-disk state are untouched.
                self.bus.write_calibration(self.calibration)
                raise

        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.calibration = self._calibrate_joints(self._primary_motors)
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def _resolve_per_motor(self, field: "int | dict[str, int]", motor: str, fallback: int) -> int:
        """Resolve a scalar-or-dict per-motor setting to a value for one motor.

        Lookup order for dicts: explicit motor name, then primary name (for
        secondaries that aren't listed explicitly), then the fallback default.
        Scalars apply to every motor.
        """
        if not isinstance(field, dict):
            return int(field)
        if motor in field:
            return int(field[motor])
        if motor.endswith(SECONDARY_SUFFIX):
            primary = motor.removesuffix(SECONDARY_SUFFIX)
            if primary in field:
                return int(field[primary])
        return fallback

    def configure(self) -> None:
        with self.bus.torque_disabled():
            # configure_motors handles Return_Delay_Time and the STS3215 Phase
            # register fix; defaults (254/254) leave Maximum_Acceleration uncapped
            # so the per-motor Acceleration writes below take full effect.
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write(
                    "P_Coefficient", motor,
                    self._resolve_per_motor(self.config.p_coefficient, motor, 16),
                )
                self.bus.write(
                    "I_Coefficient", motor,
                    self._resolve_per_motor(self.config.i_coefficient, motor, 0),
                )
                self.bus.write(
                    "D_Coefficient", motor,
                    self._resolve_per_motor(self.config.d_coefficient, motor, 32),
                )
                self.bus.write(
                    "Acceleration", motor,
                    self._resolve_per_motor(self.config.acceleration, motor, 254),
                )

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

            p_rb = self.bus.sync_read("P_Coefficient")
            i_rb = self.bus.sync_read("I_Coefficient")
            d_rb = self.bus.sync_read("D_Coefficient")
            a_rb = self.bus.sync_read("Acceleration")
            logger.info(f"{self.name} PID readback — P: {p_rb}")
            logger.info(f"{self.name} PID readback — I: {i_rb}")
            logger.info(f"{self.name} PID readback — D: {d_rb}")
            logger.info(f"{self.name} Acceleration readback: {a_rb}")

    def setup_motors(self, motors: list[str] | None = None) -> None:
        if motors is None:
            targets = list(self.bus.motors)
        else:
            for m in motors:
                if m not in self.bus.motors:
                    raise ValueError(
                        f"Motor '{m}' not found. Available motors: {list(self.bus.motors)}"
                    )
            targets = motors

        for motor in reversed(targets):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        all_pos = self.bus.sync_read("Present_Position", num_retry=3)
        obs_dict = {f"{m}.pos": v for m, v in all_pos.items() if not m.endswith(SECONDARY_SUFFIX)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def _maybe_sample_temperatures(self) -> None:
        """Read Present_Temperature on all motors every
        `temperature_sample_interval_s`, log results, warn on threshold cross.

        One batched sync_read per interval (single packet). Failures are
        logged at debug and do not propagate — temperature is observational.
        """
        interval = self.config.temperature_sample_interval_s
        if interval is None:
            return
        now = time.perf_counter()
        if self._last_temp_sample_t is not None and (now - self._last_temp_sample_t) < interval:
            return
        self._last_temp_sample_t = now

        try:
            temps = self.bus.sync_read("Present_Temperature", num_retry=3)
        except ConnectionError as e:
            logger.debug(f"{self.name}: temperature sync_read failed: {e}")
            return

        # Debounce single-sample bus glitches. STS3215 thermal time constant is
        # minutes-scale; a multi-second sample can't legitimately shift by more
        # than a few °C. Anything >10°C from the previous accepted reading for
        # the same motor is a corrupted byte (one bit flip can swap a sane temp
        # for nonsense like 73 or 200). Discard, log, keep the previous value.
        SPIKE_THRESHOLD_C = 10
        accepted: dict[str, int] = {}
        discarded: dict[str, tuple[int, int]] = {}
        for motor, t in temps.items():
            t_int = int(t)
            prev = self._last_temps.get(motor)
            if prev is not None and abs(t_int - prev) > SPIKE_THRESHOLD_C:
                discarded[motor] = (prev, t_int)
            else:
                accepted[motor] = t_int
        self._last_temps.update(accepted)

        # Multi-line block per sample, trailing separator so consecutive readings
        # are visually grouped (the line terminates each group). Order follows
        # bus motor definition (physical layout) so paired motors sit adjacent.
        sep = "-" * 50
        lines = ["", f"{self.name} temps °C"]
        for m in self.bus.motors:
            if m in accepted:
                lines.append(f"  {m:<26} = {accepted[m]:>3}°C")
        lines.append(sep)

        logger.info("\n".join(lines))
        if discarded:
            logger.warning(
                f"{self.name} discarded implausible temperature readings "
                f"(|Δ| > {SPIKE_THRESHOLD_C}°C in {interval}s, likely bus glitch): "
                + ", ".join(f"{m}: {p}→{n}" for m, (p, n) in sorted(discarded.items()))
            )

        warn_c = self.config.temperature_warning_c
        rearm_c = warn_c - 3  # hysteresis: re-arm warning once 3°C below threshold
        for motor, t in accepted.items():
            if t >= warn_c and motor not in self._temp_over_warn:
                logger.warning(
                    f"{self.name} motor '{motor}' at {t}°C (≥ warn {warn_c}°C); "
                    "STS3215 firmware thermal-protects ~70°C"
                )
                self._temp_over_warn.add(motor)
            elif t <= rearm_c and motor in self._temp_over_warn:
                self._temp_over_warn.discard(motor)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", num_retry=3)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        bus_goal = {}
        for joint, val in goal_pos.items():
            bus_goal[joint] = val
            if joint in self._secondary_of:
                bus_goal[self._secondary_of[joint]] = val

        self.bus.sync_write("Goal_Position", bus_goal, num_retry=3)

        self._maybe_sample_temperatures()

        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
