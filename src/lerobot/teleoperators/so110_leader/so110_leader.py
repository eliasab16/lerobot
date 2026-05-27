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
from queue import Queue
from threading import Lock

from pynput import keyboard

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so110_leader import SO110LeaderTeleopConfig

logger = logging.getLogger(__name__)


_BUS_RETRY_MAX_ATTEMPTS = 3


def _retry_bus(label, fn, *args, **kwargs):
    """Retry a bus operation on transient ConnectionError. Bus-side sync_read
    has num_retry=0 by default, so a single corrupt response aborts long
    calibration loops; this is the caller-side safety net.
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


class SO110Leader(Teleoperator):
    """SO-110 leader teleoperator (8 single-servo joints).

    Intervention model (matches the harness in pi_recap):
      - When NOT intervening: torque enabled, leader mirrors follower via
        `send_feedback` for haptic shadowing.
      - When intervening (key '5'): torque disabled, the human moves the leader
        and `get_action` reads its present position to drive the follower.
    """

    config_class = SO110LeaderTeleopConfig
    name = "so110_leader"

    def __init__(self, config: SO110LeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "gripper": Motor(1, "sts3215", MotorNormMode.RANGE_0_100),
                "wrist_tilt": Motor(2, "sts3215", norm_mode_body),
                "wrist_yaw": Motor(3, "sts3215", norm_mode_body),
                "wrist_roll": Motor(4, "sts3215", norm_mode_body),
                "elbow_lift": Motor(5, "sts3215", norm_mode_body),
                "shoulder_swing": Motor(6, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(7, "sts3215", norm_mode_body),
                "shoulder_yaw": Motor(8, "sts3215", norm_mode_body),
            },
            calibration=self.calibration,
        )
        self.is_intervening = False
        self.is_success = False
        self.terminate_episode = False
        self.start_episode = False
        self.listener = None
        self.event_queue = Queue()
        self.bus_lock = Lock()

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        with self.bus_lock:
            self.bus.connect()
            if not self.is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            self.configure()

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        logger.info(f"{self} connected.")

    def _on_press(self, key):
        try:
            if hasattr(key, "char"):
                if key.char == "5":
                    self.is_intervening = not self.is_intervening
                    logger.info(f"Intervention state toggled: {self.is_intervening}")

                    with self.bus_lock:
                        if self.is_intervening:
                            self.bus.disable_torque()
                            logger.info("Torque disabled for manual control.")
                        else:
                            self.bus.enable_torque()
                            logger.info("Torque enabled for feedback following.")

                elif key.char == "1":
                    self.is_success = True
                    logger.info("Success triggered manually.")
                elif key.char == "0":
                    self.terminate_episode = True
                    logger.info("Failure/Termination triggered manually.")
                elif key.char == "2":
                    self.start_episode = True
                    logger.info("Start Episode triggered manually.")

        except AttributeError:
            pass
        except Exception as e:
            logger.error(f"Error in keyboard listener: {e}")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def _record_ranges_safe(
        self, motors: list[str], retries: int = 3, sleep_ms: int = 5
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Drop-in replacement for FeetechMotorsBus.record_ranges_of_motion with
        sync_read retries and a small inter-iteration sleep, to cope with tight
        single-motor sync_read timing on this hardware.
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

    def _calibrate_motors(self, motor_names: list[str]) -> dict[str, MotorCalibration]:
        """Run center + sweep on each motor; return new MotorCalibration entries."""
        full_turn = set(self.config.full_turn_motors or [])
        new_calibration: dict[str, MotorCalibration] = {}

        self.bus.disable_torque()
        for motor in motor_names:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

            input(f"\nPosition joint '{motor}' at the middle of its range and press ENTER...")
            offsets = _retry_bus("set_half_turn_homings", self.bus.set_half_turn_homings, [motor])

            if motor in full_turn:
                range_min, range_max = 0, 4095
                print(f"'{motor}' configured as full-turn (using full 0-4095 range).")
            else:
                print(f"Sweep joint '{motor}' through its full range. Press ENTER when done.")
                mins, maxes = self._record_ranges_safe([motor])
                range_min, range_max = int(mins[motor]), int(maxes[motor])

            new_calibration[motor] = MotorCalibration(
                id=self.bus.motors[motor].id,
                drive_mode=0,
                homing_offset=int(offsets[motor]),
                range_min=range_min,
                range_max=range_max,
            )

        return new_calibration

    def calibrate(self, motors: list[str] | None = None) -> None:
        if motors is not None:
            for m in motors:
                if m not in self.bus.motors:
                    raise ValueError(
                        f"Motor '{m}' not found. Available motors: {list(self.bus.motors)}"
                    )
            if not self.calibration:
                raise ValueError(
                    "No existing calibration found. Run full calibration first before "
                    "calibrating specific motors."
                )
            logger.info(f"\nRunning partial calibration of {self} for motors: {set(motors)}")
            try:
                updates = self._calibrate_motors(motors)
                merged = dict(self.calibration)
                merged.update(updates)
                self.calibration = merged
                self.bus.write_calibration(self.calibration)
                self._save_calibration()
                print(f"Partial calibration of motors {set(motors)} saved to {self.calibration_fpath}")
                return
            except Exception:
                logger.exception(
                    "Error during partial calibration. Restoring previous calibration."
                )
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
        self.calibration = self._calibrate_motors(list(self.bus.motors))
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

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

        with self.bus_lock:
            for motor in reversed(targets):
                input(f"Connect the controller board to the '{motor}' motor only and press enter.")
                self.bus.setup_motor(motor)
                print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        with self.bus_lock:
            action = self.bus.sync_read("Present_Position", num_retry=3)
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def enable_torque(self) -> None:
        with self.bus_lock:
            self.bus.enable_torque()

    @check_if_not_connected
    def disable_torque(self) -> None:
        with self.bus_lock:
            self.bus.disable_torque()

    @check_if_not_connected
    def write_goal_positions(self, positions: dict[str, float]) -> None:
        goals: dict[str, float] = {}
        for key, val in positions.items():
            if not key.endswith(".pos"):
                continue
            motor = key.removesuffix(".pos")
            if motor in self.bus.motors:
                goals[motor] = float(val)
        if not goals:
            return
        with self.bus_lock:
            self.bus.sync_write("Goal_Position", goals)

    def get_teleop_events(self) -> dict[str, bool]:
        events = {
            TeleopEvents.IS_INTERVENTION: self.is_intervening,
            TeleopEvents.TERMINATE_EPISODE: self.terminate_episode,
            TeleopEvents.SUCCESS: self.is_success,
            TeleopEvents.START_EPISODE: self.start_episode,
            TeleopEvents.RERECORD_EPISODE: False,
        }

        if self.is_success:
            self.is_success = False
        if self.terminate_episode:
            self.terminate_episode = False
        if self.start_episode:
            self.start_episode = False

        return events

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Move the leader to follower positions for haptic shadowing.

        No-op while the user is intervening (torque is off in that mode).
        """
        if self.is_intervening:
            return

        goal_positions = {}
        for motor in self.bus.motors:
            if f"{motor}.pos" in feedback:
                goal_positions[motor] = feedback[f"{motor}.pos"]
            else:
                logger.warning(f"Missing feedback for motor {motor}")
                return

        with self.bus_lock:
            if self.is_intervening:
                return
            self.bus.sync_write("Goal_Position", goal_positions, num_retry=3)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener:
            self.listener.stop()
            self.listener = None

        with self.bus_lock:
            self.bus.disconnect()
        logger.info(f"{self} disconnected.")
