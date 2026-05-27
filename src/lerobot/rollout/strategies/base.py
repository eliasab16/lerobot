# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Base rollout strategy: autonomous policy execution with no data recording.

Also exposes a single keybinding ``r`` to release/restore follower torque so an
operator can manually re-pose the arm between attempts without aborting. While
torque is released, inference is paused and the control loop idles.
"""

from __future__ import annotations

import logging
import threading
import time

from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.

    Keybinding (when run from a terminal that has focus):
        r — toggle follower torque off/on for manual repositioning.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._toggle_request = threading.Event()
        self._torque_released = False
        self._key_listener = None

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine and start the keyboard listener."""
        self._init_engine(ctx)
        self._start_key_listener()
        logger.info("Base strategy ready (press 'r' to toggle follower torque)")

    def _start_key_listener(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            logger.warning("pynput not available — 'r' torque-toggle disabled")
            return

        def on_press(key):
            try:
                if hasattr(key, "char") and key.char == "r":
                    self._toggle_request.set()
            except AttributeError:
                pass

        self._key_listener = keyboard.Listener(on_press=on_press, daemon=True)
        self._key_listener.start()

    def _handle_torque_toggle(self, ctx: RolloutContext) -> None:
        """Process a pending 'r' keypress: flip torque state on the follower."""
        if not self._toggle_request.is_set():
            return
        self._toggle_request.clear()

        engine = self._engine
        robot = ctx.hardware.robot_wrapper
        bus = getattr(robot.inner, "bus", None)
        if bus is None or not hasattr(bus, "disable_torque"):
            logger.warning(
                "Robot %s has no .bus.disable_torque(); cannot toggle torque",
                type(robot.inner).__name__,
            )
            return

        if not self._torque_released:
            # Releasing: pause inference, give the operator 2 s to grip the arm,
            # then drop torque so it doesn't fall.
            engine.pause()
            release_delay_s = 2.0
            logger.info("Releasing torque in %.1fs — get ready to hold the arm", release_delay_s)
            time.sleep(release_delay_s)
            bus.disable_torque()
            self._torque_released = True
            logger.info("Follower torque RELEASED — reposition by hand, press 'r' to resume")
        else:
            # Re-engaging: torque on (holds current physical position), then
            # clear the inference queue so the policy plans fresh from the new
            # observation rather than slewing toward stale actions.
            bus.enable_torque()
            engine.reset()
            engine.resume()
            self._torque_released = False
            logger.info("Follower torque ENGAGED — resuming policy")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            self._handle_torque_toggle(ctx)

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            if self._torque_released:
                # Arm is limp — idle the loop until the operator re-engages.
                # Sleep longer than control_interval to avoid spinning the CPU.
                time.sleep(0.05)
                continue

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                )

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        if self._key_listener is not None:
            self._key_listener.stop()
        # If we exit mid-release, re-engage torque so the arm doesn't fall.
        if self._torque_released:
            try:
                bus = getattr(ctx.hardware.robot_wrapper.inner, "bus", None)
                if bus is not None and hasattr(bus, "enable_torque"):
                    bus.enable_torque()
            except Exception as e:
                logger.warning("Failed to re-engage torque on teardown: %s", e)
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
