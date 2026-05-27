"""Sequential subtask annotator driven by ESP32 push buttons.

Three logical buttons mirror the keyboard controls used in lerobot-record,
with added subtask semantics when recording is active:

    next  (default keyboard: right arrow)
        Recording: advance to the next subtask.  Pressing past the last
        subtask auto-ends the episode by setting events["exit_early"].
        Reset / park: set events["exit_early"] to move on to the next
        episode (same as right arrow).

    back  (default keyboard: left arrow)
        Recording: undo the previous subtask advance.  No-op at the first
        subtask.
        Reset / park: set events["rerecord_episode"] + events["exit_early"]
        so the just-recorded episode is re-recorded from the beginning
        (same as left arrow).

    stop  (default keyboard: escape)
        Any phase: set events["stop_recording"] + events["exit_early"] to
        finish the session.

Firmware contract: ESP32 emits one line per debounced press over USB CDC:
    BTN 1\n
where the integer is the button id (1..N) into the firmware's pin array.
The host config maps each id to one of next/back/stop.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field

import serial

logger = logging.getLogger(__name__)


@dataclass
class SubtaskAnnotatorConfig:
    port: str
    subtasks: list[str] = field(default_factory=list)
    baud: int = 115200
    # Button ids (1-based, as emitted by the firmware) for each function.
    back_button: int = 1
    next_button: int = 2
    stop_button: int = 3
    # Skip whole episode/phase (analogous to Down / Up arrow keys).
    skip_backward_button: int = 4
    skip_forward_button: int = 6


class SubtaskAnnotator:
    def __init__(self, config: SubtaskAnnotatorConfig):
        if not config.subtasks:
            raise ValueError("SubtaskAnnotatorConfig.subtasks must be non-empty")
        if len(set(config.subtasks)) != len(config.subtasks):
            raise ValueError(f"Subtasks must be unique: {config.subtasks}")

        self.config = config
        self.subtasks: list[str] = list(config.subtasks)
        self._serial: serial.Serial | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._events: queue.Queue[int] = queue.Queue()
        self._current_idx = 0

    def connect(self) -> None:
        self._serial = serial.Serial(self.config.port, self.config.baud, timeout=0.1)
        self._stop.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        logger.info(
            "SubtaskAnnotator connected on %s with subtasks: %s",
            self.config.port,
            self.subtasks,
        )

    def disconnect(self) -> None:
        self._stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _read_loop(self) -> None:
        assert self._serial is not None
        while not self._stop.is_set():
            try:
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
            except Exception as e:
                logger.warning("SubtaskAnnotator serial read error: %s", e)
                continue
            if not line.startswith("BTN "):
                continue
            try:
                btn_id = int(line.split()[1])
            except (IndexError, ValueError):
                logger.warning("SubtaskAnnotator malformed line: %r", line)
                continue
            self._events.put(btn_id)

    def reset_episode(self) -> None:
        """Call at the start of every recording episode."""
        self._current_idx = 0
        while not self._events.empty():
            try:
                self._events.get_nowait()
            except queue.Empty:
                break

    def process_events(self, events: dict) -> None:
        """Drain queued button events into the shared events dict.

        Sets generic step_forward / step_backward flags for the next/back
        buttons; the record_loop translates those into either a subtask
        advance/undo or a phase exit based on current annotator state. Skip
        and stop buttons set the existing exit_early / rerecord_episode /
        stop_recording flags directly.
        """
        cfg = self.config
        while True:
            try:
                btn_id = self._events.get_nowait()
            except queue.Empty:
                return

            if btn_id == cfg.stop_button:
                logger.info("SubtaskAnnotator: stop pressed — finishing session")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif btn_id == cfg.skip_forward_button:
                logger.info("SubtaskAnnotator: skip-forward pressed — ending current phase")
                events["exit_early"] = True
            elif btn_id == cfg.skip_backward_button:
                logger.info("SubtaskAnnotator: skip-backward pressed — re-recording episode")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif btn_id == cfg.next_button:
                events["step_forward"] = True
            elif btn_id == cfg.back_button:
                events["step_backward"] = True
            else:
                logger.info("SubtaskAnnotator: unmapped button id %d", btn_id)

    def advance(self) -> bool:
        """Advance to the next subtask. Returns True if past the last subtask
        (caller should treat as phase-end signal)."""
        if self._current_idx >= len(self.subtasks):
            return True
        completed = self.subtasks[self._current_idx]
        self._current_idx += 1
        if self._current_idx >= len(self.subtasks):
            logger.info("SubtaskAnnotator: completed final subtask '%s'", completed)
            return True
        logger.info(
            "SubtaskAnnotator: completed '%s' -> now on '%s'",
            completed,
            self.subtasks[self._current_idx],
        )
        return False

    def undo(self) -> bool:
        """Step back one subtask. Returns True if already at the first
        subtask (caller should treat as rerecord signal)."""
        if self._current_idx == 0:
            return True
        self._current_idx -= 1
        logger.info("SubtaskAnnotator: undone -> now on '%s'", self.subtasks[self._current_idx])
        return False

    def current_subtask(self) -> str:
        idx = min(self._current_idx, len(self.subtasks) - 1)
        return self.subtasks[idx]

    def current_index(self) -> int:
        return min(self._current_idx, len(self.subtasks) - 1)
