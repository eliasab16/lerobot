"""Standalone tester for the ESP32 subtask button box.

Reads BTN events from the serial port and prints each button's id, its physical
pin, and its role in the annotator. Use this to verify the hardware end-to-end
before plugging the annotator into lerobot-record.

Defaults match `SubtaskAnnotatorConfig`:
    1 -> back            (left arrow)
    2 -> next            (right arrow)
    3 -> stop            (escape)
    4 -> skip_backward   (rerecord episode)
    6 -> skip_forward    (end current phase)

Example:
    uv run python tools/esp32_subtask_button/test_buttons.py
    # (auto-detects port; override with --port /dev/tty.usbserial-0001)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import serial

PIN_BY_ID = {1: "D27", 2: "D26", 3: "D14", 4: "D25", 5: "D13", 6: "D12"}

DEFAULT_ROLES = {
    1: "back",
    2: "next",
    3: "stop",
    4: "skip_backward",
    6: "skip_forward",
}


def auto_detect_port() -> str | None:
    finder = Path(__file__).parent / "find_port.py"
    if not finder.exists():
        return None
    try:
        out = subprocess.check_output([sys.executable, str(finder)], text=True, timeout=5).strip()
        return out or None
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", help="serial port (auto-detected if omitted)")
    p.add_argument("--baud", type=int, default=115200)
    args = p.parse_args()

    port = args.port or auto_detect_port()
    if not port:
        print("Could not auto-detect ESP32. Pass --port /dev/tty.usbserial-XXX")
        return 1

    print(f"Opening {port} @ {args.baud}", flush=True)
    print("Default mapping (id -> role -> pin):")
    for bid in sorted(DEFAULT_ROLES):
        print(f"  {bid} -> {DEFAULT_ROLES[bid]:<14} ({PIN_BY_ID.get(bid, '?')})")
    print("\nPress each button. Ctrl-C to quit.\n", flush=True)

    s = serial.Serial(port, args.baud, timeout=0.5)
    time.sleep(1.0)  # let the ESP32 settle after DTR reset

    seen: set[int] = set()
    try:
        while True:
            line = s.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if not line.startswith("BTN "):
                print(f"  (ignored) {line!r}")
                continue
            try:
                btn = int(line.split()[1])
            except (IndexError, ValueError):
                print(f"  (malformed) {line!r}")
                continue
            pin = PIN_BY_ID.get(btn, "?")
            role = DEFAULT_ROLES.get(btn, "(unmapped)")
            new = " [first press]" if btn not in seen else ""
            seen.add(btn)
            print(f"button {btn} ({pin})  ->  {role}{new}", flush=True)
            if seen == set(DEFAULT_ROLES):
                print(f"\nAll {len(DEFAULT_ROLES)} buttons exercised at least once.")
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
