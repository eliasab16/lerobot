"""Resolve the ESP32 button-box serial port by probing each candidate.

The button-box firmware emits `ID:BUTTONS\n` once at boot (in setup()).
pyserial opening the port triggers a DTR-driven reset on the ESP32, so the
boot message is reliably observable within ~1 s of opening.

Prints the matching device path to stdout (e.g. /dev/tty.usbserial-3) and
exits 0. On failure exits non-zero with a stderr message — record_left.sh
treats that as a hard failure.

Use:
    export SUBTASK_PORT="$(uv run python tools/esp32_subtask_button/find_port.py)"
or in a shell script:
    PORT=$(python tools/esp32_subtask_button/find_port.py) || exit 1
"""

from __future__ import annotations

import glob
import sys
import time

import serial

TOKEN = "ID:BUTTONS"
PROBE_TIMEOUT_S = 1.5
BAUD = 115200


def probe(port: str) -> bool:
    try:
        s = serial.Serial(port, BAUD, timeout=0.2)
    except (OSError, serial.SerialException):
        return False
    try:
        deadline = time.monotonic() + PROBE_TIMEOUT_S
        while time.monotonic() < deadline:
            line = s.readline().decode("utf-8", errors="ignore").strip()
            if TOKEN in line:
                return True
    finally:
        s.close()
    return False


def main() -> int:
    candidates = sorted(glob.glob("/dev/tty.usbserial*"))
    if not candidates:
        print("No /dev/tty.usbserial* devices found", file=sys.stderr)
        return 2

    for port in candidates:
        if probe(port):
            print(port)
            return 0

    print(
        f"None of the {len(candidates)} candidates emitted {TOKEN!r} within "
        f"{PROBE_TIMEOUT_S}s. Is the button-box ESP32 plugged in and flashed?",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
