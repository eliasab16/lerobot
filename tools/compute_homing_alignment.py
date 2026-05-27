"""Compute corrected homing_offsets for the SO110 LEADER so that teleop produces
no motion when the arms are physically aligned.

Usage:
    1. Physically position the leader and follower in the SAME pose.
    2. Run this script.
    3. Paste the printed values into the leader's calibration JSON
       (only the `homing_offset` field per joint).
    4. Restart teleop; when prompted, press ENTER to write the JSON to the motor.

Formula: new_homing_leader = old_homing_leader + (present_leader - present_follower)
Derivation: For matching poses we need present_leader == present_follower.
Since present = raw_encoder - homing_offset (firmware), adjusting the leader's
homing_offset by the observed present delta makes the two Present_Position
streams match without touching the follower.

Pair note: the follower has paired motors with names `{joint}_secondary`. Only
primaries are compared and adjusted; secondaries are slaved to their primary
on writes (via `drive_mode`) and are not exposed to teleop, so they need no
homing alignment.
"""

import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

LEADER_PORT = "/dev/tty.usbmodem5A7A0565141"
FOLLOWER_PORT = "/dev/tty.usbmodem5AE60845471"

# Edit these to match the --teleop.id / --robot.id you calibrated with.
LEADER_JSON = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so110_leader/so110_leader_left.json"
FOLLOWER_JSON = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so110_follower/so110_follower_left.json"


# Leader: 8 motors, IDs 1-8 (matches SO110Leader bus declaration).
LEADER_MOTORS = {
    "gripper": 1,
    "wrist_tilt": 2,
    "wrist_yaw": 3,
    "wrist_roll": 4,
    "elbow_lift": 5,
    "shoulder_swing": 6,
    "shoulder_lift": 7,
    "shoulder_yaw": 8,
}

# Follower: 11 motors with `_secondary` interleaved (matches SO110Follower bus).
# Declared in full so sync_read covers the whole bus; only primaries are used
# for the alignment computation.
FOLLOWER_MOTORS = {
    "gripper": 1,
    "wrist_tilt": 2,
    "wrist_yaw": 3,
    "wrist_roll": 4,
    "elbow_lift": 5,
    "elbow_lift_secondary": 6,
    "shoulder_swing": 7,
    "shoulder_swing_secondary": 8,
    "shoulder_lift": 9,
    "shoulder_lift_secondary": 10,
    "shoulder_yaw": 11,
}

# 8 logical joints compared between leader and follower.
PRIMARIES = list(LEADER_MOTORS.keys())


def make_bus(port: str, motors: dict[str, int]) -> FeetechMotorsBus:
    return FeetechMotorsBus(
        port=port,
        motors={name: Motor(mid, "sts3215", MotorNormMode.DEGREES) for name, mid in motors.items()},
        calibration=None,
    )


def read_present(port: str, motors: dict[str, int], label: str) -> dict[str, int]:
    bus = make_bus(port, motors)
    bus.connect()
    present = bus.sync_read("Present_Position", normalize=False)
    bus.disconnect()
    print(f"\n[{label}] Present_Position (raw after firmware homing):")
    for m in motors:
        print(f"  {m:25s} = {present[m]}")
    return present


def main():
    with open(LEADER_JSON) as f:
        leader_cal = json.load(f)
    with open(FOLLOWER_JSON) as f:
        follower_cal = json.load(f)  # noqa: F841  (loaded for symmetry / future checks)

    input("Position BOTH arms in the same physical pose, then press ENTER to read positions...")

    present_leader = read_present(LEADER_PORT, LEADER_MOTORS, "LEADER")
    present_follower = read_present(FOLLOWER_PORT, FOLLOWER_MOTORS, "FOLLOWER")

    print("\n=== Corrected leader homing_offsets ===")
    print(f"Paste these into {LEADER_JSON.name} (only homing_offset fields):\n")
    for m in PRIMARIES:
        old_homing = leader_cal[m]["homing_offset"]
        delta = present_leader[m] - present_follower[m]
        new_homing = old_homing + delta
        print(f"  {m:25s}  old={old_homing:6d}  delta={delta:+6d}  new={new_homing:6d}")

    print("\nAfter editing the JSON, restart teleop and press ENTER at the calibration prompt")
    print("so the new homing is written to the motor firmware.")


if __name__ == "__main__":
    main()
