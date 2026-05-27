"""Check whether each paired joint on the SO110 follower moves its primary
and secondary motors in opposite directions (correct, drive_mode=1) or the
same direction (wrong — they'll fight under torque and the joint won't move).

Torque is disabled throughout — you move the joints by hand.

Usage:
    python tools/check_pair_directions.py
"""

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

FOLLOWER_PORT = "/dev/tty.usbmodem5AE60845471"

PAIRS = [
    ("elbow_lift", 5, 6),
    ("shoulder_swing", 7, 8),
    ("shoulder_lift", 9, 10),
]


def main():
    motors = {}
    for joint, p_id, s_id in PAIRS:
        motors[joint] = Motor(p_id, "sts3215", MotorNormMode.RANGE_M100_100)
        motors[f"{joint}_secondary"] = Motor(s_id, "sts3215", MotorNormMode.RANGE_M100_100)

    bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=motors)
    bus.connect(handshake=False)
    bus.disable_torque()

    print("\nTorque is OFF — you can move the joints by hand.\n")

    results = {}
    for joint, p_id, s_id in PAIRS:
        input(f"Place '{joint}' near its center, then press ENTER...")
        before = bus.sync_read("Present_Position", normalize=False, num_retry=3)
        p_before = before[joint]
        s_before = before[f"{joint}_secondary"]
        print(f"  before:  primary={p_before:>5}   secondary={s_before:>5}")

        input(f"Now move '{joint}' clearly in one direction (a noticeable amount), then press ENTER...")
        after = bus.sync_read("Present_Position", normalize=False, num_retry=3)
        p_after = after[joint]
        s_after = after[f"{joint}_secondary"]
        print(f"  after:   primary={p_after:>5}   secondary={s_after:>5}")

        p_delta = p_after - p_before
        s_delta = s_after - s_before
        print(f"  delta:   primary={p_delta:+5}   secondary={s_delta:+5}")

        if p_delta == 0 and s_delta == 0:
            verdict = "NO MOVEMENT DETECTED — move further and re-run"
        elif (p_delta > 0) == (s_delta > 0):
            verdict = "SAME direction → drive_mode WRONG on secondary (flip 1→0)"
        else:
            verdict = "OPPOSITE direction → drive_mode CORRECT on secondary"

        results[joint] = verdict
        print(f"  → {verdict}\n")

    print("=" * 60)
    print("Summary:")
    for joint, verdict in results.items():
        print(f"  {joint}: {verdict}")
    print("=" * 60)

    bus.disconnect()


if __name__ == "__main__":
    main()
