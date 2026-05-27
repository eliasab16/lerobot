"""Continuously print raw positions of the shoulder_lift pair while torque is
disabled, so you can verify primary + secondary track each other when the
joint is moved by hand.

After drive_mode is applied, the two motors should move in OPPOSITE raw
directions when the joint physically rotates. If they move in the same
direction or with a drifting offset, the pair will fight under torque.
"""

import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

FOLLOWER_PORT = "/dev/tty.usbmodem5AE60845471"


def main():
    bus = FeetechMotorsBus(
        port=FOLLOWER_PORT,
        motors={
            "shoulder_lift":           Motor(9,  "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift_secondary": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
        },
    )
    bus.connect(handshake=False)
    bus.disable_torque()
    print("\nTorque OFF. Move the SHOULDER_LIFT joint by hand. Press Ctrl-C to quit.\n")

    last_p, last_s = None, None
    try:
        while True:
            pos = bus.sync_read("Present_Position", normalize=False, num_retry=3)
            p = pos["shoulder_lift"]
            s = pos["shoulder_lift_secondary"]
            d_p = "  " if last_p is None else f"{p - last_p:+5d}"
            d_s = "  " if last_s is None else f"{s - last_s:+5d}"
            sum_str = f"sum={p + s:5d}"  # if mirrored, sum should be ~4096
            print(f"primary={p:5d} (Δ{d_p})    secondary={s:5d} (Δ{d_s})    {sum_str}")
            last_p, last_s = p, s
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
