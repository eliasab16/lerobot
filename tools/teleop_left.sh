#!/bin/bash
# Teleoperate the SO-110 left arm with 3 OpenCV cameras (indices 0, 1, 2).
# Adjust camera names / resolutions / fps to your setup.
#
# Tunable per-motor registers (each accepts a scalar applied to every motor,
# or a {motor_name: value} dict for overrides; secondaries inherit primary):
#     --robot.p_coefficient=8          # default 8
#     --robot.i_coefficient=0          # default 0
#     --robot.d_coefficient=64         # default 64
#     --robot.acceleration=50          # STS3215 reg 41; lower = smoother stops
#
# Example per-motor overrides:
#     --robot.p_coefficient='{gripper: 8, wrist_tilt: 8, wrist_yaw: 8, wrist_roll: 8}'
#     --robot.acceleration='{shoulder_lift: 30, elbow_lift: 30}'
#
# Flags:
#   -rzo   Enable relative zero offset (follower stays at its current pose at
#          teleop start; leader's deltas — not its absolute pose — drive it).
#
RELATIVE_ZERO_OFFSET=false
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        -rzo) RELATIVE_ZERO_OFFSET=true ;;
        *)    PASSTHROUGH+=("$arg") ;;
    esac
done

lerobot-teleoperate \
    --robot.type=so110_follower \
    --robot.port=/dev/tty.usbmodem5AE60845471 \
    --robot.id=so110_follower_left \
    --robot.p_coefficient='{shoulder_lift: 16, shoulder_swing: 16, elbow_lift: 10, gripper: 10, wrist_tilt: 10, wrist_yaw: 10, wrist_roll: 10, shoulder_yaw: 10}' \
    --robot.d_coefficient=64 \
    --robot.temperature_sample_interval_s=5.0 \
    --robot.temperature_warning_c=55 \
    --robot.cameras='{
        wrist_top:    {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true},
        wrist_bottom: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true},
        overhead:     {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true}
    }' \
    --teleop.type=so110_leader \
    --teleop.port=/dev/tty.usbmodem5A7A0565141 \
    --teleop.id=so110_leader_left \
    --display_data=true \
    --relative_zero_offset=$RELATIVE_ZERO_OFFSET \
    "${PASSTHROUGH[@]}"
