#!/bin/bash
# Record a teleoperation dataset for the SO-110 left arm.
# Cameras: wrist_top (idx 0), wrist_bottom (idx 1), overhead (idx 2)
# All cameras: 640x480 native, rotated 90° CW, center-cropped to 480x480.

# Tunable per-motor registers (each accepts a scalar applied to every motor,
# or a {motor_name: value} dict for overrides; secondaries inherit primary):
#     --robot.p_coefficient=8          # default 8
#     --robot.i_coefficient=0          # default 0
#     --robot.d_coefficient=64         # default 64
#     --robot.acceleration=50          # STS3215 reg 41; lower = smoother stops
#
# Same flags as tools/teleop_left.sh — `lerobot-record` shares the robot config.

REPO_NAME="pi05_recap_batch_1"
RESUME=true

OVERWRITE=0
DISPLAY_BACKEND=cv2
for arg in "$@"; do
    case "$arg" in
        -ow)    OVERWRITE=1 ;;
        -rerun) DISPLAY_BACKEND=rerun ;;
    esac
done

DATASET_DIR="$HOME/.cache/huggingface/lerobot/${HF_USER:-eliasab16}/${REPO_NAME}"

if [ "$OVERWRITE" -eq 1 ]; then
    if [ "$RESUME" = "true" ]; then
        echo "-ow ignored: RESUME=true (deleting the dataset would defeat resume). Set RESUME=false to overwrite."
    elif [ -d "$DATASET_DIR" ]; then
        echo "Overwrite requested. About to delete: $DATASET_DIR"
        read -r -p "Proceed? [y/N] " ans
        case "$ans" in
            y|Y|yes|YES) rm -rf "$DATASET_DIR" && echo "Deleted." ;;
            *) echo "Aborted."; exit 1 ;;
        esac
    else
        echo "Overwrite requested but $DATASET_DIR does not exist; nothing to delete."
    fi
fi

# Clean slate: kill any orphan `say` children or stale lerobot-record processes
# from a previous crashed run. These can hold serial ports / Cocoa state and
# make the next start flake until they age out.
pkill -f '^say ' 2>/dev/null || true
pkill -f 'lerobot-record' 2>/dev/null || true
sleep 0.5

# macOS fork-safety override: required when a daemon thread (pyserial in the
# subtask annotator) is alive at the moment Python forks for a subprocess
# (e.g. the `say` command from log_say). Without this, the process aborts.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Make Python print a real traceback on SIGABRT instead of just "Abort trap: 6"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Auto-detect the button-box port by probing for the ID:BUTTONS boot line.
# CP2102 devices share a default serial, so /dev/tty.usbserial-N enumeration
# is plug-order dependent; this resolves it deterministically.
SUBTASK_PORT=$(/Users/elisd/miniconda3/bin/python "$(dirname "$0")/esp32_subtask_button/find_port.py") || exit 1
echo "Detected subtask button box on $SUBTASK_PORT"

lerobot-record \
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
    --dataset.repo_id="${HF_USER:-eliasab16}/${REPO_NAME}" \
    --dataset.root="$HOME/.cache/huggingface/lerobot/${HF_USER:-eliasab16}/${REPO_NAME}" \
    --dataset.single_task="Insert the wire tip into the component hole from below." \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=150 \
    --dataset.reset_time_s=150 \
    --dataset.push_to_hub=false \
    --play_sounds=true \
    --display_data=true \
    --display_compressed_images=false \
    --dataset.vcodec=auto \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.encoder_queue_maxsize=30 \
    --dataset.video_encoding_batch_size=1 \
    --dataset.push_to_hub=false \
    --resume=$RESUME \
    --subtask_annotator.port="$SUBTASK_PORT" \
    --subtask_annotator.subtasks='["lift the wire into the workspace",
    "approach the component along the rail",
    "align the wire tip below the component hole",
    "insert the wire into the hole"]' \
    --relative_zero_offset=true \
    --display_backend=$DISPLAY_BACKEND
