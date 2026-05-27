#!/bin/bash
# Rollout SmolVLA on the SO-110 left arm via a remote policy server.
#
# Expects:
#   1) A `lerobot-policy-server` running on a GPU box (e.g. RunPod L4)
#      with the SmolVLA checkpoint already loaded via POST /setup.
#   2) An SSH port-forward so localhost:9000 on this Mac reaches the
#      server (or change SERVER_URL to wss://<host>:<port>/).
#
# DAgger strategy: policy runs autonomously until you press `tab`. Per-tab
# offset means the follower does NOT snap when you take over — leader
# motion is mapped relative to where each arm was at the moment of tab.
#
# Controls (default keyboard bindings):
#   space — pause/resume policy (no teleop, robot holds)
#   tab   — toggle correction window (leader takeover with per-tab offset)
#   enter — push the dataset to Hub on demand (we keep it local here)

REPO_ID="${HF_USER:-eliasab16}/smolvla_insert_wire_b1_10k"
TASK="Insert the wire tip into the component hole from below."
NUM_EPISODES=5
DURATION_S=600
SERVER_URL="ws://localhost:9000"

# Throwaway dataset (dagger requires one, even with record_autonomous=false).
THROWAWAY_REPO="local-only/rollout_dagger_test"

lerobot-rollout \
    --strategy.type=dagger \
    --strategy.num_episodes=$NUM_EPISODES \
    --strategy.record_autonomous=false \
    --strategy.upload_every_n_episodes=100 \
    --inference.type=remote \
    --inference.server_url="$SERVER_URL" \
    --inference.fire_after_n_actions=14 \
    --inference.rtc.enabled=true \
    --inference.rtc.execution_horizon=20 \
    --inference.log_actions_csv=outputs/rollout_actions.csv \
    --rename_map='{"observation.images.wrist_top": "observation.images.camera1", "observation.images.wrist_bottom": "observation.images.camera2", "observation.images.overhead": "observation.images.camera3"}' \
    --policy.path="$REPO_ID" \
    --robot.type=so110_follower \
    --robot.port=/dev/tty.usbmodem5AE60845471 \
    --robot.id=so110_follower_left \
    --robot.p_coefficient='{shoulder_lift: 16, shoulder_swing: 16, elbow_lift: 10, gripper: 10, wrist_tilt: 10, wrist_yaw: 10, wrist_roll: 10, shoulder_yaw: 10}' \
    --robot.d_coefficient=64 \
    --robot.temperature_sample_interval_s=5.0 \
    --robot.temperature_warning_c=55 \
    --robot.cameras='{
        wrist_top:    {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true},
        wrist_bottom: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true},
        overhead:     {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true}
    }' \
    --teleop.type=so110_leader \
    --teleop.port=/dev/tty.usbmodem5A7A0565141 \
    --teleop.id=so110_leader_left \
    --dataset.repo_id="$THROWAWAY_REPO" \
    --dataset.single_task="$TASK" \
    --dataset.push_to_hub=false \
    --task="$TASK" \
    --duration=$DURATION_S \
    --display_data=true
