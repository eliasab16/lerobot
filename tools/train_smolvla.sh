#!/bin/bash
# Finetune SmolVLA on an SO-110 dataset (e.g. so110_insert_wire_batch_1).
#
# Expected to run on a rented Linux/CUDA box, NOT the macOS workstation:
#   1) git clone this repo onto the remote box
#   2) `uv sync --locked --extra smolvla` (or pip install 'lerobot[smolvla]')
#   3) `hf auth login`  (write token, so push_to_hub works)
#   4) bash tools/train_smolvla.sh
#
# Reference: docs/source/smolvla.mdx, AGENT_GUIDE.md §7.

set -euo pipefail

# ---- Edit these per run -----------------------------------------------------
HF_USER="${HF_USER:-eliasab16}"
DATASET_REPO_ID="$HF_USER/so110_insert_wire_batch_1"
RUN_NAME="smolvla_insert_wire_b1"
STEPS=20000
BATCH_SIZE=64
SAVE_FREQ=5000          # checkpoint every N steps
SEED=1000

# Unfreeze vision encoder for real gains on specialized tasks (AGENT_GUIDE §7.6).
# Costs more VRAM (~+8 GB) and ~30% slower per step. Fits on A100 80 / A6000 48
# / 4090 24 (drop batch to 32 if you OOM on 24 GB).
UNFREEZE_VISION=true

# Push the final checkpoint to the Hub when the run finishes.
PUSH_TO_HUB=true
# -----------------------------------------------------------------------------

OUTPUT_DIR="outputs/train/${RUN_NAME}"

UNFREEZE_FLAGS=()
if [ "$UNFREEZE_VISION" = "true" ]; then
    UNFREEZE_FLAGS+=(
        --policy.freeze_vision_encoder=false
        --policy.train_expert_only=false
    )
fi

PUSH_FLAGS=()
if [ "$PUSH_TO_HUB" = "true" ]; then
    PUSH_FLAGS+=(
        --policy.push_to_hub=true
        --policy.repo_id="$HF_USER/$RUN_NAME"
    )
else
    PUSH_FLAGS+=(--policy.push_to_hub=false)
fi

uv run lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$HOME/.cache/huggingface/lerobot/$DATASET_REPO_ID" \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --policy.scheduler_decay_steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$RUN_NAME" \
    --seed=$SEED \
    --num_workers=4 \
    --wandb.enable=true \
    --wandb.project=smolvla_so110 \
    "${UNFREEZE_FLAGS[@]}" \
    "${PUSH_FLAGS[@]}"
