"""End-to-end smoke test for the remote inference server.

Sends one non-RTC InferenceRequest with dummy data and prints the
returned chunk shape + timing. Assumes the server has already loaded
a policy via POST /setup.

Usage:
    python tools/smoke_test_remote.py ws://<gpu-host>:8000
"""

from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
from websockets.sync.client import connect

from lerobot.rollout.remote.image_codec import encode_image
from lerobot.rollout.remote.protocol import InferenceRequest, InferenceResponse


def main() -> None:
    server_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000"

    dummy_state = [0.0] * 8
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "SMOKE TEST", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    encoded = encode_image(dummy_img)

    req = InferenceRequest(
        state=dummy_state,
        images={"wrist_top": encoded, "wrist_bottom": encoded, "overhead": encoded},
        task="Insert the wire tip into the component hole from below.",
        timestamp=time.perf_counter(),
    )

    print(f"connecting to {server_url}/ws ...")
    with connect(f"{server_url}/ws", open_timeout=60, ping_interval=None) as ws:
        t0 = time.perf_counter()
        ws.send(req.model_dump_json())
        raw = ws.recv()
        rtt_ms = (time.perf_counter() - t0) * 1000

    payload = json.loads(raw)
    if "error" in payload:
        print(f"server error: {payload['error']}")
        sys.exit(1)

    resp = InferenceResponse.model_validate(payload)
    chunk = np.asarray(resp.actions)
    print(f"rtt={rtt_ms:.0f}ms  server_inference={resp.inference_time_ms:.0f}ms")
    print(f"chunk shape: {chunk.shape}  (T, action_dim)")
    print(f"first action: {chunk[0].tolist()}")
    print(f"last action:  {chunk[-1].tolist()}")


if __name__ == "__main__":
    main()
