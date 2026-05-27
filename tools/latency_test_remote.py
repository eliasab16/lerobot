"""Measure round-trip latency of the remote inference server.

Sends N full InferenceRequests with dummy data and reports the
breakdown between server-side inference time and network round-trip.

Usage:
    python tools/latency_test_remote.py ws://localhost:8000           # 100 iters
    python tools/latency_test_remote.py ws://localhost:8000 500       # 500 iters

Assumes the server has already loaded a policy via POST /setup.
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


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(int(p * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


def _stats(label: str, values: list[float]) -> None:
    s = sorted(values)
    print(
        f"{label:>14s}  "
        f"min={s[0]:6.1f}  "
        f"p50={_percentile(s, 0.50):6.1f}  "
        f"p95={_percentile(s, 0.95):6.1f}  "
        f"p99={_percentile(s, 0.99):6.1f}  "
        f"max={s[-1]:6.1f}  "
        f"(ms)"
    )


def main() -> None:
    server_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000"
    n_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_warmup = 5

    dummy_state = [0.0] * 8
    img_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.putText(img_bgr, "LATENCY", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    encoded = encode_image(img_bgr)

    base_req = {
        "state": dummy_state,
        "images": {
            "wrist_top": encoded,
            "wrist_bottom": encoded,
            "overhead": encoded,
        },
        "task": "Insert the wire tip into the component hole from below.",
    }

    print(f"connecting to {server_url}/ws ...")
    rtt_ms: list[float] = []
    server_ms: list[float] = []
    network_ms: list[float] = []

    with connect(f"{server_url}/ws", open_timeout=60, ping_interval=None) as ws:
        # Warmup — discard the first few (kernel caches still settling).
        print(f"warmup: {n_warmup} discarded iterations ...")
        for _ in range(n_warmup):
            req = InferenceRequest(**base_req, timestamp=time.perf_counter())
            ws.send(req.model_dump_json())
            ws.recv()

        print(f"measuring: {n_iters} iterations ...")
        for i in range(n_iters):
            req = InferenceRequest(**base_req, timestamp=time.perf_counter())
            t0 = time.perf_counter()
            ws.send(req.model_dump_json())
            raw = ws.recv()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            payload = json.loads(raw)
            if "error" in payload:
                print(f"  iter {i}: server error: {payload['error']}")
                continue
            resp = InferenceResponse.model_validate(payload)

            rtt_ms.append(elapsed_ms)
            server_ms.append(resp.inference_time_ms)
            network_ms.append(elapsed_ms - resp.inference_time_ms)

            if (i + 1) % max(n_iters // 10, 1) == 0:
                print(f"  iter {i + 1}/{n_iters}: rtt={elapsed_ms:.0f}ms server={resp.inference_time_ms:.0f}ms")

    print()
    print(f"--- Results ({len(rtt_ms)} samples) ---")
    _stats("total RTT", rtt_ms)
    _stats("server infer", server_ms)
    _stats("network", network_ms)
    print()

    p50_rtt = _percentile(sorted(rtt_ms), 0.50)
    p50_net = _percentile(sorted(network_ms), 0.50)
    print(f"At {p50_rtt:.0f}ms median RTT with chunk_size=50 @ 30 FPS, "
          f"actions consumed during one RTT: {p50_rtt / 1000 * 30:.1f}")
    print(f"Network alone: {p50_net:.0f}ms median  "
          f"(rest is policy forward pass)")


if __name__ == "__main__":
    main()
