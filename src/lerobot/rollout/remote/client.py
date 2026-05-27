"""WebSocket inference client.

Runs locally alongside ``lerobot-rollout``:

1. Main thread (at control FPS) calls :py:meth:`update_observation` and
   :py:meth:`get_action`.
2. A background thread pulls the latest observation, ships it over the
   WebSocket, and merges the returned chunk into a local ActionQueue.

Fire policy ("when to ask the server for a new chunk"):
- ``synchronous_mode=True``: fire only when the queue is empty (arm pauses
  during RTT). Useful for isolating whether artifacts are from queue/merge
  logic vs the policy itself.
- ``fire_after_n_actions=N``: fire when N actions have been consumed since
  the last merge, or the queue is empty. N must be > observed real_delay
  (~3 at RTT=110ms) to avoid queue underflow.

Replace semantics on merge: the new chunk fully replaces the queue (sliced
by ``real_delay`` to drop actions the main thread already executed during
the RTT). The legacy ``ActionQueue.merge()`` path is retained in comments
for reference.
"""

import csv
import json
import threading
import time
from collections import deque

import requests
import torch
from torch import Tensor
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker

from .image_codec import encode_image
from .protocol import (
    InferenceRequest,
    InferenceRTCRequest,
    InferenceResponse,
    SetupRequest,
    SetupResponse,
    SetupStatus,
)

_LOG_SEPARATOR = "─" * 80


class RemoteInferenceClient:
    def __init__(
        self,
        server_url: str,
        fps: float,
        rtc_config: RTCConfig | None,
        fire_after_n_actions: int | None,
        synchronous_mode: bool,
        log_actions_csv: str | None = None,
        joint_names: list[str] | None = None,
        verbose_transitions: bool = True,
    ):
        self.server_url = server_url
        self.fps = fps
        self.rtc_config = rtc_config
        self.ws = None
        self._running = threading.Event()
        self._inference_thread: threading.Thread | None = None
        self._latest_obs: dict | None = None
        self._obs_lock = threading.Lock()
        self._action_queue: ActionQueue | None = None
        self._latency_tracker = LatencyTracker()
        self._fire_after_n_actions = fire_after_n_actions
        self._synchronous_mode = synchronous_mode
        self._warmup_latency_cutoff_s = 1.0
        self._verbose = verbose_transitions

        self._chunk_counter = 0
        self._pending_chunks: deque = deque()
        self._chunks_lock = threading.Lock()
        self._t_start: float | None = None

        self._log_actions_csv_path = log_actions_csv
        self._joint_names = joint_names
        self._csv_file = None
        self._csv_writer = None
        self._csv_lock = threading.Lock()

        self.chunk_size: int | None = None

    def setup(self, policy_path: str, action_dim: int, camera_names: list[str], task: str, device: str):
        setup_request = SetupRequest(
            policy_path=policy_path,
            action_dim=action_dim,
            camera_names=camera_names,
            task=task,
            device=device,
            compile_model=False,
        )

        http_url = self.server_url.replace("ws://", "http://").replace("wss://", "https://")
        timeout = 300
        try:
            response = requests.post(
                f"{http_url}/setup",
                data=setup_request.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            setup_response = SetupResponse.model_validate_json(response.text)
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Setup timed out after {timeout}s. Server may still be loading.")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to {http_url}. Is the server running? ({e})")
        except Exception as e:
            raise RuntimeError(f"Setup failed: {type(e).__name__}: {e}")

        if setup_response.status != SetupStatus.READY:
            raise RuntimeError(f"Setup failed on server: {setup_response.message}")

        self.chunk_size = setup_response.chunk_size
        print(f"Server setup successful. Chunk size: {self.chunk_size}")

        self._action_queue = ActionQueue(
            cfg=self.rtc_config if self.rtc_config is not None else RTCConfig(),
        )

    def start(self):
        if self._running.is_set():
            print("Inference thread already running.")
            return

        # ping_interval=None: a slow first warmup inference (30-60s) would
        # otherwise trigger keepalive timeouts before the response arrives.
        self.ws = ws_connect(
            f"{self.server_url}/ws",
            open_timeout=200,
            ping_interval=None,
        )
        self._running.set()
        self._t_start = time.perf_counter()

        if self._log_actions_csv_path is not None:
            self._csv_file = open(self._log_actions_csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            print(f"[client] logging per-action positions to {self._log_actions_csv_path}", flush=True)

        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        self._running.clear()
        if self.ws:
            self.ws.close()
        if self._inference_thread:
            self._inference_thread.join(timeout=5)
        self.ws = None
        with self._csv_lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def update_observation(self, state, images, task):
        with self._obs_lock:
            self._latest_obs = {"state": state, "images": images, "task": task}

    def get_action(self) -> Tensor | None:
        if self._action_queue is None:
            return None
        action = self._action_queue.get()
        if action is None:
            return None

        chunk_id_for_csv = None
        action_idx_for_csv = None
        chunk_fire_ms_for_csv = None
        with self._chunks_lock:
            if self._pending_chunks:
                head = self._pending_chunks[0]
                chunk_id_for_csv = head["chunk_id"]
                action_idx_for_csv = head["chunk_len"] - head["remaining"]
                chunk_fire_ms_for_csv = head["fire_time_ms"]
                head["remaining"] -= 1

                if head["remaining"] == 0:
                    next_chunk = self._pending_chunks[1] if len(self._pending_chunks) > 1 else None
                    if next_chunk is not None and self._t_start is not None and self._verbose:
                        t_rel = (time.perf_counter() - self._t_start) * 1000
                        first_new_action = next_chunk["actions"][0]
                        print(_LOG_SEPARATOR, flush=True)
                        print(
                            f"[transition @ t={t_rel:.0f}ms]  chunk_{head['chunk_id']} → chunk_{next_chunk['chunk_id']}",
                            flush=True,
                        )
                        print(
                            f"  last action (chunk_{head['chunk_id']}):  {[f'{v:+.3f}' for v in action.tolist()]}",
                            flush=True,
                        )
                        print(
                            f"  first action (chunk_{next_chunk['chunk_id']}): {[f'{v:+.3f}' for v in first_new_action.tolist()]}",
                            flush=True,
                        )
                    self._pending_chunks.popleft()

        if self._csv_writer is not None and self._t_start is not None:
            with self._csv_lock:
                if self._csv_writer is None:
                    return action
                t_ms = (time.perf_counter() - self._t_start) * 1000
                action_list = action.tolist()
                if self._csv_file.tell() == 0:
                    motor_cols = (
                        self._joint_names
                        if self._joint_names is not None and len(self._joint_names) == len(action_list)
                        else [f"m{i}" for i in range(len(action_list))]
                    )
                    self._csv_writer.writerow(
                        ["t_ms", "chunk_id", "action_idx", "chunk_fire_ms", "age_ms", *motor_cols]
                    )
                age_ms = t_ms - chunk_fire_ms_for_csv if chunk_fire_ms_for_csv is not None else ""
                self._csv_writer.writerow(
                    [
                        f"{t_ms:.2f}",
                        chunk_id_for_csv if chunk_id_for_csv is not None else "",
                        action_idx_for_csv if action_idx_for_csv is not None else "",
                        f"{chunk_fire_ms_for_csv:.2f}" if chunk_fire_ms_for_csv is not None else "",
                        f"{age_ms:.2f}" if age_ms != "" else "",
                        *[f"{v:.4f}" for v in action_list],
                    ]
                )

        return action

    def clear_queue(self):
        if self._action_queue is not None:
            self._action_queue.clear()
        with self._chunks_lock:
            self._pending_chunks.clear()

    def _build_request(self, obs: dict) -> InferenceRequest | InferenceRTCRequest:
        base_fields = {
            "state": obs["state"],
            "images": {name: encode_image(img) for name, img in obs["images"].items()},
            "task": obs["task"],
            "timestamp": time.perf_counter(),
        }

        if self.rtc_config is None or not self.rtc_config.enabled:
            return InferenceRequest(**base_fields)

        leftover = self._action_queue.get_left_over()
        if leftover is None or len(leftover) == 0:
            return InferenceRequest(**base_fields)

        return InferenceRTCRequest(
            **base_fields,
            prev_chunk_left_over=leftover.tolist(),
            inference_delay=int(self._latency_tracker.percentile(0.5) * self.fps),
            execution_horizon=self.rtc_config.execution_horizon,
        )

    def _inference_loop(self):
        while self._running.is_set():
            with self._obs_lock:
                obs = self._latest_obs
                self._latest_obs = None

            if obs is None:
                time.sleep(0.005)
                continue

            if self._synchronous_mode:
                should_fire = self._action_queue.qsize() == 0
            else:
                if self._fire_after_n_actions is None:
                    raise RuntimeError(
                        "fire_after_n_actions must be set (or enable synchronous_mode) — "
                        "the legacy queue_threshold trigger has been retired."
                    )
                should_fire = (
                    self._action_queue.qsize() == 0
                    or self._action_queue.get_action_index() >= self._fire_after_n_actions
                )
            if not should_fire:
                time.sleep(0.005)
                continue

            with self._chunks_lock:
                if self._pending_chunks:
                    head = self._pending_chunks[0]
                    exec_chunk_id = head["chunk_id"]
                    exec_action_idx = head["chunk_len"] - head["remaining"]
                    exec_chunk_len = head["chunk_len"]
                else:
                    exec_chunk_id, exec_action_idx, exec_chunk_len = None, None, None

            t_fire = time.perf_counter()
            t_fire_rel_ms = (t_fire - self._t_start) * 1000 if self._t_start is not None else 0.0

            inference_request = self._build_request(obs)
            action_index_before_inference = (
                self._action_queue.get_action_index() if self._action_queue else None
            )
            try:
                self.ws.send(inference_request.model_dump_json())
                try:
                    response_text = self.ws.recv()
                except ConnectionClosed:
                    break
                payload = json.loads(response_text)
                if "error" in payload:
                    print(f"[inference thread] server error: {payload['error']}", flush=True)
                    continue
                response = InferenceResponse.model_validate(payload)
            except Exception as e:
                print(f"[inference thread] request/response error: {type(e).__name__}: {e}", flush=True)
                continue
            rtt_ms = (time.perf_counter() - t_fire) * 1000
            if rtt_ms / 1000 < self._warmup_latency_cutoff_s:
                self._latency_tracker.add(rtt_ms / 1000)

            real_delay = (
                self._action_queue.get_action_index() - action_index_before_inference
                if action_index_before_inference is not None
                else 0
            )
            processed_actions = torch.tensor(response.actions)
            original_actions = torch.tensor(response.original_actions)

            if self._synchronous_mode:
                clamped = 0
            else:
                clamped = max(0, min(real_delay, processed_actions.shape[0]))

            old_last_commanded = None
            old_chunk_id_snapshot = None
            with self._chunks_lock:
                if self._pending_chunks:
                    old_head = self._pending_chunks[0]
                    popped_so_far = old_head["chunk_len"] - old_head["remaining"]
                    if popped_so_far > 0:
                        old_last_commanded = old_head["actions"][popped_so_far - 1].clone()
                        old_chunk_id_snapshot = old_head["chunk_id"]

            with self._action_queue.lock:
                self._action_queue.queue = processed_actions[clamped:].clone()
                self._action_queue.original_queue = original_actions[clamped:].clone()
                self._action_queue.last_index = 0
            with self._chunks_lock:
                self._pending_chunks.clear()

            if old_last_commanded is not None and clamped < processed_actions.shape[0] and self._verbose:
                new_first = processed_actions[clamped]
                print(_LOG_SEPARATOR, flush=True)
                print(
                    f"[replace @ t={(time.perf_counter() - self._t_start) * 1000:.0f}ms]  "
                    f"chunk_{old_chunk_id_snapshot} → chunk_{self._chunk_counter + 1}  "
                    f"(real_delay={real_delay})",
                    flush=True,
                )
                print(
                    f"  last commanded (chunk_{old_chunk_id_snapshot}): "
                    f"{[f'{v:+.3f}' for v in old_last_commanded.tolist()]}",
                    flush=True,
                )
                print(
                    f"  first commanded (chunk_{self._chunk_counter + 1}): "
                    f"{[f'{v:+.3f}' for v in new_first.tolist()]}",
                    flush=True,
                )

            with self._chunks_lock:
                self._chunk_counter += 1
                new_chunk_id = self._chunk_counter
                queued_actions = processed_actions[clamped:].clone()
                chunk_len_int = queued_actions.shape[0]
                self._pending_chunks.append(
                    {
                        "chunk_id": new_chunk_id,
                        "remaining": chunk_len_int,
                        "chunk_len": chunk_len_int,
                        "fire_time_ms": t_fire_rel_ms,
                        "actions": queued_actions,
                    }
                )

            if self._verbose:
                exec_desc = (
                    f"executing chunk_{exec_chunk_id} action {exec_action_idx}/{exec_chunk_len}"
                    if exec_chunk_id is not None
                    else "queue empty"
                )
                print(_LOG_SEPARATOR, flush=True)
                print(
                    f"[fire chunk_{new_chunk_id} @ t={t_fire_rel_ms:.0f}ms]  rtt={rtt_ms:.0f}ms  {exec_desc}",
                    flush=True,
                )
