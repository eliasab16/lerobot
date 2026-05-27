"""FastAPI WebSocket inference server.

Loads any lerobot ``PreTrainedPolicy`` by reading the policy config off
its Hub/path and dispatching through ``get_policy_class``. The setup
endpoint hot-swaps policies (releases the previous one before loading)
so a single server process can serve different checkpoints over its
lifetime.

Endpoints:
    GET  /health   liveness + which policy is loaded
    POST /setup    load / swap a policy
    WS   /ws       one InferenceRequest -> one InferenceResponse per turn
"""

from __future__ import annotations

import gc
import json
import time
import traceback

import torch
from fastapi import FastAPI, WebSocket

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from .image_codec import decode_image_to_tensor
from .protocol import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    InferenceRTCRequest,
    SetupRequest,
    SetupResponse,
    SetupStatus,
)


class InferenceServer:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        self.loaded_policy_path = None

    @property
    def chunk_size(self) -> int | None:
        if self.policy is None:
            return None
        return self.policy.config.chunk_size

    @property
    def ready(self) -> bool:
        return all(
            [
                self.policy is not None,
                self.preprocessor is not None,
                self.postprocessor is not None,
                self.device is not None,
            ]
        )

    @property
    def uptime_s(self) -> float:
        return time.perf_counter() - self.start_time

    def setup(self, request: SetupRequest) -> SetupResponse:
        if (
            self.ready
            and self.loaded_policy_path == request.policy_path
            and self.device == request.device
        ):
            return SetupResponse(
                status=SetupStatus.READY,
                message=f"Policy already loaded: {request.policy_path} on {request.device}",
                chunk_size=self.chunk_size,
            )

        try:
            # Swapping models OOMs if both reside on the GPU briefly.
            if self.policy is not None:
                del self.policy
                del self.preprocessor
                del self.postprocessor
                self.policy = None
                self.preprocessor = None
                self.postprocessor = None
                self.loaded_policy_path = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print("Released previous policy; VRAM cleared for new load.")

            policy_config = PreTrainedConfig.from_pretrained(request.policy_path)
            # Must be set *before* construction — the policy enables torch.compile
            # inside __init__ based on this flag; setting it after has no effect.
            policy_config.compile_model = request.compile_model
            if request.compile_model and getattr(policy_config, "compile_mode", None) == "max-autotune":
                # SmolVLM2's vision embeddings use in-place index_put_ which CUDA
                # Graphs disallow. Keep the autotuned kernels, drop the graph capture.
                policy_config.compile_mode = "max-autotune-no-cudagraphs"
            policy_class = get_policy_class(policy_config.type)
            policy = policy_class.from_pretrained(request.policy_path, config=policy_config)
            policy.to(request.device)
            policy.eval()

            device_override = {"device": request.device}
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                pretrained_path=request.policy_path,
                preprocessor_overrides={"device_processor": device_override},
                postprocessor_overrides={"device_processor": device_override},
            )

            try:
                self._warmup(
                    policy, preprocessor, request.camera_names, request.action_dim, request.device
                )
            except Exception as warmup_err:
                traceback.print_exc()
                return SetupResponse(
                    status=SetupStatus.ERROR,
                    message=f"Warmup failed: {type(warmup_err).__name__}: {warmup_err}",
                )

            self.policy = policy
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.device = request.device
            self.loaded_policy_path = request.policy_path

            return SetupResponse(
                status=SetupStatus.READY,
                message=f"Loaded {request.policy_path} ({policy_config.type}) on {request.device}",
                chunk_size=self.chunk_size,
            )
        except Exception as e:
            traceback.print_exc()
            return SetupResponse(status=SetupStatus.ERROR, message=f"{type(e).__name__}: {e}")

    def _warmup(self, policy, preprocessor, camera_names, action_dim, device):
        """Compile RTC + non-RTC kernel paths so the first real request is fast.

        Without this, the first /infer eats a ~20s kernel-compile hit which
        drains the client's action queue during warmup. Policies without
        ``predict_action_chunk`` RTC kwargs (e.g. ACT) skip the RTC branch.
        """
        # Use the robot's actual dim, not policy.config.max_*_dim — normalizer
        # stats are sized to the real dim, so passing the padded max breaks
        # broadcasting.
        state_dim = action_dim
        chunk_size = policy.config.chunk_size

        # Aspect-preserving, non-square so resize_with_pad runs (matches what
        # real requests look like).
        obs = {
            "observation.state": torch.zeros(1, state_dim, dtype=torch.float32).to(device),
            "task": "warmup",
            "robot_type": "",
        }
        for cam_name in camera_names:
            obs[f"observation.images.{cam_name}"] = torch.zeros(
                1, 3, 168, 224, dtype=torch.float32
            ).to(device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            processed_obs = preprocessor(obs)
            _ = policy.predict_action_chunk(processed_obs)

            try:
                fake_leftover = torch.zeros(
                    1, chunk_size - 5, action_dim, dtype=torch.float32
                ).to(device)
                _ = policy.predict_action_chunk(
                    processed_obs,
                    prev_chunk_left_over=fake_leftover,
                    inference_delay=5,
                    execution_horizon=20,
                )
            except TypeError:
                # Policy doesn't accept RTC kwargs (e.g. ACT). Non-RTC warmup
                # is sufficient.
                pass

        policy.reset()
        elapsed_s = time.perf_counter() - t0
        print(f"Warmup complete ({elapsed_s:.1f}s).")

    def inference(self, request: InferenceRequest | InferenceRTCRequest) -> InferenceResponse:
        obs = {
            "observation.state": torch.tensor(request.state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device),
            "task": request.task,
            "robot_type": "",
        }
        for cam_name, img_b64 in request.images.items():
            obs[f"observation.images.{cam_name}"] = decode_image_to_tensor(img_b64).to(self.device)

        rtc_kwargs = {}
        if isinstance(request, InferenceRTCRequest) and request.prev_chunk_left_over is not None:
            rtc_kwargs["prev_chunk_left_over"] = (
                torch.tensor(request.prev_chunk_left_over, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            rtc_kwargs["inference_delay"] = request.inference_delay
            rtc_kwargs["execution_horizon"] = request.execution_horizon

        with torch.inference_mode():
            obs = self.preprocessor(obs)
            t0 = time.perf_counter()
            action_chunk = self.policy.predict_action_chunk(obs, **rtc_kwargs)
            inference_time_ms = (time.perf_counter() - t0) * 1000

            original_actions = action_chunk.squeeze(0).detach().cpu()

            _, chunk_size, _ = action_chunk.shape
            processed_actions = []
            for i in range(chunk_size):
                processed_actions.append(self.postprocessor(action_chunk[:, i, :]))
            processed_tensor = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu()

        return InferenceResponse(
            actions=processed_tensor.tolist(),
            original_actions=original_actions.tolist(),
            inference_time_ms=inference_time_ms,
        )


app = FastAPI(title="LeRobot Remote Inference Server")
server = InferenceServer()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ready=server.ready,
        device=server.device or "none",
        uptime_s=server.uptime_s,
    )


@app.post("/setup", response_model=SetupResponse)
async def setup(req: SetupRequest) -> SetupResponse:
    return server.setup(req)


@app.websocket("/ws")
async def websocket_inference(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                if "prev_chunk_left_over" in payload:
                    request = InferenceRTCRequest.model_validate(payload)
                else:
                    request = InferenceRequest.model_validate(payload)
                response = server.inference(request)
                await ws.send_text(response.model_dump_json())
            except Exception as e:
                traceback.print_exc()
                await ws.send_text(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    except Exception:
        traceback.print_exc()
