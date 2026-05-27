"""WebSocket-based remote inference for lerobot-rollout.

Two pieces:

- ``server.py`` runs on a GPU box, exposes ``/setup`` (HTTP) + ``/ws`` (WebSocket)
  via FastAPI, and runs the policy forward pass on each inference request.
- ``client.py`` runs on the robot machine alongside ``lerobot-rollout`` and
  ships observations over the WebSocket in a background thread.

Both sides apply the policy's pre/postprocessor *server-side*. The client
sends raw camera images (base64 JPEG) and joint state; the server returns
processed actions ready to send to the robot, plus the raw model outputs
needed for RTC alignment.

No auth is implemented. Deploy behind a VPN or SSH tunnel.
"""

from .protocol import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    InferenceRTCRequest,
    SetupRequest,
    SetupResponse,
    SetupStatus,
)

__all__ = [
    "HealthResponse",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceRTCRequest",
    "SetupRequest",
    "SetupResponse",
    "SetupStatus",
]
