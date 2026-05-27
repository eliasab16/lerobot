from enum import Enum

from pydantic import BaseModel


class SetupStatus(str, Enum):
    READY = "ready"
    ERROR = "error"


class SetupRequest(BaseModel):
    policy_path: str
    action_dim: int = 8
    camera_names: list[str]
    task: str
    device: str = "cuda"
    compile_model: bool = False


class SetupResponse(BaseModel):
    status: SetupStatus
    message: str
    chunk_size: int | None = None


class InferenceRequest(BaseModel):
    state: list[float]
    images: dict[str, str]  # camera name -> base64 JPEG
    task: str
    timestamp: float


# RTC needs the leftover from the previous chunk so the action expert can
# align its denoising trajectory; see ActionQueue.merge() in policies/rtc/.
class InferenceRTCRequest(InferenceRequest):
    prev_chunk_left_over: list[list[float]] | None = None
    inference_delay: int | None = None
    execution_horizon: int = 20


# Two action lists because RTC's leftover tracker needs pre-postprocessor
# outputs (raw model action space) for its guidance math, while the robot
# consumes the post-processed values.
class InferenceResponse(BaseModel):
    actions: list[list[float]]
    original_actions: list[list[float]]
    inference_time_ms: float


class HealthResponse(BaseModel):
    ready: bool
    device: str
    uptime_s: float
