"""Remote WebSocket inference engine.

Adapts :class:`lerobot.rollout.remote.client.RemoteInferenceClient` to the
``InferenceEngine`` interface.  The local pre/postprocessors that the
factory passes in are *unused*: the server applies both pipelines, the
client sends raw camera frames and joint state, and the server returns
post-processed actions ready for the robot.

Caveat: ``build_rollout_context`` still loads ``--policy.path`` locally to
satisfy the engine factory signature.  On a CPU-only robot box this
download/load is wasted work, not catastrophic — use a lightweight
checkpoint path or accept the local load.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.policies.utils import make_robot_action
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.feature_utils import build_dataset_frame

from ..remote.client import RemoteInferenceClient
from .base import InferenceEngine

logger = logging.getLogger(__name__)


class RemoteInferenceEngine(InferenceEngine):
    """WebSocket-backed inference engine.

    ``notify_observation`` ships the latest raw obs to a background
    thread; ``get_action`` pops from the local action queue populated
    by that thread when the server responds.
    """

    def __init__(
        self,
        *,
        policy_path: str,
        server_url: str,
        fps: float,
        rtc_config: RTCConfig | None,
        action_dim: int,
        camera_names: list[str],
        task: str,
        device: str,
        fire_after_n_actions: int | None,
        synchronous_mode: bool,
        dataset_features: dict,
        hw_features: dict,
        ordered_action_keys: list[str],
        rename_map: dict[str, str] | None = None,
        log_actions_csv: str | None = None,
        joint_names: list[str] | None = None,
        verbose_transitions: bool = True,
    ) -> None:
        self._policy_path = policy_path
        self._action_dim = action_dim
        self._camera_names = camera_names
        self._task = task
        self._device = device
        self._dataset_features = dataset_features
        self._hw_features = hw_features
        self._ordered_action_keys = ordered_action_keys
        self._rename_map = rename_map or {}
        self._client = RemoteInferenceClient(
            server_url=server_url,
            fps=fps,
            rtc_config=rtc_config,
            fire_after_n_actions=fire_after_n_actions,
            synchronous_mode=synchronous_mode,
            log_actions_csv=log_actions_csv,
            joint_names=joint_names,
            verbose_transitions=verbose_transitions,
        )

    def start(self) -> None:
        logger.info("RemoteInferenceEngine: setup against %s", self._client.server_url)
        self._client.setup(
            policy_path=self._policy_path,
            action_dim=self._action_dim,
            camera_names=self._camera_names,
            task=self._task,
            device=self._device,
        )
        self._client.start()
        logger.info("RemoteInferenceEngine started (chunk_size=%s)", self._client.chunk_size)

    def stop(self) -> None:
        self._client.stop()
        logger.info("RemoteInferenceEngine stopped")

    def reset(self) -> None:
        self._client.clear_queue()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        action = self._client.get_action()
        if action is None:
            return None
        # Reorder server actions to match dataset action ordering so the
        # caller can treat the returned tensor uniformly across backends.
        action_dict = make_robot_action(action, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    def notify_observation(self, obs: dict) -> None:
        # obs from robot_observation_processor has per-motor keys (<motor>.pos)
        # and raw camera-name keys. Use build_dataset_frame to assemble it into
        # observation.state + observation.images.<raw_name>, then apply
        # rename_map so the server receives the keys the policy was trained on.
        frame = build_dataset_frame(self._hw_features, obs, prefix="observation")

        state = np.asarray(frame["observation.state"]).flatten().tolist()

        images = {}
        for k, v in frame.items():
            if not k.startswith("observation.images."):
                continue
            renamed = self._rename_map.get(k, k)
            cam_name = renamed.removeprefix("observation.images.")
            images[cam_name] = self._to_bgr_uint8(v)

        self._client.update_observation(state, images, self._task)

    @property
    def ready(self) -> bool:
        return self._client._action_queue is not None

    @staticmethod
    def _to_bgr_uint8(img) -> np.ndarray:
        """Coerce a rollout obs image to the HxWxC uint8 form encode_image expects.

        Handles batched tensors (1, C, H, W), float [0,1] normalized images, and
        already-uint8 numpy arrays. Color order is left as-is — most lerobot
        cameras emit BGR raw, and the server applies the policy's own image
        preprocessor on top.
        """
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            scale = 255.0 if arr.max() <= 1.0 else 1.0
            arr = (arr * scale).clip(0, 255).astype(np.uint8)
        return arr
