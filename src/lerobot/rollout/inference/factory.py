# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|remote``.  Adding a
new backend requires registering its config subclass and dispatching it
in :func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from threading import Event

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .remote import RemoteInferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


@InferenceEngineConfig.register_subclass("remote")
@dataclass
class RemoteInferenceConfig(InferenceEngineConfig):
    """WebSocket-backed remote inference against a ``lerobot-policy-server``."""

    server_url: str = "ws://localhost:8000"
    # Device the *server* runs the policy on — independent of the rollout
    # machine's device. Default cuda; override only if the server box is CPU/MPS.
    server_device: str = "cuda"
    rtc: RTCConfig = field(default_factory=RTCConfig)
    # Required (or set synchronous_mode=True). Validated at engine start.
    fire_after_n_actions: int | None = None
    synchronous_mode: bool = False
    log_actions_csv: str | None = None
    verbose_transitions: bool = True
    # When True (default), skip loading the policy weights client-side and use
    # a config-only stub in the rollout context. Saves ~60s of safetensors +
    # model-construction time on the rollout box. Set False if some local
    # consumer needs the actual policy object (e.g. action-space introspection).
    skip_local_policy_load: bool = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    pretrained_path: str | None = None,
    rename_map: dict[str, str] | None = None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    if isinstance(config, RemoteInferenceConfig):
        if not config.synchronous_mode and config.fire_after_n_actions is None:
            raise ValueError(
                "RemoteInferenceConfig: set --inference.fire_after_n_actions=<N> "
                "or --inference.synchronous_mode=true."
            )
        if pretrained_path is None:
            raise ValueError(
                "RemoteInferenceConfig requires pretrained_path (forwarded by build_rollout_context)."
            )
        # The server expects post-rename keys (matches what the policy was
        # trained on); the client applies the rename before sending.
        raw_cam_names = [
            k.removeprefix("observation.images.")
            for k in hw_features
            if k.startswith("observation.images.")
        ]
        camera_names = []
        for raw in raw_cam_names:
            raw_key = f"observation.images.{raw}"
            renamed = (rename_map or {}).get(raw_key, raw_key)
            camera_names.append(renamed.removeprefix("observation.images."))
        return RemoteInferenceEngine(
            policy_path=pretrained_path,
            server_url=config.server_url,
            fps=fps,
            rtc_config=config.rtc,
            action_dim=len(ordered_action_keys),
            camera_names=camera_names,
            task=task,
            device=config.server_device,
            fire_after_n_actions=config.fire_after_n_actions,
            synchronous_mode=config.synchronous_mode,
            dataset_features=dataset_features,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
            rename_map=rename_map,
            log_actions_csv=config.log_actions_csv,
            joint_names=ordered_action_keys,
            verbose_transitions=config.verbose_transitions,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
