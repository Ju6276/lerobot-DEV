# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""G1 whole-body hybrid normalization for FastWAM.

Matches the original G1 teleop training pipeline:
  - ``observation.state`` (32-d): min_max on every dimension
  - ``action`` dims [0, 32): min_max (hands, arms, rpy, height)
  - ``action`` dims [32, 36): mean_std (torso_vx, torso_vy, torso_vyaw, target_yaw)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.converters import from_tensor_to_numpy, to_tensor
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, PolicyAction, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

G1_DEFAULT_STATE_DIM = 32
G1_DEFAULT_ACTION_DIM = 36
G1_DEFAULT_ACTION_MIN_MAX_END = 32


def _broadcast_stat(stat: Tensor, tensor: Tensor) -> Tensor:
    """Broadcast a 1D per-dimension stat to match ``tensor`` on the last axis."""
    while stat.ndim < tensor.ndim:
        stat = stat.unsqueeze(0)
    return stat


def apply_min_max(
    tensor: Tensor,
    *,
    min_val: Tensor,
    max_val: Tensor,
    eps: float,
    inverse: bool,
) -> Tensor:
    min_val = _broadcast_stat(min_val, tensor)
    max_val = _broadcast_stat(max_val, tensor)
    denom = max_val - min_val
    denom = torch.where(denom == 0, torch.tensor(eps, device=tensor.device, dtype=tensor.dtype), denom)
    if inverse:
        return (tensor + 1) / 2 * denom + min_val
    return 2 * (tensor - min_val) / denom - 1


def apply_mean_std(
    tensor: Tensor,
    *,
    mean: Tensor,
    std: Tensor,
    eps: float,
    inverse: bool,
) -> Tensor:
    mean = _broadcast_stat(mean, tensor)
    std = _broadcast_stat(std, tensor)
    if inverse:
        return tensor * std + mean
    return (tensor - mean) / (std + eps)


@dataclass
class G1HybridNormalizationSpec:
    """Slice-wise normalization spec for G1 state/action vectors."""

    state_dim: int = G1_DEFAULT_STATE_DIM
    action_dim: int = G1_DEFAULT_ACTION_DIM
    action_min_max_end: int = G1_DEFAULT_ACTION_MIN_MAX_END

    def validate(self) -> None:
        if self.state_dim <= 0:
            raise ValueError(f"`state_dim` must be positive, got {self.state_dim}.")
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}.")
        if not 0 < self.action_min_max_end < self.action_dim:
            raise ValueError(
                f"`action_min_max_end` must satisfy 0 < end < action_dim, "
                f"got end={self.action_min_max_end}, action_dim={self.action_dim}."
            )


@dataclass
class _G1HybridNormalizationMixin:
    stats: dict[str, dict[str, Any]] | None = None
    spec: G1HybridNormalizationSpec = field(default_factory=G1HybridNormalizationSpec)
    state_key: str = OBS_STATE
    action_key: str = ACTION
    eps: float = 1e-8
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.spec.validate()
        self._stats_explicitly_provided = self.stats is not None and bool(self.stats)
        self.stats = self.stats or {}
        if self.dtype is None:
            self.dtype = torch.float32
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        self._validate_stats()

    def _validate_stats(self) -> None:
        for key, dim in ((self.state_key, self.spec.state_dim), (self.action_key, self.spec.action_dim)):
            if key not in self._tensor_stats:
                raise KeyError(f"Missing stats for `{key}` required by G1 hybrid normalization.")
            feature_stats = self._tensor_stats[key]
            for stat_name in ("min", "max", "mean", "std"):
                if stat_name not in feature_stats:
                    raise KeyError(
                        f"Stats for `{key}` must include `{stat_name}` for G1 hybrid normalization."
                    )
                if int(feature_stats[stat_name].shape[0]) != dim:
                    raise ValueError(
                        f"Stats for `{key}.{stat_name}` must have length {dim}, "
                        f"got {int(feature_stats[stat_name].shape[0])}."
                    )

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _G1HybridNormalizationMixin:
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        return self

    def state_dict(self) -> dict[str, Tensor]:
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if self._stats_explicitly_provided and self.stats is not None:
            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
            return
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )
        self.stats = {}
        for key, tensor_dict in self._tensor_stats.items():
            self.stats[key] = {stat_name: from_tensor_to_numpy(tensor) for stat_name, tensor in tensor_dict.items()}

    def get_config(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "spec": {
                "state_dim": self.spec.state_dim,
                "action_dim": self.spec.action_dim,
                "action_min_max_end": self.spec.action_min_max_end,
            },
        }

    def _ensure_stats_on_device(self, tensor: Tensor) -> None:
        if not self._tensor_stats:
            return
        first_stat = next(iter(next(iter(self._tensor_stats.values())).values()))
        if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
            self.to(device=tensor.device, dtype=tensor.dtype)

    def _normalize_state_tensor(self, tensor: Tensor, *, inverse: bool) -> Tensor:
        self._ensure_stats_on_device(tensor)
        stats = self._tensor_stats[self.state_key]
        return apply_min_max(
            tensor,
            min_val=stats["min"],
            max_val=stats["max"],
            eps=self.eps,
            inverse=inverse,
        )

    def _normalize_action_tensor(self, tensor: Tensor, *, inverse: bool) -> Tensor:
        self._ensure_stats_on_device(tensor)
        stats = self._tensor_stats[self.action_key]
        end = self.spec.action_min_max_end
        normalized = tensor.clone()
        normalized[..., :end] = apply_min_max(
            tensor[..., :end],
            min_val=stats["min"][:end],
            max_val=stats["max"][:end],
            eps=self.eps,
            inverse=inverse,
        )
        normalized[..., end:] = apply_mean_std(
            tensor[..., end:],
            mean=stats["mean"][end:],
            std=stats["std"][end:],
            eps=self.eps,
            inverse=inverse,
        )
        return normalized


@dataclass
@ProcessorStepRegistry.register(name="g1_hybrid_normalizer_processor")
class G1HybridNormalizerProcessorStep(_G1HybridNormalizationMixin, ProcessorStep):
    """Normalize G1 state/action with mixed min_max and mean_std slices."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_observation = dict(observation)
            if self.state_key in new_observation:
                tensor = torch.as_tensor(new_observation[self.state_key])
                new_observation[self.state_key] = self._normalize_state_tensor(tensor, inverse=False)
            new_transition[TransitionKey.OBSERVATION] = new_observation

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, PolicyAction):
                raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
            new_transition[TransitionKey.ACTION] = self._normalize_action_tensor(action, inverse=False)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="g1_hybrid_unnormalizer_processor")
class G1HybridUnnormalizerProcessorStep(_G1HybridNormalizationMixin, ProcessorStep):
    """Unnormalize G1 actions back to the original robot command space."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
        new_transition[TransitionKey.ACTION] = self._normalize_action_tensor(action, inverse=True)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def g1_hybrid_normalization_spec(
    *,
    proprio_dim: int,
    action_dim: int,
    action_min_max_end: int | None = None,
) -> G1HybridNormalizationSpec:
    end = action_min_max_end if action_min_max_end is not None else action_dim - 4
    return G1HybridNormalizationSpec(
        state_dim=proprio_dim,
        action_dim=action_dim,
        action_min_max_end=end,
    )


def make_g1_hybrid_normalizer_step(
    *,
    stats: dict[str, dict[str, Any]] | None,
    spec: G1HybridNormalizationSpec,
    device: torch.device | str | None,
) -> G1HybridNormalizerProcessorStep:
    return G1HybridNormalizerProcessorStep(stats=stats, spec=spec, device=device)


def make_g1_hybrid_unnormalizer_step(
    *,
    stats: dict[str, dict[str, Any]] | None,
    spec: G1HybridNormalizationSpec,
    device: torch.device | str | None,
) -> G1HybridUnnormalizerProcessorStep:
    return G1HybridUnnormalizerProcessorStep(stats=stats, spec=spec, device=device)


def reconcile_fastwam_g1_processors(
    config,
    preprocessor,
    postprocessor,
    *,
    dataset_stats: dict[str, dict[str, Any]] | None,
):
    """Replace standard normalizers with G1 hybrid steps when enabled on the config."""
    if not getattr(config, "g1_hybrid_normalization", False):
        return preprocessor, postprocessor

    from .configuration_fastwam import FastWAMConfig

    if not isinstance(config, FastWAMConfig):
        return preprocessor, postprocessor

    spec = g1_hybrid_normalization_spec(
        proprio_dim=int(config.proprio_dim or G1_DEFAULT_STATE_DIM),
        action_dim=int(config.action_dim),
        action_min_max_end=config.g1_action_min_max_end,
    )
    stats = dict(dataset_stats or {})

    pre_steps = list(preprocessor.steps)
    for idx, step in enumerate(pre_steps):
        if isinstance(step, (NormalizerProcessorStep, G1HybridNormalizerProcessorStep)):
            pre_steps[idx] = make_g1_hybrid_normalizer_step(
                stats=stats,
                spec=spec,
                device=config.device,
            )
            break
    else:
        device_idx = next(
            (idx for idx, step in enumerate(pre_steps) if step.__class__.__name__ == "DeviceProcessorStep"),
            len(pre_steps),
        )
        pre_steps.insert(
            device_idx + 1,
            make_g1_hybrid_normalizer_step(stats=stats, spec=spec, device=config.device),
        )
    preprocessor.steps = pre_steps

    post_steps = list(postprocessor.steps)
    for idx, step in enumerate(post_steps):
        if isinstance(step, (UnnormalizerProcessorStep, G1HybridUnnormalizerProcessorStep)):
            post_steps[idx] = make_g1_hybrid_unnormalizer_step(
                stats=stats,
                spec=spec,
                device=config.device,
            )
            break
    else:
        post_steps.insert(
            0,
            make_g1_hybrid_unnormalizer_step(stats=stats, spec=spec, device=config.device),
        )
    postprocessor.steps = post_steps

    return preprocessor, postprocessor
