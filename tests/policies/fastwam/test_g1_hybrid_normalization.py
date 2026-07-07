#!/usr/bin/env python

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

import pytest
import torch

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.g1_hybrid_normalization import (
    G1HybridNormalizerProcessorStep,
    G1HybridUnnormalizerProcessorStep,
    apply_mean_std,
    apply_min_max,
    g1_hybrid_normalization_spec,
)
from lerobot.policies.fastwam.processor_fastwam import (
    make_fastwam_pre_post_processors,
    reconcile_fastwam_g1_processors,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    NormalizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_STATE


def _g1_stats():
    state_dim, action_dim = 32, 36
    return {
        OBS_STATE: {
            "min": torch.arange(state_dim, dtype=torch.float32),
            "max": torch.arange(state_dim, dtype=torch.float32) + 10.0,
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim),
        },
        ACTION: {
            "min": torch.arange(action_dim, dtype=torch.float32),
            "max": torch.arange(action_dim, dtype=torch.float32) + 10.0,
            "mean": torch.arange(action_dim, dtype=torch.float32) * 0.1,
            "std": torch.arange(action_dim, dtype=torch.float32) * 0.05 + 1.0,
        },
    }


def _g1_config(**kwargs) -> FastWAMConfig:
    config = FastWAMConfig(
        action_dim=36,
        proprio_dim=32,
        g1_hybrid_normalization=True,
        base_model_id=None,
        **kwargs,
    )
    config.pretrained_path = None
    return config


def test_apply_min_max_roundtrip():
    x = torch.tensor([0.0, 5.0, 10.0])
    min_val = torch.tensor([0.0, 0.0, 0.0])
    max_val = torch.tensor([10.0, 10.0, 10.0])
    normalized = apply_min_max(x, min_val=min_val, max_val=max_val, eps=1e-8, inverse=False)
    assert torch.allclose(normalized, torch.tensor([-1.0, 0.0, 1.0]))
    restored = apply_min_max(normalized, min_val=min_val, max_val=max_val, eps=1e-8, inverse=True)
    assert torch.allclose(restored, x)


def test_g1_hybrid_action_uses_mixed_modes():
    stats = _g1_stats()
    spec = g1_hybrid_normalization_spec(proprio_dim=32, action_dim=36, action_min_max_end=32)
    normalizer = G1HybridNormalizerProcessorStep(stats=stats, spec=spec, device="cpu")
    unnormalizer = G1HybridUnnormalizerProcessorStep(stats=stats, spec=spec, device="cpu")

    raw_action = torch.zeros(36)
    raw_action[:32] = stats[ACTION]["min"][:32] + 5.0
    raw_action[32:] = stats[ACTION]["mean"][32:] + stats[ACTION]["std"][32:] * 2.0

    normalized = normalizer._normalize_action_tensor(raw_action, inverse=False)
    assert torch.allclose(normalized[:32], torch.zeros(32))
    assert torch.allclose(normalized[32:], torch.full((4,), 2.0))

    restored = unnormalizer._normalize_action_tensor(normalized, inverse=True)
    assert torch.allclose(restored, raw_action)


def test_make_fastwam_processors_use_g1_hybrid_steps():
    config = _g1_config()
    preprocessor, postprocessor = make_fastwam_pre_post_processors(config, dataset_stats=_g1_stats())
    assert isinstance(preprocessor.steps[-1], G1HybridNormalizerProcessorStep)
    assert isinstance(postprocessor.steps[0], G1HybridUnnormalizerProcessorStep)


def test_reconcile_replaces_standard_normalizers():
    config = _g1_config()
    standard_config = FastWAMConfig(action_dim=36, proprio_dim=32, base_model_id=None)
    standard_config.pretrained_path = None
    preprocessor, postprocessor = make_fastwam_pre_post_processors(standard_config, dataset_stats=_g1_stats())
    assert isinstance(preprocessor.steps[-1], NormalizerProcessorStep)

    preprocessor, postprocessor = reconcile_fastwam_g1_processors(
        config,
        preprocessor,
        postprocessor,
        dataset_stats=_g1_stats(),
    )
    assert isinstance(preprocessor.steps[-1], G1HybridNormalizerProcessorStep)
    assert isinstance(postprocessor.steps[0], G1HybridUnnormalizerProcessorStep)


def test_end_to_end_transition_roundtrip():
    config = _g1_config()
    stats = _g1_stats()
    preprocessor, postprocessor = make_fastwam_pre_post_processors(config, dataset_stats=stats)

    batch = {
        OBS_STATE: stats[OBS_STATE]["min"] + 3.0,
        ACTION: stats[ACTION]["min"] + 3.0,
    }
    transition = create_transition(observation=batch, action=batch[ACTION])
    transition = preprocessor.steps[1](transition)  # AddBatchDimensionProcessorStep
    transition = preprocessor.steps[-1](transition)

    normalized_state = transition[TransitionKey.OBSERVATION][OBS_STATE]
    normalized_action = transition[TransitionKey.ACTION]
    assert normalized_state.shape == (1, 32)
    assert normalized_action.shape == (1, 36)

    action_transition = create_transition(action=normalized_action.squeeze(0))
    action_transition = postprocessor.steps[0](action_transition)
    restored_action = action_transition[TransitionKey.ACTION]
    assert torch.allclose(restored_action, batch[ACTION], atol=1e-5)


def test_g1_hybrid_requires_proprio_dim():
    with pytest.raises(ValueError, match="requires `proprio_dim`"):
        FastWAMConfig(action_dim=36, proprio_dim=None, g1_hybrid_normalization=True, base_model_id=None)
