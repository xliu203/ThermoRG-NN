# SPDX-License-Identifier: Apache-2.0

"""Tests for profiler module."""

import torch
import pytest
import torch.nn as nn
from thermorg.analysis.profiler import (
    ArchitectureSpec,
    ProfilingResult,
    profile_architecture,
    predict_from_spec,
    compare_architectures,
)


def test_architecture_spec():
    """Test ArchitectureSpec dataclass."""
    spec = ArchitectureSpec(
        depth=4,
        width=512,
        hidden_dim=256,
        activation="relu",
    )
    assert spec.depth == 4
    assert spec.width == 512


def test_profile_architecture():
    """Test architecture profiling."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    x = torch.randn(1, 10)
    
    result = profile_architecture(model, x, manifold_dim_init=1.0)
    
    assert isinstance(result, ProfilingResult)
    assert len(result.compression_efficiencies) == 3
    assert len(result.manifold_dimensions) == 4
    assert result.predicted_performance >= 0


def test_predict_from_spec():
    """Test zero-shot prediction from spec."""
    spec = ArchitectureSpec(
        depth=4,
        width=256,
        hidden_dim=128,
    )
    pred = predict_from_spec(spec)
    assert pred >= 0


def test_compare_architectures():
    """Test architecture comparison."""
    specs = [
        ArchitectureSpec(depth=2, width=128, hidden_dim=64),
        ArchitectureSpec(depth=4, width=512, hidden_dim=256),
        ArchitectureSpec(depth=8, width=1024, hidden_dim=512),
    ]
    
    results = compare_architectures(specs)
    assert len(results) == 3
    # Results should be sorted by score (highest first)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
