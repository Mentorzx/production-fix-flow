"""Tests for PC2 (Probabilistic Circuit Variant 2) implementation.

Tests the PC2 approach from Knowledge Graph Completion research (Patil et al.)
which provides exact probability computation through tractable circuit properties.
"""

from __future__ import annotations

import numpy as np
import torch

from pff.domain.learning.ml.aggregation_strategies import NoisyOrStrategy
from pff.domain.learning.pc.compiler import RuleToCircuitCompiler
from pff.domain.learning.pc.inference import PCInferenceEngine
from pff.domain.learning.pc.npc import (
    PC2,
    CircuitProperties,
    NeuralProbabilisticCircuit,
)


def test_circuit_tractability_flags() -> None:
    """Test that PC2 returns correct tractability properties for exact inference."""
    npc = NeuralProbabilisticCircuit(
        num_attrs=3,
        pruning_threshold=0.25,
        grow_noise=0.0,
        max_depth=3,
    )
    attr_probs = torch.tensor(
        [
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
            [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8]],
            [[0.9, 0.1], [0.5, 0.5], [0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    loss = npc(attr_probs, labels)
    props = npc.properties()

    assert loss.item() > 0.0
    # PC2 properties for exact inference
    assert props.smooth, "PC2 must be smooth for exact inference"
    assert props.decomposable, "PC2 must be decomposable for exact inference"
    assert props.exact_inference, "PC2 must support exact inference"
    assert props.max_depth <= 3


def test_pc2_alias() -> None:
    """Test that PC2 alias works correctly."""
    # PC2 is the official name from KGC research
    assert PC2 is NeuralProbabilisticCircuit

    pc2 = PC2(num_attrs=4)
    props = pc2.properties()
    assert props.exact_inference


def test_pc2_exact_inference_consistency() -> None:
    """Test that PC2 produces consistent results (exact, no randomness in inference)."""
    pc = NeuralProbabilisticCircuit(num_attrs=3, grow_noise=0.0)

    attr_probs = torch.tensor(
        [[[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([1.0], dtype=torch.float32)

    # Multiple forward passes should give identical results (exact inference)
    loss1 = pc(attr_probs, labels)
    loss2 = pc(attr_probs, labels)
    loss3 = pc(attr_probs, labels)

    assert torch.allclose(loss1, loss2), "PC2 exact inference must be deterministic"
    assert torch.allclose(loss2, loss3), "PC2 exact inference must be deterministic"


def test_pc2_properties_dataclass() -> None:
    """Test CircuitProperties dataclass for PC2."""
    props = CircuitProperties(
        smooth=True,
        decomposable=True,
        exact_inference=True,
        max_depth=3,
    )

    assert props.smooth
    assert props.decomposable
    assert props.exact_inference
    assert props.max_depth == 3


def test_pc_matches_noisy_or_when_independent() -> None:
    """Test that compiled circuit matches Noisy-OR baseline for independent rules."""
    confidences = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    compiler = RuleToCircuitCompiler(
        max_rules_per_circuit=10, cache_compiled_circuits=False
    )
    circuit = compiler.compile(rule_count=len(confidences))

    engine = PCInferenceEngine(normalize_weights=True)
    pc_score = engine.infer(circuit, confidences)

    noisy_or = NoisyOrStrategy()
    baseline = noisy_or.aggregate(confidences)

    assert np.isclose(pc_score, baseline, atol=1e-6)


def test_npc_prunes_low_flow_edges() -> None:
    """Test that PC2 correctly prunes edges with low flow."""
    npc = NeuralProbabilisticCircuit(
        num_attrs=4,
        pruning_threshold=0.4,
        grow_noise=0.0,
        max_depth=3,
    )
    base = torch.full((6, 4), 0.1, dtype=torch.float32)
    attr_probs = torch.stack([base, 1.0 - base], dim=-1)
    labels = torch.ones(6, dtype=torch.float32)

    npc(attr_probs, labels)

    for idx in range(1, npc.num_attrs):
        assert npc.parents[idx] == npc.root


def test_npc_rebuild_preserves_exact_inference() -> None:
    """Test that rebuild maintains PC2 exact inference properties."""
    npc = NeuralProbabilisticCircuit(num_attrs=4)

    # Rebuild with importance scores
    mi_scores = torch.tensor([0.1, 0.5, 0.3, 0.8])
    npc.rebuild(mi_scores)

    # Properties should still hold after rebuild
    props = npc.properties()
    assert props.smooth
    assert props.decomposable
    assert props.exact_inference


def test_pc2_gradient_noise() -> None:
    """Test gradient noise application for growth."""
    npc = NeuralProbabilisticCircuit(num_attrs=3, grow_noise=0.1)

    # Get initial parameters
    initial_params = [p.clone() for p in npc.parameters()]

    # Apply noise
    npc.gradient_noise()

    # Parameters should have changed
    changed = False
    for initial, current in zip(initial_params, npc.parameters()):
        if not torch.allclose(initial, current):
            changed = True
            break

    assert changed, "Gradient noise should modify parameters"


def test_pc2_zero_grow_noise() -> None:
    """Test that zero grow_noise doesn't modify parameters."""
    npc = NeuralProbabilisticCircuit(num_attrs=3, grow_noise=0.0)

    # Get initial parameters
    initial_params = [p.clone() for p in npc.parameters()]

    # Apply noise (should do nothing)
    npc.gradient_noise()

    # Parameters should be unchanged
    for initial, current in zip(initial_params, npc.parameters()):
        assert torch.allclose(initial, current)
