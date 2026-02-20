"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_npc_edge_cases.py

"""

from __future__ import annotations

import torch

from pff.domain.learning.pc.npc import NeuralProbabilisticCircuit


def test_npc_single_attribute_runs_and_is_finite() -> None:
    """Execute test npc single attribute runs and is finite.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    npc = NeuralProbabilisticCircuit(num_attrs=1)
    attr_probs = torch.tensor(
        [[[0.7, 0.3]], [[0.2, 0.8]], [[0.9, 0.1]], [[0.5, 0.5]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    loss = npc(attr_probs, labels)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_npc_handles_empty_batch() -> None:
    """Execute test npc handles empty batch.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    npc = NeuralProbabilisticCircuit(num_attrs=1)
    attr_probs = torch.empty((0, 1, 2), dtype=torch.float32)
    labels = torch.empty((0,), dtype=torch.float32)

    loss = npc(attr_probs, labels)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_npc_sanitizes_nan_inputs() -> None:
    """Execute test npc sanitizes nan inputs.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    npc = NeuralProbabilisticCircuit(num_attrs=1)
    attr_probs = torch.tensor([[[float("nan"), 1.0]], [[0.0, float("nan")]]], dtype=torch.float32)
    labels = torch.tensor([1, 0], dtype=torch.float32)

    loss = npc(attr_probs, labels)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
