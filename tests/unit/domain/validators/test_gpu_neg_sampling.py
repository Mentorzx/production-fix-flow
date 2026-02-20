"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_gpu_neg_sampling.py

"""

from __future__ import annotations

import torch
import pytest
from pff.domain.learning.dslfm.neg_sampling import DegreeBasedSampler, SamplerConfig


def test_degree_based_sampler_gpu_distribution() -> None:
    """Regression test: Ensures multinomial sampling on GPU respects weights."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    num_entities = 10
    # Weights: entity 0 is 100x more likely than others
    degrees = torch.ones(num_entities, device=device)
    degrees[0] = 100.0

    config = SamplerConfig(num_entities=num_entities, alpha=1.0)
    sampler = DegreeBasedSampler(config, entity_degrees=degrees)

    heads = torch.zeros(1, device=device).long()
    rels = torch.zeros(1, device=device).long()
    tails = torch.zeros(1, device=device).long()

    # Sample many times
    num_negatives = 1000
    samples = sampler.sample_negatives(heads, rels, tails, num_negatives)

    # Count occurrences of entity 0
    count_0 = (samples == 0).sum().item()

    # Expectation: ~90% should be entity 0
    assert count_0 > 800
