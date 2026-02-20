"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/bench/test_dslfm_eval_bench.py

"""

import time

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.shared.acceleration.triton_kernels import TRITON_AVAILABLE


@pytest.mark.skipif(
    not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton/GPU required"
)
def test_dslfm_eval_integration():
    """Execute test dslfm eval integration.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    device = torch.device("cuda")
    num_entities = 10_000
    config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=5,
        entity_dim=128,
        feature_dim=128,
        max_communities=8,
        triton_min_entities=100,
    )
    model = DSLFMKGCModel(config).to(device)
    model.eval()

    # Precompute mock latents
    model.precompute_entity_latents(batch_size=512)

    # Mock triples
    eval_triples = torch.randint(0, num_entities, (512, 3)).to(device)
    eval_triples[:, 1] = 0

    torch.cuda.synchronize()
    start = time.perf_counter()
    metrics = model.evaluate(eval_triples, batch_size=256)
    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"\nEval metrics: {metrics}")
    print(f"Eval time: {(end - start) * 1000:.2f} ms")

    assert "mrr" in metrics
