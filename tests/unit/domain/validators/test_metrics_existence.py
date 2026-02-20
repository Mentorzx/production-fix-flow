"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_metrics_existence.py

"""

import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


def test_evaluate_returns_expected_metrics():
    """Execute test evaluate returns expected metrics.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = DSLFMKGCConfig(
        num_entities=10,
        num_relations=2,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
    )
    model = DSLFMKGCModel(config)

    # Create dummy triples
    triples = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)

    metrics = model.evaluate(triples, batch_size=2)

    # These are the metrics expected by model.evaluate() (Ranking metrics)
    expected_metrics = ["mrr", "hits@1", "hits@3", "hits@10", "ap@10"]

    missing = [m for m in expected_metrics if m not in metrics]

    assert not missing, f"Missing metrics: {missing}. Available: {list(metrics.keys())}"
