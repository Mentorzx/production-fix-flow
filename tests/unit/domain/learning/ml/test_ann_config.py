"""Tests for ANNConfig defaults parsing."""

from __future__ import annotations


def test_ann_config_defaults_loaded() -> None:
    """Ensure ANNConfig picks up new default keys from dslfm config."""
    from pff.domain.learning.ml.ann_evaluator import ANNConfig

    cfg = ANNConfig.from_defaults()
    assert cfg.backend in {"faiss", "scann", "cuvs"}
    assert cfg.index_type in {"flat", "ivf", "ivfpq", "hnsw", "cagra"}
    assert cfg.metric in {"ip", "l2"}
    assert cfg.pq_bits >= 4
    assert cfg.scann_num_leaves >= 0
    assert cfg.cagra_graph_degree >= 1


def test_ann_backend_available_unknown_backend() -> None:
    from pff.domain.learning.ml.ann_evaluator import ann_backend_available

    assert ann_backend_available("unknown-backend") is False
