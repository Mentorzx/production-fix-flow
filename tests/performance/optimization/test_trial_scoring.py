"""Tests for the unified HPO scoring helpers."""

from __future__ import annotations

from pff.domain.hpo.scoring import (
    build_weights_from_settings,
    compute_score,
    rename_metric_keys,
)


def test_rename_metric_keys_maps_legacy_names():
    """Legacy metric keys should be mapped to the new short names."""
    raw = {
        "score_composto": 0.5,
        "kge_mrr": 0.3,
        "kge_best_mrr": 0.35,
        "hits@1": 0.2,
        "hits@3": 0.4,
        "hits@10": 0.6,
        "elapsed_time": 12.0,
    }
    renamed = rename_metric_keys(raw)
    assert renamed["score"] == 0.5
    assert renamed["mrr"] == 0.3
    assert renamed["best_mrr"] == 0.35
    assert renamed["hits1"] == 0.2
    assert renamed["hits3"] == 0.4
    assert renamed["hits10"] == 0.6
    assert renamed["duration"] == 12.0


def test_compute_score_uses_all_blocks_and_open_interval():
    """Score should reflect rank, classification and duration with open-interval bounds."""
    weights = build_weights_from_settings({})
    history = [
        {
            "mrr": 0.25,
            "best_mrr": 0.3,
            "hits1": 0.1,
            "hits3": 0.2,
            "hits10": 0.4,
            "auc": 0.6,
            "pr_auc": 0.5,
            "precision": 0.45,
            "recall": 0.5,
            "duration": 80.0,
        }
    ]
    current = {
        "mrr": 0.4,
        "best_mrr": 0.45,
        "hits1": 0.25,
        "hits3": 0.35,
        "hits10": 0.55,
        "auc": 0.75,
        "pr_auc": 0.7,
        "precision": 0.6,
        "recall": 0.65,
        "duration": 40.0,
    }
    score, normalized, components = compute_score(current, history, weights=weights)
    assert 0.0 < score < 1.0
    assert normalized["duration"] > 0.5  # faster than history -> higher normalized score
    assert components.rank > 0
    assert components.classification > 0
    assert components.efficiency > 0
    assert score < 0.99  # sem cap, mas sem atingir 1.0


def test_score_avoids_extremes_and_weights_mrr_highest():
    """Score must not collapse to 0/1 and MRR weight dominates rank metrics."""
    weights = build_weights_from_settings({})
    zero_metrics = {
        "mrr": 0.0,
        "best_mrr": 0.0,
        "hits1": 0.0,
        "hits3": 0.0,
        "hits10": 0.0,
        "auc": 0.0,
        "pr_auc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "duration": 300.0,
    }
    score, normalized, _ = compute_score(zero_metrics, [], weights=weights)
    assert weights.rank_metrics["mrr"] >= max(
        v for k, v in weights.rank_metrics.items() if k != "mrr"
    )
    assert 0.0 < score < 0.98
    assert score > weights.eps  # should not hit the floor or midpoint artificially
