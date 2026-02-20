"""Tests for deterministic symbolic features (Sprint 27).

Validates that RuleEncoder + check_violations_batch produce deterministic
results across multiple runs with the same input.
"""

from __future__ import annotations

import numpy as np
import pytest

from pff_rust import RuleEncoder


@pytest.fixture
def sample_rules():
    """Sample rules for determinism testing."""
    return [
        {
            "head": {"subject": "?a", "predicate": "hasType", "object": "Premium"},
            "body": [
                {"subject": "?a", "predicate": "hasAge", "object": "?b"},
                {"subject": "?a", "predicate": "hasBalance", "object": "?c"},
            ],
        },
        {
            "head": {"subject": "?x", "predicate": "needsReview", "object": "Yes"},
            "body": [{"subject": "?x", "predicate": "hasRevenue", "object": "?y"}],
        },
        {
            "head": {"subject": "?e", "predicate": "isEligible", "object": "True"},
            "body": [{"subject": "?e", "predicate": "hasScore", "object": "?s"}],
        },
    ]


def test_vocabulary_building_is_deterministic(sample_rules) -> None:
    """Vocabulary building should produce identical stats across runs."""
    stats_list = []
    for _ in range(5):
        enc = RuleEncoder()
        enc.build_vocabulary_from_rules(sample_rules)
        stats_list.append(enc.get_stats())

    for i in range(1, len(stats_list)):
        assert stats_list[i] == stats_list[0], (
            f"Run {i + 1} stats differ from Run 1: {stats_list[i]} vs {stats_list[0]}"
        )


def test_encode_entity_deterministic_across_instances(sample_rules) -> None:
    """Same entity should produce same index across encoder instances."""
    indices = []
    for _ in range(5):
        enc = RuleEncoder()
        enc.build_vocabulary_from_rules(sample_rules)
        idx = enc.encode_entity("Premium")
        indices.append(idx)

    assert all(idx == indices[0] for idx in indices), f"Entity index not deterministic: {indices}"


def test_variable_encoding_deterministic() -> None:
    """Variable encoding (BLAKE3 hash) should be stable across instances."""
    var_indices = []
    for _ in range(5):
        enc = RuleEncoder()
        x = enc.encode_entity("?a")
        y = enc.encode_entity("?b")
        var_indices.append((x, y))

    for pair in var_indices:
        assert pair == var_indices[0], f"Variable indices not deterministic: {var_indices}"


def test_encode_triples_deterministic(sample_rules) -> None:
    """Encoding the same triples should produce identical arrays."""
    triples = [
        ("entity1", "hasAge", "35"),
        ("entity1", "hasBalance", "1000"),
        ("entity2", "hasRevenue", "50000"),
    ]

    arrays = []
    for _ in range(3):
        enc = RuleEncoder()
        enc.build_vocabulary_from_rules(sample_rules)
        encoded = enc.encode_triples(triples)
        arrays.append(np.array(encoded, dtype=np.int32))

    for i in range(1, len(arrays)):
        assert np.array_equal(arrays[i], arrays[0]), (
            f"Run {i + 1} encoded triples differ: {arrays[i]} vs {arrays[0]}"
        )
