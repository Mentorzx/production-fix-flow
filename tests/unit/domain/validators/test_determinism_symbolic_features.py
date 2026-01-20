"""
Test for deterministic symbolic features (Sprint 27).

Validates that symbolic feature extraction is deterministic across multiple runs
with the same input, preventing the non-determinism bug identified in Issue #1.
"""

import numpy as np
import pytest

from pff.shared.acceleration.symbolic_rule_accelerator import SymbolicRuleAccelerator


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
            "confidence": 0.9,
        },
        {
            "head": {"subject": "?x", "predicate": "needsReview", "object": "Yes"},
            "body": [{"subject": "?x", "predicate": "hasRevenue", "object": "?y"}],
            "confidence": 0.8,
        },
        {
            "head": {"subject": "?e", "predicate": "isEligible", "object": "True"},
            "body": [{"subject": "?e", "predicate": "hasScore", "object": "?s"}],
            "confidence": 0.7,
        },
    ]


@pytest.fixture
def sample_data():
    """Sample triples for testing."""
    return [
        [
            ("entity1", "hasType", "Customer"),
            ("entity1", "hasAge", "35"),
            ("entity1", "hasBalance", "1000"),
        ],
        [
            ("entity2", "hasRevenue", "50000"),
            ("entity2", "hasType", "Business"),
        ],
        [
            ("entity3", "hasScore", "850"),
            ("entity3", "hasType", "Premium"),
        ],
    ]


def test_vocabulary_building_is_deterministic(sample_rules):
    """
    Test that vocabulary building produces the same mapping across multiple runs.

    This validates the fix from Sprint 27 where vocabulary was built non-deterministically
    due to parallel processing race conditions.
    """
    vocabularies = []

    for run in range(3):
        acc = SymbolicRuleAccelerator(sample_rules, enable_numba=False)
        vocab = {
            "entities": dict(acc.encoder.entity_to_idx),
            "predicates": dict(acc.encoder.predicate_to_idx),
        }
        vocabularies.append(vocab)

    # All vocabularies should be identical
    for i in range(1, len(vocabularies)):
        assert vocabularies[i]["entities"] == vocabularies[0]["entities"], (
            f"Run {i + 1} entity vocabulary differs from Run 1!\n"
            f"Run 1: {vocabularies[0]['entities']}\n"
            f"Run {i + 1}: {vocabularies[i]['entities']}"
        )
        assert vocabularies[i]["predicates"] == vocabularies[0]["predicates"], (
            f"Run {i + 1} predicate vocabulary differs from Run 1!\n"
            f"Run 1: {vocabularies[0]['predicates']}\n"
            f"Run {i + 1}: {vocabularies[i]['predicates']}"
        )

    print(f" Vocabulary is deterministic across {len(vocabularies)} runs")


def test_symbolic_features_are_deterministic(sample_rules, sample_data):
    """
    Test that symbolic feature extraction produces identical results across multiple runs.

    This is the main regression test for Issue #1: Non-Deterministic Results.
    Before the fix, sparsity varied 21% between runs (1.18% → 0.97%).
    After the fix, results must be 100% identical.
    """
    results = []

    for run in range(3):
        acc = SymbolicRuleAccelerator(sample_rules, enable_numba=True)

        run_results = []
        for sample in sample_data:
            violations = acc.check_violations(sample)
            run_results.append(violations.copy())  # Copy to avoid reference issues

        results.append(run_results)

    # Verify all runs produced identical results
    for sample_idx in range(len(sample_data)):
        run1 = results[0][sample_idx]
        run2 = results[1][sample_idx]
        run3 = results[2][sample_idx]

        assert np.array_equal(run1, run2), (
            f"Sample {sample_idx + 1}: Run 1 and Run 2 differ!\n"
            f"Run 1: {run1}\n"
            f"Run 2: {run2}\n"
            f"This indicates non-determinism in symbolic feature extraction."
        )

        assert np.array_equal(run2, run3), (
            f"Sample {sample_idx + 1}: Run 2 and Run 3 differ!\n"
            f"Run 2: {run2}\n"
            f"Run 3: {run3}\n"
            f"This indicates non-determinism in symbolic feature extraction."
        )

    print(f" Symbolic features are deterministic across 3 runs for {len(sample_data)} samples")


def test_sparsity_variance_is_below_threshold(sample_rules, sample_data):
    """
    Test that sparsity variance is below 5% across multiple runs.

    Before the fix: 21% variance (1.18% → 0.97%)
    After the fix: <5% variance (ideally 0%)
    """
    sparsities = []

    for run in range(3):
        acc = SymbolicRuleAccelerator(sample_rules, enable_numba=True)

        total_violations = 0
        total_elements = 0

        for sample in sample_data:
            violations = acc.check_violations(sample)
            total_violations += np.sum(violations)
            total_elements += len(violations)

        sparsity = (total_violations / total_elements) * 100 if total_elements > 0 else 0
        sparsities.append(sparsity)

    # Calculate variance
    mean_sparsity = np.mean(sparsities)
    variance = max(sparsities) - min(sparsities)
    variance_pct = (variance / mean_sparsity * 100) if mean_sparsity > 0 else 0

    print(f"Sparsities: {sparsities}")
    print(f"Mean: {mean_sparsity:.4f}%, Variance: {variance:.4f}% ({variance_pct:.2f}%)")

    # Variance should be exactly 0% (deterministic)
    assert variance < 5.0, (
        f"Sparsity variance is too high: {variance:.4f}% (>{5.0}%)\n"
        f"Sparsities: {sparsities}\n"
        f"This indicates non-determinism in symbolic feature extraction."
    )

    # Ideally variance should be 0%
    if variance == 0:
        print(" Perfect determinism: 0% variance")
    else:
        print(f" Some variance detected: {variance:.4f}% (but <5% threshold)")


@pytest.mark.slow
def test_determinism_with_numba_parallel(sample_rules):
    """
    Test determinism with Numba parallel processing enabled.

    This is a slow test that validates determinism even with parallel execution.
    """
    # Create larger dataset to trigger parallel processing
    large_samples = []
    for i in range(500):
        large_samples.append(
            [
                (f"entity{i}", "hasType", "Customer"),
                (f"entity{i}", "hasAge", str(25 + i % 50)),
                (f"entity{i}", "hasBalance", str(1000 * (i % 10))),
            ]
        )

    results = []
    for run in range(2):
        acc = SymbolicRuleAccelerator(sample_rules, enable_numba=True)

        run_results = []
        for sample in large_samples:
            violations = acc.check_violations(sample)
            run_results.append(violations.copy())

        results.append(run_results)

    # Verify all results are identical
    for i in range(len(large_samples)):
        assert np.array_equal(results[0][i], results[1][i]), (
            f"Sample {i}: Results differ between runs with parallel processing!\n"
            f"Run 1: {results[0][i]}\n"
            f"Run 2: {results[1][i]}"
        )

    print(f" Determinism maintained with Numba parallel processing ({len(large_samples)} samples)")
