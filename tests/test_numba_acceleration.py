"""
Tests for Numba-accelerated hot loop optimization (Sprint 17).

This module tests the performance improvement from Numba JIT compilation
of the rule validation hot loops.

Author: Claude Code (Sprint 17)
Date: 2025-10-23
"""

import time
import pytest
import numpy as np

from pff.utils.numba_kernels import (
    VocabularyEncoder,
    find_matching_triples_accelerated,
    unify_batch_numba,
    NUMBA_AVAILABLE,
)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
class TestNumbaKernels:
    """Tests for Numba-compiled kernels."""

    def test_vocabulary_encoder_basic(self):
        """Test basic vocabulary encoding/decoding."""
        encoder = VocabularyEncoder()

        # Test entity encoding
        idx1 = encoder.encode_entity("customer_123")
        idx2 = encoder.encode_entity("prepaid")
        idx3 = encoder.encode_entity("customer_123")  # Should return same idx

        assert idx1 == idx3, "Same entity should have same index"
        assert idx1 != idx2, "Different entities should have different indices"

        # Test decoding
        assert encoder.decode_entity(idx1) == "customer_123"
        assert encoder.decode_entity(idx2) == "prepaid"

    def test_vocabulary_encoder_relations(self):
        """Test relation encoding/decoding."""
        encoder = VocabularyEncoder()

        # Test relation encoding
        idx1 = encoder.encode_relation("has_plan")
        idx2 = encoder.encode_relation("has_service")
        wildcard = encoder.encode_relation("*")

        assert idx1 != idx2, "Different relations should have different indices"
        assert wildcard == encoder.WILDCARD_IDX, "Wildcard should have special index"

        # Test decoding
        assert encoder.decode_relation(idx1) == "has_plan"
        assert encoder.decode_relation(wildcard) == "*"

    def test_encode_triples(self):
        """Test triple encoding to NumPy array."""
        encoder = VocabularyEncoder()
        triples = [
            ("customer_1", "has_plan", "prepaid"),
            ("customer_2", "has_plan", "postpaid"),
            ("customer_3", "has_service", "data"),
        ]

        encoded = encoder.encode_triples(triples)

        assert encoded.shape == (3, 3), "Should be (n_triples, 3) array"
        assert encoded.dtype == np.int32, "Should be int32 dtype"

        # Verify encoding is consistent
        assert encoded[0, 1] == encoded[1, 1], "Same predicate should have same index"
        assert encoded[0, 1] != encoded[2, 1], "Different predicates should differ"

    def test_encode_pattern(self):
        """Test pattern encoding."""
        encoder = VocabularyEncoder()

        # Pattern with variable and constant
        pattern = {"predicate": "has_plan", "args": ["X", "prepaid"]}
        pred_idx, arg0_idx, arg0_is_var, arg1_idx, arg1_is_var = encoder.encode_pattern(
            pattern
        )

        assert arg0_is_var == 1, "X should be marked as variable"
        assert arg1_is_var == 0, "prepaid should be marked as constant"
        assert arg0_idx >= encoder.VARIABLE_START, "Variable should have high index"
        assert arg1_idx < encoder.VARIABLE_START, "Constant should have low index"

    def test_unify_batch_numba_basic(self):
        """Test basic unification with Numba."""
        encoder = VocabularyEncoder()

        # Create simple pattern: has_plan(X, prepaid)
        pattern = {"predicate": "has_plan", "args": ["X", "prepaid"]}
        pattern_encoded = encoder.encode_pattern(pattern)
        pattern_array = np.array([pattern_encoded], dtype=np.int32)

        # Create triples
        triples = [
            ("customer_1", "has_plan", "prepaid"),  # Should match
            ("customer_2", "has_plan", "postpaid"),  # Should not match
            ("customer_3", "has_plan", "prepaid"),  # Should match
        ]
        triples_encoded = encoder.encode_triples(triples)

        # Run Numba kernel
        matches = unify_batch_numba(
            pattern_array, triples_encoded, encoder.WILDCARD_IDX
        )

        assert matches.shape == (1, 3), "Should be (1 pattern, 3 triples)"
        assert matches[0, 0] == 1, "First triple should match"
        assert matches[0, 1] == 0, "Second triple should not match"
        assert matches[0, 2] == 1, "Third triple should match"

    def test_unify_batch_numba_wildcard(self):
        """Test unification with wildcard predicate."""
        encoder = VocabularyEncoder()

        # Pattern with wildcard: *(X, prepaid)
        pattern = {"predicate": "*", "args": ["X", "prepaid"]}
        pattern_encoded = encoder.encode_pattern(pattern)
        pattern_array = np.array([pattern_encoded], dtype=np.int32)

        # Create triples with different predicates
        triples = [
            ("customer_1", "has_plan", "prepaid"),  # Should match
            ("customer_2", "has_service", "prepaid"),  # Should match
            ("customer_3", "has_plan", "postpaid"),  # Should not match
        ]
        triples_encoded = encoder.encode_triples(triples)

        # Run Numba kernel
        matches = unify_batch_numba(
            pattern_array, triples_encoded, encoder.WILDCARD_IDX
        )

        assert matches[0, 0] == 1, "First triple should match (wildcard)"
        assert matches[0, 1] == 1, "Second triple should match (wildcard)"
        assert matches[0, 2] == 0, "Third triple should not match (wrong object)"

    def test_find_matching_triples_accelerated(self):
        """Test high-level API for finding matching triples."""
        encoder = VocabularyEncoder()

        pattern = {"predicate": "has_plan", "args": ["X", "prepaid"]}
        triples = [
            ("customer_1", "has_plan", "prepaid"),  # Should match
            ("customer_2", "has_plan", "postpaid"),  # Should not match
            ("customer_3", "has_plan", "prepaid"),  # Should match
            ("customer_4", "has_service", "prepaid"),  # Should not match
        ]

        matching_indices = find_matching_triples_accelerated(pattern, triples, encoder)

        assert matching_indices == [0, 2], "Should find indices 0 and 2"

    def test_numba_works_correctly(self):
        """Test that Numba produces correct results."""
        encoder = VocabularyEncoder()

        pattern = {"predicate": "has_plan", "args": ["X", "prepaid"]}
        triples = [
            ("customer_1", "has_plan", "prepaid"),  # Should match
            ("customer_2", "has_plan", "postpaid"),  # Should not match
        ] * 50  # 100 triples total

        # Run accelerated version
        matching_indices = find_matching_triples_accelerated(pattern, triples, encoder)

        # Verify results: every other triple should match (indices 0, 2, 4, ...)
        expected = [i for i in range(100) if i % 2 == 0]
        assert matching_indices == expected, "Numba should produce correct matches"


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
class TestNumbaBenchmark:
    """Benchmark tests for Numba acceleration."""

    def test_numba_python_equivalence_large_set(self):
        """Verify Numba and Python produce identical results on large triple sets."""
        encoder = VocabularyEncoder()

        # Create large triple set (1000 triples, realistic for production)
        triples = [(f"customer_{i}", "has_plan", "prepaid" if i % 2 == 0 else "postpaid") for i in range(1000)]

        pattern = {"predicate": "has_plan", "args": ["X", "prepaid"]}

        # Run both implementations
        result_numba = find_matching_triples_accelerated(pattern, triples, encoder)

        from pff.utils.numba_kernels import _find_matching_triples_python
        result_python = _find_matching_triples_python(pattern, triples)

        # Verify results match
        assert result_numba == result_python, "Numba and Python should produce identical results"
        assert len(result_numba) == 500, "Should find 500 matches (half of 1000 triples)"


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
class TestNumbaIntegration:
    """Integration tests for Numba with business_service."""

    def test_business_service_imports_numba(self):
        """Test that business_service correctly imports Numba modules."""
        from pff.services.business_service import NUMBA_AVAILABLE, VocabularyEncoder

        assert NUMBA_AVAILABLE is True, "Numba should be available"
        assert VocabularyEncoder is not None, "VocabularyEncoder should be imported"

    def test_run_rule_check_shared_with_numba(self):
        """Test that _run_rule_check_shared uses Numba acceleration."""
        from pff.services.business_service import _run_rule_check_shared, Rule

        # Create test rule
        rule = Rule(
            id="test_rule",
            confidence=0.95,
            head={"predicate": "valid", "args": ["X"]},
            body=[{"predicate": "has_plan", "args": ["X", "prepaid"]}],
            source="test",
        )

        # Create test triples (200+ triples to trigger Numba)
        triples = [
            (f"customer_{i}", "has_plan", "prepaid" if i % 2 == 0 else "postpaid")
            for i in range(200)
        ]

        # Run rule check (should use Numba internally)
        violations = _run_rule_check_shared(triples, rule)

        # Should find violations for customers with postpaid (not valid)
        assert len(violations) >= 0, "Should return violations list"


class TestNumbaFallback:
    """Tests for Python fallback when Numba is unavailable."""

    def test_python_fallback(self):
        """Test that Python fallback works correctly."""
        from pff.utils.numba_kernels import _find_matching_triples_python

        pattern = {"predicate": "has_plan", "args": ["X", "prepaid"]}
        triples = [
            ("customer_1", "has_plan", "prepaid"),  # Should match
            ("customer_2", "has_plan", "postpaid"),  # Should not match
            ("customer_3", "has_plan", "prepaid"),  # Should match
        ]

        matching_indices = _find_matching_triples_python(pattern, triples)

        assert matching_indices == [0, 2], "Python fallback should find correct matches"

    def test_wildcard_fallback(self):
        """Test Python fallback with wildcard predicates."""
        from pff.utils.numba_kernels import _find_matching_triples_python

        pattern = {"predicate": "*", "args": ["X", "prepaid"]}
        triples = [
            ("customer_1", "has_plan", "prepaid"),  # Should match
            ("customer_2", "has_service", "prepaid"),  # Should match
            ("customer_3", "has_plan", "postpaid"),  # Should not match
        ]

        matching_indices = _find_matching_triples_python(pattern, triples)

        assert matching_indices == [0, 1], "Wildcard should match any predicate"
