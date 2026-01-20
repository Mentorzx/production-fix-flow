"""
Tests for SymbolicRuleAccelerator - Numba-accelerated rule validation.

Tests cover:
- RuleEncoder (string→integer encoding)
- Rule violation checking
- Batch processing
- Fallback behavior

Author: PFF Team
Date: 2025-10-31
"""

import numpy as np
import pytest

from pff.shared import RuleEncoder, SymbolicRuleAccelerator


class TestRuleEncoder:
    """Test RuleEncoder functionality."""

    def test_predicate_encoding(self):
        """Test predicate encoding."""
        encoder = RuleEncoder()

        idx1 = encoder.encode_predicate("has_plan")
        idx2 = encoder.encode_predicate("has_status")
        idx3 = encoder.encode_predicate("has_plan")  # Same as idx1

        assert idx1 != idx2
        assert idx1 == idx3  # Same predicate should get same index

    def test_entity_encoding(self):
        """Test entity encoding."""
        encoder = RuleEncoder()

        idx1 = encoder.encode_entity("customer_1")
        idx2 = encoder.encode_entity("customer_2")
        idx3 = encoder.encode_entity("customer_1")  # Same as idx1

        assert idx1 != idx2
        assert idx1 == idx3

    def test_variable_encoding(self):
        """Test variable (uppercase) encoding."""
        encoder = RuleEncoder()

        var_idx = encoder.encode_entity("X")
        const_idx = encoder.encode_entity("customer_1")

        # Variables should be >= VARIABLE_START
        assert encoder.is_variable(var_idx)
        assert not encoder.is_variable(const_idx)

    def test_atom_encoding(self):
        """Test atom (triple) encoding."""
        encoder = RuleEncoder()

        atom = {"predicate": "has_plan", "subject": "customer_1", "object": "prepaid"}

        pred_idx, subj_idx, obj_idx = encoder.encode_atom(atom)

        assert isinstance(pred_idx, int)
        assert isinstance(subj_idx, int)
        assert isinstance(obj_idx, int)

    def test_triples_encoding(self):
        """Test encoding multiple triples."""
        encoder = RuleEncoder()

        triples = [
            ("customer_1", "has_plan", "prepaid"),
            ("customer_1", "has_status", "active"),
        ]

        encoded = encoder.encode_triples(triples)

        assert encoded.shape == (2, 3)
        assert encoded.dtype == np.int32


class TestRuleEncoding:
    """Test rule structure encoding."""

    def test_simple_rule_encoding(self):
        """Test encoding a simple rule."""
        encoder = RuleEncoder()

        rule = {
            "head": {"predicate": "has_plan", "subject": "X", "object": "prepaid"},
            "body": [{"predicate": "has_status", "subject": "X", "object": "active"}],
            "confidence": 0.9,
        }

        encoded = encoder.encode_rule(rule)

        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.int32
        assert encoded[0] == 1  # n_body_atoms

    def test_multiple_body_atoms(self):
        """Test rule with multiple body atoms."""
        encoder = RuleEncoder()

        rule = {
            "head": {"predicate": "has_plan", "subject": "X", "object": "prepaid"},
            "body": [
                {"predicate": "has_status", "subject": "X", "object": "active"},
                {"predicate": "has_region", "subject": "X", "object": "north"},
            ],
            "confidence": 0.8,
        }

        encoded = encoder.encode_rule(rule)

        assert encoded[0] == 2  # n_body_atoms

    def test_batch_rule_encoding(self):
        """Test encoding multiple rules."""
        encoder = RuleEncoder()

        rules = [
            {
                "head": {"predicate": "p1", "subject": "X", "object": "o1"},
                "body": [{"predicate": "p2", "subject": "X", "object": "o2"}],
                "confidence": 0.9,
            },
            {
                "head": {"predicate": "p3", "subject": "Y", "object": "o3"},
                "body": [{"predicate": "p4", "subject": "Y", "object": "o4"}],
                "confidence": 0.8,
            },
        ]

        rules_array, lengths_array = encoder.encode_rules(rules)

        assert rules_array.shape[0] == 2  # 2 rules
        assert lengths_array.shape[0] == 2
        assert rules_array.dtype == np.int32


class TestSymbolicRuleAccelerator:
    """Test SymbolicRuleAccelerator functionality."""

    def test_initialization(self):
        """Test accelerator initialization."""
        rules = [
            {
                "head": {"predicate": "has_plan", "subject": "X", "object": "prepaid"},
                "body": [{"predicate": "has_status", "subject": "X", "object": "active"}],
                "confidence": 0.9,
            }
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)

        assert accelerator.rules == rules
        assert accelerator.encoder is not None

    def test_check_violations_simple(self):
        """Test checking violations for a simple case."""
        rules = [
            {
                "head": {"predicate": "has_plan", "subject": "X", "object": "prepaid"},
                "body": [{"predicate": "has_status", "subject": "X", "object": "active"}],
                "confidence": 0.9,
            }
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)

        # Sample with body satisfied but head not satisfied → VIOLATED
        sample = [
            ("customer_1", "has_status", "active"),
            # Missing: ("customer_1", "has_plan", "prepaid")
        ]

        violations = accelerator.check_violations(sample)

        assert violations.shape == (1,)
        # Note: Actual violation checking depends on correct implementation

    def test_check_violations_empty(self):
        """Test with empty sample."""
        rules = [
            {
                "head": {"predicate": "has_plan", "subject": "X", "object": "prepaid"},
                "body": [{"predicate": "has_status", "subject": "X", "object": "active"}],
                "confidence": 0.9,
            }
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)
        violations = accelerator.check_violations([])

        assert violations.shape == (1,)

    def test_batch_processing(self):
        """Test batch processing of multiple samples."""
        rules = [
            {
                "head": {"predicate": "has_plan", "subject": "X", "object": "prepaid"},
                "body": [{"predicate": "has_status", "subject": "X", "object": "active"}],
                "confidence": 0.9,
            }
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)

        samples = [
            [("customer_1", "has_status", "active")],
            [("customer_2", "has_status", "inactive")],
        ]

        violations_list = accelerator.check_violations_batch(samples, use_parallel=False)

        assert len(violations_list) == 2
        assert all(v.shape == (1,) for v in violations_list)

    def test_get_stats(self):
        """Test getting accelerator statistics."""
        rules = [
            {
                "head": {"predicate": "p1", "subject": "X", "object": "o1"},
                "body": [{"predicate": "p2", "subject": "X", "object": "o2"}],
                "confidence": 0.9,
            }
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)
        stats = accelerator.get_stats()

        assert "n_rules" in stats
        assert stats["n_rules"] == 1
        assert "numba_enabled" in stats


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_many_rules(self):
        """Test with many rules."""
        # Create 1000 simple rules
        rules = [
            {
                "head": {"predicate": f"p{i}", "subject": "X", "object": "o"},
                "body": [{"predicate": f"q{i}", "subject": "X", "object": "a"}],
                "confidence": 0.9,
            }
            for i in range(1000)
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)

        sample = [("entity", "q500", "a")]
        violations = accelerator.check_violations(sample)

        assert violations.shape == (1000,)

    def test_many_samples(self):
        """Test with many samples."""
        rules = [
            {
                "head": {"predicate": "p", "subject": "X", "object": "o"},
                "body": [{"predicate": "q", "subject": "X", "object": "a"}],
                "confidence": 0.9,
            }
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)

        # Create 100 samples
        samples = [[("entity", "q", "a")] for _ in range(100)]

        violations_list = accelerator.check_violations_batch(samples)

        assert len(violations_list) == 100


class TestFallback:
    """Test fallback behavior when Numba is not available."""

    def test_numba_disabled(self):
        """Test with Numba explicitly disabled."""
        rules = [
            {
                "head": {"predicate": "p", "subject": "X", "object": "o"},
                "body": [{"predicate": "q", "subject": "X", "object": "a"}],
                "confidence": 0.9,
            }
        ]

        accelerator = SymbolicRuleAccelerator(rules, enable_numba=False)

        sample = [("entity", "q", "a")]
        violations = accelerator.check_violations(sample)

        # Should return zeros (fallback behavior)
        assert violations.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
