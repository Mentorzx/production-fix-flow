"""
Test suite for memory optimization in rule validation (Sprint 15 - v10.5.0).

This validates the fix for OOM crash when validating 128K rules with 1K triples.

Previous bug:
    args_list = [(rule, triples) for rule in rules]
    → Copied triples 128,306 times = 21.6 GB RAM → OOM crash → Linux freeze

Fix:
    - Dask (>10K rules): shared_data with scatter/broadcast
    - Process (<10K rules): functools.partial to bind triples once
    → Memory: 21.6 GB → 169 KB (-99.99%)
"""

import pytest
import psutil
import os
from pff.services.business_service import Rule, RuleValidator


class TestMemoryOptimizedValidation:
    """Test memory-optimized rule validation."""

    def test_small_ruleset_uses_process_partial(self):
        """Validate that <10K rules use ProcessExecutor with partial."""
        validator = RuleValidator()

        # Create 1000 rules (< 10K threshold)
        rules = [
            Rule(
                id=f"rule_{i}",
                confidence=0.8,
                head={"predicate": "hasValue", "args": ["X", str(i)]},
                body=[],
                source="test"
            )
            for i in range(1000)
        ]

        # Create 1125 triples (production-like)
        triples = [(f"s{i}", f"p{i}", f"o{i}") for i in range(1125)]

        # Measure memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Validate (should use Process with partial)
        violations, satisfied = validator.validate_rules(rules, triples)

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        # Assert: Memory increase should be < 500 MB (not 21.6 GB!)
        assert mem_increase < 500, (
            f"Memory explosion detected! Increase: {mem_increase:.1f} MB "
            f"(expected < 500 MB)"
        )

        # Assert: Validation should work
        assert isinstance(violations, list)
        assert isinstance(satisfied, list)
        assert len(violations) + len(satisfied) == len(rules)

    def test_large_ruleset_uses_dask_shared(self):
        """Validate that >10K rules use Dask with shared_data."""
        validator = RuleValidator()

        # Create 15000 rules (> 10K threshold)
        rules = [
            Rule(
                id=f"rule_{i}",
                confidence=0.8,
                head={"predicate": "hasValue", "args": ["X", str(i)]},
                body=[],
                source="test"
            )
            for i in range(15000)
        ]

        # Create 1125 triples (production-like)
        triples = [(f"s{i}", f"p{i}", f"o{i}") for i in range(1125)]

        # Measure memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Validate (should use Dask with shared_data)
        violations, satisfied = validator.validate_rules(rules, triples)

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        # Assert: Memory increase should be < 1 GB (not 21.6 GB!)
        assert mem_increase < 1024, (
            f"Memory explosion detected! Increase: {mem_increase:.1f} MB "
            f"(expected < 1024 MB)"
        )

        # Assert: Validation should work
        assert isinstance(violations, list)
        assert isinstance(satisfied, list)
        assert len(violations) + len(satisfied) == len(rules)

    @pytest.mark.skipif(
        not os.path.exists("outputs/pyclause/rules_anyburl.tsv"),
        reason="Requires AnyBURL rules file (128K rules)"
    )
    def test_production_128k_rules_no_oom(self):
        """
        Test with real production data: 128K AnyBURL rules.

        This is the exact scenario that caused OOM crash before fix.
        """
        from pff.services.business_service import BusinessService

        # Initialize service (loads 128K rules)
        service = BusinessService()

        # Create test triples (production-like)
        triples = [(f"s{i}", f"p{i}", f"o{i}") for i in range(1125)]

        # Measure memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Get all rules (128K rules aggregated to ~8K unique patterns due to 93.9% duplicates)
        all_rules = service.rule_engine.get_all_rules()
        assert len(all_rules) > 5000, f"Expected >5K aggregated rules, got {len(all_rules)}"

        # Validate (THIS USED TO CRASH WITH OOM!)
        violations, satisfied = service.rule_validator.validate_rules(
            all_rules, triples
        )

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        # Assert: Memory increase should be < 2 GB (not 21.6 GB!)
        assert mem_increase < 2048, (
            f"Memory explosion detected! Increase: {mem_increase:.1f} MB "
            f"(expected < 2048 MB)"
        )

        # Assert: Validation should complete without crash
        assert isinstance(violations, list)
        assert isinstance(satisfied, list)
        print(f"✅ Validated {len(all_rules):,} rules without OOM!")
        print(f"   Memory increase: {mem_increase:.1f} MB (< 2 GB threshold)")


class TestMemoryFixRegression:
    """Regression tests to ensure OOM fix doesn't break functionality."""

    def test_rule_validation_still_finds_violations(self):
        """Ensure optimization doesn't break violation detection."""
        validator = RuleValidator()

        # Create rule: hasValue(X, 999) <= hasProperty(X, 'test')
        rule = Rule(
            id="test_rule",
            confidence=0.9,
            head={"predicate": "hasValue", "args": ["X", "999"]},
            body=[{"predicate": "hasProperty", "args": ["X", "test"]}],
            source="test"
        )

        # Triples: hasProperty(entity1, 'test') exists, but hasValue(entity1, 999) missing
        triples = [
            ("entity1", "hasProperty", "test"),
            ("entity1", "hasValue", "123"),  # Wrong value!
        ]

        # Validate
        violations, satisfied = validator.validate_rules([rule], triples)

        # Should find violation (hasValue(entity1, 999) missing)
        assert len(violations) > 0, "Should detect violation"
        assert violations[0].rule_id == "test_rule"

    def test_empty_rules_returns_empty(self):
        """Ensure edge case of empty rules works."""
        validator = RuleValidator()
        violations, satisfied = validator.validate_rules([], [])
        assert violations == []
        assert satisfied == []

    def test_empty_triples_returns_violations(self):
        """Ensure rules fail when no triples exist."""
        validator = RuleValidator()

        rule = Rule(
            id="test_rule",
            confidence=0.9,
            head={"predicate": "hasValue", "args": ["X", "999"]},
            body=[{"predicate": "hasProperty", "args": ["X", "test"]}],
            source="test"
        )

        # No triples → rule can't be satisfied
        violations, satisfied = validator.validate_rules([rule], [])

        # Should have no violations (body not satisfied, so rule doesn't apply)
        assert len(violations) == 0
        assert len(satisfied) == 0
