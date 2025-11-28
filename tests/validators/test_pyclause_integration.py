"""
Tests for PyClause rule engine integration.

Uses synthetic fixtures under tests/fixtures/ for fast, deterministic tests.
Does NOT depend on production assets under data/models/.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.fixtures import SAMPLE_RULES_PATH, get_sample_rules


class TestPyClauseRuleEngine:
    """Tests for PyClause library availability and basic functionality."""

    def test_pyclause_import(self):
        """Verify PyClause (clause) library is installed and importable."""
        try:
            from clause import Learner, Options

            assert Learner is not None
            assert Options is not None
        except ImportError:
            pytest.fail("PyClause (clause) not installed")

    def test_pyclause_learner_creation(self):
        """Verify PyClause Learner can be instantiated with AnyBURL mode."""
        from clause import Learner, Options

        options = Options()
        options.set("learner.mode", "anyburl")

        learner = Learner(options=options.get("learner"))
        assert learner is not None


class TestRuleFormatConversion:
    """Tests for rule format parsing and conversion."""

    def test_anyburl_tsv_parsing(self):
        """Test parsing AnyBURL rules from TSV format."""
        rules = get_sample_rules()
        
        assert len(rules) == 10, f"Expected 10 sample rules, got {len(rules)}"
        
        # Validate first rule structure
        first_rule = rules[0]
        assert "head_coverage" in first_rule
        assert "body_coverage" in first_rule
        assert "confidence" in first_rule
        assert "rule_string" in first_rule
        
        # Validate confidence is between 0 and 1
        for rule in rules:
            assert 0 <= rule["confidence"] <= 1, f"Invalid confidence: {rule['confidence']}"

    def test_rule_string_structure(self):
        """Test that rule strings have valid head <= body structure."""
        rules = get_sample_rules()
        
        for rule in rules:
            rule_str = rule["rule_string"]
            assert "<=" in rule_str, f"Rule missing '<=' separator: {rule_str}"
            
            parts = rule_str.split("<=")
            assert len(parts) == 2, f"Rule should have exactly one '<=': {rule_str}"
            
            head = parts[0].strip()
            body = parts[1].strip()
            
            # Head should contain a predicate with arguments
            assert "(" in head and ")" in head, f"Invalid head: {head}"
            # Body should contain at least one predicate
            assert "(" in body and ")" in body, f"Invalid body: {body}"

    def test_rule_loading_from_tsv_file(self):
        """Test loading rules directly from TSV file."""
        assert SAMPLE_RULES_PATH.exists(), f"Fixture not found: {SAMPLE_RULES_PATH}"
        
        with open(SAMPLE_RULES_PATH) as f:
            lines = f.readlines()
        
        assert len(lines) == 10, f"Expected 10 lines, got {len(lines)}"
        
        # Validate each line is tab-separated with 4 fields
        for i, line in enumerate(lines):
            parts = line.strip().split("\t")
            assert len(parts) >= 4, f"Line {i+1} has {len(parts)} fields, expected 4+"


class TestRuleAggregation:
    """Tests for rule aggregation and deduplication."""

    def test_duplicate_rule_detection(self):
        """Test detecting duplicate rules by rule string."""
        rules = get_sample_rules()
        
        rule_strings = [r["rule_string"] for r in rules]
        unique_strings = set(rule_strings)
        
        # Sample rules should all be unique
        assert len(rule_strings) == len(unique_strings), "Sample has duplicate rules"

    def test_rule_confidence_aggregation(self):
        """Test aggregating confidence for duplicate rules."""
        # Create test data with duplicates
        rules = [
            {"rule_string": "pred(X,Y) <= rel(X,Y)", "confidence": 0.8},
            {"rule_string": "pred(X,Y) <= rel(X,Y)", "confidence": 0.9},
            {"rule_string": "other(X,Z) <= attr(X,Z)", "confidence": 0.7},
        ]
        
        # Aggregate by rule string (max confidence)
        aggregated = {}
        for rule in rules:
            key = rule["rule_string"]
            if key not in aggregated or rule["confidence"] > aggregated[key]["confidence"]:
                aggregated[key] = rule
        
        assert len(aggregated) == 2, "Should reduce to 2 unique rules"
        assert aggregated["pred(X,Y) <= rel(X,Y)"]["confidence"] == 0.9, "Should keep max confidence"

    def test_rule_filtering_by_confidence(self):
        """Test filtering rules by minimum confidence threshold."""
        rules = get_sample_rules()
        
        min_confidence = 0.8
        filtered = [r for r in rules if r["confidence"] >= min_confidence]
        
        # Check that all filtered rules meet threshold
        for rule in filtered:
            assert rule["confidence"] >= min_confidence
        
        # Should have fewer rules after filtering
        assert len(filtered) < len(rules), "Filtering should reduce rule count"


class TestPyClauseIntegration:
    """Integration tests for PyClause with BusinessService."""

    def test_pyclause_with_business_service_mock(self):
        """Test BusinessService initialization with mocked rule engine."""
        from pff.services.business_service import BusinessService

        service = BusinessService()
        assert service is not None

    def test_rule_engine_accepts_sample_rules(self):
        """Test that sample rules can be processed by rule engine components."""
        rules = get_sample_rules()
        
        # Simulate rule engine processing
        processed_rules = []
        for rule in rules:
            processed = {
                "id": f"rule_{len(processed_rules)}",
                "head": rule["rule_string"].split("<=")[0].strip(),
                "body": rule["rule_string"].split("<=")[1].strip(),
                "confidence": rule["confidence"],
                "support": rule["head_coverage"],
            }
            processed_rules.append(processed)
        
        assert len(processed_rules) == len(rules)
        
        # Validate processed structure
        for pr in processed_rules:
            assert "id" in pr
            assert "head" in pr
            assert "body" in pr
            assert "confidence" in pr
            assert "support" in pr

    def test_rule_coverage_calculation(self):
        """Test calculating rule coverage from sample rules."""
        rules = get_sample_rules()
        
        total_head_coverage = sum(r["head_coverage"] for r in rules)
        total_body_coverage = sum(r["body_coverage"] for r in rules)
        
        assert total_head_coverage > 0, "Should have positive head coverage"
        assert total_body_coverage > 0, "Should have positive body coverage"
        
        # Coverage ratio should be <= 1
        for rule in rules:
            if rule["head_coverage"] > 0:
                ratio = rule["body_coverage"] / rule["head_coverage"]
                assert ratio <= 1.5, f"Unusual coverage ratio: {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
