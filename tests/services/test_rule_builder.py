"""
Tests for RuleBuilder and RuleSourceFactory.

Tests cover:
    - Builder pattern for Rule construction
    - Factory pattern for loading rules from files
    - Pattern parsing for Datalog-like strings
"""

import tempfile
from pathlib import Path

import pytest

from pff.services.rule_builder import (
    AnyBURLRuleSource,
    ManualRuleSource,
    Rule,
    RuleBuilder,
    RuleSourceFactory,
    _parse_pattern,
)


class TestRuleBuilder:
    """Tests for RuleBuilder fluent interface."""

    def test_build_minimal_rule(self):
        """Test building a rule with minimal required fields."""
        rule = (RuleBuilder()
            .with_id("test_001")
            .with_head("knows", ["A", "B"])
            .build())

        assert rule.id == "test_001"
        assert rule.head == {"predicate": "knows", "args": ["A", "B"]}
        assert rule.confidence == 0.0
        assert rule.source == "unknown"

    def test_build_full_rule(self):
        """Test building a rule with all fields."""
        rule = (RuleBuilder()
            .with_id("full_001")
            .with_confidence(0.85)
            .with_head("knows", ["A", "B"])
            .with_body_clause("friend", ["A", "C"])
            .with_body_clause("friend", ["C", "B"])
            .from_source("anyburl")
            .with_predictions(100, 85)
            .with_occurrences(3, 2.55)
            .build())

        assert rule.id == "full_001"
        assert rule.confidence == 0.85
        assert rule.head == {"predicate": "knows", "args": ["A", "B"]}
        assert len(rule.body) == 2
        assert rule.source == "anyburl"
        assert rule.total_predictions == 100
        assert rule.correct_predictions == 85
        assert rule.occurrences == 3
        assert rule.aggregated_confidence == 2.55

    def test_confidence_clamping(self):
        """Test that confidence is clamped to 0-1 range."""
        rule_low = RuleBuilder().with_id("low").with_confidence(-0.5).with_head("p", ["A"]).build()
        rule_high = RuleBuilder().with_id("high").with_confidence(1.5).with_head("p", ["A"]).build()

        assert rule_low.confidence == 0.0
        assert rule_high.confidence == 1.0

    def test_missing_id_raises(self):
        """Test that missing ID raises ValueError."""
        with pytest.raises(ValueError, match="Rule ID is required"):
            RuleBuilder().with_head("knows", ["A", "B"]).build()

    def test_missing_head_raises(self):
        """Test that missing head raises ValueError."""
        with pytest.raises(ValueError, match="Rule head is required"):
            RuleBuilder().with_id("test").build()

    def test_with_head_dict(self):
        """Test setting head from dictionary."""
        head = {"predicate": "knows", "args": ["X", "Y"]}
        rule = RuleBuilder().with_id("dict_001").with_head_dict(head).build()

        assert rule.head == head

    def test_with_body_list(self):
        """Test setting entire body from list."""
        body = [
            {"predicate": "friend", "args": ["A", "B"]},
            {"predicate": "lives", "args": ["B", "City"]},
        ]
        rule = RuleBuilder().with_id("body_001").with_head("knows", ["A"]).with_body(body).build()

        assert rule.body == body

    def test_from_pattern_string(self):
        """Test parsing Datalog-like pattern string."""
        rule = (RuleBuilder()
            .with_id("pattern_001")
            .with_confidence(0.9)
            .from_pattern_string("knows(A, B) <= friend(A, C), friend(C, B)")
            .from_source("manual")
            .build())

        assert rule.head["predicate"] == "knows"
        assert rule.head["args"] == ["A", "B"]
        assert len(rule.body) == 2
        assert rule.body[0]["predicate"] == "friend"
        assert rule.body[1]["predicate"] == "friend"

    def test_from_pattern_string_arrow_syntax(self):
        """Test parsing with <- arrow syntax."""
        rule = (RuleBuilder()
            .with_id("arrow_001")
            .from_pattern_string("result(X) <- input(X)")
            .build())

        assert rule.head["predicate"] == "result"
        assert len(rule.body) == 1


class TestParsePattern:
    """Tests for the _parse_pattern helper function."""

    def test_simple_pattern(self):
        """Test parsing simple pattern."""
        head, body = _parse_pattern("knows(A, B)")
        assert head["predicate"] == "knows"
        assert head["args"] == ["A", "B"]
        assert body == []

    def test_pattern_with_body(self):
        """Test parsing pattern with body clauses."""
        head, body = _parse_pattern("knows(A, B) <= friend(A, C), friend(C, B)")
        assert head["predicate"] == "knows"
        assert len(body) == 2

    def test_single_body_clause(self):
        """Test parsing with single body clause."""
        head, body = _parse_pattern("derived(X) <= source(X)")
        assert len(body) == 1
        assert body[0]["predicate"] == "source"


class TestRuleSourceFactory:
    """Tests for RuleSourceFactory."""

    def test_get_available_sources(self):
        """Test listing available source types."""
        sources = RuleSourceFactory.get_available_sources()
        assert "manual" in sources
        assert "anyburl" in sources
        assert "json" in sources
        assert "tsv" in sources

    def test_auto_detect_json(self):
        """Test auto-detection of JSON extension."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write('{"rules": []}')
            filepath = Path(f.name)

        try:
            # Should not raise - just return empty list
            rules = RuleSourceFactory.load_rules(filepath)
            assert isinstance(rules, list)
        finally:
            filepath.unlink()

    def test_auto_detect_tsv(self):
        """Test auto-detection of TSV extension."""
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
            # Empty file or minimal content
            f.write("")
            filepath = Path(f.name)

        try:
            rules = RuleSourceFactory.load_rules(filepath)
            assert isinstance(rules, list)
        finally:
            filepath.unlink()

    def test_unknown_extension_raises(self):
        """Test that unknown extension raises ValueError."""
        filepath = Path("/tmp/rules.unknown")
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            RuleSourceFactory.load_rules(filepath)

    def test_unknown_source_type_raises(self):
        """Test that unknown source type raises ValueError."""
        filepath = Path("/tmp/rules.json")
        with pytest.raises(ValueError, match="Unknown rule source type"):
            RuleSourceFactory.load_rules(filepath, source_type="nonexistent")

    def test_register_custom_source(self):
        """Test registering a custom source type."""
        from pff.services.rule_builder import RuleSource

        class CustomSource(RuleSource):
            def load(self, filepath: Path) -> list[Rule]:
                return []

        RuleSourceFactory.register_source("custom", CustomSource)
        assert "custom" in RuleSourceFactory.get_available_sources()


class TestManualRuleSource:
    """Tests for ManualRuleSource."""

    def test_load_valid_rules(self):
        """Test loading valid manual rules."""
        rules_json = """{
            "category1": [
                {"id": "rule_1", "confidence": 0.9, "pattern": "knows(A,B) <= friend(A,B)"},
                {"id": "rule_2", "confidence": 0.8, "pattern": "likes(A,B) <= follows(A,B)"}
            ]
        }"""

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write(rules_json)
            filepath = Path(f.name)

        try:
            source = ManualRuleSource()
            rules = source.load(filepath)
            assert len(rules) == 2
            assert rules[0].id == "rule_1"
            assert rules[0].confidence == 0.9
            assert rules[0].source == "manual"
        finally:
            filepath.unlink()

    def test_load_missing_file(self):
        """Test loading from non-existent file returns empty list."""
        source = ManualRuleSource()
        rules = source.load(Path("/nonexistent/path/rules.json"))
        assert rules == []


class TestAnyBURLRuleSource:
    """Tests for AnyBURLRuleSource."""

    def test_load_valid_tsv(self):
        """Test loading valid AnyBURL TSV."""
        tsv_content = "100\t85\t0.85\tknows(A,B) <= friend(A,C), friend(C,B)\n"
        tsv_content += "50\t40\t0.80\tlikes(X,Y) <= follows(X,Y)\n"

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
            f.write(tsv_content)
            filepath = Path(f.name)

        try:
            source = AnyBURLRuleSource()
            rules = source.load(filepath)
            assert len(rules) == 2
            assert rules[0].total_predictions == 100
            assert rules[0].correct_predictions == 85
            assert rules[0].confidence == 0.85
            assert rules[0].source == "anyburl"
        finally:
            filepath.unlink()

    def test_load_missing_file(self):
        """Test loading from non-existent file returns empty list."""
        source = AnyBURLRuleSource()
        rules = source.load(Path("/nonexistent/path/rules.tsv"))
        assert rules == []
