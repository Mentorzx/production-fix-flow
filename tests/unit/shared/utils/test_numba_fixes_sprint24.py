"""Tests for RuleEncoder determinism (Sprint 24 fixes).

Validates:
1. Variable encoding is deterministic (BLAKE3 hash-based).
2. Vocabulary building via build_vocabulary_from_rules is stable.
3. Constants vs variables produce distinct index ranges.
"""

from __future__ import annotations

from pff_rust import RuleEncoder


class TestVariableEncodingDeterministic:
    """Test that variable encoding is deterministic."""

    def test_same_variable_same_encoding(self) -> None:
        """Same variable should get same encoding every time."""
        encoder = RuleEncoder()
        x1 = encoder.encode_entity("X")
        x2 = encoder.encode_entity("X")
        assert x1 == x2
        assert encoder.is_variable(x1)

    def test_different_variables_different_encoding(self) -> None:
        """Different variables should get different encodings."""
        encoder = RuleEncoder()
        x = encoder.encode_entity("X")
        y = encoder.encode_entity("Y")
        z = encoder.encode_entity("Z")
        assert x != y
        assert y != z
        assert x != z
        assert all(encoder.is_variable(v) for v in [x, y, z])

    def test_deterministic_across_instances(self) -> None:
        """Same variable should encode same way in different encoder instances."""
        encoder1 = RuleEncoder()
        encoder2 = RuleEncoder()
        x1 = encoder1.encode_entity("X")
        x2 = encoder2.encode_entity("X")
        assert x1 == x2

    def test_constant_different_from_variable(self) -> None:
        """Constants should encode differently from variables."""
        encoder = RuleEncoder()
        x_var = encoder.encode_entity("X")
        x_const = encoder.encode_entity("x")
        assert x_var != x_const
        assert encoder.is_variable(x_var)
        assert not encoder.is_variable(x_const)


class TestBuildVocabularyFromRules:
    """Test build_vocabulary_from_rules determinism."""

    SAMPLE_RULES = [
        {
            "head": {"subject": "X", "predicate": "hasType", "object": "Premium"},
            "body": [{"subject": "X", "predicate": "hasValue", "object": "Y"}],
        },
        {
            "head": {"subject": "X", "predicate": "invalid", "object": "Y"},
            "body": [
                {"subject": "X", "predicate": "pred1", "object": "Z"},
                {"subject": "Z", "predicate": "pred2", "object": "Y"},
            ],
        },
    ]

    def test_vocabulary_built_flag(self) -> None:
        """build_vocabulary_from_rules should set vocabulary_built."""
        encoder = RuleEncoder()
        encoder.build_vocabulary_from_rules(self.SAMPLE_RULES)
        n_preds, n_ents, built = encoder.get_stats()
        assert built is True
        assert n_preds > 0

    def test_deterministic_across_runs(self) -> None:
        """Multiple builds should produce identical stats."""
        stats_list = []
        for _ in range(3):
            enc = RuleEncoder()
            enc.build_vocabulary_from_rules(self.SAMPLE_RULES)
            stats_list.append(enc.get_stats())
        assert all(s == stats_list[0] for s in stats_list)

    def test_encode_predicate_after_build(self) -> None:
        """Pre-built predicates should have stable indices."""
        enc1 = RuleEncoder()
        enc1.build_vocabulary_from_rules(self.SAMPLE_RULES)
        enc2 = RuleEncoder()
        enc2.build_vocabulary_from_rules(self.SAMPLE_RULES)
        assert enc1.encode_predicate("hasType") == enc2.encode_predicate("hasType")
        assert enc1.encode_predicate("pred1") == enc2.encode_predicate("pred1")


class TestEncodeTriples:
    """Test encode_triples output."""

    def test_encode_triples_returns_flat_array(self) -> None:
        """encode_triples should return a flat i32 array of length 3*n."""
        encoder = RuleEncoder()
        triples = [
            ("entity1", "hasValue", "value1"),
            ("entity2", "hasType", "Customer"),
        ]
        encoded = encoder.encode_triples(triples)
        assert len(encoded) == 6
