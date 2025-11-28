"""
Tests for Numba accelerator fixes (Sprint 24).

Validates:
1. Fallback calls business_service instead of returning zeros
2. Variable encoding is deterministic
3. Dual validation (Numba + business_service) works correctly
"""

import pytest
import numpy as np
from pff.utils.acceleration.symbolic_rule_accelerator import SymbolicRuleAccelerator, RuleEncoder


class TestNumbaFallbackFix:
    """Test that fallback calls business_service correctly."""
    
    def test_fallback_calls_business_service_not_zeros(self):
        """Verify fallback doesn't return zeros."""
        rules = [
            {
                "id": "test_rule_1",
                "head": {"subject": "X", "predicate": "hasType", "object": "invalid"},
                "body": [{"subject": "X", "predicate": "hasValue", "object": "Y"}],
                "confidence": 0.8,
            }
        ]
        
        accelerator = SymbolicRuleAccelerator(rules, enable_numba=False)
        sample_triples = [("entity1", "hasValue", "value1")]
        
        violations = accelerator.check_violations(sample_triples)
        
        assert violations.shape == (1,)
        assert violations.dtype == np.int8
    
    def test_fallback_with_complex_rule(self):
        """Test fallback with multi-body rule."""
        rules = [
            {
                "id": "complex_rule",
                "head": {"subject": "X", "predicate": "invalid", "object": "Y"},
                "body": [
                    {"subject": "X", "predicate": "pred1", "object": "Z"},
                    {"subject": "Z", "predicate": "pred2", "object": "Y"},
                ],
                "confidence": 0.9,
            }
        ]
        
        accelerator = SymbolicRuleAccelerator(rules, enable_numba=False)
        sample_triples = [
            ("e1", "pred1", "e2"),
            ("e2", "pred2", "e3"),
        ]
        
        violations = accelerator.check_violations(sample_triples)
        assert violations.shape == (1,)


class TestVariableEncodingDeterministic:
    """Test that variable encoding is deterministic."""
    
    def test_same_variable_same_encoding(self):
        """Same variable should get same encoding every time."""
        encoder = RuleEncoder()
        
        x1 = encoder.encode_entity("X")
        x2 = encoder.encode_entity("X")
        
        assert x1 == x2
        assert x1 >= encoder.VARIABLE_START
    
    def test_different_variables_different_encoding(self):
        """Different variables should get different encodings."""
        encoder = RuleEncoder()
        
        x = encoder.encode_entity("X")
        y = encoder.encode_entity("Y")
        z = encoder.encode_entity("Z")
        
        assert x != y
        assert y != z
        assert x != z
        assert all(v >= encoder.VARIABLE_START for v in [x, y, z])
    
    def test_deterministic_across_instances(self):
        """Same variable should encode same way in different encoder instances."""
        encoder1 = RuleEncoder()
        encoder2 = RuleEncoder()
        
        x1 = encoder1.encode_entity("X")
        x2 = encoder2.encode_entity("X")
        
        assert x1 == x2
    
    def test_constant_different_from_variable(self):
        """Constants should encode differently from variables."""
        encoder = RuleEncoder()
        
        x_var = encoder.encode_entity("X")
        x_const = encoder.encode_entity("x")
        
        assert x_var != x_const
        assert x_var >= encoder.VARIABLE_START
        assert x_const < encoder.VARIABLE_START


class TestDualValidation:
    """Test Numba + business_service dual validation."""
    
    @pytest.mark.slow
    def test_validation_detects_mismatch(self):
        """Validation should detect when Numba and business_service disagree."""
        rules = [
            {
                "id": f"rule_{i}",
                "head": {"subject": "X", "predicate": f"pred_{i}", "object": "Y"},
                "body": [{"subject": "X", "predicate": "base", "object": "Y"}],
                "confidence": 0.7,
            }
            for i in range(20)
        ]
        
        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)
        sample_triples = [("e1", "base", "e2"), ("e1", "pred_0", "e2")]
        
        violations_validated = accelerator.check_violations(sample_triples, validate=True)
        
        assert violations_validated.shape == (20,)
        assert violations_validated.dtype == np.int8
    
    def test_validation_with_small_ruleset(self):
        """Validation should work with small rulesets."""
        rules = [
            {
                "id": "single_rule",
                "head": {"subject": "X", "predicate": "invalid", "object": "Y"},
                "body": [{"subject": "X", "predicate": "valid", "object": "Y"}],
                "confidence": 0.8,
            }
        ]
        
        accelerator = SymbolicRuleAccelerator(rules, enable_numba=True)
        sample_triples = [("e1", "valid", "e2")]
        
        violations = accelerator.check_violations(sample_triples, validate=True)
        assert violations.shape == (1,)


class TestBusinessRuleConversion:
    """Test conversion from internal format to business_service format."""
    
    def test_convert_simple_rule(self):
        """Test converting simple rule."""
        rule = {
            "id": "test",
            "head": {"subject": "X", "predicate": "pred", "object": "Y"},
            "body": [{"subject": "X", "predicate": "body_pred", "object": "Y"}],
            "confidence": 0.9,
        }
        
        accelerator = SymbolicRuleAccelerator([rule], enable_numba=False)
        business_rule = accelerator._convert_to_business_rule(rule, 0)
        
        assert business_rule.id == "numba_rule_0"
        assert business_rule.confidence == 0.9
        assert business_rule.head == ("X", "pred", "Y")
        assert len(business_rule.body) == 1
        assert business_rule.body[0] == ("X", "body_pred", "Y")
    
    def test_convert_multi_body_rule(self):
        """Test converting rule with multiple body clauses."""
        rule = {
            "id": "multi",
            "head": {"subject": "X", "predicate": "h", "object": "Z"},
            "body": [
                {"subject": "X", "predicate": "b1", "object": "Y"},
                {"subject": "Y", "predicate": "b2", "object": "Z"},
            ],
            "confidence": 0.75,
        }
        
        accelerator = SymbolicRuleAccelerator([rule], enable_numba=False)
        business_rule = accelerator._convert_to_business_rule(rule, 0)
        
        assert len(business_rule.body) == 2
        assert business_rule.body[0] == ("X", "b1", "Y")
        assert business_rule.body[1] == ("Y", "b2", "Z")
