"""Test XGBoost rule extraction bug fix."""
import pytest
from pff.validators.ensembles.ensemble_rules_extractor import EnsembleRulesExtractor


def test_normalize_tree_node_with_int():
    """Test that _normalize_tree_node handles integer nodes gracefully."""
    extractor = EnsembleRulesExtractor()
    
    # Test with integer (should return empty dict)
    result = extractor._normalize_tree_node(5)
    assert result == {}
    
    # Test with None
    result = extractor._normalize_tree_node(None)
    assert result == {}
    
    # Test with string
    result = extractor._normalize_tree_node("test")
    assert result == {}
    
    # Test with valid dict
    result = extractor._normalize_tree_node({"leaf": 0.5})
    assert result == {"leaf": 0.5}
    
    # Test with split node
    result = extractor._normalize_tree_node({
        "split": "f151",
        "split_condition": 0.5,
        "yes": 1,
        "no": 2
    })
    assert "split" in result
    assert "split_condition" in result
    assert "yes" in result
    assert "no" in result


def test_normalize_tree_node_various_formats():
    """Test normalization of different XGBoost tree formats."""
    extractor = EnsembleRulesExtractor()
    
    # Format 1: "split" + "split_condition" + "yes"/"no"
    node1 = {
        "split": "f10",
        "split_condition": 0.75,
        "yes": {"leaf": 0.5},
        "no": {"leaf": -0.3}
    }
    result1 = extractor._normalize_tree_node(node1)
    assert result1["split"] == "f10"
    assert result1["split_condition"] == 0.75
    
    # Format 2: "feature" + "threshold" + "left"/"right"
    node2 = {
        "feature": 5,
        "threshold": 0.5,
        "left": {"leaf": 0.2},
        "right": {"leaf": -0.1}
    }
    result2 = extractor._normalize_tree_node(node2)
    assert result2["split"] == 5
    assert result2["split_condition"] == 0.5
    assert "yes" in result2  # Normalized from "left"
    assert "no" in result2   # Normalized from "right"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
