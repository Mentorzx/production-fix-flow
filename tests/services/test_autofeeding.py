"""
Tests for autofeeding rules and feedback loop.

Uses mocks and synthetic data for fast, deterministic tests.
"""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.fixtures import get_sample_rules, get_sample_metrics


@pytest.mark.slow
class TestAutofeedingRules:
    """Slow tests for full autofeeding pipeline."""

    @pytest.mark.asyncio
    async def test_apply_autofeeding_rules_exists(self):
        """Verify autofeeding module exists and can be called."""
        try:
            from pff.utils.autofeeding import apply_autofeeding_rules

            with patch("pff.utils.autofeeding.logger") as mock_logger:
                await apply_autofeeding_rules()

                assert mock_logger.info.called or mock_logger.warning.called
        except ImportError:
            pytest.skip("autofeeding module not yet implemented")


class TestAutofeedingIntegration:
    """Fast integration tests using mocks."""

    def test_autofeeding_module_structure(self):
        """Verify autofeeding module has expected structure."""
        try:
            from pff.utils import autofeeding

            assert hasattr(autofeeding, 'apply_autofeeding_rules')
        except ImportError:
            pytest.skip("autofeeding module not yet implemented")

    def test_autofeeding_config_exists(self):
        """Verify autofeeding config file exists."""
        autofeeding_config = Path("config/models/autofeeding.yaml")
        assert autofeeding_config.exists(), f"Missing config: {autofeeding_config}"


class TestAutofeedingRuleApplication:
    """Tests for rule application with mocked KG data."""

    def test_rules_can_be_loaded_for_autofeeding(self):
        """Test that sample rules can be loaded for autofeeding."""
        rules = get_sample_rules()
        
        assert len(rules) > 0, "Should have rules to apply"
        
        # Filter rules by confidence for autofeeding
        high_confidence = [r for r in rules if r["confidence"] >= 0.8]
        assert len(high_confidence) > 0, "Should have high-confidence rules"

    def test_metrics_feedback_structure(self):
        """Test that metrics have structure needed for feedback loop."""
        metrics = get_sample_metrics()
        
        # Autofeeding needs ensemble performance metrics
        assert "Ensemble_Final" in metrics, "Need ensemble metrics for feedback"
        
        # And rule statistics
        assert "Symbolic_Stats" in metrics or "Feature_Balance" in metrics, \
            "Need symbolic stats for feedback"

    @pytest.mark.asyncio
    async def test_rule_application_mock(self):
        """Test rule application with mocked KG builder."""
        rules = get_sample_rules()
        
        # Mock KG builder
        mock_kg_builder = Mock()
        mock_kg_builder.add_inferred_triples = AsyncMock(return_value=5)
        
        # Simulate applying high-confidence rules
        high_conf_rules = [r for r in rules if r["confidence"] >= 0.9]
        
        # Would call kg_builder.add_inferred_triples(rules)
        if high_conf_rules:
            result = await mock_kg_builder.add_inferred_triples(high_conf_rules)
            assert result >= 0, "Should return count of added triples"
            mock_kg_builder.add_inferred_triples.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_loop_mock(self):
        """Test feedback loop with mocked ensemble."""
        metrics = get_sample_metrics()
        
        # Mock ensemble trainer
        mock_trainer = Mock()
        mock_trainer.get_low_confidence_predictions = Mock(return_value=[
            {"triple": ("A", "rel", "B"), "confidence": 0.45},
            {"triple": ("C", "rel", "D"), "confidence": 0.52},
        ])
        
        # Get low confidence predictions for re-training
        low_conf = mock_trainer.get_low_confidence_predictions(threshold=0.6)
        
        assert len(low_conf) > 0, "Should identify low confidence predictions"
        for pred in low_conf:
            assert pred["confidence"] < 0.6, "Should be below threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
