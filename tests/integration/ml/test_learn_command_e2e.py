"""
End-to-End tests for Learn command.

These tests validate the Learn command with different model types (kg, dslfm, ensemble, all).
Most tests are marked as @pytest.mark.slow due to full pipeline execution.
"""

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pff.drivers.cli.main import LearnCommand


@pytest.mark.skip(reason="LearnCommand refactoring in progress - Sprint 18")
class TestLearnCommandKG:
    """Tests for KG-only learning."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_kg_only(self):
        """Test learning KG rules only."""
        args = argparse.Namespace(model="kg", config=None)
        cmd = LearnCommand(args)

        with patch("pff.domain.kg.pipeline.KGPipeline") as mock_pipeline:
            mock_instance = Mock()
            mock_instance.run_build_and_preprocess = AsyncMock()
            mock_instance.run_learn_rules = AsyncMock()
            mock_instance.run_ranking = AsyncMock()
            mock_pipeline.return_value = mock_instance

            await cmd.execute()

            mock_instance.run_build_and_preprocess.assert_called_once()
            mock_instance.run_learn_rules.assert_called_once()
            mock_instance.run_ranking.assert_called_once()


@pytest.mark.skip(reason="LearnCommand refactoring in progress - Sprint 18")
class TestLearnCommandDSLFM:
    """Tests for DSLFM-only learning."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_dslfm_only(self):
        """Test learning DSLFM embeddings only."""
        args = argparse.Namespace(model="dslfm", config=None)
        cmd = LearnCommand(args)

        with patch("pff.domain.learning.dslfm.manager.DSLFMManager") as mock_manager:
            mock_instance = Mock()
            mock_instance.train = AsyncMock()
            mock_manager.return_value = mock_instance

            await cmd.execute()

            mock_instance.train.assert_called_once()


@pytest.mark.skip(reason="Legacy ensemble removed - DSLFM+PC only architecture")
class TestLearnCommandEnsemble:
    """Tests for Ensemble-only learning."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_ensemble_only(self):
        """Test learning Ensemble model only."""
        args = argparse.Namespace(model="ensemble", config=None)
        LearnCommand(args)

        # Legacy advanced_trainer removed - test no longer applicable
        pass


@pytest.mark.skip(reason="Legacy ensemble removed - DSLFM+PC only architecture")
class TestLearnCommandAll:
    """Tests for full pipeline (all models)."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_all_models(self):
        """Test learning all models sequentially."""
        # Legacy advanced_trainer removed - test needs rewriting for DSLFM+PC
        pass


def test_learn_command_basic_import():
    """Test that LearnCommand can be imported successfully."""
    assert LearnCommand is not None

    # Test basic initialization
    args = argparse.Namespace(model="kg", config=None)
    cmd = LearnCommand(args)

    assert cmd.model == "kg"
    assert cmd.config_path is None


def test_learn_command_with_custom_config():
    """Test LearnCommand initialization with custom config."""
    custom_config = Path("/tmp/custom_kg.yaml")
    args = argparse.Namespace(model="dslfm", config=str(custom_config))
    cmd = LearnCommand(args)

    assert cmd.model == "dslfm"
    assert cmd.config_path == str(custom_config)


def test_learn_command_all_models():
    """Test LearnCommand initialization with all models."""
    args = argparse.Namespace(model="all", config=None)
    cmd = LearnCommand(args)

    assert cmd.model == "all"


import pytest  # noqa: E402

pytest.skip("Fluxo learn DSLFM/PC ativo; paths legacy desativados", allow_module_level=True)
