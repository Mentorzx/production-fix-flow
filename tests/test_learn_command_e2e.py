"""
End-to-End tests for Learn command.

These tests validate the Learn command with different model types (kg, transe, ensemble, all).
Most tests are marked as @pytest.mark.slow due to full pipeline execution.
"""
import argparse
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pff.cli import LearnCommand


@pytest.mark.skip(reason="LearnCommand refactoring in progress - Sprint 18")
class TestLearnCommandKG:
    """Tests for KG-only learning."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_kg_only(self):
        """Test learning KG rules only."""
        args = argparse.Namespace(model="kg", config=None)
        cmd = LearnCommand(args)
        
        with patch("pff.validators.kg.pipeline.KGPipeline") as mock_pipeline:
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
class TestLearnCommandTransE:
    """Tests for TransE-only learning."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_transe_only(self):
        """Test learning TransE embeddings only."""
        args = argparse.Namespace(model="transe", config=None)
        cmd = LearnCommand(args)
        
        with patch("pff.validators.transe.transe_pipeline.TransEPipeline") as mock_pipeline:
            mock_instance = Mock()
            mock_instance.train_transe = AsyncMock()
            mock_pipeline.return_value = mock_instance
            
            await cmd.execute()
            
            mock_instance.train_transe.assert_called_once()


@pytest.mark.skip(reason="LearnCommand refactoring in progress - Sprint 18")
class TestLearnCommandEnsemble:
    """Tests for Ensemble-only learning."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_ensemble_only(self):
        """Test learning Ensemble model only."""
        args = argparse.Namespace(model="ensemble", config=None)
        cmd = LearnCommand(args)
        
        with patch("pff.validators.ensembles.advanced_trainer.run_standalone_ensemble_pipeline") as mock_func:
            mock_func.return_value = {"status": "success"}
            
            await cmd.execute()
            
            mock_func.assert_called_once()


@pytest.mark.skip(reason="LearnCommand refactoring in progress - Sprint 18")
class TestLearnCommandAll:
    """Tests for full pipeline (all models)."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_learn_all_models(self):
        """Test learning all models sequentially."""
        args = argparse.Namespace(model="all", config=None)
        cmd = LearnCommand(args)
        
        with patch("pff.validators.kg.pipeline.KGPipeline") as mock_kg, \
             patch("pff.validators.transe.transe_pipeline.TransEPipeline") as mock_transe, \
             patch("pff.validators.ensembles.advanced_trainer.run_standalone_ensemble_pipeline") as mock_ensemble:
            
            mock_kg_instance = Mock()
            mock_kg_instance.run_build_and_preprocess = AsyncMock()
            mock_kg_instance.run_learn_rules = AsyncMock()
            mock_kg_instance.run_ranking = AsyncMock()
            mock_kg.return_value = mock_kg_instance
            
            mock_transe_instance = Mock()
            mock_transe_instance.train_transe = AsyncMock()
            mock_transe.return_value = mock_transe_instance
            
            mock_ensemble.return_value = {"status": "success"}
            
            await cmd.execute()
            
            # Validate all pipelines were called
            mock_kg_instance.run_build_and_preprocess.assert_called_once()
            mock_transe_instance.train_transe.assert_called_once()
            mock_ensemble.assert_called_once()


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
    args = argparse.Namespace(model="transe", config=str(custom_config))
    cmd = LearnCommand(args)
    
    assert cmd.model == "transe"
    assert cmd.config_path == str(custom_config)


def test_learn_command_all_models():
    """Test LearnCommand initialization with all models."""
    args = argparse.Namespace(model="all", config=None)
    cmd = LearnCommand(args)
    
    assert cmd.model == "all"
