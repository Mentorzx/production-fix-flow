"""Integration tests for learn command (DSLFM+PC architecture).

Validates the LearnCommand → LearnUseCase → Strategy wiring with
mocked strategy execution (no GPU / DB required).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pff.application.errors import StrategyResolutionError
from pff.application.learn_use_case import LearnUseCase
from pff.application.strategy_registry import KGCStrategyRegistry
from pff.drivers.cli.internal.commands import LearnCommand


class TestLearnCommandInit:
    """Verify LearnCommand initialization for each model type."""

    def test_learn_command_kg(self):
        """LearnCommand accepts model='kg'."""
        args = argparse.Namespace(model="kg", config=None)
        cmd = LearnCommand(args)
        assert cmd.model == "kg"
        assert cmd.config_path is None

    def test_learn_command_kgc(self):
        """LearnCommand accepts model='kgc'."""
        args = argparse.Namespace(model="kgc", config=None)
        cmd = LearnCommand(args)
        assert cmd.model == "kgc"

    def test_learn_command_all(self):
        """LearnCommand accepts model='all'."""
        args = argparse.Namespace(model="all", config=None)
        cmd = LearnCommand(args)
        assert cmd.model == "all"

    def test_learn_command_custom_config(self):
        """LearnCommand stores a custom config path."""
        args = argparse.Namespace(model="kgc", config="/tmp/custom.yaml")
        cmd = LearnCommand(args)
        assert cmd.config_path == "/tmp/custom.yaml"


class TestLearnUseCaseStrategyResolution:
    """Verify LearnUseCase routes to the correct strategy."""

    @pytest.mark.asyncio
    async def test_kg_strategy_resolved(self):
        """LearnUseCase resolves 'kg' to KGTrainingStrategy."""
        mock_execute = AsyncMock()
        registry = KGCStrategyRegistry()
        registry.register("kg", _make_stub_strategy(mock_execute))

        use_case = LearnUseCase(
            config_path=Path("/dev/null"),
            strategy_registry=registry,
        )
        await use_case.execute("kg")
        mock_execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kgc_strategy_resolved(self):
        """LearnUseCase resolves 'kgc' to KGCTrainingStrategy."""
        mock_execute = AsyncMock()
        registry = KGCStrategyRegistry()
        registry.register("kgc", _make_stub_strategy(mock_execute))

        use_case = LearnUseCase(
            config_path=Path("/dev/null"),
            strategy_registry=registry,
        )
        await use_case.execute("kgc")
        mock_execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_strategy_resolved(self):
        """LearnUseCase resolves 'all' to FullPipelineStrategy."""
        mock_execute = AsyncMock()
        registry = KGCStrategyRegistry()
        registry.register("all", _make_stub_strategy(mock_execute))

        use_case = LearnUseCase(
            config_path=Path("/dev/null"),
            strategy_registry=registry,
        )
        await use_case.execute("all")
        mock_execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_model_raises(self):
        """LearnUseCase raises StrategyResolutionError for unknown models."""
        registry = KGCStrategyRegistry()
        use_case = LearnUseCase(
            config_path=Path("/dev/null"),
            strategy_registry=registry,
        )
        with pytest.raises(StrategyResolutionError, match="Unknown training strategy"):
            await use_case.execute("nonexistent")


class TestLearnCommandE2EWiring:
    """Verify full wiring: LearnCommand.execute → _run_learn → LearnUseCase."""

    @pytest.mark.asyncio
    async def test_execute_delegates_to_use_case(self):
        """LearnCommand.execute() calls LearnUseCase.execute(model)."""
        args = argparse.Namespace(model="kgc", config=None)
        cmd = LearnCommand(args)

        with patch(
            "pff.application.learn_use_case.LearnUseCase.execute",
            new_callable=AsyncMock,
        ) as mock_uc:
            await cmd.execute()
            mock_uc.assert_awaited_once_with("kgc")

    @pytest.mark.asyncio
    async def test_execute_passes_config_path(self):
        """LearnCommand forwards config_path to _run_learn."""
        args = argparse.Namespace(model="kg", config="/tmp/my.yaml")
        cmd = LearnCommand(args)

        with patch(
            "pff.drivers.cli.internal.commands._run_learn",
            new_callable=AsyncMock,
        ) as mock_run:
            await cmd.execute()
            mock_run.assert_awaited_once_with("kg", config_path="/tmp/my.yaml")


def _make_stub_strategy(mock_execute: AsyncMock):
    """Create a strategy class stub that delegates execute to a mock."""

    class StubStrategy:
        def __init__(self, *args, **kwargs):
            pass

        async def execute(self):
            await mock_execute()

    return StubStrategy
