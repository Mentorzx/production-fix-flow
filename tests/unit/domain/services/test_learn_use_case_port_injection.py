"""Regression tests for LearnUseCase file-manager port forwarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pff.application.learn_use_case import LearnUseCase
from pff.application.strategy_registry import KGCStrategyRegistry


class _FakeFileManager:
    def read(self, path: Path | str, **kwargs: Any) -> Any:  # noqa: ARG002
        return {}

    def save(self, data: Any, path: Path | str, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    def exists(self, path: Path | str) -> bool:  # noqa: ARG002
        return True

    def get_mtime(self, path: Path | str) -> float | None:  # noqa: ARG002
        return None

    def mkdir(self, path: Path | str, parents: bool = True, exist_ok: bool = True) -> None:  # noqa: ARG002
        return None

    def delete_file(self, path: Path | str, ignore_errors: bool = True) -> None:  # noqa: ARG002
        return None

    def glob(self, directory: Path | str, pattern: str) -> list[Path]:  # noqa: ARG002
        return []

    def read_bytes(self, path: Path | str) -> bytes:  # noqa: ARG002
        return b""

    def write_bytes(self, data: bytes, path: Path | str) -> None:  # noqa: ARG002
        return None


@pytest.mark.asyncio
async def test_learn_use_case_forwards_injected_file_manager_to_strategy() -> None:
    """LearnUseCase must pass injected file_manager to strategy constructor."""
    captured = {}
    execute_mock = AsyncMock()

    class StubStrategy:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            captured["file_manager"] = kwargs.get("file_manager")
            captured["settings_obj"] = kwargs.get("settings_obj")

        async def execute(self):
            await execute_mock()

    class _FakeSettings:
        DATA_DIR = Path("/tmp/pff_data")
        OUTPUTS_DIR = Path("/tmp/pff_outputs")
        CACHE_DIR = Path("/tmp/pff_cache")
        PATTERNS_DIR = Path("/tmp/pff_patterns")

    registry = KGCStrategyRegistry()
    registry.register("kg", StubStrategy)

    fake_manager = _FakeFileManager()
    fake_settings = _FakeSettings()
    use_case = LearnUseCase(
        config_path=Path("/dev/null"),
        strategy_registry=registry,
        file_manager=fake_manager,
        settings_obj=fake_settings,
    )

    await use_case.execute("kg")

    assert captured["file_manager"] is fake_manager
    assert captured["settings_obj"] is fake_settings
    execute_mock.assert_awaited_once()
