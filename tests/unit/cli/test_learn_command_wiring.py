from __future__ import annotations

from pathlib import Path

import pytest

from pff.drivers.cli.internal import commands


@pytest.mark.asyncio
async def test_run_learn_wires_use_case(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}
    sentinel_registry = object()
    config_path = Path("config/models/kg.yaml")

    class DummyUseCase:
        def __init__(
            self,
            config_path: Path | None = None,
            strategy_registry=None,
            checkpoints_repo=None,
            splits_repo=None,
        ) -> None:
            self.config_path = config_path
            self.strategy_registry = strategy_registry
            self.called_with: str | None = None
            created["instance"] = self

        async def execute(self, model: str) -> None:
            self.called_with = model

    monkeypatch.setattr(commands, "LearnUseCase", DummyUseCase)
    monkeypatch.setattr(commands, "get_strategy_registry", lambda: sentinel_registry)

    await commands._run_learn("kgc", config_path=config_path)

    instance = created.get("instance")
    assert instance is not None
    assert instance.config_path == config_path
    assert instance.strategy_registry is sentinel_registry
    assert instance.called_with == "kgc"
