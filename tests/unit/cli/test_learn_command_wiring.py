"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/cli/test_learn_command_wiring.py

"""

from __future__ import annotations

from pathlib import Path

import pytest

from pff.drivers.cli.internal import commands


@pytest.mark.asyncio
async def test_run_learn_wires_use_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute test run learn wires use case.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    created: dict[str, object] = {}
    sentinel_registry = object()
    config_path = Path("config/models/kg.yaml")

    class DummyUseCase:
        """Represent DummyUseCase.



        Notes:

            Encapsulates behavior while preserving architecture boundaries.

        """

        def __init__(
            self,
            config_path: Path | None = None,
            strategy_registry=None,
            checkpoints_repo=None,
            splits_repo=None,
        ) -> None:
            """Execute init.



            Args:

                config_path: Optional input value.

                strategy_registry: Optional input value.

                checkpoints_repo: Optional input value.

                splits_repo: Optional input value.

            """

            self.config_path = config_path
            self.strategy_registry = strategy_registry
            self.called_with: str | None = None
            created["instance"] = self

        async def execute(self, model: str) -> None:
            """Execute execute.



            Args:

                model: Input value used by this callable.

            """

            self.called_with = model

    monkeypatch.setattr("pff.application.learn_use_case.LearnUseCase", DummyUseCase)
    monkeypatch.setattr(
        "pff.application.strategy_registry.get_strategy_registry",
        lambda: sentinel_registry,
    )

    await commands._run_learn("kgc", config_path=config_path)

    instance = created.get("instance")
    assert instance is not None
    assert instance.config_path == config_path
    assert instance.strategy_registry is sentinel_registry
    assert instance.called_with == "kgc"
