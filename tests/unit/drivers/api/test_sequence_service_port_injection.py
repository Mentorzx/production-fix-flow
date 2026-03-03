"""Regression tests for SequenceService file-manager port injection."""

from __future__ import annotations

import pytest

from pff.application.services.sequence_service import SequenceService


class _FakeFileManager:
    def __init__(self, sequences: dict[str, list[dict]]) -> None:
        self._sequences = sequences
        self.read_calls = 0

    def read(self, _path, **_kwargs):  # noqa: ANN001
        self.read_calls += 1
        return self._sequences


@pytest.mark.asyncio
async def test_sequence_service_uses_injected_file_manager_port() -> None:
    """SequenceService must read sequence definitions through injected file-manager port."""

    fake_manager = _FakeFileManager({"demo": [{"set": "flag", "value": True}]})
    service = SequenceService({}, file_manager=fake_manager)

    await service.run("5511999999999", "demo")

    assert fake_manager.read_calls == 1
