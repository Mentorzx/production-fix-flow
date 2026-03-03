"""Regression tests for LineService port injection support."""

from __future__ import annotations

import pytest

from pff.application.services.line_service import LineService


class _FakeHttpClient:
    def __init__(self) -> None:
        self.closed = False
        self.requests: list[tuple[dict, dict]] = []

    async def close(self) -> None:
        self.closed = True

    async def make_request(self, endpoint_config: dict, subscriber_data: dict) -> dict | None:
        self.requests.append((endpoint_config, subscriber_data))
        return {"ok": True}

    def _generate_unique_path(self, folder, stem: str, suffix: str):  # noqa: ANN001
        return folder / f"{stem}{suffix}"


class _FakeFileManager:
    def ensure_dir(self, _path):  # noqa: ANN001
        return None

    def save(self, _data, _path, **_kwargs):  # noqa: ANN001
        return None

    def exists(self, _path):  # noqa: ANN001
        return False

    def assert_supported_path(self, _path, **_kwargs):  # noqa: ANN001
        return None

    def read(self, _path, **_kwargs):  # noqa: ANN001
        return {}


class _FakeAPI:
    CUSTOMER_ENQUIRY = {"type": "TEST"}


@pytest.mark.asyncio
async def test_line_service_uses_injected_ports() -> None:
    """LineService must bind injected HTTP and file-manager ports."""

    http_client = _FakeHttpClient()
    file_manager = _FakeFileManager()
    service = LineService(
        http_client=http_client,
        file_manager=file_manager,
        api_client=_FakeAPI(),
    )

    assert service._http_client is http_client
    assert service._file_manager is file_manager

    result = await service.make_request({"type": "TEST"}, {"msisdn": "5511"})
    assert result == {"ok": True}

    await service.close()
    assert http_client.closed is True
