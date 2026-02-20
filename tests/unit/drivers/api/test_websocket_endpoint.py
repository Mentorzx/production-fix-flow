"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/drivers/api/test_websocket_endpoint.py

"""

from __future__ import annotations

import pytest
from fastapi import WebSocketDisconnect

from pff.drivers.api.routers import websocket


class _FakeManager:
    def __init__(self) -> None:
        """Execute init."""

        self.connected: list[str] = []
        self.disconnected: list[str] = []
        self.subscriptions: list[tuple[str, str]] = []
        self.unsubscriptions: list[tuple[str, str]] = []

    async def connect(self, _websocket, client_id: str) -> None:
        """Execute connect.



        Args:

            _websocket: Input value used by this callable.

            client_id: Input value used by this callable.

        """

        self.connected.append(client_id)

    def disconnect(self, client_id: str) -> None:
        """Execute disconnect.



        Args:

            client_id: Input value used by this callable.

        """

        self.disconnected.append(client_id)

    async def subscribe_to_execution(self, client_id: str, exec_id: str) -> None:
        """Execute subscribe to execution.



        Args:

            client_id: Input value used by this callable.

            exec_id: Input value used by this callable.

        """

        self.subscriptions.append((client_id, exec_id))

    async def unsubscribe_from_execution(self, client_id: str, exec_id: str) -> None:
        """Execute unsubscribe from execution.



        Args:

            client_id: Input value used by this callable.

            exec_id: Input value used by this callable.

        """

        self.unsubscriptions.append((client_id, exec_id))


class _FakeWebSocket:
    def __init__(
        self, receive_json_values: list[object], *, receive_text_value: str = "not-json"
    ) -> None:
        """Execute init.



        Args:

            receive_json_values: Input value used by this callable.

            receive_text_value: Optional input value.

        """

        self._receive_json_values = list(receive_json_values)
        self._receive_text_value = receive_text_value
        self.sent_json: list[dict] = []

    async def receive_json(self):
        """Execute receive json.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not self._receive_json_values:
            raise WebSocketDisconnect(code=1000)
        value = self._receive_json_values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def receive_text(self) -> str:
        """Execute receive text.



        Returns:

            Return value produced by the callable.

        """

        return self._receive_text_value

    async def send_json(self, payload: dict) -> None:
        """Execute send json.



        Args:

            payload: Input value used by this callable.

        """

        self.sent_json.append(payload)


@pytest.mark.asyncio
async def test_websocket_endpoint_subscribe_and_disconnect(monkeypatch) -> None:
    """Execute test websocket endpoint subscribe and disconnect.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    fake_manager = _FakeManager()
    fake_socket = _FakeWebSocket(
        [
            {"action": "subscribe", "execution_id": "exec-1"},
            WebSocketDisconnect(code=1000),
        ]
    )
    monkeypatch.setattr(websocket, "manager", fake_manager)

    await websocket.websocket_endpoint(fake_socket, "client-1")

    assert fake_manager.connected == ["client-1"]
    assert fake_manager.subscriptions == [("client-1", "exec-1")]
    assert fake_manager.disconnected == ["client-1"]
    assert fake_socket.sent_json[0]["type"] == "subscribed"


@pytest.mark.asyncio
async def test_websocket_endpoint_handles_invalid_json(monkeypatch) -> None:
    """Execute test websocket endpoint handles invalid json.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    fake_manager = _FakeManager()
    fake_socket = _FakeWebSocket(
        [
            Exception("invalid json body"),
            WebSocketDisconnect(code=1000),
        ],
        receive_text_value="{invalid-json}",
    )
    monkeypatch.setattr(websocket, "manager", fake_manager)

    await websocket.websocket_endpoint(fake_socket, "client-2")

    assert fake_manager.connected == ["client-2"]
    assert fake_manager.disconnected == ["client-2"]
    assert fake_socket.sent_json[0] == {"type": "error", "message": "Invalid JSON format"}
