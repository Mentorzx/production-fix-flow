from __future__ import annotations

import asyncio
from typing import Any

import orjson
import redis.asyncio as redis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from pff.config import settings
from pff.utils import logger
from pff.utils.cache import CacheManager
from pff.utils.concurrency import ConcurrencyManager

"""
WebSocket module for real-time communication in PFF.

Provides real-time updates for execution progress using WebSocket
connections and Redis pub/sub for message distribution.
"""

router = APIRouter()
cache = CacheManager()
concurrency = ConcurrencyManager()


class ConnectionManager:
    """
    Manages WebSocket connections and message routing.

    Handles connection lifecycle, subscription management, and
    message broadcasting to connected clients.
    """

    def __init__(self):
        """Initialize connection manager with empty connection pools"""
        self.active_connections: dict[str, set[WebSocket]] = {}
        self.user_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection instance
            client_id: Unique identifier for the client
        """
        await websocket.accept()
        self.user_connections[client_id] = websocket
        logger.info(f"WebSocket conectado: {client_id}")

    def disconnect(self, client_id: str) -> None:
        """
        Remove a WebSocket connection and clean up subscriptions.

        Args:
            client_id: Unique identifier for the client
        """
        if client_id in self.user_connections:
            websocket = self.user_connections[client_id]
            del self.user_connections[client_id]

            # Remove from all execution subscriptions
            for exec_id in list(self.active_connections.keys()):
                self.active_connections[exec_id].discard(websocket)
                if not self.active_connections[exec_id]:
                    del self.active_connections[exec_id]

        logger.info(f"WebSocket desconectado: {client_id}")

    async def subscribe_to_execution(self, client_id: str, exec_id: str) -> None:
        """
        Subscribe a client to receive updates for a specific execution.

        Args:
            client_id: Unique identifier for the client
            exec_id: Execution ID to subscribe to
        """
        websocket = self.user_connections.get(client_id)
        if websocket:
            if exec_id not in self.active_connections:
                self.active_connections[exec_id] = set()
            self.active_connections[exec_id].add(websocket)
            logger.info(f"Cliente {client_id} inscrito na execução {exec_id}")

    async def unsubscribe_from_execution(self, client_id: str, exec_id: str) -> None:
        """
        Unsubscribe a client from execution updates.

        Args:
            client_id: Unique identifier for the client
            exec_id: Execution ID to unsubscribe from
        """
        websocket = self.user_connections.get(client_id)
        if websocket and exec_id in self.active_connections:
            self.active_connections[exec_id].discard(websocket)
            if not self.active_connections[exec_id]:
                del self.active_connections[exec_id]
            logger.info(f"Cliente {client_id} desinscrito da execução {exec_id}")

    async def send_personal_message(self, message: str, client_id: str) -> None:
        """
        Send a message to a specific client.

        Args:
            message: Message content to send
            client_id: Target client identifier
        """
        websocket = self.user_connections.get(client_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.exception(f"Erro ao enviar mensagem para {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_to_execution(
        self, exec_id: str, message: dict[str, Any]
    ) -> None:
        """
        Broadcast a message to all clients monitoring an execution.

        Args:
            exec_id: Execution ID to broadcast to
            message: Message data to broadcast
        """
        if exec_id in self.active_connections:
            dead_connections = set()
            for websocket in self.active_connections[exec_id]:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(message)
                except Exception as e:
                    logger.exception(f"Erro ao broadcast para execução {exec_id}: {e}")
                    dead_connections.add(websocket)

            # Remove dead connections
            for websocket in dead_connections:
                self.active_connections[exec_id].discard(websocket)

            if not self.active_connections[exec_id]:
                del self.active_connections[exec_id]


# Global connection manager instance
manager = ConnectionManager()


async def get_redis_pubsub():
    """
    Create Redis connection for pub/sub operations.

    Returns:
        Redis client configured for pub/sub
    """
    client = await redis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/2",  # Using db 2 for pubsub
        decode_responses=True,
    )
    return client


async def redis_listener():
    """
    Background task to listen for Redis pub/sub messages.

    Subscribes to execution_updates channel and broadcasts
    messages to relevant WebSocket clients.
    """
    try:
        redis_client = await get_redis_pubsub()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("execution_updates")

        logger.info("Redis listener iniciado para atualizações de execução")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = orjson.loads(message["data"])
                    exec_id = data.get("execution_id")
                    if exec_id:
                        await manager.broadcast_to_execution(exec_id, data)
                except Exception as e:
                    logger.exception(f"Erro ao processar mensagem Redis: {e}")
    except asyncio.CancelledError:
        logger.info("Redis listener cancelado")
        raise
    except Exception as e:
        logger.error(f"Redis não disponível para WebSocket: {e}")
        logger.warning("WebSocket funcionará sem Redis pub/sub")


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time communication.

    Protocol:
    - Client -> Server:
        {"action": "subscribe", "execution_id": "abc123"}
        {"action": "unsubscribe", "execution_id": "abc123"}
        {"action": "ping"}

    - Server -> Client:
        {"type": "execution_update", "execution_id": "abc123", "status": "running", ...}
        {"type": "execution_complete", "execution_id": "abc123", ...}
        {"type": "execution_error", "execution_id": "abc123", "error": "...", ...}
        {"type": "pong", "timestamp": "..."}

    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
            except Exception:
                # Try to parse as text if JSON fails
                text = await websocket.receive_text()
                try:
                    data = orjson.loads(text)
                except Exception:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON format"}
                    )
                    continue

            action = data.get("action")

            if action == "subscribe":
                exec_id = data.get("execution_id")
                if exec_id:
                    await manager.subscribe_to_execution(client_id, exec_id)
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "execution_id": exec_id,
                            "message": f"Subscribed to execution {exec_id}",
                        }
                    )
                    logger.success(f"Cliente {client_id} inscrito em {exec_id}")
                else:
                    await websocket.send_json(
                        {"type": "error", "message": "Missing execution_id"}
                    )

            elif action == "unsubscribe":
                exec_id = data.get("execution_id")
                if exec_id:
                    await manager.unsubscribe_from_execution(client_id, exec_id)
                    await websocket.send_json(
                        {
                            "type": "unsubscribed",
                            "execution_id": exec_id,
                            "message": f"Unsubscribed from execution {exec_id}",
                        }
                    )
                else:
                    await websocket.send_json(
                        {"type": "error", "message": "Missing execution_id"}
                    )

            elif action == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": asyncio.get_event_loop().time()}
                )

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown action: {action}"}
                )
                logger.warning(f"Ação desconhecida recebida: {action}")

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"Erro no WebSocket para cliente {client_id}: {e}")
        manager.disconnect(client_id)


async def publish_execution_update(
    exec_id: str,
    status: str,
    progress: int,
    current_step: str | None = None,
    error: str | None = None,
    output_file: str | None = None,
):
    """
    Publish execution update to Redis for WebSocket broadcast.

    Args:
        exec_id: Execution identifier
        status: Current execution status
        progress: Progress percentage (0-100)
        current_step: Description of current step
        error: Error message if failed
        output_file: Output file path if completed
    """
    redis_client = await get_redis_pubsub()

    update = {
        "type": "execution_update",
        "execution_id": exec_id,
        "status": status,
        "progress": progress,
        "timestamp": asyncio.get_event_loop().time(),
    }

    if current_step:
        update["current_step"] = current_step
    if error:
        update["error"] = error
        update["type"] = "execution_error"
    if output_file:
        update["output_file"] = output_file
    if status == "completed":
        update["type"] = "execution_complete"

    message = orjson.dumps(update).decode()
    await redis_client.publish("execution_updates", message)
    await redis_client.close()

    logger.info(
        f"Atualização publicada para execução {exec_id}: {status} ({progress}%)"
    )


_listener_task: asyncio.Task | None = None


async def start_redis_listener():
    """
    Start the Redis listener background task.

    Should be called on application startup to enable
    WebSocket message distribution.
    """
    global _listener_task
    _listener_task = asyncio.create_task(redis_listener())
    logger.success("WebSocket Redis listener iniciado")


async def stop_redis_listener():
    """
    Stop the Redis listener background task.

    Should be called on application shutdown to cleanup
    Redis connections gracefully.
    """
    global _listener_task
    if _listener_task and not _listener_task.done():
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        logger.info("Redis listener stopped successfully")
