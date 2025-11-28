"""Lightweight observer interfaces for progress reporting."""

from __future__ import annotations

from typing import Protocol


class ProgressObserver(Protocol):
    """Observer interface for long-running operations."""

    def on_start(self, context: dict | None = None) -> None: ...

    def on_step(self, context: dict | None = None) -> None: ...

    def on_complete(self, context: dict | None = None) -> None: ...

    def on_error(self, context: dict | None = None) -> None: ...


class NullObserver:
    """No-op observer used as default."""

    def on_start(self, context: dict | None = None) -> None:
        return

    def on_step(self, context: dict | None = None) -> None:
        return

    def on_complete(self, context: dict | None = None) -> None:
        return

    def on_error(self, context: dict | None = None) -> None:
        return
