"""Lightweight observer interfaces for progress reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from pff.shared.core.logging import logger

T = TypeVar("T", contravariant=True)


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


class EventObserver(Protocol[T]):
    """Protocol for event-based observers with an on_event hook."""

    def on_event(self, event: T) -> None: ...


class CompositeObserver:
    """Composite observer for dispatching events to multiple observers."""

    def __init__(self, observers: Sequence[EventObserver[Any]] | None = None) -> None:
        self._observers: list[EventObserver[Any]] = list(observers) if observers else []

    def add(self, observer: EventObserver[Any]) -> CompositeObserver:
        """Add observer to composite (fluent)."""
        self._observers.append(observer)
        return self

    def remove(self, observer: EventObserver[Any]) -> CompositeObserver:
        """Remove observer from composite (fluent)."""
        if observer in self._observers:
            self._observers.remove(observer)
        return self

    def add_observer(self, observer: EventObserver[Any]) -> None:
        """Add observer to composite."""
        self._observers.append(observer)

    def remove_observer(self, observer: EventObserver[Any]) -> None:
        """Remove observer from composite."""
        if observer in self._observers:
            self._observers.remove(observer)

    def on_event(self, event: Any) -> None:
        """Dispatch event to all observers."""
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception as exc:
                logger.warning(
                    f"Observer failure observer={type(observer).__name__} error={exc}"
                )
