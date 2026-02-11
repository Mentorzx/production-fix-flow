"""
Global interrupt manager to coordinate SIGINT/SIGTERM handling across the stack.

Design Pattern: Singleton, Observer
- Singleton centralizes shared interrupt state.
- Observer dispatches shutdown callbacks in priority order (lower runs first).
"""

from __future__ import annotations

import asyncio
import functools
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar, cast

from ..core.logging import logger

P = ParamSpec("P")
T = TypeVar("T")

PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 10
PRIORITY_NORMAL = 20
PRIORITY_LOW = 30


@dataclass(frozen=True, slots=True)
class RegisteredCallback:
    """Registered callback with priority and label for debugging."""

    callback: Callable[[], None]
    priority: int
    label: str
    order: int


class GlobalInterruptManager:
    """Singleton that tracks and propagates interrupt signals (thread-safe, process-local)."""

    _instance: GlobalInterruptManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> GlobalInterruptManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._stop_event = threading.Event()
        self._signal_received = False
        self._interruption_warning_emitted = False
        self._callbacks: list[RegisteredCallback] = []
        self._callback_counter = 0
        self._callbacks_lock = threading.Lock()
        self._original_handlers: dict[int, signal.Handlers] = {}
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers with asyncio support when available."""
        self._original_handlers = {}
        if threading.current_thread() is not threading.main_thread():
            return

        def sync_signal_handler(signum: int, frame: object | None) -> None:
            self._handle_signal(signum)

        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, lambda s=sig: sync_signal_handler(s, None))  # type: ignore[arg-type,return-value]
                return
            except RuntimeError:
                pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[int(sig)] = cast(
                signal.Handlers, signal.signal(sig, sync_signal_handler)
            )

    def _handle_signal(self, signum: int) -> None:
        """Handle SIGINT/SIGTERM once to avoid duplicate shutdown work."""
        if self._signal_received:
            return

        should_log = False
        with self._lock:
            if self._signal_received:
                return
            self._stop_event.set()
            self._signal_received = True
            should_log = True

        if should_log:
            try:
                signal_name = signal.Signals(signum).name
                logger.warning(f"{signal_name} received - starting coordinated shutdown")
            except Exception:
                pass

        self._execute_callbacks()

    def _execute_callbacks(self) -> None:
        """Execute callbacks in priority order (stable by registration).

        Guards against multiple executions in multi-process scenarios.
        """
        if getattr(self, "_callbacks_executed", False):
            return
        self._callbacks_executed = True
        with self._callbacks_lock:
            sorted_callbacks = sorted(self._callbacks, key=lambda cb: (cb.priority, cb.order))
        for cb in sorted_callbacks:
            try:
                cb.callback()
            except Exception as exc:
                logger.error(
                    f"Error in shutdown callback '{cb.label}' (priority={cb.priority}): {exc}"
                )
        if sorted_callbacks:
            logger.info("Sinal de parada propagado para todos os componentes")

    @property
    def should_stop(self) -> bool:
        """True when an interrupt was received or forced. Semantics preserved."""
        return self._stop_event.is_set()

    @property
    def signal_received(self) -> bool:
        """True when a signal handler was triggered."""
        return self._signal_received

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Block until stop signal or timeout. Returns True if stopped."""
        return self._stop_event.wait(timeout)

    def register_callback(
        self,
        callback: Callable[[], None],
        priority: int = PRIORITY_NORMAL,
        label: str | None = None,
    ) -> str:
        """Register a shutdown callback.

        Args:
            callback: Function to call on shutdown.
            priority: Execution priority (lower executes earlier).
            label: Optional label for debugging. Auto-generated if None.

        Returns:
            str: Assigned label (useful for unregistering).
        """
        with self._callbacks_lock:
            assigned_label = label or f"callback_{self._callback_counter}"
            self._callbacks.append(
                RegisteredCallback(
                    callback=callback,
                    priority=priority,
                    label=assigned_label,
                    order=self._callback_counter,
                )
            )
            self._callback_counter += 1
            return assigned_label

    def register_callback_once(
        self,
        callback: Callable[[], None],
        *,
        priority: int = PRIORITY_NORMAL,
        label: str,
    ) -> str:
        """Register a shutdown callback only if the label is not already present.

        Args:
            callback: Function to call on shutdown.
            priority: Execution priority (lower executes earlier).
            label: Stable label used for deduplication.

        Returns:
            str: The registered (or existing) label.
        """
        with self._callbacks_lock:
            if any(existing.label == label for existing in self._callbacks):
                return label
            self._callbacks.append(
                RegisteredCallback(
                    callback=callback,
                    priority=priority,
                    label=label,
                    order=self._callback_counter,
                )
            )
            self._callback_counter += 1
            return label

    def unregister_callback(self, callback_or_label: Callable[[], None] | str) -> bool:
        """Remove callback by reference or label. Idempotent."""
        with self._callbacks_lock:
            for idx, registered in enumerate(self._callbacks):
                if (
                    registered.callback is callback_or_label
                    or registered.label == callback_or_label
                ):
                    self._callbacks.pop(idx)
                    return True
            return False

    def force_stop(self, reason: str = "Manual") -> None:
        """Force an interrupt state and trigger callbacks."""
        logger.warning(f"Forced stop requested: {reason}")
        self._stop_event.set()
        self._signal_received = True
        self._execute_callbacks()

    def reset(self) -> None:
        """Clear interrupt state and callbacks (mainly for tests)."""
        logger.debug("Resetting GlobalInterruptManager")
        self._stop_event.clear()
        self._signal_received = False
        self._interruption_warning_emitted = False
        self._callbacks_executed = False
        self._callbacks.clear()
        self._callback_counter = 0

    def restore_original_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        logger.debug("Original signal handlers restored")

    def __del__(self) -> None:
        try:
            self.restore_original_handlers()
        except Exception:
            pass


def get_interrupt_manager() -> GlobalInterruptManager:
    """Return the singleton interrupt manager."""
    return GlobalInterruptManager()


def should_stop() -> bool:
    """Convenience helper to check stop flag."""
    return get_interrupt_manager().should_stop


def register_interrupt_callback(callback: Callable[[], None]) -> str:
    """Register a callback on the singleton manager."""
    return get_interrupt_manager().register_callback(callback)


def check_interruption() -> None:
    """Raise KeyboardInterrupt if a stop request was captured."""
    manager = get_interrupt_manager()
    if manager.should_stop:
        if not manager._interruption_warning_emitted:
            logger.warning("Operation interrupted by GlobalInterruptManager")
            manager._interruption_warning_emitted = True
        raise KeyboardInterrupt("Operation interrupted by GlobalInterruptManager")


def interruptible(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to abort execution when a stop flag is set."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if should_stop():
            logger.warning(f"Function {func.__name__} interrupted by GlobalInterruptManager")
            raise KeyboardInterrupt(f"Function {func.__name__} interrupted")
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.debug(f"Graceful interrupt captured for function={func.__name__}")
            raise

    return wrapper
