"""
Global interrupt manager to coordinate SIGINT/SIGTERM handling across the stack.

Design Pattern: Singleton, Observer
- Singleton centraliza estado de interrupção.
- Observer distribui callbacks de desligamento em ordem de prioridade.
"""

from __future__ import annotations

import asyncio
import functools
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Callable, ParamSpec, TypeVar

from ..core.logger import logger

P = ParamSpec("P")
T = TypeVar("T")

# Prioridade: menor valor = executa primeiro
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
    """Singleton that tracks and propagates interrupt signals."""

    _instance: "GlobalInterruptManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "GlobalInterruptManager":
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
        self._callbacks: list[RegisteredCallback] = []
        self._callback_counter = 0
        self._original_handlers: dict[int, signal.Handlers] = {}
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers with asyncio support when available."""
        self._original_handlers = {}

        def sync_signal_handler(signum: int, frame: object | None) -> None:  # noqa: ARG001
            signal_name = signal.Signals(signum).name
            logger.warning(f"{signal_name} received - starting coordinated shutdown")
            self._stop_event.set()
            self._signal_received = True
            self._execute_callbacks()

        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, lambda s=sig: sync_signal_handler(s, None))
                logger.debug("Signal handlers registered via asyncio event loop")
                return
            except RuntimeError:
                pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[int(sig)] = signal.signal(sig, sync_signal_handler)
        logger.debug("Signal handlers registered via signal.signal()")

    def _execute_callbacks(self) -> None:
        """Execute callbacks in priority order (stable by registration)."""
        sorted_callbacks = sorted(self._callbacks, key=lambda cb: (cb.priority, cb.order))
        for cb in sorted_callbacks:
            try:
                cb.callback()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"Error in shutdown callback '{cb.label}' (priority={cb.priority}): {exc}"
                )
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

    def unregister_callback(self, callback_or_label: Callable[[], None] | str) -> bool:
        """Remove callback by reference or label. Idempotent."""
        for idx, registered in enumerate(self._callbacks):
            if registered.callback is callback_or_label or registered.label == callback_or_label:
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
        logger.warning("Operation interrupted by GlobalInterruptManager")
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
            logger.info(f"{func.__name__} interrompida graciosamente")
            raise

    return wrapper
