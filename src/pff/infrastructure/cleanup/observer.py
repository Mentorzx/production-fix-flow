"""Observer pattern implementation for cleanup events.

Design Pattern: Observer. Decouples cleanup execution from logging and metrics.
"""

from __future__ import annotations

from typing import Protocol

from pff.shared.core.logging import logger

from .commands.base import CleanupCommand
from .utils import format_size


class CleanupObserver(Protocol):
    """Protocol defining the observer interface for cleanup events.

    Implementations receive notifications at key points during cleanup execution.
    """

    def on_command_start(self, cmd: CleanupCommand) -> None:
        """Called before a command begins execution.

        Args:
            cmd: The command about to execute.
        """
        ...

    def on_command_complete(self, cmd: CleanupCommand, duration_ms: float) -> None:
        """Called after a command completes successfully.

        Args:
            cmd: The command that completed.
            duration_ms: Execution duration in milliseconds.
        """
        ...

    def on_command_error(self, cmd: CleanupCommand, error: Exception) -> None:
        """Called when a command fails with an exception.

        Args:
            cmd: The command that failed.
            error: The exception that was raised.
        """
        ...

    def on_cleanup_complete(self, total_freed_bytes: int) -> None:
        """Called when all cleanup operations are complete.

        Args:
            total_freed_bytes: Total disk space freed in bytes.
        """
        ...


class LoggingCleanupObserver:
    """Observer that logs cleanup events to the standard logger.

    Logs command start at info level (PT-BR), completion at debug level (EN),
    errors at error level (EN), and final summary at success level (PT-BR).
    """

    def on_command_start(self, cmd: CleanupCommand) -> None:
        """Log command start at info level.

        Args:
            cmd: The command about to execute.
        """
        logger.info(f"Executando: {cmd.label}")

    def on_command_complete(self, cmd: CleanupCommand, duration_ms: float) -> None:
        """Log command completion at debug level.

        Args:
            cmd: The command that completed.
            duration_ms: Execution duration in milliseconds.
        """
        logger.debug(f"Completed {cmd.label} in {duration_ms:.1f}ms")

    def on_command_error(self, cmd: CleanupCommand, error: Exception) -> None:
        """Log command error at error level.

        Args:
            cmd: The command that failed.
            error: The exception that was raised.
        """
        logger.error(f"Error in {cmd.label}: {error}")

    def on_cleanup_complete(self, total_freed_bytes: int) -> None:
        """Log cleanup completion at success level.

        Args:
            total_freed_bytes: Total disk space freed in bytes.
        """
        logger.success(
            f"Limpeza finalizada: {format_size(total_freed_bytes)} liberados"
        )


class CompositeCleanupObserver:
    """Dispatch events to multiple observers.

    Design Pattern: Composite Observer. Allows multiple observers to receive
    the same events without coupling the engine to specific implementations.

    Args:
        observers: List of observers to dispatch events to.
    """

    def __init__(self, observers: list[CleanupObserver]):
        """Initialize with a list of observers.

        Args:
            observers: List of CleanupObserver implementations.
        """
        self._observers = observers

    def on_command_start(self, cmd: CleanupCommand) -> None:
        """Dispatch command start event to all observers.

        Args:
            cmd: The command about to execute.
        """
        for obs in self._observers:
            obs.on_command_start(cmd)

    def on_command_complete(self, cmd: CleanupCommand, duration_ms: float) -> None:
        """Dispatch command completion event to all observers.

        Args:
            cmd: The command that completed.
            duration_ms: Execution duration in milliseconds.
        """
        for obs in self._observers:
            obs.on_command_complete(cmd, duration_ms)

    def on_command_error(self, cmd: CleanupCommand, error: Exception) -> None:
        """Dispatch command error event to all observers.

        Args:
            cmd: The command that failed.
            error: The exception that was raised.
        """
        for obs in self._observers:
            obs.on_command_error(cmd, error)

    def on_cleanup_complete(self, total_freed_bytes: int) -> None:
        """Dispatch cleanup completion event to all observers.

        Args:
            total_freed_bytes: Total disk space freed in bytes.
        """
        for obs in self._observers:
            obs.on_cleanup_complete(total_freed_bytes)


__all__ = ["CleanupObserver", "LoggingCleanupObserver", "CompositeCleanupObserver"]
