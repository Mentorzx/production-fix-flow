"""
Debug utilities for symbolic feature processors.

This module implements the Command pattern for debugging operations
and provides a centralized debug management system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import DebugCommand


class SaveDebugInfoCommand(DebugCommand):
    """Command to save debug information to a file."""

    def __init__(self, filename_prefix: str = "numba_accel_debug"):
        self.filename_prefix = filename_prefix

    def execute(self, context: dict[str, Any]) -> None:
        """Execute the debug save command."""
        data = context.get("data", [])
        error = context.get("error", Exception("Unknown error"))
        config = context.get("config", {})

        self.save_debug_info(data, error, config)

    def save_debug_info(self, data: Any, error: Exception, config: dict[str, Any]) -> None:
        """Save debug information to a JSON file."""
        try:
            debug_dir = Path(config.get("debug_output_dir", "debug"))
            debug_dir.mkdir(exist_ok=True)

            dump = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_samples": len(data) if hasattr(data, "__len__") else 0,
                "sample_preview": [repr(s) for s in data[:5]] if hasattr(data, "__getitem__") else [],
                "exception": repr(error),
                "n_rules": config.get("n_rules", 0),
                "rule_index_exists": config.get("rule_index") is not None,
                "strategy": config.get("strategy", "unknown"),
                "processing_time": config.get("processing_time", 0.0),
            }

            filename = f"{self.filename_prefix}.json"
            filepath = debug_dir / filename

            from pff.utils import FileManager
            FileManager().save_json(dump, filepath)

        except Exception as e:
            # Don't let debug operations fail the main processing
            pass


class LogPerformanceCommand(DebugCommand):
    """Command to log performance metrics."""

    def __init__(self, level: str = "info"):
        self.level = level.lower()

    def execute(self, context: dict[str, Any]) -> None:
        """Execute the performance logging command."""
        from pff.utils.core.logger import logger

        strategy = context.get("strategy", "unknown")
        processing_time = context.get("processing_time", 0.0)
        samples_processed = context.get("samples_processed", 0)
        metadata = context.get("metadata", {})

        if samples_processed > 0:
            rate = samples_processed / processing_time if processing_time > 0 else 0
            message = (
                f" Performance metrics - Strategy: {strategy}, "
                f"Samples: {samples_processed}, "
                f"Time: {processing_time:.3f}s, "
                f"Rate: {rate:.1f} samples/s"
            )

            if self.level == "debug":
                logger.debug(message)
            elif self.level == "info":
                logger.info(message)
            elif self.level == "warning":
                logger.warning(message)
            else:
                logger.error(message)

    def save_debug_info(self, data: Any, error: Exception, context: dict[str, Any]) -> None:
        """This command doesn't save debug info."""
        pass


class ValidateOutputCommand(DebugCommand):
    """Command to validate output and log statistics."""

    def execute(self, context: dict[str, Any]) -> None:
        """Execute the output validation command."""
        from pff.utils.core.logger import logger

        data = context.get("data", [])
        config = context.get("config", {})

        if isinstance(data, list) and data:
            # Analyze the output
            sample_shape = data[0].shape if hasattr(data[0], "shape") else len(data[0])
            total_elements = sum(hasattr(arr, "size") and arr.size or len(arr) for arr in data)
            non_zero_elements = sum(
                hasattr(arr, "nnz") and arr.nnz or
                (hasattr(arr, "nonzero") and len(arr.nonzero()[0])) or
                sum(1 for x in (arr.flatten() if hasattr(arr, "flatten") else arr) if x != 0)
                for arr in data
            )

            sparsity = (non_zero_elements / total_elements * 100) if total_elements > 0 else 0

            logger.info(
                f" Output validation - Shape: ({len(data)}, {sample_shape}), "
                f"Non-zero: {non_zero_elements}/{total_elements} ({sparsity:.2f}%)"
            )

    def save_debug_info(self, data: Any, error: Exception, context: dict[str, Any]) -> None:
        """This command doesn't save debug info."""
        pass


class DebugManager:
    """Manager for debug commands and operations."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.commands: dict[str, DebugCommand] = {}
        self._register_default_commands()

    def register_command(self, name: str, command: DebugCommand) -> None:
        """Register a debug command."""
        self.commands[name] = command

    def execute_command(self, name: str, context: dict[str, Any]) -> None:
        """Execute a debug command if debugging is enabled."""
        if not self.enabled:
            return

        command = self.commands.get(name)
        if command:
            try:
                command.execute(context)
            except Exception as e:
                # Don't let debug operations fail the main processing
                from pff.utils.core.logger import logger
                logger.warning(f"Debug command '{name}' failed: {e}")

    def execute_all_commands(self, context: dict[str, Any]) -> None:
        """Execute all registered debug commands."""
        for name in self.commands:
            self.execute_command(name, context)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable debugging."""
        self.enabled = enabled

    def is_enabled(self) -> bool:
        """Check if debugging is enabled."""
        return self.enabled

    def _register_default_commands(self) -> None:
        """Register default debug commands."""
        self.register_command("save_debug", SaveDebugInfoCommand())
        self.register_command("log_performance", LogPerformanceCommand())
        self.register_command("validate_output", ValidateOutputCommand())


def create_debug_manager(config: dict[str, Any]) -> DebugManager:
    """Create a debug manager based on configuration."""
    enabled = config.get("enable_debug", False)
    manager = DebugManager(enabled)

    # Register additional commands based on configuration
    if config.get("verbose_performance", False):
        manager.register_command("detailed_performance", LogPerformanceCommand("debug"))

    return manager
