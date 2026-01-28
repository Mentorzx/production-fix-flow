"""Command factory for CLI commands."""

from __future__ import annotations

from typing import Any

from pff.shared.factory import GenericFactory

from .commands import (
    APICommand,
    CleanCommand,
    Command,
    GenerateCommand,
    HpoCommand,
    HpoProxyCommand,
    LearnCommand,
    LogsCommand,
    ResetMLCommand,
    RunCommand,
    WorkerCommand,
)


class CommandFactory(GenericFactory[Command]):
    """
    Factory class for creating Command instances.

    Pattern: Factory Pattern + Registry Pattern
    """

    _command_registry: dict[str, type[Command]] = {
        "run": RunCommand,
        "generate": GenerateCommand,
        "worker": WorkerCommand,
        "api": APICommand,
        "clean": CleanCommand,
        "reset-ml": ResetMLCommand,
        "logs": LogsCommand,
        "learn": LearnCommand,
        "hpo": HpoCommand,
        "hpo-proxy": HpoProxyCommand,
    }

    @classmethod
    def create(cls, key: str, *args: Any, **kwargs: Any) -> Command:
        """
        Create a command instance based on command name.

        Args:
            key: Name of the command
            *args: Positional arguments (first is expected to be args)
            **kwargs: Additional keyword arguments (e.g., launcher)

        Returns:
            Command instance

        Raises:
            ValueError: If command name is not registered
        """
        from typing import cast
        import argparse

        command_class = cls._command_registry.get(key)
        parsed_args = cast(argparse.Namespace, args[0] if args else kwargs.get("args"))

        if not command_class:
            raise ValueError(f"Unknown command: {key}")

        if key == "run" and "launcher" in kwargs:
            # Special case for RunCommand which takes a launcher
            from .commands import RunCommand

            return RunCommand(parsed_args, launcher=kwargs["launcher"])

        return command_class(parsed_args)

    @classmethod
    def register(cls, command_name: str, command_class: type[Command]) -> None:  # type: ignore[override]
        """
        Register a new command class.

        Args:
            command_name: Name of the command
            command_class: Command class to register
        """
        cls._command_registry[command_name] = command_class

    @classmethod
    def get_all_commands(cls) -> list[str]:
        """Get list of all registered command names."""
        return list(cls._command_registry.keys())
