"""Command factory for CLI commands."""

from __future__ import annotations


from pff.shared.factory import GenericFactory

from .commands import (
    APICommand,
    CleanCommand,
    Command,
    GenerateCommand,
    HpoProxyCommand,
    HpoCommand,
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
    def create(cls, command_name: str, args, **kwargs) -> Command:
        """
        Create a command instance based on command name.

        Args:
            command_name: Name of the command
            args: Parsed arguments
            **kwargs: Additional keyword arguments (e.g., launcher)

        Returns:
            Command instance

        Raises:
            ValueError: If command name is not registered
        """
        command_class = cls._command_registry.get(command_name)

        if not command_class:
            raise ValueError(f"Unknown command: {command_name}")

        # Special case for RunCommand (needs launcher)
        if command_name == "run" and "launcher" in kwargs:
            return command_class(args, launcher=kwargs["launcher"])  # type: ignore[call-arg]

        return command_class(args)

    @classmethod
    def register(cls, command_name: str, command_class: type[Command]) -> None:
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
