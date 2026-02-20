"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/cli/test_command_registry.py

"""

from pff.drivers.cli.internal.factory import CommandFactory


def test_dashboard_command_removed_from_registry() -> None:
    """Execute test dashboard command removed from registry."""

    commands = CommandFactory.get_all_commands()
    assert "dashboard" not in commands
