from pff.drivers.cli.internal.factory import CommandFactory


def test_dashboard_command_removed_from_registry() -> None:
    commands = CommandFactory.get_all_commands()
    assert "dashboard" not in commands
