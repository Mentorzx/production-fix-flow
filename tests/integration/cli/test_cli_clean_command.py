"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/integration/cli/test_cli_clean_command.py

"""

from pff.drivers.cli.internal.parser import CLIParserBuilder


def test_clean_command_accepts_strategy() -> None:
    """Execute test clean command accepts strategy."""

    parser = CLIParserBuilder().add_version().create_subparsers().add_commands().build()

    args = parser.parse_args(["clean", "deep", "--dry-run"])

    assert args.command == "clean"
    assert args.strategy == "deep"
    assert args.dry_run is True
