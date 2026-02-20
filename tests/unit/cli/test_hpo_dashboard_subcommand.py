"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/cli/test_hpo_dashboard_subcommand.py

"""

from pff.drivers.cli.internal.parser import CLIParserBuilder


def test_hpo_dashboard_subcommand_parses() -> None:
    """Execute test hpo dashboard subcommand parses."""

    parser = CLIParserBuilder().add_version().create_subparsers().add_commands().build()
    args = parser.parse_args(["hpo", "dashboard", "on"])
    assert args.command == "hpo"
    assert args.hpo_subcommand == "dashboard"
    assert args.dashboard_action == "on"
