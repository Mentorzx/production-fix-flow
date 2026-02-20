"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/integration/cli/test_cli_entrypoint_import.py

"""


def test_cli_entrypoint_module_import() -> None:
    """Execute test cli entrypoint module import."""

    import pff.drivers.cli.main

    assert hasattr(pff.drivers.cli.main, "cli_entrypoint")
