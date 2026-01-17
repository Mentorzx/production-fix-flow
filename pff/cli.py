"""Compatibility entrypoint for legacy console scripts."""

import importlib


def cli_entrypoint() -> None:
    """Run the PFF CLI entrypoint."""
    module = importlib.import_module("pff.drivers.cli.main")
    module.cli_entrypoint()


__all__ = ["cli_entrypoint"]
