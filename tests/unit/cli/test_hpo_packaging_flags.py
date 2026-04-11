"""Tests for packaging-friendly HPO CLI flags."""

from __future__ import annotations

import argparse

from pff.drivers.cli.internal.commands import HpoCommand


def _parse_hpo_args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    HpoCommand.configure_parser(subparsers)
    return parser.parse_args(["hpo", *argv])


def test_hpo_parser_accepts_packaging_flags() -> None:
    """The HPO parser should expose synthetic-data and no-dashboard switches."""
    args = _parse_hpo_args(
        "--trials",
        "1",
        "--synthetic-data",
        "--no-dashboard",
        "--no-update-config",
    )

    assert args.synthetic_data is True
    assert args.no_dashboard is True
    assert args.no_update_config is True


def test_hpo_command_reads_packaging_env(monkeypatch) -> None:
    """Environment flags should enable packaging-friendly HPO defaults."""
    monkeypatch.setenv("PFF_HPO_USE_SYNTHETIC", "1")
    monkeypatch.setenv("PFF_HPO_DISABLE_DASHBOARD", "1")

    command = HpoCommand(_parse_hpo_args("--trials", "1"))

    assert command.synthetic_data is True
    assert command.no_dashboard is True
