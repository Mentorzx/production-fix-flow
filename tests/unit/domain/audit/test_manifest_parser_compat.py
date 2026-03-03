"""Regression tests for ManifestParser backward compatibility."""

from __future__ import annotations

from pff.domain.audit.manifest import ManifestParser


def test_manifest_parser_supports_zero_arg_constructor() -> None:
    """ManifestParser() must remain constructible without explicit dependencies."""
    parser = ManifestParser()
    assert parser is not None
