"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/shared/core/__init__.py

"""

from __future__ import annotations

from .logging import (
    FORMAT,
    LogReorderer,
    catch,
    create_isolated_logger,
    critical,
    debug,
    error,
    exception,
    info,
    local_timestamp,
    logger,
    silence_libs,
    suppress_output,
    timeit,
    warning,
)

__all__ = [
    "logger",
    "timeit",
    "catch",
    "suppress_output",
    "silence_libs",
    "local_timestamp",
    "create_isolated_logger",
    "LogReorderer",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
    "FORMAT",
]
