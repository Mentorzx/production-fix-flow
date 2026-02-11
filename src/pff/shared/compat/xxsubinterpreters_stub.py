"""Compatibility stub for builds without the stdlib _xxsubinterpreters module."""

from __future__ import annotations

from typing import Iterable


def get_current() -> int:
    """Return a sentinel interpreter id for environments without subinterpreters."""
    return 0


def list_all() -> Iterable[int]:
    """Return the list of available interpreter ids (stubbed to the main one)."""
    return (0,)


def is_running(interpreter_id: int | None = None) -> bool:
    """Report whether a subinterpreter is running (always False for stub)."""
    _ = interpreter_id
    return False


def create(*args, **kwargs) -> int:
    """Subinterpreter creation is unavailable in this Python build."""
    raise RuntimeError("_xxsubinterpreters is unavailable in this Python build")


def destroy(*args, **kwargs) -> None:
    """Subinterpreter destruction is unavailable in this Python build."""
    raise RuntimeError("_xxsubinterpreters is unavailable in this Python build")


def run_string(*args, **kwargs) -> None:
    """Executing code in subinterpreters is unavailable in this build."""
    raise RuntimeError("_xxsubinterpreters is unavailable in this Python build")
