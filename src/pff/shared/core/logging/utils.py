from __future__ import annotations

import io
import logging
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def timeit(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator that measures execution time."""

    @wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs):
        t0 = time.perf_counter()
        result: R = fn(*args, **kwargs)
        logger.debug(
            f"{fn.__qualname__} took {(time.perf_counter() - t0) * 1000:,.1f} ms"
        )
        return result

    return _wrapper


def catch(
    *, reraise: bool = False, default: R | None = None, level: str = "ERROR"
) -> Callable[[Callable[P, R]], Callable[P, R | None]]:
    """Decorator to catch exceptions."""

    def _decor(fn: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(fn)
        def _inner(*args: P.args, **kwargs: P.kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.log(level, f"Error in {fn.__qualname__}: {exc}", exc_info=True)
                if reraise:
                    raise
                return default

        return _inner

    return _decor


@contextmanager
def suppress_output(suppress: bool = True):
    """Context manager to suppress stdout/stderr."""
    if not suppress:
        yield
        return
    _stdout, _stderr = sys.stdout, sys.stderr
    try:
        devnull = io.StringIO()
        sys.stdout = sys.stderr = devnull
        yield
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr


def silence_libs(*modules: str, level: str = "WARNING") -> None:
    """Silence specific libraries."""

    lvl = getattr(logging, level.upper(), logging.WARNING)
    for name in modules:
        logging.getLogger(name).setLevel(lvl)


def local_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds")
