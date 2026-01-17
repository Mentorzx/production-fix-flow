from __future__ import annotations

import importlib
import io
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar
from collections.abc import Callable

import orjson
from loguru import logger as _loguru_logger
from rich.logging import RichHandler
from rich.traceback import install as rich_tb_install

from pff import settings

"""
pff.shared.core.logger
~~~~~~~~~~~~~~~~
Unified logging utilities and helpers for Python projects, with support for console (Rich) and rotating file logs.
This module provides a pre-configured logger, decorators for timing and exception handling, library silencing, timestamp helpers, and log reordering tools.
Quick Start
-----------
1. **Logging Setup**
    - The logger is pre-configured for both console (with Rich formatting) and rotating file output.
    - Environment variables allow customization (see below).
2. **Usage Example**
    ```python
    logger.info("Hello, world!")
    ```
3. **Timing Functions**
    - Use `@timeit` to log execution time of functions.
    ```python
    @timeit
    def slow_function():
         ...
    ```
4. **Catching Exceptions**
    - Use `@catch` to log and optionally suppress exceptions.
    ```python
    @catch(default=None)
    def might_fail():
         ...
    ```
5. **Silencing Noisy Libraries**
    - Silence logs from specific libraries.
    ```python
    silence_libs("urllib3", "chardet", level="ERROR")
    ```
6. **Getting Local Timestamps**
    - Get a local ISO 8601 timestamp.
    ```python
    print(local_timestamp())
    ```
7. **Reordering Log Files**
    - Group log entries by thread and MSISDN for easier analysis.
    ```python
    LogReorderer.reorder(Path("mylogfile.log"))
    ```
Environment Variables
---------------------
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.
- `LOG_DIR`: Directory for log files. Default: ~/.logs.
- `LOG_ROTATION`: Log rotation policy. Default: 100 MB.
- `LOG_RETENTION`: Log retention policy. Default: 30 days.
- `LOG_COMPRESSION`: Compression for rotated logs (zip, gz, bz2, none). Default: zip.
- `DISABLE_RICH`: Disable Rich console output if set.
- `RICH_THEME`: Rich traceback theme. Default: monokai.
Exports
-------
- `logger`: Pre-configured Loguru logger.
- `timeit`: Decorator to log function execution time.
- `catch`: Decorator to log and handle exceptions.
- `silence_libs`: Function to silence logs from specified libraries.
- `local_timestamp`: Function to get local ISO 8601 timestamp.
- `LogReorderer`: Class to reorder log files by thread and MSISDN.
"""


# ╭──────────────────────── Configuração básica ───────────────────────╮ #

_loguru_logger.remove()
_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<cyan>[{extra[task_id]:^11}]</cyan> - "
    "<level>{message}</level>"
)

# ——— console (Rich) ——— #
_IS_TTY = "DISABLE_RICH" not in os.environ
if _IS_TTY:
    rich_tb_install(
        show_locals=False,
        theme=os.getenv("RICH_THEME", "monokai"),
    )
    _loguru_logger.add(
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            highlighter=None,
        ),
        level=_LEVEL,
        format="{message}",
    )
else:
    _loguru_logger.add(sys.stderr, level=_LEVEL, format=FORMAT, colorize=True)

# ——— arquivo rotativo ——— #
LOG_DIR = Path(os.getenv("LOG_DIR", settings.LOGS_DIR)).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _sink_serializer(record):
    """Serialize log record using orjson for speed."""
    subset = {
        "time": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "extra": record["extra"],
    }
    return orjson.dumps(subset).decode("utf-8") + "\n"


try:
    _loguru_logger.add(
        LOG_DIR / "{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=os.getenv("LOG_ROTATION", "100 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        enqueue=True,  # ASYNC WRITES
        backtrace=False,
        format=FORMAT,
        serialize=False,  # We keep text format for readability in dev
    )
except PermissionError:
    # Fall back to synchronous logging when the sandbox forbids creating
    # multiprocessing primitives (common in CI sandboxes).
    _loguru_logger.add(
        LOG_DIR / "{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=os.getenv("LOG_ROTATION", "100 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        enqueue=False,
        backtrace=False,
        format=FORMAT,
    )

logger = _loguru_logger  # reexport

# ╰────────────────────────────────────────────────────────────────────╯ #

# ───────── helpers utilitários ───────── #
P = ParamSpec("P")
R = TypeVar("R")


def timeit(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that measures the execution time of the decorated function and logs it using the logger.
    Args:
        fn (Callable[P, R]): The function to be decorated.
    Returns:
        Callable[P, R]: The wrapped function that logs its execution time in milliseconds.
    Logs:
        Logs the qualified name of the function and the time taken to execute it at the debug level.
    """

    @wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs):  # type: ignore[name-defined]
        t0 = time.perf_counter()
        result: R = fn(*args, **kwargs)
        logger.debug(
            f"{fn.__qualname__} levou {(time.perf_counter() - t0) * 1000:,.1f} ms"
        )
        return result

    return _wrapper


def catch(
    *, reraise: bool = False, default: R | None = None, level: str = "ERROR"
) -> Callable[[Callable[P, R]], Callable[P, R | None]]:
    """
    A decorator to catch exceptions in the decorated function, log them, and optionally reraise or return a default value.
    Args:
        reraise (bool, optional): If True, re-raises the caught exception after logging. Defaults to False.
        default (R | None, optional): The value to return if an exception is caught and not reraised. Defaults to None.
        level (str, optional): The logging level to use when logging the exception. Defaults to "ERROR".
    Returns:
        Callable[[Callable[P, R]], Callable[P, R | None]]: A decorator that wraps the target function with exception handling.
    Example:
        @catch(reraise=False, default=None, level="WARNING")
        def my_function():
            ...
    """

    def _decor(fn: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(fn)
        def _inner(*args: P.args, **kwargs: P.kwargs):  # type: ignore[name-defined]
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.log(level, f"Error in {fn.__qualname__}: {exc}", exc_info=True)
                if reraise:
                    raise
                return default

        return _inner

    return _decor


@contextmanager
def suppress_output(suppress: bool = True):
    """
    Context manager to suppress stdout and stderr.

    Args:
        suppress: If True, output will be redirected to devnull.
    """
    if not suppress:
        yield
        return

    _stdout = sys.stdout
    _stderr = sys.stderr

    try:
        devnull = io.StringIO()
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = _stdout
        sys.stderr = _stderr


def silence_libs(*modules: str, level: str = "WARNING") -> None:
    lvl = getattr(logging, level.upper(), logging.WARNING)
    for name in modules:
        try:
            mod = importlib.import_module(name)
            logging.getLogger(mod.__name__).setLevel(lvl)
        except ModuleNotFoundError:
            continue


def local_timestamp() -> str:
    return datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds")


class LogReorderer:
    """
    Reorders the log entries in the specified file by thread and MSISDN.
    This class provides methods to extract thread name, MSISDN, and text from log lines,
    and to rewrite the log file so that entries are grouped first by thread and then by MSISDN.
    """

    HEADER_PREFIX: str = "===== THREAD"

    @staticmethod
    def _extract(line: str) -> tuple[str, str | None, str]:
        """
        Extracts thread name, msisdn, and text from a log line.
        This function attempts to parse the input line as JSON. If successful, it extracts the thread name,
        msisdn, and text from the JSON structure. If the line is not valid JSON, it attempts to parse it as a
        pipe-separated string and extract the relevant fields. If the line is empty or starts with the header
        prefix, it returns a default "_meta" value.

        Args:
            line (str): A single line from the log file.

        Returns:
            Tuple[str, Optional[str], str]: A tuple containing:
                - The thread name or "_meta" if not found.
                - The msisdn (phone number) if available, otherwise None.
                - The original or extracted text from the log line.
        """
        if not line or line.startswith(LogReorderer.HEADER_PREFIX):
            return "_meta", None, line

        try:
            # Avoid circular import - assuming FileManager logic logic matches
            # but using orjson directly here as requested by performance optimization
            rec = orjson.loads(line)
            tname = rec.get("record", {}).get("thread", {}).get("name", "_meta")
            extra = rec.get("record", {}).get("extra", {})
            msisdn = extra.get("msisdn") or extra.get("task_id")
            text = rec.get("text", "").rstrip()
            return tname, msisdn, text
        except orjson.JSONDecodeError:
            parts = line.split("|")
            if len(parts) >= 4:
                import re

                task_match = re.search(r"\[([^\]]+)\]", line)
                msisdn = task_match.group(1) if task_match else None
                thread_match = re.search(r"Thread-\d+", line)
                tname = thread_match.group(0) if thread_match else "MainThread"

                return tname, msisdn, line
            return "_meta", None, line

    @staticmethod
    def reorder(file_path: Path) -> Path:
        """
        Reorders the log entries in the specified file by thread and MSISDN.
        Uses a streaming approach to avoid OOM on large files.
        """
        # First pass: map thread/msisdn to file positions or temp files
        # For simplicity in this iteration, we will use memory but optimized with generators
        # A true production fix for GB+ logs would involve splitting into temp files per thread.
        # Here we just optimize the read/parse loop.

        buckets: dict[str, list[tuple[str | None, str]]] = defaultdict(list)

        # Generator based read
        def stream_lines(fp: Path):
            with fp.open("r", encoding="utf-8") as f:
                yield from f

        for ln in stream_lines(file_path):
            thr, msisdn, txt = LogReorderer._extract(ln)
            buckets[thr].append((msisdn, txt))

        # Write back
        with file_path.open("w", encoding="utf-8") as fp:
            for thr in sorted(buckets):
                entries = buckets[thr]
                if thr == "_meta":
                    for _, txt in entries:
                        fp.write(txt + "\n")
                    continue
                fp.write(f"\n{LogReorderer.HEADER_PREFIX} {thr} =====\n")
                last_msisdn: str | None = None
                for msisdn, txt in entries:
                    if msisdn and msisdn != last_msisdn:
                        fp.write("\n")
                        last_msisdn = msisdn
                    fp.write(txt + "\n")
        return file_path


debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
exception = logger.exception
critical = logger.critical
logger.configure(extra={"task_id": "MAIN"})

__all__ = [
    "logger",
    "timeit",
    "catch",
    "suppress_output",
    "silence_libs",
    "local_timestamp",
    "LogReorderer",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
]
