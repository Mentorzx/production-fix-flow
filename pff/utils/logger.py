from __future__ import annotations

import importlib
# Sprint 16.5: Removed stdlib json import, using FileManager.json_loads() instead
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from loguru import logger as _loguru_logger
from rich.logging import RichHandler
from rich.traceback import install as rich_tb_install

from pff import settings

"""
pff.utils.logger
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
_loguru_logger.add(
    LOG_DIR / "{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation=os.getenv("LOG_ROTATION", "100 MB"),
    retention=os.getenv("LOG_RETENTION", "30 days"),
    compression=os.getenv("LOG_COMPRESSION", "zip"),
    enqueue=True,
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
                logger.log(level, f"Erro em {fn.__qualname__}: {exc}", exc_info=True)
                if reraise:
                    raise
                return default

        return _inner

    return _decor


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
            # Sprint 16.5: Use FileManager for faster JSON parsing (msgspec)
            from pff.utils import FileManager
            rec = FileManager.json_loads(line)
            tname = rec.get("record", {}).get("thread", {}).get("name", "_meta")
            extra = rec.get("record", {}).get("extra", {})
            msisdn = extra.get("msisdn") or extra.get("task_id")
            text = rec.get("text", "").rstrip()
            return tname, msisdn, text
        except json.JSONDecodeError:
            parts = line.split("|")
            if len(parts) >= 4:
                import re
                task_match = re.search(r'\[([^\]]+)\]', line)
                msisdn = task_match.group(1) if task_match else None
                thread_match = re.search(r'Thread-\d+', line)
                tname = thread_match.group(0) if thread_match else "MainThread"
                
                return tname, msisdn, line
            return "_meta", None, line

    @staticmethod
    def reorder(file_path: Path) -> Path:
        """
        Reorders the log entries in the specified file by thread and MSISDN.
        This method reads a log file, groups log entries by thread identifier,
        and writes them back to the file in a sorted order. Entries with the thread
        identifier "_meta" are written first, preserving their original order.
        Other threads are written in sorted order, with a header for each thread.
        Within each thread, entries are grouped by MSISDN, and a blank line is
        inserted when the MSISDN changes.

        Args:
            file_path (Path): The path to the log file to reorder.

        Returns:
            Path: The path to the reordered log file.
        """
        buckets: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
        for ln in file_path.read_text(encoding="utf-8").splitlines():
            thr, msisdn, txt = LogReorderer._extract(ln)
            buckets[thr].append((msisdn, txt))

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
