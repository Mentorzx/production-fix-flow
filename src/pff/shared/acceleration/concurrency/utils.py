"""Concurrency module utilities and helpers."""

from __future__ import annotations

import shutil
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Sized
from typing import Any

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:
    Progress = None  # type: ignore[assignment,misc]
    BarColumn = MofNCompleteColumn = SpinnerColumn = TaskProgressColumn = TextColumn = (  # type: ignore[misc, assignment]
        TimeElapsedColumn  # type: ignore[misc]
    ) = TimeRemainingColumn = None  # type: ignore[assignment, misc]

from ...core.logging import logger

Args = tuple[Any, ...]

# Lazy-loaded module cache
_duckdb = None
_joblib = None
_polars = None
_psutil = None
_pynvml = None
_ray = None
_dask_client = None
_dask_as_completed = None


class GlobalLock:
    """
    A wrapper around threading.Lock to provide a consistent interface
    and avoid direct threading imports in business logic.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._lock.__exit__(exc_type, exc_val, exc_tb)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._lock.acquire(blocking, timeout)  # type: ignore[no-any-return]

    def release(self) -> None:
        self._lock.release()


def get_lock() -> GlobalLock:
    """Returns a new GlobalLock instance."""
    return GlobalLock()


def _require_duckdb():
    """Lazy import duckdb."""
    global _duckdb
    if _duckdb is None:
        try:
            import duckdb as _duckdb_mod
        except ImportError as exc:
            raise RuntimeError(
                "duckdb não está disponível; instale a dependência para usar query_lazyframe."
            ) from exc
        _duckdb = _duckdb_mod
    return _duckdb


def _require_joblib():
    """Lazy import joblib."""
    global _joblib
    if _joblib is None:
        try:
            import joblib as _joblib_mod
        except ImportError as exc:
            raise RuntimeError(
                "joblib não está disponível; instale a dependência para usar JoblibExecutor."
            ) from exc
        _joblib = _joblib_mod
    return _joblib


def _require_polars():
    """Lazy import polars."""
    global _polars
    if _polars is None:
        try:
            import polars as _polars_mod
        except ImportError as exc:
            raise RuntimeError(
                "polars não está disponível; instale a dependência para usar query_lazyframe."
            ) from exc
        _polars = _polars_mod
    return _polars


def _require_psutil():
    """Lazy import psutil."""
    global _psutil
    if _psutil is None:
        try:
            import psutil as _psutil_mod
        except ImportError as exc:
            raise RuntimeError(
                "psutil não está disponível; instale a dependência para usar ConcurrencyManager."
            ) from exc
        _psutil = _psutil_mod
    return _psutil


def _try_import_pynvml() -> Any:
    """Try to import pynvml, return None if not available."""
    global _pynvml
    if _pynvml is None:
        try:
            import pynvml as _pynvml_mod
        except ImportError:
            _pynvml = False
        else:
            _pynvml = _pynvml_mod
    return _pynvml if _pynvml is not False else None


def _require_ray():
    """Lazy import ray."""
    global _ray
    if _ray is None:
        try:
            import ray as _ray_mod
        except ImportError as exc:
            raise RuntimeError(
                "ray não está disponível; instale a dependência para usar RayExecutor."
            ) from exc
        _ray = _ray_mod
    return _ray


def _require_dask():
    """Lazy import dask distributed client."""
    global _dask_client, _dask_as_completed
    if _dask_client is None or _dask_as_completed is None:
        try:
            from dask.distributed import Client as DaskClient
            from dask.distributed import (
                as_completed as dask_as_completed,
            )
        except ImportError as exc:
            raise RuntimeError(
                "dask.distributed não está disponível; instale a dependência para usar DaskExecutor."
            ) from exc
        _dask_client = DaskClient
        _dask_as_completed = dask_as_completed
    return _dask_client, _dask_as_completed


def _format_time(seconds: float) -> str:
    """
    Formats a time duration given in seconds into a human-readable string.
    If the duration is negative, returns "--:--".
    If the duration is one hour or more, returns a string in the format "HH:MM:SS".
    If the duration is less than one hour, returns a string in the format "MM:SS".
    Args:
        seconds (float): The time duration in seconds.
    Returns:
        str: The formatted time string.
    """
    if seconds < 0:
        return "--:--"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def progress_bar(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str | None = None,
    enabled: bool = True,
) -> Iterator[Any]:
    """
    Iterates over an iterable while displaying a progress bar in the terminal.
    This function provides a visual progress indicator for long-running iterations.
    If enabled and the terminal supports it, it uses the Rich library for a modern progress bar.
    Otherwise, it falls back to a simple text-based progress bar or spinner.
    Progress is displayed on stderr and updates periodically or at the end of the iteration.
    Args:
        iterable (Iterable[Any]): The iterable to process.
        total (int | None, optional): The total number of items. If not provided, tries to infer using len().
        desc (str | None, optional): Description to display alongside the progress bar.
        enabled (bool, optional): If False, disables the progress bar and yields items directly. Defaults to True.
    Yields:
        Any: Items from the input iterable, one by one.
    Notes:
        - If the Rich library is available and the terminal supports it, a Rich progress bar is shown.
        - If not, a fallback text-based progress bar or spinner is used.
        - Progress is only shown if `enabled` is True.
        - Handles both sized and unsized iterables.
        - Displays elapsed time and estimated time remaining (ETA) when possible.
        - Displays elapsed time and estimated time remaining (ETA) when possible.
    Examples:
        >>> for item in progress_bar(range(100), desc="Processing"):
        ...     process(item)
    """
    if not enabled:
        yield from iterable
        return
    if total is None and isinstance(iterable, Sized):
        try:
            total = len(iterable)
        except Exception:
            total = None
    if Progress is not None and sys.stderr.isatty():
        Spinner = SpinnerColumn
        Text = TextColumn
        Bar = BarColumn
        TaskProgress = TaskProgressColumn
        MofN = MofNCompleteColumn
        Elapsed = TimeElapsedColumn
        Remaining = TimeRemainingColumn

        columns = [
            Spinner() if Spinner is not None else None,
            (
                Text("[progress.description]{task.description}")
                if Text is not None
                else None
            ),
            Bar(bar_width=40) if Bar is not None else None,
            TaskProgress() if TaskProgress is not None else None,
            Text("•") if Text is not None else None,
            MofN() if MofN is not None else None,
            Elapsed() if Elapsed is not None else None,
            Remaining() if Remaining is not None else None,
        ]
        columns = [c for c in columns if c is not None]
        try:
            with Progress(*columns, transient=False, refresh_per_second=4) as progress:
                task = progress.add_task(desc or "Processando...", total=total)
                for item in iterable:
                    yield item
                    progress.update(task, advance=1)
                progress.update(
                    task, completed=total if total else progress.tasks[task].completed
                )
            sys.stderr.write("\n")
            sys.stderr.flush()
            return
        except Exception as e:
            logger.debug(f"Rich progress failed: {e}, using fallback")
            pass
    try:
        terminal_width = shutil.get_terminal_size().columns
    except Exception:
        terminal_width = 80
    start_time = time.time()
    last_update = start_time
    items_processed = 0
    for idx, item in enumerate(iterable, start=1):
        yield item
        items_processed = idx
        current_time = time.time()
        if current_time - last_update >= 0.5 or (total and idx == total):
            last_update = current_time
            elapsed = current_time - start_time
            if total and total > 0:
                percentage = (idx / total) * 100
                if idx > 1 and elapsed > 1:
                    rate = idx / elapsed
                    if rate > 0:
                        eta_seconds = (total - idx) / rate
                        eta_str = f" ETA: {_format_time(eta_seconds)}"
                    else:
                        eta_str = " ETA: calculando..."
                else:
                    eta_str = " ETA: calculando..."

                bar_width = min(30, terminal_width - 60)
                filled = int((percentage / 100) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                status = (
                    f"\r{desc or 'Progresso'}: {percentage:5.1f}% "
                    f"|{bar}| {idx}/{total} "
                    f"[{_format_time(elapsed)}{eta_str}]"
                )
            else:
                spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                spinner = spinner_chars[idx % len(spinner_chars)]
                status = f"\r{desc or 'Processando'} {spinner} {idx} items [{_format_time(elapsed)}]"
            clear_line = "\r" + " " * (terminal_width - 1) + "\r"
            sys.stderr.write(clear_line + status)
            sys.stderr.flush()
    if total:
        elapsed = time.time() - start_time
        final_msg = (
            f"\r{desc or 'Concluído'}: 100.0% "
            f"|{'█' * 30}| {total}/{total} "
            f"[{_format_time(elapsed)} total]"
        )
        clear_line = "\r" + " " * (terminal_width - 1) + "\r"
        sys.stderr.write(clear_line + final_msg + "\n")
    else:
        elapsed = time.time() - start_time
        final_msg = f"\r{desc or 'Concluído'}: {items_processed} items em {_format_time(elapsed)}"
        clear_line = "\r" + " " * (terminal_width - 1) + "\r"
        sys.stderr.write(clear_line + final_msg + "\n")
    sys.stderr.flush()
