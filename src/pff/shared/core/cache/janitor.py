"""Cache janitor for background cleanup of stale entries."""

from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging import logger

if TYPE_CHECKING:
    pass


_CACHE_JANITORS: list[CacheJanitor] = []
_CACHE_JANITORS_LOCK = threading.Lock()


def shutdown_all_cache_janitors() -> None:
    """
    Stop all running cache janitor threads.

    Call this before process exit to prevent segfaults during Python interpreter shutdown.
    """
    with _CACHE_JANITORS_LOCK:
        janitors = list(_CACHE_JANITORS)
        _CACHE_JANITORS.clear()

    for janitor in janitors:
        try:
            janitor.stop()
        except Exception as exc:
            logger.debug(f"Error stopping cache janitor: {exc}")


class CacheJanitor:
    """Background task for cleaning up stale cache entries."""

    def __init__(self, cache_root: Path, max_age_seconds: int, interval_seconds: int):
        """
        Initialize the cache janitor.

        Args:
            cache_root: Root directory of the cache
            max_age_seconds: Maximum age for cache files
            interval_seconds: How often to run cleanup
        """
        self.cache_root = cache_root
        self.max_age_seconds = max_age_seconds
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __getstate__(self):
        """Prepare object for pickling."""
        state = self.__dict__.copy()
        state["_stop_event"] = None
        state["_thread"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickle."""
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        """Start the janitor thread."""
        if self.interval_seconds <= 0:
            return

        import threading

        self._thread = threading.Thread(
            target=self._run_cleanup_loop, name="CacheJanitor", daemon=True
        )
        self._thread.start()

        with _CACHE_JANITORS_LOCK:
            if self not in _CACHE_JANITORS:
                _CACHE_JANITORS.append(self)

        atexit.register(self.stop)

    def stop(self) -> None:
        """Stop the janitor thread gracefully."""
        if self._stop_event is None:
            return
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _run_cleanup_loop(self) -> None:
        """Main cleanup loop running in background thread."""
        while not self._stop_event.wait(self.interval_seconds):
            self._purge_stale_entries()

    def _purge_stale_entries(self) -> None:
        """Remove cache files older than the maximum age."""
        current_time = time.time()
        removed_count = 0

        if not self.cache_root.exists():
            return

        with os.scandir(self.cache_root) as entries:
            for entry in entries:
                if not entry.name.endswith((".pkl", ".pkl.gz")):
                    continue

                try:
                    file_age = current_time - entry.stat().st_mtime
                    if file_age > self.max_age_seconds:
                        Path(entry.path).unlink(missing_ok=True)
                        base_name = entry.name
                        if base_name.endswith(".pkl.gz"):
                            base_name = base_name[: -len(".pkl.gz")]
                        elif base_name.endswith(".pkl"):
                            base_name = base_name[: -len(".pkl")]
                        parquet_sidecar = self.cache_root / f"{base_name}.parquet"
                        parquet_sidecar.unlink(missing_ok=True)
                        removed_count += 1
                except FileNotFoundError:
                    pass
                except Exception as error:
                    logger.debug(f"Error checking cache file {entry.name}: {error}")

        if removed_count:
            logger.debug(f"[CacheJanitor] Purged {removed_count} stale entries")
