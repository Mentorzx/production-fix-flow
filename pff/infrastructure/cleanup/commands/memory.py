from __future__ import annotations

import atexit
import gc
import logging
import sys
import time

from pff import settings
from pff.shared.acceleration.concurrency import ConcurrencyManager
from pff.shared.core.cache import DiskCache
from pff.shared.core.logger import logger

from .base import CleanupCommand


class CloseLoggerCommand(CleanupCommand):
    """Gracefully shut down logger sinks.

    Closes all logging handlers and runs registered atexit functions to
    ensure log files are properly flushed before process termination.

    Attributes:
        label: Display label for UI.
    """

    label = "Fechando coletores de log ativos"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Removes loguru sinks, shuts down stdlib logging, and runs atexit handlers.
        """
        try:
            logger.remove()
            logging.shutdown()
            atexit._run_exitfuncs()  # noqa: SLF001
            time.sleep(0.2)
        except Exception:
            return


class FlushMemoryCommand(CleanupCommand):
    """Flush in-memory caches and trigger garbage collection.

    Purges disk cache, clears functools.lru_cache entries from all loaded
    modules, and runs Python garbage collection.

    Attributes:
        label: Display label for UI.
    """

    label = "Liberando caches de memória"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Purges DiskCache, clears module caches, and triggers gc.collect().
        """
        DiskCache(settings.CACHE_DIR).purge()
        for obj in list(sys.modules.values()):
            if callable(getattr(obj, "cache_clear", None)):
                obj.cache_clear()  # type: ignore[arg-type]
        gc.collect()


class ConcurrencyShutdownCommand(CleanupCommand):
    """Shutdown ConcurrencyManager workers.

    Gracefully terminates any active Ray, Dask, or thread pool workers
    managed by the ConcurrencyManager singleton.

    Attributes:
        label: Display label for UI.
    """

    label = "Encerrando gerenciador de concorrência"

    def execute(self) -> None:
        """Execute the cleanup operation.

        Calls `ConcurrencyManager().shutdown()` to terminate workers.
        """
        ConcurrencyManager().shutdown()


__all__ = ["CloseLoggerCommand", "FlushMemoryCommand", "ConcurrencyShutdownCommand"]
