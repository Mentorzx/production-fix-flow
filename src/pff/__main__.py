from __future__ import annotations

import asyncio
import signal
import sys

from pff.drivers.orchestrator import Orchestrator
from pff.shared.core.logging import logger
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.config import settings


class AppLauncher:
    """Prepares the application environment and delegates execution to the CLI."""

    def __init__(self):
        self.orchestrator: Orchestrator | None = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """
        Configures signal handlers for SIGINT and SIGTERM to enable graceful shutdown.
        When a termination signal is received, logs a warning and attempts to initiate
        a graceful shutdown of the orchestrator using the running asyncio event loop.
        If no event loop is running, logs a warning and exits the process immediately.
        This method should be called during application startup to ensure proper
        handling of shutdown signals.
        """

        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.warning(f"Signal {signal_name} received, initiating shutdown...")

            try:
                loop = asyncio.get_running_loop()
                if self.orchestrator and not loop.is_closed():
                    loop.create_task(self._graceful_shutdown())
            except RuntimeError:
                logger.warning("No event loop running. Exiting directly.")
                sys.exit(128 + signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _graceful_shutdown(self) -> None:
        """Performs graceful shutdown by calling the orchestrator's shutdown method."""
        logger.info("Iniciando graceful shutdown...")
        if self.orchestrator:
            await self.orchestrator.shutdown()

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            logger.debug(f"Cancelando {len(tasks)} tarefas pendentes...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.success("Shutdown concluído.")
        loop = asyncio.get_running_loop()
        loop.stop()

        try:
            from pff.infrastructure.persistence.db.ingestion import (
                TelecomDataIngestion,
            )

            if hasattr(TelecomDataIngestion, "_pool") and TelecomDataIngestion._pool:
                await TelecomDataIngestion._pool.close()
                logger.debug("Database connection pool closed successfully")
        except Exception as e:
            logger.warning(f"Database cleanup error (non-critical): {e}")

    def _run_health_checks(self) -> bool:
        """Executa verificações rápidas de sanidade do ambiente."""
        logger.debug("Running health checks...")
        all_ok = True
        try:
            from pff.shared.core.config import get_redis_client

            get_redis_client(db=5, decode_responses=True).ping()
            logger.debug("Redis connection OK.")
        except Exception:
            logger.warning("Redis connection failed. Worker mode will not function.")

        if not settings.DATA_DIR.exists():
            logger.error(f"Data directory not found at: {settings.DATA_DIR}")
            all_ok = False
        else:
            logger.debug("Data directory OK.")

        return all_ok

    async def launch(self) -> None:
        """Main entry point to start the application."""
        self._run_health_checks()

        try:
            from pff.drivers.cli.main import main

            await main(launcher=self)
        except KeyboardInterrupt:
            logger.warning("Execution interrupted by user.")
            sys.exit(130)
        except Exception as e:
            logger.exception(f"Critical unhandled error in execution: {e}", exc_info=True)
            sys.exit(1)


async def bootstrap():
    """Initializes the application environment and launches the core logic."""
    from pff import __version__
    from pff.shared.determinism import (
        configure_numba_threads,
        configure_torch_determinism,
    )
    from pff.shared.system.runtime import initialize_runtime

    configure_torch_determinism(enforce=True)
    initialize_runtime(__version__)
    configure_numba_threads()
    if sys.platform != "win32":
        try:
            import uvloop

            uvloop.install()
            logger.info(" uvloop instalado com sucesso (ambiente não-Windows).")
        except ImportError:
            logger.warning(" uvloop not found. Using default asyncio loop.")
    else:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.debug("Windows Proactor event loop policy configured.")

    launcher = AppLauncher()
    await launcher.launch()


if __name__ == "__main__":
    run_coroutine_sync(bootstrap())
