from __future__ import annotations

import asyncio
import signal
import sys

from pff import settings
from pff.orchestrator import Orchestrator
from pff.utils import logger


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
            logger.warning(f"Sinal {signal_name} recebido, iniciando desligamento...")

            try:
                loop = asyncio.get_running_loop()
                if self.orchestrator and not loop.is_closed():
                    loop.create_task(self._graceful_shutdown())
            except RuntimeError:
                logger.warning(
                    "Nenhum loop de eventos rodando. Encerrando diretamente."
                )
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
            from pff.db.ingestion import TelecomDataIngestion
            if hasattr(TelecomDataIngestion, '_pool') and TelecomDataIngestion._pool:
                await TelecomDataIngestion._pool.close()
                logger.debug("Database connection pool closed successfully")
        except Exception as e:
            logger.warning(f"Database cleanup error (non-critical): {e}")

    def _run_health_checks(self) -> bool:
        """Executa verificações rápidas de sanidade do ambiente."""
        logger.debug("Executando health checks...")
        all_ok = True
        try:
            from pff.config import rds

            rds.ping()
            logger.debug("✅ Conexão com Redis OK.")
        except Exception:
            logger.warning(
                "⚠️ Conexão com Redis falhou. O modo 'worker' não funcionará."
            )

        if not settings.DATA_DIR.exists():
            logger.error(
                f"❌ Diretório de dados não encontrado em: {settings.DATA_DIR}"
            )
            all_ok = False
        else:
            logger.debug("✅ Diretório de dados OK.")

        return all_ok

    async def launch(self) -> None:
        """Main entry point to start the application."""
        self._run_health_checks()

        try:
            from pff.cli import main

            await main(launcher=self)
        except KeyboardInterrupt:
            logger.warning("Execução interrompida pelo usuário.")
            sys.exit(130)
        except Exception as e:
            logger.exception(f"Erro crítico não tratado na execução: {e}", exc_info=True)
            sys.exit(1)


async def bootstrap():
    """Initializes the application environment and launches the core logic."""

    if sys.platform != "win32":
        try:
            import uvloop

            uvloop.install()
            logger.info("✅ uvloop instalado com sucesso (ambiente não-Windows).")
        except ImportError:
            logger.warning("⚠️ uvloop não encontrado. Usando o loop padrão do asyncio.")
    else:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.debug("✅ Política de loop Proactor do Windows configurada.")

    launcher = AppLauncher()
    await launcher.launch()


if __name__ == "__main__":
    asyncio.run(bootstrap())
