"""CLI command implementations."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from dependency_injector.wiring import Provide, inject
from pff import IntelligentPreprocessor, ManifestParser, Orchestrator, settings
from pff.__main__ import AppLauncher
from pff.shared import logger
from pff.shared.ops.global_interrupt_manager import (
    check_interruption,
    get_interrupt_manager,
    should_stop,
)

from pff.application.container import ApplicationContainer
from pff.application.learn_use_case import LearnUseCase
from pff.application.optimize_use_case import OptimizeUseCase
from pff.shared.core.config import ACCELERATION_CONFIG_PATH, OPTIMIZATION_CONFIG_PATH
from pff.infrastructure.hpo.runner import HpoRunner
from pff.infrastructure.hpo.grpc_proxy import run_optuna_grpc_proxy
from pff.infrastructure.persistence.db.connection import close_connection_pool
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.cache import shutdown_all_cache_janitors
from pff.shared.core.file_manager import FileManager
from pff.shared.determinism import set_global_seed


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def is_vpn_up() -> bool:
    """
    Check if a VPN connection is currently active (DISABLED).

    Returns:
        bool: Always returns False (VPN check disabled)
    """
    # VPN check disabled
    return False


def _apply_numba_thread_override(file_manager: FileManager | None = None) -> None:
    if "NUMBA_NUM_THREADS" in os.environ:
        return
    fm = file_manager or FileManager()
    if not fm.exists(ACCELERATION_CONFIG_PATH):
        return
    try:
        cfg = fm.read(ACCELERATION_CONFIG_PATH, return_native=True) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Falha ao carregar config de aceleracao: {exc}")
        return
    if not isinstance(cfg, dict):
        return
    numba_cfg = cfg.get("numba", {})
    if not isinstance(numba_cfg, dict):
        return
    num_threads = numba_cfg.get("num_threads")
    if num_threads is None:
        return
    try:
        os.environ["NUMBA_NUM_THREADS"] = str(int(num_threads))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"numba.num_threads invalido: value={num_threads!r} erro={exc}")


def _resolve_hpo_seed(file_manager: FileManager | None = None) -> int | None:
    fm = file_manager or FileManager()
    if not fm.exists(OPTIMIZATION_CONFIG_PATH):
        return None
    try:
        cfg = fm.read(OPTIMIZATION_CONFIG_PATH, return_native=True) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Falha ao carregar config HPO: {exc}")
        return None
    if not isinstance(cfg, dict):
        return None
    sampler_cfg = cfg.get("sampler", {})
    if not isinstance(sampler_cfg, dict):
        return None
    seed = sampler_cfg.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"sampler.seed invalido: value={seed!r} erro={exc}")
        return None


def _cleanup_hpo_resources() -> None:
    try:
        shutdown_all_cache_janitors()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Falha ao encerrar cache janitor: {exc}")

    try:
        run_coroutine_sync(close_connection_pool())
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Falha ao encerrar pool Postgres: {exc}")

    try:
        from numba.core.runtime import nrt  # type: ignore

        nrt.rtsys.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Falha ao encerrar runtime Numba: {exc}")


# ============================================================================
# COMMAND PATTERN - Base Classes
# ============================================================================


class Command(ABC):
    """
    Abstract base class for CLI commands (Command Pattern).

    Each command encapsulates:
    - Argument parsing logic
    - Execution logic
    - Error handling
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize command with parsed arguments.

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.interrupt_manager = get_interrupt_manager()

    @abstractmethod
    async def execute(self) -> None:
        """Execute the command asynchronously."""
        pass

    @staticmethod
    @abstractmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """
        Configure argument parser for this command.

        Args:
            subparsers: Subparsers action from main parser
        """
        pass

    def check_interruption(self) -> None:
        """Check if user requested interruption."""
        check_interruption()


class SyncCommand(Command):
    """Base class for synchronous commands."""

    async def execute(self) -> None:
        """Execute synchronously (wrapper for sync commands)."""
        self.execute_sync()

    @abstractmethod
    def execute_sync(self) -> None:
        """Execute the command synchronously."""
        pass


# ============================================================================
# CONCRETE COMMANDS - Implementation
# ============================================================================


class RunCommand(Command):
    """
    Command to execute a task manifest (Orchestrator).

    Pattern: Command Pattern
    """

    def __init__(self, args: argparse.Namespace, launcher: AppLauncher | None = None):
        super().__init__(args)
        self.launcher = launcher

    async def execute(self) -> None:
        """Execute the orchestrator workflow."""
        logger.debug(
            f"component=cli command=run evento=selecionado manifesto={self.args.manifest_file}"
        )

        # VPN check disabled
        # if not is_vpn_up():
        #     logger.critical("Nenhuma interface VPN detectada – conecte-se à VPN antes de continuar.")
        #     sys.exit(1)

        try:
            await self._run_orchestrator()
            logger.info("component=cli command=run status=sucesso")
        except FileNotFoundError:
            logger.error(
                "component=cli command=run stop_reason=manifesto_nao_encontrado "
                f"manifesto={self.args.manifest_file}"
            )
            logger.warning(
                "component=cli command=run acao_sugerida=gerar_manifesto comando=generate"
            )
            sys.exit(1)
        except Exception as e:
            logger.exception(
                f"component=cli command=run stop_reason=erro_critico erro={e}"
            )
            sys.exit(1)

    async def _run_orchestrator(self) -> None:
        """Initialize and run the orchestrator."""
        parser = ManifestParser()
        manifest = parser.parse(self.args.manifest_file)
        orchestrator = Orchestrator(
            exec_id=manifest.execution_id,
            tasks=manifest.tasks,
            max_workers=manifest.max_workers,
            resource_usage=manifest.resource_usage,
        )

        if self.launcher:
            self.launcher.orchestrator = orchestrator

        await orchestrator.run()

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'run' command parser."""
        parser = subparsers.add_parser("run", help="Executa um manifesto de tarefas.")
        parser.add_argument(
            "manifest_file",
            type=Path,
            nargs="?",
            default=settings.DEFAULT_MANIFEST_PATH,
            help=f"Caminho para o manifesto. Padrão: {settings.DEFAULT_MANIFEST_PATH}",
        )


class GenerateCommand(SyncCommand):
    """
    Command to generate a manifest file from text.

    Pattern: Command Pattern
    """

    def execute_sync(self) -> None:
        """Generate manifest from text file."""
        logger.info("component=cli command=generate status=iniciando")

        input_file = self.args.input_file
        output_file = self.args.output_file

        if not input_file.exists():
            logger.error(
                "component=cli command=generate stop_reason=entrada_nao_encontrada "
                f"arquivo={input_file}"
            )
            sys.exit(1)

        preprocessor = IntelligentPreprocessor()

        try:
            preprocessor.process_text(input_file, output_file)
            logger.success(
                f"component=cli command=generate status=sucesso arquivo={output_file}"
            )
        except Exception as e:
            logger.exception(
                f"component=cli command=generate stop_reason=erro_geracao erro={e}"
            )
            sys.exit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'generate' command parser."""
        parser = subparsers.add_parser(
            "generate",
            help="Gera o manifesto padrao a partir de texto bruto.",
        )
        parser.add_argument(
            "input_file", type=Path, help="Arquivo de texto com descrição"
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output_file",
            type=Path,
            default=settings.DEFAULT_MANIFEST_PATH,
            help=f"Arquivo de saída do manifesto (padrão: {settings.DEFAULT_MANIFEST_PATH})",
        )


class WorkerCommand(SyncCommand):
    """
    Command to start Celery worker.

    Pattern: Command Pattern
    """

    def execute_sync(self) -> None:
        """Start Celery worker."""
        logger.info("component=cli command=worker status=iniciando")

        from pff.celery_app import celery_app

        worker_args = [
            "worker",
            "--loglevel=info",
        ]

        celery_app.worker_main(worker_args)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'worker' command parser."""
        subparsers.add_parser("worker", help="Inicia um worker Celery.")


class APICommand(Command):
    """
    Command to start FastAPI server.

    Pattern: Command Pattern
    """

    async def execute(self) -> None:
        """Start Granian server (High Performance)."""
        logger.info(
            "component=cli command=api status=iniciando_granian "
            f"host={self.args.host} port={self.args.port} reload={self.args.reload}"
        )

        from granian import Granian
        from granian.constants import Interfaces

        # Granian configuration for maximum performance
        # Using 'pff.drivers.api.main:app' as target
        server = Granian(
            target="pff.drivers.api.main:app",
            address=self.args.host,
            port=self.args.port,
            interface=Interfaces.ASGI,
            websockets=True,
            reload=self.args.reload,
            workers=int(os.getenv("WEB_CONCURRENCY", 1)),
            loop="uvloop",  # Enforce uvloop
            log_level="info",
        )

        await asyncio.to_thread(server.serve)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'api' command parser."""
        parser = subparsers.add_parser("api", help="Inicia o servidor da API.")
        parser.add_argument("--host", default="0.0.0.0", help="Host do servidor")
        parser.add_argument("--port", type=int, default=8000, help="Porta do servidor")
        parser.add_argument(
            "--reload", action="store_true", help="Auto-reload em desenvolvimento"
        )


class CleanCommand(Command):
    """
    Command to run cleanup operations.

    Pattern: Command Pattern
    """

    async def execute(self) -> None:
        """Execute cleanup command."""
        from pff.infrastructure.cleanup.engine import build_engine

        logger.info(
            "component=cli command=clean status=iniciando "
            f"strategy={self.args.strategy} dry_run={self.args.dry_run} auto_yes={self.args.yes}"
        )
        engine = build_engine(
            self.args.strategy,
            auto_yes=self.args.yes,
            dry_run=self.args.dry_run,
        )
        await engine.run()

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'clean' command parser."""
        parser = subparsers.add_parser(
            "clean",
            help="Limpa caches, logs, outputs e artefatos temporarios.",
        )
        parser.add_argument(
            "strategy",
            choices=["standard", "deep", "ml", "shutdown"],
            nargs="?",
            default="standard",
            help="A estratégia de limpeza a ser utilizada.",
        )
        parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Não pedir confirmação.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Simular execução sem deletar.",
        )


class ResetMLCommand(Command):
    """
    Command to reset ML state and data.

    Pattern: Command Pattern
    """

    async def execute(self) -> None:
        """Execute ML reset command."""
        from pff.infrastructure.cleanup.reset_ml import run_reset_ml

        logger.info("component=cli command=reset-ml status=iniciando")
        await run_reset_ml()

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'reset-ml' command parser."""
        subparsers.add_parser(
            "reset-ml",
            help="Reseta completamente o ambiente de ML/DSLFM+PC",
        )


class LogsCommand(Command):
    """
    Command to view logs and metrics.

    Pattern: Command Pattern
    """

    async def execute(self) -> None:
        """Execute logs command."""
        from pff.application.services.execution_log_service import ExecutionLogService
        from pff.application.services.training_metrics_service import (
            TrainingMetricsService,
        )

        log_service = ExecutionLogService()
        metrics_service = TrainingMetricsService()

        if self.args.subcommand == "list":
            logs = log_service.list_logs(
                operation=self.args.operation,
                status=self.args.status,
                last_hours=self.args.last_hours,
                limit=self.args.limit,
            )

            for log in logs:
                logger.info(f"component=cli command=logs evento=lista item={log}")

        elif self.args.subcommand == "stats":
            stats = log_service.get_statistics(operation=self.args.operation)
            logger.info(f"component=cli command=logs evento=stats dados={stats}")

        elif self.args.subcommand == "metrics":
            metrics = metrics_service.list_metrics(
                log_id=self.args.log_id,
                model=self.args.model,
            )

            for metric in metrics:
                logger.info(f"component=cli command=logs evento=metricas item={metric}")

        elif self.args.subcommand == "cleanup":
            deleted = log_service.cleanup_old_logs(days=self.args.days)
            logger.success(
                f"component=cli command=logs evento=cleanup status=sucesso removidos={deleted}"
            )

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'logs' command parser."""
        parser = subparsers.add_parser(
            "logs",
            help="Visualizar e gerenciar logs de execucao e metricas de treinamento",
            description="Comandos para visualizar execution logs e training metrics do PostgreSQL",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        logs_subparsers = parser.add_subparsers(dest="subcommand", required=True)

        # pff logs list
        list_parser = logs_subparsers.add_parser("list", help="Listar logs de execução")
        list_parser.add_argument("--operation", type=str, help="Filtrar por operação")
        list_parser.add_argument(
            "--status",
            type=str,
            choices=["running", "success", "failed"],
            help="Filtrar por status",
        )
        list_parser.add_argument(
            "--last-hours", type=int, help="Mostrar logs das últimas N horas"
        )
        list_parser.add_argument(
            "--limit", type=int, default=50, help="Número máximo de logs (padrão: 50)"
        )

        # pff logs stats
        stats_parser = logs_subparsers.add_parser(
            "stats", help="Estatísticas de execução"
        )
        stats_parser.add_argument("--operation", type=str, help="Filtrar por operação")

        # pff logs metrics
        metrics_parser = logs_subparsers.add_parser(
            "metrics", help="Visualizar métricas de treinamento"
        )
        metrics_parser.add_argument("--log-id", type=int, help="ID do execution log")
        metrics_parser.add_argument(
            "--model", type=str, help="Filtrar por modelo (dslfm)"
        )

        # pff logs cleanup
        cleanup_parser = logs_subparsers.add_parser(
            "cleanup", help="Deletar logs antigos"
        )
        cleanup_parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Deletar logs mais antigos que N dias (padrão: 30)",
        )


class LearnCommand(Command):
    """
    Command to train AI models.

    Pattern: Command Pattern + Strategy Pattern (model types) + Template Method
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.model = getattr(args, "model", "all")
        self.config_path = getattr(args, "config", None)

    async def execute(self) -> None:
        """Execute training based on model type (Strategy Pattern)."""
        logger.info(
            "component=cli command=learn status=iniciando info=global_interrupt_manager_ativo"
        )

        # Register interrupt callback
        def learn_interrupt_callback():
            logger.info("component=cli command=learn evento=interrompendo")

        self.interrupt_manager.register_callback_once(
            learn_interrupt_callback, label="learn_cli_interrupt"
        )

        container = ApplicationContainer()
        container.config_path.override(self.config_path)
        container.wire(modules=[__name__])
        try:
            await _run_learn(self.model)
        except KeyboardInterrupt:
            logger.warning(
                "component=cli command=learn stop_reason=interrompido_usuario"
            )
            logger.info("component=cli command=learn evento=limpeza_graceful")
            await asyncio.sleep(0.5)
            logger.success("component=cli command=learn status=interrompido_tratado")
            sys.exit(128)
        except Exception as e:
            logger.exception(
                f"component=cli command=learn stop_reason=erro_critico erro={e}"
            )
            sys.exit(1)
        finally:
            container.unwire()
            if should_stop():
                logger.info("component=cli command=learn evento=limpeza_final")

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'learn' command parser."""
        parser = subparsers.add_parser("learn", help="Treinar modelos de IA")
        parser.add_argument(
            "model",
            nargs="?",
            default="kgc",
            choices=["kg", "kgc", "all"],
            help=(
                "Modelo a treinar: kg=preprocess, kgc=DSLFM-KGC (BERT+VAE+IBP+PC, padrão), "
                "all=pipeline completa"
            ),
        )
        parser.add_argument("-c", "--config", type=Path, help="Arquivo de configuração")


class HpoCommand(Command):
    """Command to run HPO for DSLFM-KGC."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.model = getattr(args, "model", "dslfm-kgc")
        self.trials = int(getattr(args, "trials", 50))
        self.study_name = getattr(args, "study_name", None)
        self.no_update_config = bool(getattr(args, "no_update_config", False))
        self.no_bert = bool(getattr(args, "no_bert", False))

    async def execute(self) -> None:
        """Execute HPO workflow."""
        logger.info(
            "component=cli command=hpo status=iniciando info=global_interrupt_manager_ativo"
        )

        def hpo_interrupt_callback():
            logger.info("component=cli command=hpo evento=interrompendo")

        self.interrupt_manager.register_callback_once(
            hpo_interrupt_callback, label="hpo_cli_interrupt"
        )

        seed = _resolve_hpo_seed()
        if seed is not None:
            set_global_seed(seed)
        _apply_numba_thread_override()
        try:
            from pff.infrastructure.hpo.dashboard import ensure_optuna_dashboard_running

            ensure_optuna_dashboard_running()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"component=cli command=hpo dashboard_optuna status=erro erro={exc}"
            )

        study_name = self.study_name or f"pff_kg_real_{self.model.replace('-', '_')}"

        logger.info(
            f"hpo_iniciado modelo={self.model.upper()} trials={self.trials} "
            f"fonte={settings.DATA_DIR / 'models' / 'kg'}"
        )

        runner = HpoRunner()
        use_case = OptimizeUseCase(runner)

        try:
            result = use_case.execute(
                n_trials=self.trials,
                strategy="optuna",
                enable_mlflow=True,
                enable_visualization=False,
                study_name=study_name,
                target_entity_ratio=0.7,
                kge_model=self.model,
                no_update_config=self.no_update_config,
                no_bert=self.no_bert,
            )
        except KeyboardInterrupt:
            logger.warning("component=cli command=hpo stop_reason=user_interrupted")
            sys.exit(128)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                f"component=cli command=hpo stop_reason=erro_critico erro={exc}"
            )
            sys.exit(1)
        finally:
            _cleanup_hpo_resources()

        logger.success(
            f"hpo_concluido trials={result.get('n_trials', 0)} "
            f"tempo_s={result.get('optimization_time', 0):.1f}"
        )

        if "real_data_info" in result:
            info = result["real_data_info"]
            logger.info(
                f"dados_reais n_train={info.get('n_train', 'N/A')} "
                f"n_valid={info.get('n_valid', 'N/A')} "
                f"n_entities={info.get('n_entities', 'N/A')} "
                f"n_predicates={info.get('n_predicates', 'N/A')}"
            )

        mo = result.get("multi_objective", {}) or {}
        best_tradeoff = mo.get("best_tradeoff") or {}
        best_time = mo.get("best_time_aware") or {}
        best_quality = mo.get("best_quality") or {}

        if best_tradeoff:
            logger.info(
                f"melhor_tradeoff score_time={best_tradeoff.get('score_time', 0.0):.4f} "
                f"tradeoff_score={best_tradeoff.get('tradeoff_score', 0.0):.4f} "
                f"trial={best_tradeoff.get('trial_number', 'N/A')} "
                f"duracao_s={best_tradeoff.get('duration', 0.0):.1f}"
            )
        elif result.get("best_value") is not None:
            logger.info(f"melhor_score score={result['best_value']:.4f}")
        else:
            logger.warning(
                "Best score unavailable (optimization failed or no solution found)"
            )

        if best_time:
            logger.info(
                f"campeao_tempoaware trial={best_time.get('trial_number', 'N/A')} "
                f"score={best_time.get('score_time', 0.0):.4f} "
                f"duracao_s={best_time.get('duration', 0.0):.1f}"
            )
        if best_quality:
            logger.info(
                f"campeao_sem_tempo trial={best_quality.get('trial_number', 'N/A')} "
                f"score={best_quality.get('score_quality', 0.0):.4f}"
            )

        if self.no_update_config:
            logger.info("auto_update_config desabilitado")

        dashboard_url = os.getenv(
            "OPTUNA_DASHBOARD_URL", "http://localhost:8080/dashboard"
        )
        if result.get("live_dashboard"):
            logger.info(
                f"dashboard_optuna url={dashboard_url} html={result.get('live_dashboard')}"
            )

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'hpo' command parser."""
        parser = subparsers.add_parser(
            "hpo", help="Otimizar hiperparametros (DSLFM-KGC)"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="dslfm-kgc",
            choices=["dslfm-kgc"],
            help="Modelo KGE (DSLFM-KGC com BERT + VAE + IBP + PC)",
        )
        parser.add_argument("--trials", type=int, default=50, help="Numero de trials")
        parser.add_argument(
            "--study-name", type=str, default=None, help="Nome do estudo Optuna"
        )
        parser.add_argument(
            "--no-update-config",
            action="store_true",
            help="Nao atualizar automaticamente o config/models/dslfm.yaml",
        )
        parser.add_argument(
            "--no-bert",
            action="store_true",
            help="Desabilitar encoder BERT para relacoes (usa defaults do YAML quando aplicavel)",
        )


class HpoProxyCommand(Command):
    """Command to run Optuna gRPC storage proxy."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.host = getattr(args, "host", None)
        self.port = getattr(args, "port", None)
        self.storage_url = getattr(args, "storage_url", None)

    async def execute(self) -> None:
        """Start the Optuna gRPC proxy server."""
        logger.info("component=cli command=hpo-proxy status=iniciando")
        try:
            run_optuna_grpc_proxy(
                host=self.host,
                port=self.port,
                storage_url=self.storage_url,
            )
        except KeyboardInterrupt:
            logger.warning(
                "component=cli command=hpo-proxy stop_reason=user_interrupted"
            )
            raise SystemExit(128)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                f"component=cli command=hpo-proxy stop_reason=erro_critico erro={exc}"
            )
            raise SystemExit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'hpo-proxy' command parser."""
        parser = subparsers.add_parser("hpo-proxy", help="Iniciar proxy gRPC do Optuna")
        parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="Host do proxy gRPC (default: config storage.grpc_proxy.host)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="Porta do proxy gRPC (default: config storage.grpc_proxy.port)",
        )
        parser.add_argument(
            "--storage-url",
            type=str,
            default=None,
            help="URL RDB do Optuna (default: config storage.url ou Postgres settings)",
        )


@inject
async def _run_learn(
    model: str,
    use_case: LearnUseCase = Provide[ApplicationContainer.learn_use_case],
) -> None:
    """Execute LearnUseCase with dependency injection."""
    await use_case.execute(model)
