"""CLI command implementations."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pff.shared.core.logging import logger
from pff.shared.ops.global_interrupt_manager import (
    check_interruption,
    get_interrupt_manager,
    should_stop,
)

if TYPE_CHECKING:
    from pff.__main__ import AppLauncher
    from pff.shared.core.file_manager import FileManager


def is_vpn_up() -> bool:
    """
    Check if a VPN connection is currently active (DISABLED).

    Returns:
        bool: Always returns False (VPN check disabled)
    """

    return False


def _resolve_hpo_seed(file_manager: "FileManager | None" = None) -> int | None:
    """Execute resolve hpo seed.



    Args:

        file_manager: Optional input value.



    Returns:

        Return value produced by the callable.

    """

    from pff.shared.core.config import OPTIMIZATION_CONFIG_PATH
    from pff.shared.core.config_loader import load_config

    cfg = load_config(OPTIMIZATION_CONFIG_PATH)
    if not cfg:
        return None
    sampler_cfg = cfg.get("sampler", {})
    if not isinstance(sampler_cfg, dict):
        return None
    seed = sampler_cfg.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except Exception as exc:
        logger.warning(f"Invalid sampler.seed: value={seed!r} error={exc}")
        return None


def _cleanup_hpo_resources() -> None:
    """Execute cleanup hpo resources."""

    from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
    from pff.shared.core.cache import shutdown_all_cache_janitors
    from pff.infrastructure.persistence.db.connection import close_connection_pool

    try:
        shutdown_all_cache_janitors()
    except Exception as exc:
        logger.debug(f"Failed to shut down cache janitor: {exc}")

    try:
        run_coroutine_sync(close_connection_pool())
    except Exception as exc:
        logger.debug(f"Failed to shut down Postgres pool: {exc}")


_HPO_DASHBOARD_DEFAULT_PORT = 8766
_HPO_DASHBOARD_DEFAULT_BIND = "127.0.0.1"
_HPO_DASHBOARD_HEALTHCHECK_TIMEOUT_S = 20.0


def _hpo_dashboard_pid_path() -> Path:
    """Execute hpo dashboard pid path.



    Returns:

        Return value produced by the callable.

    """

    from pff.shared.core.config import settings

    return settings.CACHE_DIR / "hpo" / "dashboard_server.pid"


def _load_hpo_dashboard_pid(pid_path: Path) -> int | None:
    """Execute load hpo dashboard pid.



    Args:

        pid_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    from pff.shared.core.file_manager import FileManager

    if not FileManager.exists(pid_path):
        return None
    try:
        raw = FileManager.read_bytes(pid_path).decode("utf-8", errors="ignore").strip()
        return int(raw)
    except (ValueError, OSError):
        return None


def _is_pid_running(pid: int) -> bool:
    """Execute is pid running.



    Args:

        pid: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _hpo_dashboard_healthcheck(bind: str, port: int, timeout_s: float = 10.0) -> bool:
    from pff.infrastructure.hpo.dashboard_healthcheck import is_dashboard_healthy

    return is_dashboard_healthy(bind=bind, port=port, timeout_s=timeout_s)


def _hpo_dashboard_build_script_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent.parent
        / "infrastructure"
        / "hpo"
        / "dashboard"
        / "build_dashboard.sh"
    )


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


class RunCommand(Command):
    """
    Command to execute a task manifest (Orchestrator).

    Pattern: Command Pattern
    """

    def __init__(self, args: argparse.Namespace, launcher: "AppLauncher | None" = None):
        """Execute init.



        Args:

            args: Input value used by this callable.

            launcher: Optional input value.

        """

        super().__init__(args)
        self.launcher = launcher

    async def execute(self) -> None:
        """Execute the orchestrator workflow."""
        logger.debug(f"Manifesto selecionado: {self.args.manifest_file}")

        try:
            await self._run_orchestrator()
            logger.success("Workflow do orquestrador concluído com sucesso")
        except FileNotFoundError:
            logger.error(f"Manifest not found: {self.args.manifest_file}")
            logger.warning("Run 'pff generate' to create a manifest first")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Critical error in orchestrator: {e}")
            sys.exit(1)

    async def _run_orchestrator(self) -> None:
        """Initialize and run the orchestrator."""
        from pff import ManifestParser, Orchestrator

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
        from pff.shared.core.config import settings

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
        from pff import IntelligentPreprocessor

        input_file = self.args.input_file
        output_file = self.args.output_file

        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)

        preprocessor = IntelligentPreprocessor()
        process_text = getattr(preprocessor, "process_text", None)
        if not callable(process_text):
            raise RuntimeError("IntelligentPreprocessor.process_text is not available")

        try:
            process_text(input_file, output_file)
            logger.success(f"Manifesto gerado: {output_file}")
        except Exception as e:
            logger.exception(f"Manifest generation failed: {e}")
            sys.exit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'generate' command parser."""
        from pff.shared.core.config import settings

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
        logger.info("Iniciando worker Celery...")

        from pff import celery_app

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
            f"Iniciando servidor Granian: host={self.args.host} porta={self.args.port} reload={self.args.reload}"
        )

        from granian import Granian
        from granian.constants import Interfaces, Loops
        from granian.log import LogLevels

        server = Granian(
            target="pff.drivers.api.main:app",
            address=self.args.host,
            port=self.args.port,
            interface=Interfaces.ASGI,
            websockets=True,
            reload=self.args.reload,
            workers=int(os.getenv("WEB_CONCURRENCY", 1)),
            loop=Loops.uvloop,
            log_level=LogLevels.info,
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
            f"Iniciando limpeza: estrategia={self.args.strategy} dry_run={self.args.dry_run}"
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

        logger.info("Resetando ambiente ML/DSLFM+PC...")
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
        from datetime import datetime, timedelta

        from pff.infrastructure.persistence.db.repositories.execution_logs import (
            ExecutionLogsRepository,
        )
        from pff.infrastructure.persistence.db.repositories.training_metrics import (
            TrainingMetricsRepository,
        )

        log_repository = ExecutionLogsRepository()
        metrics_repository = TrainingMetricsRepository()

        if self.args.subcommand == "list":
            since = None
            if self.args.last_hours:
                since = datetime.now() - timedelta(hours=self.args.last_hours)

            logs = await log_repository.get_logs(
                operation=self.args.operation,
                status=self.args.status,
                since=since,
                limit=self.args.limit,
            )

            for log in logs:
                logger.info(f"{log}")

        elif self.args.subcommand == "stats":
            stats = await log_repository.get_statistics(operation=self.args.operation)
            logger.info(f"Estatísticas: {stats}")

        elif self.args.subcommand == "metrics":
            metrics = await metrics_repository.get_metrics(
                execution_log_id=self.args.log_id,
                model_name=self.args.model,
            )

            for metric in metrics:
                logger.info(f"{metric}")

        elif self.args.subcommand == "cleanup":
            deleted = await log_repository.delete_old_logs(
                older_than_days=self.args.days
            )
            logger.success(f"Logs antigos removidos: {deleted} registros")

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

        stats_parser = logs_subparsers.add_parser(
            "stats", help="Estatísticas de execução"
        )
        stats_parser.add_argument("--operation", type=str, help="Filtrar por operação")

        metrics_parser = logs_subparsers.add_parser(
            "metrics", help="Visualizar métricas de treinamento"
        )
        metrics_parser.add_argument("--log-id", type=int, help="ID do execution log")
        metrics_parser.add_argument(
            "--model", type=str, help="Filtrar por modelo (dslfm)"
        )

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
        """Execute init.



        Args:

            args: Input value used by this callable.

        """

        super().__init__(args)
        self.model = getattr(args, "model", "all")
        self.config_path = getattr(args, "config", None)

    async def execute(self) -> None:
        """Execute training based on model type (Strategy Pattern)."""
        from pff.shared.determinism import configure_torch_determinism

        configure_torch_determinism(enforce=True)
        logger.info("Iniciando treinamento...")

        def learn_interrupt_callback():
            """Execute learn interrupt callback."""

            logger.info("Treinamento interrompido pelo usuário")

        self.interrupt_manager.register_callback_once(
            learn_interrupt_callback, label="learn_cli_interrupt"
        )
        try:
            await _run_learn(self.model, config_path=self.config_path)
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            await asyncio.sleep(0.5)
            logger.success("Treinamento interrompido com segurança")
            sys.exit(128)
        except Exception as e:
            logger.exception(f"Critical training error: {e}")
            sys.exit(1)
        finally:
            if should_stop():
                logger.debug("Final cleanup after interruption")

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
        """Execute init.



        Args:

            args: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        super().__init__(args)
        from pff.shared.core.config import settings

        self.subcommand = getattr(args, "hpo_subcommand", None)
        self.model = getattr(args, "model", "dslfm-kgc")
        _cli_trials = getattr(args, "trials", None)
        if _cli_trials is None:
            _cli_trials = settings.HPO_CONFIG.get("defaults", {}).get("n_trials", 50)
        self.trials = int(_cli_trials)
        self.study_name = getattr(args, "study_name", None)
        self.no_update_config = bool(getattr(args, "no_update_config", False))
        self.no_bert = bool(getattr(args, "no_bert", False))
        self.dashboard_action = getattr(args, "dashboard_action", None)
        self.dashboard_bind = getattr(
            args, "dashboard_bind", _HPO_DASHBOARD_DEFAULT_BIND
        )
        self.dashboard_port = int(
            getattr(args, "dashboard_port", _HPO_DASHBOARD_DEFAULT_PORT)
        )
        self.dashboard_no_healthcheck = bool(
            getattr(args, "dashboard_no_healthcheck", False)
        )
        self.dashboard_healthcheck_timeout = float(
            getattr(
                args,
                "dashboard_healthcheck_timeout",
                _HPO_DASHBOARD_HEALTHCHECK_TIMEOUT_S,
            )
        )

    async def execute(self) -> None:
        """Execute HPO workflow."""
        if self.subcommand == "dashboard":
            self._execute_dashboard_action()
            return

        from pff.application.optimize_use_case import OptimizeUseCase
        from pff.infrastructure.hpo.background_process import BackgroundProcess
        from pff.infrastructure.hpo.runner import HpoRunner
        from pff.shared.core.config import settings

        self._prepare_hpo_runtime()

        study_name = self.study_name or f"pff_kg_real_{self.model.replace('-', '_')}"
        logger.info(
            f"HPO configurado: modelo={self.model.upper()}, trials={self.trials}, "
            f"fonte={settings.DATA_DIR / 'models' / 'kg'}"
        )
        runner = HpoRunner()
        use_case = OptimizeUseCase(runner)
        self._build_dashboard_if_available()

        async with BackgroundProcess(
            [
                sys.executable,
                "-m",
                "pff.infrastructure.hpo.dashboard.server",
                "--bind",
                "127.0.0.1",
                "--parent-pid",
                str(os.getpid()),
            ],
            name="HPO Dashboard Server",
        ):
            result = self._execute_hpo_use_case(use_case, study_name)
        self._log_hpo_result(result)

    def _prepare_hpo_runtime(self) -> None:
        """Execute prepare hpo runtime."""

        from pff.shared.determinism import configure_torch_determinism, set_global_seed

        configure_torch_determinism(enforce=True)
        logger.info("Iniciando workflow HPO...")
        self.interrupt_manager.register_callback_once(
            lambda: logger.info("HPO interrompido pelo usuário"),
            label="hpo_cli_interrupt",
        )
        seed = _resolve_hpo_seed()
        if seed is not None:
            set_global_seed(seed)

    def _build_dashboard_if_available(self) -> None:
        """Execute build dashboard if available."""

        build_script = _hpo_dashboard_build_script_path()
        if not build_script.exists():
            return
        try:
            logger.info("Compilando dashboard HPO...")
            subprocess.run(
                ["bash", str(build_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.success("Dashboard HPO compilado")
        except subprocess.CalledProcessError as exc:
            logger.error(f"Dashboard build failed: {exc.stderr}")

    def _execute_hpo_use_case(self, use_case: Any, study_name: str) -> dict[str, Any]:
        """Execute execute hpo use case.



        Args:

            use_case: Input value used by this callable.

            study_name: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        self._start_dashboard_healthcheck_thread()
        try:
            return use_case.execute(
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
            logger.warning("HPO interrupted by user")
            sys.exit(128)
        except Exception as exc:
            logger.exception(f"Critical HPO error: {exc}")
            sys.exit(1)
        finally:
            _cleanup_hpo_resources()

    @staticmethod
    def _start_dashboard_healthcheck_thread() -> None:
        """Execute start dashboard healthcheck thread."""

        from pff.infrastructure.hpo.dashboard_healthcheck import (
            start_dashboard_healthcheck_thread,
        )

        dashboard_bind = _HPO_DASHBOARD_DEFAULT_BIND
        dashboard_port = _HPO_DASHBOARD_DEFAULT_PORT
        dashboard_timeout = _HPO_DASHBOARD_HEALTHCHECK_TIMEOUT_S
        dashboard_url = f"http://{dashboard_bind}:{dashboard_port}/api/status"

        def _on_success() -> None:
            logger.success("Dashboard HPO está saudável")

        def _on_timeout() -> None:
            logger.warning(
                f"Dashboard health check failed: url={dashboard_url} timeout_s={dashboard_timeout}"
            )

        start_dashboard_healthcheck_thread(
            bind=dashboard_bind,
            port=dashboard_port,
            timeout_s=dashboard_timeout,
            on_success=_on_success,
            on_timeout=_on_timeout,
        )

    def _log_hpo_result(self, result: dict[str, Any]) -> None:
        """Execute log hpo result.



        Args:

            result: Input value used by this callable.

        """

        logger.success(
            f"HPO concluído: {result.get('n_trials', 0)} trials em "
            f"{result.get('optimization_time', 0):.1f}s"
        )
        self._log_real_data_info(result)
        self._log_multi_objective_summary(result)
        if self.no_update_config:
            logger.info("Auto-update do config desabilitado (--no-update-config)")
        dashboard_url = os.getenv(
            "OPTUNA_DASHBOARD_URL", "http://localhost:8080/dashboard"
        )
        if result.get("live_dashboard"):
            logger.info(
                f"Dashboard Optuna: url={dashboard_url} html={result.get('live_dashboard')}"
            )

    @staticmethod
    def _log_real_data_info(result: dict[str, Any]) -> None:
        """Execute log real data info.



        Args:

            result: Input value used by this callable.

        """

        if "real_data_info" not in result:
            return
        info = result["real_data_info"]
        logger.info(
            f"Dados reais: train={info.get('n_train', 'N/A')}, "
            f"valid={info.get('n_valid', 'N/A')}, entidades={info.get('n_entities', 'N/A')}"
        )

    @staticmethod
    def _log_multi_objective_summary(result: dict[str, Any]) -> None:
        """Execute log multi objective summary.



        Args:

            result: Input value used by this callable.

        """

        mo = result.get("multi_objective", {}) or {}
        best_tradeoff = mo.get("best_tradeoff") or {}
        best_time = mo.get("best_time_aware") or {}
        best_quality = mo.get("best_quality") or {}
        if best_tradeoff:
            logger.info(
                f"Melhor tradeoff: score_time={best_tradeoff.get('score_time', 0.0):.4f}, "
                f"tradeoff={best_tradeoff.get('tradeoff_score', 0.0):.4f}, "
                f"trial #{best_tradeoff.get('trial_number', 'N/A')}, "
                f"duração={best_tradeoff.get('duration', 0.0):.1f}s"
            )
        elif result.get("best_value") is not None:
            logger.info(f"Melhor score: {result['best_value']:.4f}")
        else:
            logger.warning("No best score available from optimization")
        if best_time:
            logger.info(
                f"Campeão time-aware: trial #{best_time.get('trial_number', 'N/A')}, "
                f"score={best_time.get('score_time', 0.0):.4f}, "
                f"duração={best_time.get('duration', 0.0):.1f}s"
            )
        if best_quality:
            logger.info(
                f"Campeão qualidade: trial #{best_quality.get('trial_number', 'N/A')}, "
                f"score={best_quality.get('score_quality', 0.0):.4f}"
            )

    def _execute_dashboard_action(self) -> None:
        """Execute execute dashboard action."""

        action = self.dashboard_action or "status"
        logger.info(
            f"Comando dashboard HPO: acao={action}, bind={self.dashboard_bind}:{self.dashboard_port}"
        )

        if action in {"on", "restart"}:
            self._start_or_restart_dashboard()
            return
        if action == "off":
            self._stop_dashboard()
            return
        if action == "build":
            self._build_dashboard()
            return
        if action == "status":
            self._dashboard_status()
            return

        logger.error(f"Invalid dashboard action: {action}")
        sys.exit(2)

    def _dashboard_status(self) -> None:
        """Execute dashboard status."""

        pid_path = _hpo_dashboard_pid_path()
        pid = _load_hpo_dashboard_pid(pid_path)
        if pid is None:
            logger.info("Dashboard parece desligado (PID ausente)")
            return
        if not _is_pid_running(pid):
            logger.info(f"Dashboard parece desligado (PID {pid} stale)")
            return
        logger.success(
            f"Dashboard ativo: PID={pid}, bind={self.dashboard_bind}:{self.dashboard_port}"
        )

    def _start_or_restart_dashboard(self) -> None:
        """Execute start or restart dashboard."""

        pid_path = _hpo_dashboard_pid_path()
        pid = _load_hpo_dashboard_pid(pid_path)
        if pid is not None and _is_pid_running(pid):
            logger.info(f"Dashboard já ativo (PID={pid}); reiniciando...")
            self._stop_dashboard()
        self._start_dashboard()

    def _start_dashboard(self) -> None:
        """Execute start dashboard."""

        from pff.shared.core.file_manager import FileManager

        pid_path = _hpo_dashboard_pid_path()
        cmd = [
            sys.executable,
            "-m",
            "pff.infrastructure.hpo.dashboard.server",
            "--bind",
            self.dashboard_bind,
            "--port",
            str(self.dashboard_port),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            logger.error(f"Failed to start dashboard server: {exc}")
            sys.exit(1)

        FileManager.write_text(str(proc.pid), pid_path)
        logger.success(
            f"Dashboard iniciado: PID={proc.pid}, bind={self.dashboard_bind}:{self.dashboard_port}"
        )

        if self.dashboard_no_healthcheck:
            return

        if _hpo_dashboard_healthcheck(
            self.dashboard_bind,
            self.dashboard_port,
            timeout_s=self.dashboard_healthcheck_timeout,
        ):
            logger.success("Dashboard saudável")
        else:
            logger.warning(
                f"Dashboard healthcheck failed: bind={self.dashboard_bind}:{self.dashboard_port}, "
                f"timeout={self.dashboard_healthcheck_timeout}s"
            )

    def _stop_dashboard(self) -> None:
        """Execute stop dashboard."""

        from pff.shared.core.file_manager import FileManager

        pid_path = _hpo_dashboard_pid_path()
        pid = _load_hpo_dashboard_pid(pid_path)
        if pid is None:
            logger.info("Dashboard já está desligado")
            return
        if not _is_pid_running(pid):
            if FileManager.exists(pid_path):
                pid_path.unlink(missing_ok=True)
            logger.info(f"PID {pid} não está ativo; dashboard já desligado")
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            logger.error(f"Failed to terminate dashboard process (PID={pid}): {exc}")
            return

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not _is_pid_running(pid):
                break
            time.sleep(0.1)

        if _is_pid_running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception as exc:
                logger.error(f"Failed to kill dashboard process (PID={pid}): {exc}")
                return

        pid_path.unlink(missing_ok=True)
        logger.success(f"Dashboard desligado (PID={pid})")

    def _build_dashboard(self) -> None:
        """Execute build dashboard."""

        build_script = _hpo_dashboard_build_script_path()
        if not build_script.exists():
            logger.warning(f"Dashboard build script not found: {build_script}")
            return

        logger.info("Compilando dashboard HPO...")
        try:
            subprocess.run(
                ["bash", str(build_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.success("Build do dashboard concluído")
        except subprocess.CalledProcessError as exc:
            logger.error(f"Dashboard build failed: {exc.stderr}")
            sys.exit(1)

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
        parser.add_argument(
            "--trials",
            type=int,
            default=None,
            help="Numero de trials (default: config/hpo/optimization.yaml)",
        )
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
        hpo_subparsers = parser.add_subparsers(
            dest="hpo_subcommand",
            help="Subcomandos HPO",
        )
        dashboard_parser = hpo_subparsers.add_parser(
            "dashboard",
            help="Controla o servidor do dashboard HPO.",
        )
        dashboard_parser.add_argument(
            "dashboard_action",
            choices=["on", "off", "status", "restart", "build"],
            help="Ação do servidor (on/off/status/restart/build).",
        )
        dashboard_parser.add_argument(
            "--bind",
            dest="dashboard_bind",
            type=str,
            default=_HPO_DASHBOARD_DEFAULT_BIND,
            help=f"Endereço de bind (padrão: {_HPO_DASHBOARD_DEFAULT_BIND})",
        )
        dashboard_parser.add_argument(
            "--port",
            dest="dashboard_port",
            type=int,
            default=_HPO_DASHBOARD_DEFAULT_PORT,
            help=f"Porta do servidor (padrão: {_HPO_DASHBOARD_DEFAULT_PORT})",
        )
        dashboard_parser.add_argument(
            "--no-healthcheck",
            dest="dashboard_no_healthcheck",
            action="store_true",
            help="Desativa o healthcheck após iniciar.",
        )
        dashboard_parser.add_argument(
            "--healthcheck-timeout",
            dest="dashboard_healthcheck_timeout",
            type=float,
            default=_HPO_DASHBOARD_HEALTHCHECK_TIMEOUT_S,
            help=(
                "Timeout do healthcheck em segundos "
                f"(padrão: {_HPO_DASHBOARD_HEALTHCHECK_TIMEOUT_S})"
            ),
        )


class HpoProxyCommand(Command):
    """Command to run Optuna gRPC storage proxy."""

    def __init__(self, args: argparse.Namespace):
        """Execute init.



        Args:

            args: Input value used by this callable.

        """

        super().__init__(args)
        self.host = getattr(args, "host", None)
        self.port = getattr(args, "port", None)
        self.storage_url = getattr(args, "storage_url", None)

    async def execute(self) -> None:
        """Start the Optuna gRPC proxy server."""
        from pff.infrastructure.hpo.grpc_proxy import run_optuna_grpc_proxy

        logger.info("Iniciando proxy gRPC do Optuna...")
        try:
            run_optuna_grpc_proxy(
                host=self.host,
                port=self.port,
                storage_url=self.storage_url,
            )
        except KeyboardInterrupt:
            logger.warning("gRPC proxy interrupted by user")
            raise SystemExit(128)
        except Exception as exc:
            logger.exception(f"Critical gRPC proxy error: {exc}")
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


async def _run_learn(
    model: str,
    config_path: Path | None = None,
) -> None:
    """Execute LearnUseCase with explicit wiring."""
    from pff.application.learn_use_case import LearnUseCase
    from pff.application.strategy_registry import get_strategy_registry

    use_case = LearnUseCase(
        config_path=config_path,
        strategy_registry=get_strategy_registry(),
    )
    await use_case.execute(model)
