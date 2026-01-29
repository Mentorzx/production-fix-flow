"""CLI command implementations."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from pff import IntelligentPreprocessor, ManifestParser, Orchestrator, settings
from pff.__main__ import AppLauncher
from pff.application.learn_use_case import LearnUseCase
from pff.application.optimize_use_case import OptimizeUseCase
from pff.application.strategy_registry import get_strategy_registry
from pff.infrastructure.hpo.background_process import BackgroundProcess
from pff.infrastructure.hpo.grpc_proxy import run_optuna_grpc_proxy
from pff.infrastructure.hpo.runner import HpoRunner
from pff.infrastructure.persistence.db.connection import close_connection_pool
from pff.shared import logger
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.cache import shutdown_all_cache_janitors
from pff.shared.core.config import OPTIMIZATION_CONFIG_PATH
from pff.shared.core.file_manager import FileManager
from pff.shared.determinism import set_global_seed
from pff.shared.ops.global_interrupt_manager import (
    check_interruption,
    get_interrupt_manager,
    should_stop,
)


def is_vpn_up() -> bool:
    """
    Check if a VPN connection is currently active (DISABLED).

    Returns:
        bool: Always returns False (VPN check disabled)
    """

    return False


def _resolve_hpo_seed(file_manager: FileManager | None = None) -> int | None:
    fm = file_manager or FileManager()
    if not fm.exists(OPTIMIZATION_CONFIG_PATH):
        return None
    try:
        cfg = fm.read(OPTIMIZATION_CONFIG_PATH, return_native=True)
        if cfg is None:
            cfg = {}
    except Exception as exc:
        logger.warning(f"Failed to load HPO config: {exc}")
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
    except Exception as exc:
        logger.warning(f"Invalid sampler.seed: value={seed!r} error={exc}")
        return None


def _cleanup_hpo_resources() -> None:
    try:
        shutdown_all_cache_janitors()
    except Exception as exc:
        logger.debug(f"Failed to shut down cache janitor: {exc}")

    try:
        run_coroutine_sync(close_connection_pool())
    except Exception as exc:
        logger.debug(f"Failed to shut down Postgres pool: {exc}")

    try:
        from numba.core.runtime import nrt

        nrt.rtsys.shutdown()
    except Exception as exc:
        logger.debug(f"Failed to shut down Numba runtime: {exc}")


_HPO_DASHBOARD_DEFAULT_PORT = 8766
_HPO_DASHBOARD_DEFAULT_BIND = "127.0.0.1"
_HPO_DASHBOARD_HEALTHCHECK_TIMEOUT_S = 20.0


def _hpo_dashboard_pid_path() -> Path:
    return settings.CACHE_DIR / "hpo" / "dashboard_server.pid"


def _load_hpo_dashboard_pid(pid_path: Path) -> int | None:
    if not FileManager.exists(pid_path):
        return None
    try:
        raw = FileManager.read_bytes(pid_path).decode("utf-8", errors="ignore").strip()
        return int(raw)
    except (ValueError, OSError):
        return None


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _hpo_dashboard_healthcheck(bind: str, port: int, timeout_s: float = 10.0) -> bool:
    url = f"http://{bind}:{port}/api/status"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
        except Exception:
            time.sleep(0.25)
    return False


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

    def __init__(self, args: argparse.Namespace, launcher: AppLauncher | None = None):
        super().__init__(args)
        self.launcher = launcher

    async def execute(self) -> None:
        """Execute the orchestrator workflow."""
        logger.debug(
            f"component=cli command=run evento=selecionado manifesto={self.args.manifest_file}"
        )

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
            "component=cli command=api status=iniciando_granian "
            f"host={self.args.host} port={self.args.port} reload={self.args.reload}"
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
                logger.info(f"component=cli command=logs evento=lista item={log}")

        elif self.args.subcommand == "stats":
            stats = await log_repository.get_statistics(operation=self.args.operation)
            logger.info(f"component=cli command=logs evento=stats dados={stats}")

        elif self.args.subcommand == "metrics":
            metrics = await metrics_repository.get_metrics(
                execution_log_id=self.args.log_id,
                model_name=self.args.model,
            )

            for metric in metrics:
                logger.info(f"component=cli command=logs evento=metricas item={metric}")

        elif self.args.subcommand == "cleanup":
            deleted = await log_repository.delete_old_logs(
                older_than_days=self.args.days
            )
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
        super().__init__(args)
        self.model = getattr(args, "model", "all")
        self.config_path = getattr(args, "config", None)

    async def execute(self) -> None:
        """Execute training based on model type (Strategy Pattern)."""
        logger.info(
            "component=cli command=learn status=iniciando info=global_interrupt_manager_ativo"
        )

        def learn_interrupt_callback():
            logger.info("component=cli command=learn evento=interrompendo")

        self.interrupt_manager.register_callback_once(
            learn_interrupt_callback, label="learn_cli_interrupt"
        )
        try:
            await _run_learn(self.model, config_path=self.config_path)
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
        self.subcommand = getattr(args, "hpo_subcommand", None)
        self.model = getattr(args, "model", "dslfm-kgc")
        self.trials = int(getattr(args, "trials", 50))
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

        logger.info(
            "component_name=cli key_parameters={'command': 'hpo'} message='Iniciando workflow HPO (interrupt manager ativo)'"
        )

        def hpo_interrupt_callback():
            logger.info("component=cli command=hpo evento=interrompendo")

        self.interrupt_manager.register_callback_once(
            hpo_interrupt_callback, label="hpo_cli_interrupt"
        )

        seed = _resolve_hpo_seed()
        if seed is not None:
            set_global_seed(seed)

        study_name = self.study_name or f"pff_kg_real_{self.model.replace('-', '_')}"

        logger.info(
            f"component=cli command=hpo evento=iniciado modelo={self.model.upper()} "
            f"trials={self.trials} fonte={settings.DATA_DIR / 'models' / 'kg'}"
        )

        runner = HpoRunner()
        use_case = OptimizeUseCase(runner)

        build_script = _hpo_dashboard_build_script_path()
        if build_script.exists():
            try:
                logger.info(
                    "component=cli command=hpo evento=build_dashboard status=iniciando"
                )
                subprocess.run(
                    ["bash", str(build_script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.success(
                    "component=cli command=hpo evento=build_dashboard status=sucesso"
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"component=cli command=hpo evento=build_dashboard status=falha erro={e.stderr}"
                )

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
            try:
                dashboard_url = "http://127.0.0.1:8766/api/status"
                logger.info(
                    f"component=cli command=hpo evento=dashboard_healthcheck status=iniciando url={dashboard_url}"
                )

                t0 = time.time()
                dashboard_ready = False
                while time.time() - t0 < 20:
                    try:
                        with urllib.request.urlopen(dashboard_url, timeout=1) as resp:
                            if resp.status == 200:
                                dashboard_ready = True
                                break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.25)
                    except Exception:
                        time.sleep(0.25)

                if dashboard_ready:
                    logger.success(
                        f"component=cli command=hpo evento=dashboard_healthcheck status=sucesso url={dashboard_url}"
                    )
                else:
                    logger.warning(
                        f"Dashboard health check failed: url={dashboard_url} timeout_s=20"
                    )

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
            except Exception as exc:
                logger.exception(
                    f"component=cli command=hpo stop_reason=erro_critico erro={exc}"
                )
                sys.exit(1)
            finally:
                _cleanup_hpo_resources()

        logger.success(
            f"component=cli command=hpo evento=concluido trials={result.get('n_trials', 0)} "
            f"tempo_s={result.get('optimization_time', 0):.1f}"
        )

        if "real_data_info" in result:
            info = result["real_data_info"]
            logger.info(
                f"component_name=cli message='Dados reais carregados: n_train={info.get('n_train', 'N/A')} "
                f"n_valid={info.get('n_valid', 'N/A')} n_entities={info.get('n_entities', 'N/A')}'"
            )

        mo = result.get("multi_objective", {}) or {}
        best_tradeoff = mo.get("best_tradeoff") or {}
        best_time = mo.get("best_time_aware") or {}
        best_quality = mo.get("best_quality") or {}

        if best_tradeoff:
            logger.info(
                "component=cli command=hpo evento=melhor_tradeoff "
                f"score_time={best_tradeoff.get('score_time', 0.0):.4f} "
                f"tradeoff_score={best_tradeoff.get('tradeoff_score', 0.0):.4f} "
                f"trial={best_tradeoff.get('trial_number', 'N/A')} "
                f"duracao_s={best_tradeoff.get('duration', 0.0):.1f}"
            )
        elif result.get("best_value") is not None:
            logger.info(
                f"component=cli command=hpo evento=melhor_score score={result['best_value']:.4f}"
            )
        else:
            logger.warning("component=cli command=hpo evento=melhor_score_ausente")

        if best_time:
            logger.info(
                "component=cli command=hpo evento=campeao_tempoaware "
                f"trial={best_time.get('trial_number', 'N/A')} "
                f"score={best_time.get('score_time', 0.0):.4f} "
                f"duracao_s={best_time.get('duration', 0.0):.1f}"
            )
        if best_quality:
            logger.info(
                "component=cli command=hpo evento=campeao_sem_tempo "
                f"trial={best_quality.get('trial_number', 'N/A')} "
                f"score={best_quality.get('score_quality', 0.0):.4f}"
            )

        if self.no_update_config:
            logger.info(
                "component=cli command=hpo evento=auto_update status=desabilitado"
            )

        dashboard_url = os.getenv(
            "OPTUNA_DASHBOARD_URL", "http://localhost:8080/dashboard"
        )
        if result.get("live_dashboard"):
            logger.info(
                f"dashboard_optuna url={dashboard_url} html={result.get('live_dashboard')}"
            )

    def _execute_dashboard_action(self) -> None:
        action = self.dashboard_action or "status"
        key_params = {
            "action": action,
            "bind": self.dashboard_bind,
            "port": self.dashboard_port,
        }
        logger.info(
            "component_name=cli_hpo_dashboard "
            f"key_parameters={key_params} stop_reason=none "
            "message='Comando de dashboard (HPO) acionado'"
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

        logger.error(
            "component_name=cli_hpo_dashboard "
            f"key_parameters={key_params} stop_reason=invalid_action "
            f"message='Invalid dashboard action: {action}'"
        )
        sys.exit(2)

    def _dashboard_status(self) -> None:
        pid_path = _hpo_dashboard_pid_path()
        pid = _load_hpo_dashboard_pid(pid_path)
        if pid is None:
            logger.info(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'bind': '{self.dashboard_bind}', 'port': {self.dashboard_port}}} "
                "stop_reason=none message='Dashboard parece desligado (PID ausente)'"
            )
            return
        if not _is_pid_running(pid):
            logger.info(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'pid': {pid}}} stop_reason=none "
                "message='Dashboard parece desligado (PID stale)'"
            )
            return
        logger.success(
            "component_name=cli_hpo_dashboard "
            f"key_parameters={{'pid': {pid}, 'bind': '{self.dashboard_bind}', "
            f"'port': {self.dashboard_port}}} stop_reason=none message='Dashboard ativo'"
        )

    def _start_or_restart_dashboard(self) -> None:
        pid_path = _hpo_dashboard_pid_path()
        pid = _load_hpo_dashboard_pid(pid_path)
        if pid is not None and _is_pid_running(pid):
            logger.info(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'pid': {pid}}} stop_reason=none "
                "message='Dashboard já estava ativo; reiniciando'"
            )
            self._stop_dashboard()
        self._start_dashboard()

    def _start_dashboard(self) -> None:
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
            logger.error(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'bind': '{self.dashboard_bind}', 'port': {self.dashboard_port}}} "
                f"stop_reason=spawn_failed message='Failed to start dashboard server: {exc}'"
            )
            sys.exit(1)

        FileManager.write_text(str(proc.pid), pid_path)
        logger.success(
            "component_name=cli_hpo_dashboard "
            f"key_parameters={{'pid': {proc.pid}, 'bind': '{self.dashboard_bind}', "
            f"'port': {self.dashboard_port}}} stop_reason=none message='Dashboard iniciado'"
        )

        if self.dashboard_no_healthcheck:
            return

        if _hpo_dashboard_healthcheck(
            self.dashboard_bind,
            self.dashboard_port,
            timeout_s=self.dashboard_healthcheck_timeout,
        ):
            logger.success(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'bind': '{self.dashboard_bind}', 'port': {self.dashboard_port}}} "
                "stop_reason=none message='Dashboard saudável'"
            )
        else:
            logger.warning(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'bind': '{self.dashboard_bind}', 'port': {self.dashboard_port}, "
                f"'timeout_s': {self.dashboard_healthcheck_timeout}}} "
                "stop_reason=healthcheck_timeout message='Dashboard healthcheck failed'"
            )

    def _stop_dashboard(self) -> None:
        pid_path = _hpo_dashboard_pid_path()
        pid = _load_hpo_dashboard_pid(pid_path)
        if pid is None:
            logger.info(
                "component_name=cli_hpo_dashboard key_parameters={} stop_reason=none "
                "message='Dashboard já está desligado'"
            )
            return
        if not _is_pid_running(pid):
            if FileManager.exists(pid_path):
                pid_path.unlink(missing_ok=True)
            logger.info(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'pid': {pid}}} stop_reason=none "
                "message='PID não está ativo; dashboard parece desligado'"
            )
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            logger.error(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={{'pid': {pid}}} stop_reason=terminate_failed "
                f"message='Failed to terminate dashboard process: {exc}'"
            )
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
                logger.error(
                    "component_name=cli_hpo_dashboard "
                    f"key_parameters={{'pid': {pid}}} stop_reason=kill_failed "
                    f"message='Failed to kill dashboard process: {exc}'"
                )
                return

        pid_path.unlink(missing_ok=True)
        logger.success(
            "component_name=cli_hpo_dashboard "
            f"key_parameters={{'pid': {pid}}} stop_reason=none "
            "message='Dashboard desligado'"
        )

    def _build_dashboard(self) -> None:
        build_script = _hpo_dashboard_build_script_path()
        key_params = {"script": str(build_script)}
        if not build_script.exists():
            logger.warning(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={key_params} stop_reason=script_missing "
                f"message='Dashboard build script not found: path={build_script}'"
            )
            return

        logger.info(
            "component_name=cli_hpo_dashboard "
            f"key_parameters={key_params} stop_reason=none "
            "message='Iniciando build do dashboard'"
        )
        try:
            subprocess.run(
                ["bash", str(build_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.success(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={key_params} stop_reason=none "
                "message='Build do dashboard concluido'"
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "component_name=cli_hpo_dashboard "
                f"key_parameters={key_params} stop_reason=build_failed "
                f"message='Dashboard build failed: {exc.stderr}'"
            )
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
        except Exception as exc:
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


async def _run_learn(
    model: str,
    config_path: Path | None = None,
) -> None:
    """Execute LearnUseCase with explicit wiring."""
    use_case = LearnUseCase(
        config_path=config_path,
        strategy_registry=get_strategy_registry(),
    )
    await use_case.execute(model)
