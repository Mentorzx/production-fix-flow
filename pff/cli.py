"""
PFF CLI - Production Fix Flow Command Line Interface

Refactored with Design Patterns:
- Command Pattern: Each CLI command is a Command class
- Factory Pattern: CommandFactory creates commands
- Strategy Pattern: Different execution strategies (learn models, cleanup types)
- Template Method: Base command class with common behavior
- Dependency Injection: Commands receive dependencies
- Single Responsibility: Each class has one clear purpose
"""

import argparse
import asyncio
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type

from pff import (
    IntelligentPreprocessor,
    ManifestParser,
    Orchestrator,
    __version__,
    settings,
)
from pff.__main__ import AppLauncher
from pff.utils import logger
from pff.utils.global_interrupt_manager import (
    check_interruption,
    get_interrupt_manager,
    should_stop,
)
from pff.validators.ensembles.advanced_trainer import run_standalone_ensemble_pipeline

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

    # Original implementation kept for reference:
    # for name, addrs in psutil.net_if_addrs().items():
    #     ipv4_addrs = [addr for addr in addrs if addr.family == socket.AF_INET]
    #     if not ipv4_addrs:
    #         continue
    #     if _VPN_PATTERN.search(name):
    #         logger.debug(f"VPN detectada por padrão: {name}")
    #         return True
    #     for addr in ipv4_addrs:
    #         ip = addr.address
    #         if ip.startswith("172.") and "ethernet" in name.lower():
    #             logger.info(f"🔗 FortiClient VPN detectada: {name} ({ip})")
    #             return True
    #         if (
    #             ip.startswith("10.")
    #             and not ip.startswith("10.0.0.")
    #             and "ethernet" in name.lower()
    #         ):
    #             logger.info(f"🔗 VPN corporativa detectada: {name} ({ip})")
    #             return True
    # return False


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

    def __init__(self, args: argparse.Namespace, launcher: Optional[AppLauncher] = None):
        super().__init__(args)
        self.launcher = launcher

    async def execute(self) -> None:
        """Execute the orchestrator workflow."""
        logger.debug(f"Comando 'run' selecionado. Usando manifesto: {self.args.manifest_file}")

        # VPN check disabled
        # if not is_vpn_up():
        #     logger.critical("Nenhuma interface VPN detectada – conecte-se à VPN antes de continuar.")
        #     sys.exit(1)

        try:
            await self._run_orchestrator()
            logger.info("Execução do pipeline concluída com sucesso.")
        except FileNotFoundError:
            logger.error(f"Arquivo de manifesto não encontrado em: {self.args.manifest_file}")
            logger.warning("Você talvez precise gerar o manifesto primeiro com o comando 'generate'.")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Uma falha crítica impediu a execução. Erro: {e}")
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

    Pattern: Command Pattern + Strategy Pattern (clipboard vs file)
    """

    def execute_sync(self) -> None:
        """Generate manifest from input source."""
        logger.info("Comando 'generate' selecionado.")
        output_path = settings.DEFAULT_MANIFEST_PATH
        preprocessor = IntelligentPreprocessor()

        if self.args.from_clipboard:
            self._process_from_clipboard(preprocessor, output_path)
        elif self.args.from_file:
            self._process_from_file(preprocessor, output_path)

    def _process_from_clipboard(self, preprocessor: IntelligentPreprocessor, output_path: Path) -> None:
        """Strategy: Process from clipboard."""
        preprocessor.process_from_clipboard(
            default_sequence=self.args.sequence,
            output_path=output_path.name
        )

    def _process_from_file(self, preprocessor: IntelligentPreprocessor, output_path: Path) -> None:
        """Strategy: Process from file."""
        try:
            raw_text = self.args.from_file.read_text(encoding="utf-8")
            tasks = preprocessor.parse_text(raw_text, default_sequence=self.args.sequence)
            preprocessor.generate_manifest_file(tasks, output_path=output_path.name)
        except FileNotFoundError:
            logger.error(f"Arquivo de entrada não encontrado em: {self.args.from_file}")
            sys.exit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'generate' command parser."""
        parser = subparsers.add_parser("generate", help="Gera o manifesto padrão a partir de texto bruto.")

        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument("--from-clipboard", action="store_true", help="Lê da área de transferência.")
        input_group.add_argument("--from-file", type=Path, help="Lê de um arquivo de texto.")

        parser.add_argument("--sequence", type=str, help="Sequência padrão para linhas com apenas MSISDN.")


class WorkerCommand(SyncCommand):
    """
    Command to start Celery worker.

    Pattern: Command Pattern
    """

    def execute_sync(self) -> None:
        """Start Celery worker."""
        logger.info("Iniciando modo Worker (Celery)...")

        try:
            from pff.celery_app import celery_app

            worker_args = ["worker", f"--queues={self.args.queues}", "--loglevel=info"]
            if self.args.concurrency:
                worker_args.append(f"--concurrency={self.args.concurrency}")

            logger.info(f"Iniciando worker com args: {' '.join(worker_args)}")
            celery_app.worker_main(worker_args)
        except ImportError as e:
            logger.error(f"Celery não disponível: {e}")
            sys.exit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'worker' command parser."""
        parser = subparsers.add_parser("worker", help="Inicia um worker Celery.")
        parser.add_argument("-c", "--concurrency", type=int, help="Número de processos do worker.")
        parser.add_argument(
            "-Q",
            "--queues",
            type=str,
            default="default,high,low",
            help="Filas a serem consumidas.",
        )


class APICommand(Command):
    """
    Command to start FastAPI server.

    Pattern: Command Pattern
    """

    async def execute(self) -> None:
        """Start FastAPI server."""
        logger.info("Iniciando modo API (FastAPI)...")

        import redis

        try:
            import uvicorn
            from pff.api.main import app

            host = getattr(self.args, "host", "0.0.0.0")
            port = getattr(self.args, "port", 8000)

            app.state.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=5
            )

            logger.info(f"Servidor API iniciando em http://{host}:{port}")
            logger.info(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")

            config = uvicorn.Config(app, host=host, port=port, reload=self.args.reload)
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError as e:
            logger.exception(f"FastAPI/Uvicorn não disponível: {e}")
            sys.exit(1)
        except redis.ConnectionError:
            logger.error(f"Redis não está rodando em {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            logger.info("Instale e inicie o Redis: sudo apt install redis-server && sudo service redis-server start")
            sys.exit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'api' command parser."""
        parser = subparsers.add_parser("api", help="Inicia o servidor da API.")
        parser.add_argument("--host", default="0.0.0.0", help="Host do servidor.")
        parser.add_argument("-p", "--port", type=int, default=8000, help="Porta do servidor.")
        parser.add_argument("--reload", action="store_true", help="Ativa auto-reload para desenvolvimento.")


class CleanCommand(Command):
    """
    Command to clean cache, logs, and temporary files.

    Pattern: Command Pattern + Strategy Pattern (cleanup strategies)
    """

    async def execute(self) -> None:
        """Execute cleanup with specified strategy."""
        logger.info(f"Comando 'clean' selecionado (estratégia: {self.args.strategy}).")

        from importlib import import_module

        build_engine = import_module("pff.utils.cleanup").build_engine
        engine = build_engine(
            self.args.strategy,
            auto_yes=self.args.yes,
            dry_run=self.args.dry_run
        )

        try:
            await engine.run()
        except Exception as exc:
            logger.exception(f"Falha durante a limpeza: {exc}")
            sys.exit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'clean' command parser."""
        parser = subparsers.add_parser(
            "clean",
            help="Limpa caches, logs, outputs e artefatos temporários.",
            description="""
        Estratégias de limpeza disponíveis:

        • standard: Limpeza básica (cache, logs, outputs básicos)
        • deep: Limpeza agressiva (inclui artefatos de desenvolvimento + ML completo)
        • ml: Limpeza focada em ML/TransE (checkpoints, experimentos, cache de modelos)
        • shutdown: Limpeza seletiva para shutdown graceful

        A estratégia 'deep' agora inclui limpeza completa de ML/TransE!
        """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "strategy",
            choices=["standard", "deep", "ml", "shutdown"],
            nargs="?",
            default="standard",
            help="Estratégia de limpeza a ser utilizada.",
        )
        parser.add_argument("-y", "--yes", action="store_true", help="Não pedir confirmação.")
        parser.add_argument("--dry-run", action="store_true", help="Simular execução sem deletar.")


class ResetMLCommand(Command):
    """
    Command to reset ML/TransE environment (alias for clean ml).

    Pattern: Command Pattern
    """

    async def execute(self) -> None:
        """Reset ML environment."""
        logger.info("🧹 Resetando ambiente de ML/TransE...")

        from importlib import import_module

        build_engine = import_module("pff.utils.cleanup").build_engine
        engine = build_engine("ml", auto_yes=self.args.yes, dry_run=self.args.dry_run)

        try:
            await engine.run()
            logger.success("✅ Ambiente de ML resetado com sucesso!")
            logger.info("💡 Agora você pode treinar do zero com: pff learn transe")
        except Exception as exc:
            logger.exception(f"❌ Falha durante o reset: {exc}")
            sys.exit(1)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'reset-ml' command parser."""
        parser = subparsers.add_parser(
            "reset-ml",
            help="Reseta completamente o ambiente de ML/TransE",
            description="""
            Remove todos os artefatos de treinamento:
            • Checkpoints do TransE (.pt, .pth)
            • Experimentos MLflow (mlruns/)
            • Cache de modelos PyTorch/HuggingFace
            • Artefatos temporários de treinamento
            • Bancos de dados Optuna
            • Outputs do TransE (outputs/transe/)

            Use este comando quando quiser começar o treinamento do zero.
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("-y", "--yes", action="store_true", help="Não pedir confirmação.")
        parser.add_argument("--dry-run", action="store_true", help="Simular execução sem deletar.")


class LogsCommand(Command):
    """
    Command to view execution logs and training metrics.

    Pattern: Command Pattern + Query Pattern
    """

    async def execute(self) -> None:
        """Display execution logs and metrics."""
        from pff.db.repositories.execution_logs import ExecutionLogsRepository
        from pff.db.repositories.training_metrics import TrainingMetricsRepository
        from datetime import datetime, timedelta

        repo = ExecutionLogsRepository()
        metrics_repo = TrainingMetricsRepository()

        if self.args.subcommand == "list":
            # List execution logs
            since = None
            if self.args.last_hours:
                since = datetime.now() - timedelta(hours=self.args.last_hours)

            logs = await repo.get_logs(
                operation=self.args.operation,
                status=self.args.status,
                since=since,
                limit=self.args.limit
            )

            if not logs:
                logger.info("📭 Nenhum log encontrado")
                return

            logger.info(f"\n📋 {len(logs)} logs encontrados:\n")
            for log in logs:
                status_icon = "✅" if log['status'] == 'success' else "❌" if log['status'] == 'failed' else "⏳"
                duration = f"{log['duration_seconds']:.1f}s" if log['duration_seconds'] else "N/A"
                logger.info(
                    f"{status_icon} [{log['id']}] {log['operation']}: {log['status']} "
                    f"({duration}) - {log['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                if log['error_message']:
                    logger.info(f"   Error: {log['error_message']}")

        elif self.args.subcommand == "stats":
            # Show statistics
            stats = await repo.get_statistics(operation=self.args.operation)

            logger.info("\n📊 Estatísticas de Execução:\n")
            logger.info(f"Total de execuções: {stats['total_executions']}")
            logger.info(f"Sucessos: {stats['successful']} ({stats['success_rate']*100:.1f}%)")
            logger.info(f"Falhas: {stats['failed']}")
            logger.info(f"Em execução: {stats['running']}")

            if stats['avg_duration'] > 0:
                logger.info(f"\nDuração média: {stats['avg_duration']:.1f}s")
                logger.info(f"Min: {stats['min_duration']:.1f}s | Max: {stats['max_duration']:.1f}s")

            if stats['by_operation']:
                logger.info("\n📈 Por operação:")
                for op in stats['by_operation']:
                    logger.info(
                        f"  • {op['operation']}: {op['count']} execuções "
                        f"(média: {op['avg_duration']:.1f}s)"
                    )

        elif self.args.subcommand == "metrics":
            # Show training metrics
            if self.args.log_id:
                # Show metrics for specific execution
                metrics = await metrics_repo.get_metrics(
                    execution_log_id=self.args.log_id,
                    limit=1000
                )

                if not metrics:
                    logger.info(f"📭 Nenhuma métrica encontrada para log_id={self.args.log_id}")
                    return

                logger.info(f"\n📊 Métricas para execution_log_id={self.args.log_id}:\n")

                # Group by epoch
                by_epoch: dict[int | str, list] = {}
                for m in metrics:
                    epoch = m['epoch'] if m['epoch'] is not None else 'final'
                    if epoch not in by_epoch:
                        by_epoch[epoch] = []
                    by_epoch[epoch].append(m)

                for epoch in sorted(by_epoch.keys(), key=lambda x: x if isinstance(x, int) else 999999):
                    logger.info(f"Época {epoch}:")
                    for m in by_epoch[epoch]:
                        split_label = f"[{m['split']}]" if m['split'] else ""
                        logger.info(
                            f"  • {m['metric_name']}{split_label}: {m['metric_value']:.4f}"
                        )

            else:
                # Show summary statistics
                summary = await metrics_repo.get_summary_statistics(model_name=self.args.model)

                logger.info("\n📊 Resumo de Métricas de Treinamento:\n")
                logger.info(f"Total de métricas: {summary['total_metrics']}")
                logger.info(f"Total de modelos: {summary['total_models']}")
                logger.info(f"Total de épocas: {summary['total_epochs']}")

                if summary['by_model']:
                    logger.info("\n📈 Por modelo:")
                    for model in summary['by_model']:
                        logger.info(
                            f"  • {model['model_name']}: {model['metric_count']} métricas, "
                            f"{model['epoch_count']} épocas"
                        )

        elif self.args.subcommand == "cleanup":
            # Cleanup old logs
            days = self.args.days or 30
            logger.info(f"🗑️  Deletando logs mais antigos que {days} dias...")

            deleted = await repo.delete_old_logs(older_than_days=days)

            logger.success(f"✅ {deleted} logs deletados")

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'logs' command parser."""
        parser = subparsers.add_parser(
            "logs",
            help="Visualizar e gerenciar logs de execução e métricas de treinamento",
            description="Comandos para visualizar execution logs e training metrics do PostgreSQL",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        logs_subparsers = parser.add_subparsers(dest="subcommand", required=True)

        # pff logs list
        list_parser = logs_subparsers.add_parser(
            "list",
            help="Listar logs de execução"
        )
        list_parser.add_argument("--operation", type=str, help="Filtrar por operação")
        list_parser.add_argument("--status", type=str, choices=["running", "success", "failed"], help="Filtrar por status")
        list_parser.add_argument("--last-hours", type=int, help="Mostrar logs das últimas N horas")
        list_parser.add_argument("--limit", type=int, default=50, help="Número máximo de logs (padrão: 50)")

        # pff logs stats
        stats_parser = logs_subparsers.add_parser(
            "stats",
            help="Estatísticas de execução"
        )
        stats_parser.add_argument("--operation", type=str, help="Filtrar por operação")

        # pff logs metrics
        metrics_parser = logs_subparsers.add_parser(
            "metrics",
            help="Visualizar métricas de treinamento"
        )
        metrics_parser.add_argument("--log-id", type=int, help="ID do execution log")
        metrics_parser.add_argument("--model", type=str, help="Filtrar por modelo (transe, lightgbm, ensemble)")

        # pff logs cleanup
        cleanup_parser = logs_subparsers.add_parser(
            "cleanup",
            help="Deletar logs antigos"
        )
        cleanup_parser.add_argument("--days", type=int, default=30, help="Deletar logs mais antigos que N dias (padrão: 30)")


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
        logger.info("🛡️ GlobalInterruptManager ativo - CTRL+C irá parar toda a pipeline")

        # Register interrupt callback
        def learn_interrupt_callback():
            logger.info("🛑 Learn command será interrompido...")

        self.interrupt_manager.register_callback(learn_interrupt_callback)

        try:
            # Strategy Pattern: Select training strategy based on model type
            strategy = self._get_training_strategy()
            await strategy.execute()
        except KeyboardInterrupt:
            logger.warning("🛑 Pipeline interrompida pelo usuário (CTRL+C)")
            logger.info("🧹 Executando limpeza graceful...")
            await asyncio.sleep(0.5)
            logger.success("✅ Interrupção tratada com sucesso")
            sys.exit(128)
        except Exception as e:
            logger.exception(f"Erro crítico durante o processo de treinamento: {e}")
            sys.exit(1)
        finally:
            if should_stop():
                logger.info("🧹 Limpeza final do GlobalInterruptManager")

    def _get_training_strategy(self) -> "TrainingStrategy":
        """
        Factory method to get training strategy.

        Pattern: Factory Method + Strategy Pattern
        """
        if self.config_path is None:
            self.config_path = settings.CONFIG_DIR / "kg.yaml"

        strategy_map = {
            "kg": KGTrainingStrategy,
            "transe": TransETrainingStrategy,
            "ensemble": EnsembleTrainingStrategy,
            "all": FullPipelineStrategy,
            "both": FullPipelineStrategy,
            "": FullPipelineStrategy,
        }

        strategy_class = strategy_map.get(self.model.lower())

        if not strategy_class:
            logger.error(f"Modelo desconhecido: {self.model}. Use 'kg', 'transe' ou 'all'.")
            sys.exit(1)

        return strategy_class(self.config_path)

    @staticmethod
    def configure_parser(subparsers: argparse._SubParsersAction) -> None:
        """Configure 'learn' command parser."""
        parser = subparsers.add_parser("learn", help="Treinar modelos de IA")
        parser.add_argument(
            "model",
            nargs="?",
            default="all",
            choices=["kg", "transe", "ensemble", "all"],
            help="Modelo a treinar",
        )
        parser.add_argument("-c", "--config", type=Path, help="Arquivo de configuração")


# ============================================================================
# STRATEGY PATTERN - Training Strategies
# ============================================================================


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.

    Pattern: Strategy Pattern + Template Method
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path

    @abstractmethod
    async def execute(self) -> None:
        """Execute training strategy."""
        pass

    def check_interruption(self) -> None:
        """Check if user requested interruption."""
        check_interruption()


class KGTrainingStrategy(TrainingStrategy):
    """Strategy for Knowledge Graph training."""

    async def execute(self) -> None:
        """Train KG model."""
        from pff.validators.kg.config import KGConfig
        from pff.validators.kg.pipeline import KGPipeline

        logger.info("🧠 Executando pipeline do Knowledge Graph (KG)...")

        self.check_interruption()
        kg_pipeline = KGPipeline(KGConfig(self.config_path))

        await kg_pipeline.run_build_and_preprocess()
        self.check_interruption()

        await kg_pipeline.run_learn_rules()
        self.check_interruption()

        await kg_pipeline.run_ranking()
        logger.success("✅ Pipeline do KG concluída.")


class TransETrainingStrategy(TrainingStrategy):
    """Strategy for TransE training."""

    async def execute(self) -> None:
        """Train TransE model."""
        from pff.validators.transe.transe_pipeline import TransEPipeline

        logger.info("🤖 Executando pipeline do TransE (autossuficiente)...")

        self.check_interruption()
        transe_pipeline = TransEPipeline(self.config_path)

        await transe_pipeline.train_transe()
        self.check_interruption()

        transe_pipeline.rank_and_evaluate_transe()
        logger.success("✅ Pipeline do TransE concluída.")


class EnsembleTrainingStrategy(TrainingStrategy):
    """Strategy for Ensemble training."""

    async def execute(self) -> None:
        """Train Ensemble model."""
        logger.info("✨ Executando pipeline de Ensemble...")

        # Check if TransE dependencies exist
        required_files = [
            settings.OUTPUTS_DIR / "transe" / "transe_entity_map.parquet",
            settings.OUTPUTS_DIR / "transe" / "transe_relation_map.parquet",
            settings.OUTPUTS_DIR / "transe" / "node_embeddings.pkl",
        ]

        missing = [f for f in required_files if not f.exists()]

        if missing:
            logger.error("❌ Ensemble requer que o TransE seja treinado primeiro!")
            logger.error(f"Arquivos faltando: {[f.name for f in missing]}")
            logger.info("💡 Execute: pff learn transe")
            logger.info("💡 Ou execute: pff learn all  (pipeline completa)")
            sys.exit(1)

        await run_standalone_ensemble_pipeline()
        logger.success("✅ Pipeline do Ensemble concluída.")


class FullPipelineStrategy(TrainingStrategy):
    """
    Strategy for full pipeline (KG + TransE + Ensemble + Autofeeding).

    Pattern: Composite Strategy (combines multiple strategies)
    """

    async def execute(self) -> None:
        """Execute full training pipeline with autofeeding."""
        from pff.utils.autofeeding import apply_autofeeding_rules
        from pff.validators.kg.config import KGConfig
        from pff.validators.kg.pipeline import KGPipeline
        from pff.validators.transe.transe_pipeline import TransEPipeline

        logger.info("🚀 Executando pipeline completa com autofeeding")

        # Step 1/4: KG Pipeline
        logger.info("🧠 1/4: Executando pipeline do Knowledge Graph...")
        self.check_interruption()

        kg_pipeline = KGPipeline(KGConfig(self.config_path))
        await kg_pipeline.run_build_and_preprocess()
        self.check_interruption()

        await kg_pipeline.run_learn_rules()
        self.check_interruption()

        # await kg_pipeline.run_ranking()  # Commented out

        # Step 2/4: TransE Pipeline
        logger.info("🤖 2/4: Executando pipeline do TransE...")
        transe_pipeline = TransEPipeline(self.config_path)

        await transe_pipeline.train_transe()
        self.check_interruption()

        transe_pipeline.rank_and_evaluate_transe()
        self.check_interruption()

        # Step 3/4: Ensemble
        logger.info("✨ 3/4: Executando Ensemble...")
        await run_standalone_ensemble_pipeline()
        self.check_interruption()

        # Step 4/4: Autofeeding
        logger.info("🤖 4/4: Aplicando autofeeding...")
        await apply_autofeeding_rules()

        logger.success("✅ Pipeline completo com autofeeding concluído!")


# ============================================================================
# FACTORY PATTERN - Command Factory
# ============================================================================


class CommandFactory:
    """
    Factory class for creating Command instances.

    Pattern: Factory Pattern + Registry Pattern
    """

    _command_registry: dict[str, Type[Command]] = {
        "run": RunCommand,
        "generate": GenerateCommand,
        "worker": WorkerCommand,
        "api": APICommand,
        "clean": CleanCommand,
        "reset-ml": ResetMLCommand,
        "logs": LogsCommand,
        "learn": LearnCommand,
    }

    @classmethod
    def create(cls, command_name: str, args: argparse.Namespace, **kwargs) -> Command:
        """
        Create a command instance based on command name.

        Args:
            command_name: Name of the command
            args: Parsed arguments
            **kwargs: Additional keyword arguments (e.g., launcher)

        Returns:
            Command instance

        Raises:
            ValueError: If command name is not registered
        """
        command_class = cls._command_registry.get(command_name)

        if not command_class:
            raise ValueError(f"Unknown command: {command_name}")

        # Special case for RunCommand (needs launcher)
        if command_name == "run" and "launcher" in kwargs:
            return command_class(args, launcher=kwargs["launcher"])  # type: ignore[call-arg]

        return command_class(args)

    @classmethod
    def register(cls, command_name: str, command_class: Type[Command]) -> None:
        """
        Register a new command class.

        Args:
            command_name: Name of the command
            command_class: Command class to register
        """
        cls._command_registry[command_name] = command_class

    @classmethod
    def get_all_commands(cls) -> list[str]:
        """Get list of all registered command names."""
        return list(cls._command_registry.keys())


# ============================================================================
# BUILDER PATTERN - Argument Parser Builder
# ============================================================================


class CLIParserBuilder:
    """
    Builder class for constructing the CLI argument parser.

    Pattern: Builder Pattern
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="pff",
            description=f"PFF - Production Fix Flow v{__version__}",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.subparsers = None

    def add_version(self) -> "CLIParserBuilder":
        """Add version argument."""
        self.parser.add_argument(
            "--version",
            action="version",
            version=f"PFF v{__version__}"
        )
        return self

    def create_subparsers(self) -> "CLIParserBuilder":
        """Create subparsers for commands."""
        self.subparsers = self.parser.add_subparsers(
            dest="command",
            help="Comandos principais"
        )
        self.subparsers.required = True
        return self

    def add_commands(self) -> "CLIParserBuilder":
        """Add all registered commands."""
        if not self.subparsers:
            raise RuntimeError("Must call create_subparsers() first")

        # Configure all registered commands
        for command_name in CommandFactory.get_all_commands():
            command_class = CommandFactory._command_registry[command_name]
            command_class.configure_parser(self.subparsers)

        return self

    def build(self) -> argparse.ArgumentParser:
        """Build and return the parser."""
        return self.parser


# ============================================================================
# MAIN ENTRY POINT - CLI Runner
# ============================================================================


class CLIRunner:
    """
    Main CLI runner class.

    Pattern: Facade Pattern (simplifies complex subsystems)
    """

    def __init__(self, launcher: Optional[AppLauncher] = None):
        self.launcher = launcher
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser using builder."""
        builder = CLIParserBuilder()
        return (
            builder
            .add_version()
            .create_subparsers()
            .add_commands()
            .build()
        )

    async def run(self, argv: Optional[list[str]] = None) -> None:
        """
        Run CLI application.

        Args:
            argv: Optional list of arguments (defaults to sys.argv)
        """
        args = self.parser.parse_args(argv)

        try:
            # Create command using factory
            command = CommandFactory.create(
                args.command,
                args,
                launcher=self.launcher
            )

            # Execute command
            await command.execute()

        except KeyboardInterrupt:
            logger.warning("🛑 Aplicação interrompida pelo usuário")
            sys.exit(128)
        except Exception as e:
            logger.exception(f"Erro crítico na aplicação: {e}")
            sys.exit(1)


# ============================================================================
# PUBLIC API
# ============================================================================


async def main(launcher: Optional[AppLauncher] = None, argv: Optional[list[str]] = None):
    """
    Main entry point for the PFF CLI application.

    Args:
        launcher: Optional AppLauncher instance
        argv: Optional list of command-line arguments
    """
    runner = CLIRunner(launcher)
    await runner.run(argv)


def cli_entrypoint():
    """Entry point for poetry script."""
    asyncio.run(main(launcher=None))


if __name__ == "__main__":
    cli_entrypoint()
