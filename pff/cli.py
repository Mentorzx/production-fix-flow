import argparse
import asyncio
import re
import socket
import sys
from pathlib import Path

import psutil

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

_VPN_PATTERN = re.compile(
    r"(tun|tap|ppp|wg|vpn|utun|fortissl|fortinet|forti|ssl|remote|cisco|anyconnect)",
    re.I,
)
interrupt_manager = get_interrupt_manager()


def is_vpn_up() -> bool:
    """
    Check if a VPN connection is currently active on the system.
    This function examines all network interfaces to determine if a VPN is up by checking:
    - Interface names matching a VPN pattern (via _VPN_PATTERN)
    - IPv4 addresses starting with '172.' on ethernet interfaces (FortiClient VPN)
    - IPv4 addresses starting with '10.' (excluding '10.0.0.') on ethernet interfaces (Corporate VPN)
    Returns:
        bool: True if a VPN connection is detected, False otherwise.
    Note:
        - Requires psutil library for network interface information
        - Uses socket.AF_INET to filter IPv4 addresses
        - Logs debug/info messages when VPN connections are detected
    """

    for name, addrs in psutil.net_if_addrs().items():
        ipv4_addrs = [addr for addr in addrs if addr.family == socket.AF_INET]
        if not ipv4_addrs:
            continue
        if _VPN_PATTERN.search(name):
            logger.debug(f"VPN detectada por padrão: {name}")
            return True
        for addr in ipv4_addrs:
            ip = addr.address
            if ip.startswith("172.") and "ethernet" in name.lower():
                logger.info(f"🔗 FortiClient VPN detectada: {name} ({ip})")
                return True
            if (
                ip.startswith("10.")
                and not ip.startswith("10.0.0.")
                and "ethernet" in name.lower()
            ):
                logger.info(f"🔗 VPN corporativa detectada: {name} ({ip})")
                return True

    return False


async def _async_run_orchestrator(manifest_file: Path, launcher: "AppLauncher | None"):
    """
    Asynchronously initializes and runs the Orchestrator.

    This function is designed to be called by `asyncio.run()` to ensure an event
    loop is active before the Orchestrator is instantiated, as its constructor
    may create asyncio tasks.

    Args:
        manifest_file (Path): The path to the manifest file to be executed.
    """
    parser = ManifestParser()
    manifest = parser.parse(manifest_file)
    orchestrator = Orchestrator(
        exec_id=manifest.execution_id,
        tasks=manifest.tasks,
        max_workers=manifest.max_workers,  # Legacy support
        resource_usage=manifest.resource_usage,  # New percentage-based (preferred)
    )
    if launcher:
        launcher.orchestrator = orchestrator

    await orchestrator.run()


async def _run_orchestrator(args: argparse.Namespace, launcher: "AppLauncher | None"):
    """
    Executes the orchestrator workflow using the provided command-line arguments.

    Logs the start of the 'run' command, parses the specified manifest file, and initializes

    Logs the start of the 'run' command, parses the specified manifest file, and initializes
    the Orchestrator with the manifest's execution ID, tasks, and maximum number of workers.
    Handles errors related to missing manifest files and other critical failures, logging
    appropriate error messages and exiting the program with a non-zero status code.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing at least the
            'manifest_file' attribute specifying the path to the manifest file.

    Raises:
        SystemExit: Exits the program with status code 1 if the manifest file is not found
            or if a critical error occurs during execution.
    """
    logger.debug(f"Comando 'run' selecionado. Usando manifesto: {args.manifest_file}")
    # if not is_vpn_up():
    #     logger.critical(
    #         "Nenhuma interface VPN detectada – conecte-se à VPN antes de continuar."
    #     )
    #     sys.exit(1)
    try:
        await _async_run_orchestrator(args.manifest_file, launcher)
        logger.info("Execução do pipeline concluída com sucesso.")
    except FileNotFoundError:
        logger.error(f"Arquivo de manifesto não encontrado em: {args.from_file}")
        logger.warning(
            "Você talvez precise gerar o manifesto primeiro com o comando 'generate'."
        )
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Uma falha crítica impediu a execução. Erro: {e}")
        sys.exit(1)


def _generate_manifest(args: argparse.Namespace):
    """
    Generates a manifest file based on user input from either the clipboard or a file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing options for input source,
            default sequence, and other relevant parameters.

    Behavior:
        - If 'from_clipboard' is specified in args, processes text from the clipboard and generates a manifest file.
        - If 'from_file' is specified in args, reads text from the provided file, parses it, and generates a manifest file.
        - Logs the selected command and errors if the input file is not found.

    Raises:
        SystemExit: If the specified input file does not exist.
    """
    logger.info("Comando 'generate' selecionado.")
    output_path = settings.DEFAULT_MANIFEST_PATH
    preprocessor = IntelligentPreprocessor()
    if args.from_clipboard:
        preprocessor.process_from_clipboard(
            default_sequence=args.sequence, output_path=output_path.name
        )
    elif args.from_file:
        try:
            raw_text = args.from_file.read_text(encoding="utf-8")
            tasks = preprocessor.parse_text(raw_text, default_sequence=args.sequence)
            preprocessor.generate_manifest_file(tasks, output_path=output_path.name)
        except FileNotFoundError:
            logger.error(f"Arquivo de entrada não encontrado em: {args.from_file}")
            sys.exit(1)


def _run_worker_mode(args: argparse.Namespace):
    """
    Starts the application in Worker mode using Celery.
    This function initializes and runs a Celery worker with the specified command-line arguments.
    It configures the worker's queue, log level, and optionally its concurrency level.
    If Celery is not available, it logs an error and exits the program.
    Args:
        args (argparse.Namespace): Parsed command-line arguments containing at least
            'queues' (str): The name(s) of the Celery queue(s) to listen to.
            'concurrency' (Optional[int]): The number of concurrent worker processes (optional).
    Raises:
        SystemExit: If Celery is not installed or cannot be imported.
    """
    logger.info("Iniciando modo Worker (Celery)...")
    try:
        from pff.celery_app import celery_app

        worker_args = ["worker", f"--queues={args.queues}", "--loglevel=info"]
        if args.concurrency:
            worker_args.append(f"--concurrency={args.concurrency}")
        logger.info(f"Iniciando worker com args: {' '.join(worker_args)}")
        celery_app.worker_main(worker_args)
    except ImportError as e:
        logger.error(f"Celery não disponível: {e}")
        sys.exit(1)


async def _run_api_mode(args: argparse.Namespace):
    """
    Starts the API mode using FastAPI and Uvicorn.
    Logs the startup process and attempts to launch the FastAPI application using Uvicorn with the specified host, port, and reload options from the provided arguments.
    If FastAPI or Uvicorn is not installed, logs an error and exits the program.
    Args:
        args (argparse.Namespace): Command-line arguments containing 'host', 'port', and 'reload' attributes for server configuration.
    Raises:
        SystemExit: If FastAPI or Uvicorn is not available.
    """
    logger.info("Iniciando modo API (FastAPI)...")
    import redis

    try:
        import uvicorn

        from pff.api.main import app

        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 8000)

        app.state.redis = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=5
        )

        logger.info(f"Servidor API iniciando em http://{host}:{port}")
        logger.info(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")

        config = uvicorn.Config(app, host=host, port=port, reload=args.reload)
        server = uvicorn.Server(config)
        await server.serve()
    except ImportError as e:
        logger.exception(f"FastAPI/Uvicorn não disponível: {e}")
        sys.exit(1)
    except redis.ConnectionError:
        logger.error(
            f"Redis não está rodando em {settings.REDIS_HOST}:{settings.REDIS_PORT}"
        )
        logger.info(
            "Instale e inicie o Redis: sudo apt install redis-server && sudo service redis-server start"
        )
        sys.exit(1)


async def _run_cleanup(args: argparse.Namespace):
    """
    Execute the cleanup command with the specified strategy.
    This function handles the 'clean' command by dynamically importing the cleanup
    module, building an engine with the provided arguments, and running the cleanup
    process asynchronously.
    Args:
        args (argparse.Namespace): Command line arguments containing:
            - strategy: The cleanup strategy to use
            - yes: Auto-confirm prompts if True
            - dry_run: Perform a dry run without actual changes if True
    Raises:
        SystemExit: Exits with code 1 if cleanup fails
    Note:
        Uses late import for the cleanup module to avoid circular dependencies.
        Logs success message on completion or exception details on failure.
    """

    logger.info("Comando 'clean' selecionado.")
    from importlib import import_module

    build_engine = import_module("pff.utils.cleanup").build_engine  # import tardio
    engine = build_engine(args.strategy, auto_yes=args.yes, dry_run=args.dry_run)
    try:
        await engine.run()
    except Exception as exc:
        logger.exception(f"Falha durante a limpeza: {exc}")
        sys.exit(1)


def _reset_ml_command(args: argparse.Namespace):
    """
    Resets the ML/TransE environment by invoking the cleanup utility.
    This function logs the reset process, dynamically imports the cleanup engine,
    and executes the reset operation. If the reset is successful, it logs a success
    message and provides guidance for retraining. If an error occurs during the reset,
    it logs the exception and exits the program.
    Args:
        args (argparse.Namespace): Command-line arguments containing options such as
            'yes' (auto-confirmation) and 'dry_run' (simulation mode).
    Raises:
        SystemExit: Exits the program with status 1 if the reset fails.
    """
    logger.info("🧹 Resetando ambiente de ML/TransE...")
    from importlib import import_module

    build_engine = import_module("pff.utils.cleanup").build_engine
    engine = build_engine("ml", auto_yes=args.yes, dry_run=args.dry_run)

    try:
        asyncio.run(engine.run())
        logger.success("✅ Ambiente de ML resetado com sucesso!")
        logger.info(
            "💡 Agora você pode treinar do zero com: python -m pff learn transe"
        )
    except Exception as exc:
        logger.exception(f"❌ Falha durante o reset: {exc}")
        sys.exit(1)


def check_pipeline_interruption(stage_name: str) -> None:
    if should_stop():
        logger.warning(
            f"🛑 Etapa '{stage_name}' interrompida por solicitação do usuário"
        )
        raise KeyboardInterrupt(f"Etapa '{stage_name}' foi interrompida")


async def learn_command(model: str = "all", config_path: Path | None = None):
    """Trains AI models for data validation and correction."""
    from pff.utils import logger
    from pff.utils.autofeeding import apply_autofeeding_rules
    from pff.validators.kg.config import KGConfig
    from pff.validators.kg.pipeline import KGPipeline
    from pff.validators.transe.transe_pipeline import TransEPipeline

    _interrupt_manager = get_interrupt_manager()
    logger.info("🛡️ GlobalInterruptManager ativo - CTRL+C irá parar toda a pipeline")

    try:
        if config_path is None:
            config_path = settings.CONFIG_DIR / "kg.yaml"
        if model.lower() == "kg":
            logger.info("🧠 Executando pipeline do Knowledge Graph (KG)...")
            check_interruption()
            kg_pipeline = KGPipeline(KGConfig(config_path))
            await kg_pipeline.run_build_and_preprocess()
            check_interruption()
            await kg_pipeline.run_learn_rules()
            check_interruption()
            await kg_pipeline.run_ranking()
            logger.success("✅ Pipeline do KG concluída.")
        elif model.lower() == "transe":
            logger.info("🤖 Executando pipeline do TransE (autossuficiente)...")
            check_interruption()
            transe_pipeline = TransEPipeline(config_path)
            await transe_pipeline.train_transe()
            check_interruption()
            transe_pipeline.rank_and_evaluate_transe()
            logger.success("✅ Pipeline do TransE concluída.")
        elif model.lower() == "ensemble":
            logger.info("✨ Executando pipeline de Ensemble...")
            _results = await run_standalone_ensemble_pipeline()
            logger.success("✅ Pipeline do Ensemble concluída.")
        elif model.lower() in ["all", "both", ""]:
            logger.info("🚀 Executando pipeline completa com autofeeding")
            logger.info("🧠 1/4: Executando pipeline do Knowledge Graph...")
            check_interruption()
            kg_pipeline = KGPipeline(KGConfig(config_path))
            await kg_pipeline.run_build_and_preprocess()
            check_interruption()
            await kg_pipeline.run_learn_rules()
            check_interruption()
            # await kg_pipeline.run_ranking()
            logger.info("🤖 2/4: Executando pipeline do TransE...")
            transe_pipeline = TransEPipeline(config_path)
            await transe_pipeline.train_transe()
            check_interruption()
            transe_pipeline.rank_and_evaluate_transe()
            check_interruption()
            logger.info("✨ 3/4: Executando Ensemble...")
            _ensemble_results = await run_standalone_ensemble_pipeline()
            check_interruption()
            logger.info("🤖 4/4: Aplicando autofeeding...")
            await apply_autofeeding_rules()
            logger.success("✅ Pipeline completo com autofeeding concluído!")
        else:
            logger.error(f"Modelo desconhecido: {model}. Use 'kg', 'transe' ou 'all'.")
            sys.exit(1)
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


async def main(launcher: "AppLauncher | None", argv: list[str] | None = None):
    """
    Main entry point for the PFF CLI application.
    Parses command-line arguments and dispatches to the appropriate subcommand handler.
    Main entry point for the PFF CLI application.
    Parses command-line arguments and dispatches to the appropriate subcommand handler.
    Args:
        argv (list[str] | None): Optional list of command-line arguments. If None, uses sys.argv.
    Subcommands:
        run:       Executes a task manifest.
            - manifest_file: Path to the manifest file (default: settings.DEFAULT_MANIFEST_PATH).
        generate:  Generates a default manifest from raw text.
            - --from-clipboard: Reads input from clipboard.
            - --from-file: Reads input from a text file.
            - --sequence: Default sequence for lines with only MSISDN.
        worker:    Starts a Celery worker.
            - -c / --concurrency: Number of worker processes.
            - -Q / --queues: Queues to consume (default: "default,high,low").
        api:       Starts the API server.
            - --host: Server host (default: "0.0.0.0").
            - -p / --port: Server port (default: 8000).
            - --reload: Enables auto-reload for development.
        cleanup:   Cleans caches, logs and outputs.
            - strategy: standard | deep | shutdown (default: standard)
            - -y / --yes: Skip confirmation.
            - --dry-run: Simulate deletion without removing files.
    Raises:
        SystemExit: If argument parsing fails or upon completion of the selected command.
    """
    parser = argparse.ArgumentParser(
        prog="pff",
        description=f"PFF - Production Fix Flow v{__version__}",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"PFF v{__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Comandos principais")
    subparsers.required = True

    # --- Comando: run ---
    parser_run = subparsers.add_parser("run", help="Executa um manifesto de tarefas.")
    parser_run.add_argument(
        "manifest_file",
        type=Path,
        nargs="?",
        default=settings.DEFAULT_MANIFEST_PATH,
        help=f"Caminho para o manifesto. Padrão: {settings.DEFAULT_MANIFEST_PATH}",
    )
    parser_run.set_defaults(func=_run_orchestrator)

    # --- Comando: generate ---
    parser_generate = subparsers.add_parser(
        "generate", help="Gera o manifesto padrão a partir de texto bruto."
    )
    input_group = parser_generate.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--from-clipboard", action="store_true", help="Lê da área de transferência."
    )
    input_group.add_argument(
        "--from-file", type=Path, help="Lê de um arquivo de texto."
    )
    parser_generate.add_argument(
        "--sequence", type=str, help="Sequência padrão para linhas com apenas MSISDN."
    )
    parser_generate.set_defaults(func=_generate_manifest)

    # --- Comando: worker ---
    parser_worker = subparsers.add_parser("worker", help="Inicia um worker Celery.")
    parser_worker.add_argument(
        "-c", "--concurrency", type=int, help="Número de processos do worker."
    )
    parser_worker.add_argument(
        "-Q",
        "--queues",
        type=str,
        default="default,high,low",
        help="Filas a serem consumidas.",
    )
    parser_worker.set_defaults(func=_run_worker_mode)

    # --- Comando: api ---
    parser_api = subparsers.add_parser("api", help="Inicia o servidor da API.")
    parser_api.add_argument("--host", default="0.0.0.0", help="Host do servidor.")
    parser_api.add_argument(
        "-p", "--port", type=int, default=8000, help="Porta do servidor."
    )
    parser_api.add_argument(
        "--reload", action="store_true", help="Ativa auto-reload para desenvolvimento."
    )
    parser_api.set_defaults(func=_run_api_mode)

    # --- Comando: clean ---
    parser_cleanup = subparsers.add_parser(
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
    parser_cleanup.add_argument(
        "strategy",
        choices=["standard", "deep", "ml", "shutdown"],
        nargs="?",
        default="standard",
        help="Estratégia de limpeza a ser utilizada.",
    )
    parser_cleanup.add_argument(
        "-y", "--yes", action="store_true", help="Não pedir confirmação."
    )
    parser_cleanup.add_argument(
        "--dry-run", action="store_true", help="Simular execução sem deletar."
    )
    parser_cleanup.set_defaults(func=_run_cleanup)

    learn_parser = subparsers.add_parser("learn", help="Treinar modelos de IA")
    learn_parser.add_argument(
        "model",
        nargs="?",
        default="all",
        choices=["kg", "transe", "all"],
        help="Modelo a treinar",
    )
    learn_parser.add_argument(
        "-c", "--config", type=Path, help="Arquivo de configuração"
    )

    parser_reset_ml = subparsers.add_parser(
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
    parser_reset_ml.add_argument(
        "-y", "--yes", action="store_true", help="Não pedir confirmação."
    )
    parser_reset_ml.add_argument(
        "--dry-run", action="store_true", help="Simular execução sem deletar."
    )
    parser_reset_ml.set_defaults(func=_reset_ml_command)

    args = parser.parse_args(argv)

    try:
        if hasattr(args, "func"):
            if args.command == "run":
                await args.func(args, launcher)
            elif args.command == "learn":

                def learn_interrupt_callback():
                    logger.info("🛑 Learn command será interrompido...")

                interrupt_manager.register_callback(learn_interrupt_callback)
                await learn_command(args.model, args.config)
            elif args.command == "reset-ml":
                args.func(args)
            else:
                await args.func(args)
        elif args.command == "learn":  # Fallback for older structure
            await learn_command(args.model, args.config)

    except KeyboardInterrupt:
        logger.warning("🛑 Aplicação interrompida pelo usuário")
        sys.exit(128)
    except Exception as e:
        logger.exception(f"Erro crítico na aplicação: {e}")
        sys.exit(1)


def cli_entrypoint():
    """Entry point for poetry script."""
    asyncio.run(main(launcher=None))


if __name__ == "__main__":
    cli_entrypoint()
