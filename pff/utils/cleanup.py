from __future__ import annotations

import argparse
import gc
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Protocol

from rich.console import Console

from pff import settings
from pff.utils import ConcurrencyManager, DiskCache, logger


def _format_size(size_bytes: int) -> str:
    """Formats a size in bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


class CleanupCommand(Protocol):
    """Protocol for a command that can be executed to perform a cleanup action."""

    label: str

    def execute(self) -> None: ...


class CompositeCommand:
    def __init__(self, label: str, children: Iterable[CleanupCommand]):
        self.label = label
        self._children = list(children)

    def execute(self) -> None:
        for cmd in self._children:
            cmd.execute()


class CloseLoggerCommand(CleanupCommand):
    """A command to gracefully shut down the logger sinks."""

    label = "Fechando coletores de log ativos"

    def execute(self) -> None:
        try:
            logger.remove()
            import atexit
            import logging

            logging.shutdown()
            atexit._run_exitfuncs()
            time.sleep(0.2)
        except Exception as e:
            print(f"Alerta: Falha ao fechar loggers: {e}")


class DirCleanCommand(CleanupCommand):
    """A command to clean files and directories based on a pattern."""

    def __init__(
        self,
        label: str,
        directory: Path,
        pattern: str | None = None,
        recursive: bool = False,
    ):
        self.label = label
        self._dir = directory
        self._pattern = pattern
        self._recursive = recursive

    def execute(self) -> None:
        if not self._dir.exists():
            return
        iterator = (
            self._dir.rglob(self._pattern or "*")
            if self._recursive
            else self._dir.glob(self._pattern or "*")
        )
        for item in iterator:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink(missing_ok=True)
                except PermissionError:
                    try:
                        item.open("w").close()
                        item.unlink(missing_ok=True)
                    except Exception as exc:
                        if not item.suffix == ".log":
                            logger.warning(f"Não foi possível remover {item} – {exc}")


class NestedDirCleanCommand(CleanupCommand):
    """A command to clean all nested directories with a specific name."""

    def __init__(self, dirname: str, label: str):
        self.dirname = dirname
        self.label = label

    def execute(self) -> None:
        for d in settings.ROOT_DIR.rglob(self.dirname):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


class PyCacheCleanCommand(CleanupCommand):
    """A command to remove all __pycache__ directories."""

    label = "Removendo __pycache__"

    def execute(self) -> None:
        for p in settings.ROOT_DIR.rglob("__pycache__"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)


class FlushMemoryCommand(CleanupCommand):
    """A command to flush in-memory caches and trigger garbage collection."""

    label = "Liberando caches de memória"

    def execute(self) -> None:
        DiskCache(settings.ROOT_DIR / ".cache").purge()
        for obj in list(sys.modules.values()):
            if callable(getattr(obj, "cache_clear", None)):
                obj.cache_clear()  # type: ignore[arg-type]
        gc.collect()


class CleanupStrategy(Protocol):
    """Protocol for a strategy that builds a list of cleanup commands."""

    def build_commands(self) -> list[CleanupCommand]: ...


class StandardCleanup(CleanupStrategy):
    def build_commands(self) -> list[CleanupCommand]:
        return [
            PyCacheCleanCommand(),
            DirCleanCommand("Limpando outputs", settings.OUTPUTS_DIR),
            DirCleanCommand("Limpando cache em disco", settings.ROOT_DIR / ".cache"),
            FlushMemoryCommand(),
            CloseLoggerCommand(),
            DirCleanCommand("Limpando logs", settings.LOGS_DIR, "*.log"),
            NestedDirCleanCommand(".cache", "Limpando todos os .cache"),
            DirCleanCommand(
                "Limpando pytest cache",
                settings.ROOT_DIR / ".pytest_cache",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando mypy cache", settings.ROOT_DIR / ".mypy_cache", recursive=True
            ),
            DirCleanCommand(
                "Limpando checkpoints Jupyter",
                settings.ROOT_DIR,
                "**/.ipynb_checkpoints",
                recursive=True,
            ),
            NestedDirCleanCommand(".pytest_cache", "Limpando todos os .pytest_cache"),
            NestedDirCleanCommand(".mypy_cache", "Limpando todos os .mypy_cache"),
            NestedDirCleanCommand("node_modules", "Limpando todos os node_modules"),
            NestedDirCleanCommand("dist", "Limpando todos os dist"),
            NestedDirCleanCommand(".coverage", "Limpando todos os .coverage"),
            NestedDirCleanCommand("htmlcov", "Limpando todos os htmlcov"),
            DirCleanCommand("Limpando mlruns", settings.ROOT_DIR / "mlruns"),
            DirCleanCommand(
                "Limpando pip cache", settings.PIP_CACHE_DIR, recursive=True
            ),
        ]


class MLFlowCleanCommand(CleanupCommand):
    label = "Limpando experimentos MLflow"

    def execute(self) -> None:
        mlruns_dir = settings.ROOT_DIR / "mlruns"
        if mlruns_dir.exists():
            logger.info(f"Removendo MLflow experiments: {mlruns_dir}")
            shutil.rmtree(mlruns_dir, ignore_errors=True)
            logger.info("✅ Experimentos MLflow removidos")
        else:
            logger.debug("MLflow directory não encontrado")


class TransECheckpointsCleanCommand(CleanupCommand):
    label = "Limpando checkpoints TransE"

    def execute(self) -> None:
        locations: list[Path] = [
            settings.ROOT_DIR / "checkpoints",
            settings.OUTPUTS_DIR / "transe",
            Path.cwd() / "checkpoints",
        ]
        file_patterns = [
            "*.pt",
            "*.pth",
            "checkpoint_*.pt",
            "checkpoint_*.pth",
            "best_model.pt",
            "latest_checkpoint.pt",
        ]
        for location in locations:
            if not location.exists():
                continue
            logger.info(f"Limpando checkpoints em: {location}")
            for pattern in file_patterns:
                for fp in location.rglob(pattern):
                    try:
                        fp.unlink(missing_ok=True)
                        logger.debug(f"Removido arquivo de checkpoint: {fp}")
                    except Exception as e:
                        logger.warning(f"Não foi possível remover {fp}: {e}")
            try:
                if not any(location.iterdir()):
                    shutil.rmtree(location, ignore_errors=True)
                    logger.info(f"✅ Diretório de checkpoints removido: {location}")
            except Exception as e:
                logger.warning(f"Não foi possível remover diretório {location}: {e}")

        logger.info("✅ Checkpoints TransE removidos")


class TransparentCompositeCommand:
    def __init__(self, label: str, children: Iterable[CleanupCommand]):
        self.label = label
        self._children = list(children)

    def execute(self) -> None:
        for cmd in self._children:
            cmd.execute()

    def get_all_leaf_commands(self) -> list[CleanupCommand]:
        """Retorna todos os comandos folha (não-composite) para transparência."""
        leaf_commands = []
        for child in self._children:
            if isinstance(child, TransparentCompositeCommand):
                leaf_commands.extend(child.get_all_leaf_commands())
            elif isinstance(child, CompositeCommand):
                # Recursively collect from CompositeCommand if needed
                for subchild in child._children:
                    if isinstance(subchild, TransparentCompositeCommand):
                        leaf_commands.extend(subchild.get_all_leaf_commands())
                    else:
                        leaf_commands.append(subchild)
            else:
                leaf_commands.append(child)
        return leaf_commands


class ModelCacheCleanCommand(CleanupCommand):
    label = "Limpando cache de modelos"

    def execute(self) -> None:
        cache_locations = [
            settings.OUTPUTS_DIR / "transe" / "temp_models",
            settings.ROOT_DIR / ".cache" / "torch",
            settings.ROOT_DIR / ".cache" / "huggingface",
            Path.home() / ".cache" / "torch",
            Path.home() / ".cache" / "huggingface",
        ]

        for cache_dir in cache_locations:
            if cache_dir.exists():
                logger.info(f"Removendo cache: {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)

        logger.info("✅ Cache de modelos removido")


class TrainingArtifactsCleanCommand(CleanupCommand):
    label = "Limpando artefatos de treinamento"

    def execute(self) -> None:
        # Artefatos temporários e intermediários
        artifacts_patterns = [
            settings.OUTPUTS_DIR / "transe" / "temp_*",  # Arquivos temporários
            settings.OUTPUTS_DIR / "transe" / "*_temp.yaml",  # Configs temporárias
            settings.OUTPUTS_DIR / "temp_config_trial_*.yaml",  # Configs do Optuna
            settings.ROOT_DIR / "temp_config_trial_*.yaml",
            settings.OUTPUTS_DIR / "**" / "*.tmp",  # Arquivos temporários
            settings.OUTPUTS_DIR / "**" / "training_state_*.json",  # Estados de treino
        ]

        for pattern in artifacts_patterns:
            if "*" in str(pattern):
                # É um pattern, usar glob
                parent = pattern.parent
                pattern_name = pattern.name
                if parent.exists():
                    for item in parent.glob(pattern_name):
                        try:
                            if item.is_file():
                                item.unlink(missing_ok=True)
                            elif item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                        except Exception as e:
                            logger.warning(f"Não foi possível remover {item}: {e}")
            else:
                # É um path direto
                if pattern.exists():
                    if pattern.is_file():
                        pattern.unlink(missing_ok=True)
                    elif pattern.is_dir():
                        shutil.rmtree(pattern, ignore_errors=True)

        logger.info("✅ Artefatos de treinamento removidos")


class OptunaDatabaseCleanCommand(CleanupCommand):
    label = "Limpando bancos Optuna"

    def execute(self) -> None:
        # Bancos de dados do Optuna
        optuna_files = [
            settings.ROOT_DIR / "optuna.db",
            settings.ROOT_DIR / "**/*.db",  # Qualquer arquivo .db
            settings.OUTPUTS_DIR / "**/*.db",
        ]

        for pattern in optuna_files:
            if "*" in str(pattern):
                parent = pattern.parent
                pattern_name = pattern.name
                if parent.exists():
                    for item in parent.rglob(pattern_name):
                        try:
                            item.unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Não foi possível remover {item}: {e}")
            else:
                if pattern.exists():
                    pattern.unlink(missing_ok=True)

        logger.info("✅ Bancos de dados Optuna removidos")


class MLTrainingCleanCommand(CompositeCommand):
    def __init__(self):
        super().__init__(
            "Limpeza completa de ML/TransE",
            [
                TransECheckpointsCleanCommand(),
                MLFlowCleanCommand(),
                ModelCacheCleanCommand(),
                TrainingArtifactsCleanCommand(),
                OptunaDatabaseCleanCommand(),
                DirCleanCommand(
                    "Limpando outputs TransE", settings.OUTPUTS_DIR / "transe"
                ),
                DirCleanCommand("Limpando PyClause outputs", settings.PYCLAUSE_DIR),
            ],
        )


class DeepCleanup(StandardCleanup):
    """A more aggressive cleanup strategy including developer artifacts."""

    def build_commands(self) -> list[CleanupCommand]:
        base = super().build_commands()
        ml_commands = [
            MLTrainingCleanCommand(),
            DirCleanCommand(
                "Limpando dados KG processados",
                settings.DATA_DIR / "models" / "kg",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando cache PyTorch",
                Path.home() / ".cache" / "torch",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando cache HuggingFace",
                Path.home() / ".cache" / "huggingface",
                recursive=True,
            ),
            DirCleanCommand(
                "Limpando logs de treinamento", settings.LOGS_DIR, "training_*.log"
            ),
            DirCleanCommand("Limpando logs MLflow", settings.LOGS_DIR, "mlflow_*.log"),
        ]
        base[-2:-2] = ml_commands
        return base


class MLCleanup(CleanupStrategy):
    def build_commands(self) -> list[CleanupCommand]:
        return [
            FlushMemoryCommand(),
            MLTrainingCleanCommand(),
            DirCleanCommand("Limpando logs ML", settings.LOGS_DIR, "*training*.log"),
            DirCleanCommand("Limpando logs MLflow", settings.LOGS_DIR, "*mlflow*.log"),
            CloseLoggerCommand(),
        ]


class ShutdownCleanup(CleanupStrategy):
    """A selective cleanup strategy for graceful shutdown."""

    def build_commands(self) -> list[CleanupCommand]:
        logger.info("Building selective commands for graceful shutdown...")
        return [
            FlushMemoryCommand(),
            DirCleanCommand("Limpando cache em disco", settings.ROOT_DIR / ".cache"),
            PyCacheCleanCommand(),
        ]


class CleanupEngine:
    """The engine that orchestrates the cleanup process."""

    def __init__(
        self, strategy: CleanupStrategy, auto_yes: bool = False, dry_run: bool = False
    ):
        self._commands = strategy.build_commands()
        self._console = Console()
        self._auto_yes = auto_yes
        self._dry_run = dry_run

    def _flatten_commands(self, commands: list[CleanupCommand]) -> list[CleanupCommand]:
        flattened = []
        for cmd in commands:
            if isinstance(cmd, TransparentCompositeCommand):
                flattened.extend(cmd.get_all_leaf_commands())
            elif isinstance(cmd, CompositeCommand):
                flattened.extend(self._flatten_commands(cmd._children))
            else:
                flattened.append(cmd)
        return flattened

    def _calculate_target_size(self, cmd: CleanupCommand) -> int:
        """
        Calculates the total size in bytes of files and directories that would be affected by the given cleanup command.
        The method determines the type of the provided command and computes the cumulative size of all target files and directories:
        - For `DirCleanCommand`, it sums the sizes of files matching the specified pattern in the target directory (recursively if specified).
        - For `NestedDirCleanCommand`, it sums the sizes of all files within directories matching the given name under the root path.
        - For `PyCacheCleanCommand`, it sums the sizes of all files within `__pycache__` directories under the root path.
        - For `CompositeCommand`, it recursively calculates the total size for each child command.
        Files that cannot be accessed (e.g., due to being deleted during iteration) are skipped.
        Args:
            cmd (CleanupCommand): The cleanup command specifying the target files and directories.
        Returns:
            int: The total size in bytes of all files that would be affected by the command.
        """
        total_size = 0
        if isinstance(cmd, DirCleanCommand):
            if cmd._dir.exists():
                glob_fn = cmd._dir.rglob if cmd._recursive else cmd._dir.glob
                pattern = cmd._pattern or "*"
                for item in glob_fn(pattern):
                    try:
                        if item.is_file():
                            total_size += item.stat().st_size
                        elif item.is_dir():
                            total_size += sum(
                                f.stat().st_size for f in item.rglob("*") if f.is_file()
                            )
                    except FileNotFoundError:
                        continue
        elif isinstance(cmd, NestedDirCleanCommand):
            for item in settings.ROOT_DIR.rglob(f"**/{cmd.dirname}"):
                if item.is_dir():
                    try:
                        total_size += sum(
                            f.stat().st_size for f in item.rglob("*") if f.is_file()
                        )
                    except FileNotFoundError:
                        continue
        elif isinstance(cmd, PyCacheCleanCommand):
            for item in settings.ROOT_DIR.rglob("__pycache__"):
                if item.is_dir():
                    try:
                        total_size += sum(
                            f.stat().st_size for f in item.rglob("*") if f.is_file()
                        )
                    except FileNotFoundError:
                        continue
        elif isinstance(cmd, CompositeCommand):
            total_size += sum(self._calculate_target_size(c) for c in cmd._children)

        return total_size

    async def _filter_commands(self) -> list[tuple[CleanupCommand, int]]:
        flat_commands = self._flatten_commands(self._commands)

        cm = ConcurrencyManager()
        command_sizes = await cm.execute(
            lambda cmd: self._calculate_target_size(cmd),
            [(cmd,) for cmd in flat_commands],
            task_type="thread",
            desc="Scanning file sizes",
        )

        return [
            (cmd, size) for cmd, size in zip(flat_commands, command_sizes) if size > 0
        ]

    async def _confirm(self) -> list[tuple[CleanupCommand, int]]:
        """
        Confirms with the user before deleting files or directories.
        This method filters the list of commands representing files or directories to be deleted,
        displays them along with their sizes, and shows the total space that will be freed.
        If there are no items to delete, it notifies the user and exits.
        Otherwise, it prompts the user for confirmation to proceed with the deletion.
        If the user does not confirm, the operation is aborted.
        Side Effects:
            - Prints information to the console.
            - Exits the program if there is nothing to delete or if the user aborts.
        Returns:
            None
        """
        visible_commands_with_sizes = await self._filter_commands()

        if not visible_commands_with_sizes:
            self._console.print(
                "[bold green]✅ Nenhum arquivo ou diretório para limpar.[/]"
            )
            sys.exit(0)

        self._console.print(
            "[bold yellow]Os diretórios/arquivos a seguir serão apagados:[/]"
        )

        total_size_to_delete = 0
        for (
            cmd,
            size,
        ) in visible_commands_with_sizes:
            total_size_to_delete += size
            size_str = f"({_format_size(size)})"
            target_path = getattr(cmd, "_dir", None)
            if not target_path and hasattr(cmd, "dirname"):
                target_path = f"**/{getattr(cmd, 'dirname')}"
            if target_path:
                self._console.print(
                    f" • {cmd.label}: {target_path} [bold cyan]{size_str}[/]"
                )
            else:
                self._console.print(f" • {cmd.label} [bold cyan]{size_str}[/]")

        self._console.print("-" * 30)
        self._console.print(
            f"[bold red]Total a ser liberado: {_format_size(total_size_to_delete)}[/]"
        )

        ans = self._console.input("Prosseguir? (y/N): ").strip().lower()
        if ans != "y":
            self._console.print("Abortado.")
            sys.exit(130)
        return visible_commands_with_sizes

    async def run(self, confirm: bool = True) -> None:
        """Executes the cleanup commands."""
        if confirm and not self._auto_yes:
            visible_commands_with_sizes = await self._confirm()
        else:
            visible_commands_with_sizes = await self._filter_commands()

        if self._dry_run:
            self._console.print(
                "[bold yellow]Execução simulada: Os seguintes comandos seriam executados:[/]"
            )
            for cmd, _ in visible_commands_with_sizes:
                self._console.print(f" • {cmd.label}")
            sys.exit(0)

        if not visible_commands_with_sizes:
            logger.info("Nenhuma tarefa de limpeza a ser executada.")
            return

        cm = ConcurrencyManager()
        await cm.execute(
            lambda cmd: cmd.execute(),
            [(cmd,) for cmd, _ in visible_commands_with_sizes],
            task_type="thread",
            desc="Limpando",
        )
        logger.success("Limpeza finalizada com sucesso.")


def build_engine(strategy_name: str, **kwargs) -> CleanupEngine:
    """Builds a CleanupEngine with the specified strategy."""
    strategies = {
        "standard": StandardCleanup,
        "deep": DeepCleanup,
        "ml": MLCleanup,
        "shutdown": ShutdownCleanup,
    }
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Estratégia de limpeza desconhecida: {strategy_name}. Disponíveis: {available}"
        )
    return CleanupEngine(strategy_class(), **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Limpa caches antigos, logs e outputs.")
    p.add_argument(
        "strategy",
        choices=["standard", "deep", "ml", "shutdown"],  # 🔥 ADICIONAR "ml" AQUI
        nargs="?",
        default="standard",
        help="A estratégia de limpeza a ser utilizada.",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Não pedir confirmação.")
    p.add_argument(
        "--dry-run", action="store_true", help="Simular execução sem deletar."
    )
    ns = p.parse_args()
    engine = build_engine(ns.strategy, auto_yes=ns.yes, dry_run=ns.dry_run)
    import asyncio

    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
