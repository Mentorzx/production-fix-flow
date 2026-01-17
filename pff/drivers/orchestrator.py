import asyncio
import datetime
import time
from datetime import timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable, Sequence
from contextvars import ContextVar

import polars as pl

from pff.domain.audit.manifest import TaskModel
from pff.shared.core.config import settings
from pff.application.services import BusinessService, LineService, SequenceService
from pff.shared import ConcurrencyManager, LogReorderer, logger
from pff.shared.core.file_manager import FileManager
from pff.shared.system.resource_manager import HardwareDetector

Task = TaskModel


_ENGINE_CTX: ContextVar[SequenceService | None] = ContextVar("engine_ctx", default=None)


def _make_rotated_path(path: Path, idx: int) -> Path:
    return path.with_name(f"{path.stem}_{idx:04d}{path.suffix}")


class BufferedWriter:
    """
    AsyncBufferedWriter is a fully asynchronous buffered writer for CSV, JSONL, and Parquet files.
    This class accumulates rows in memory and periodically flushes them to disk based on
    row count or elapsed time. It supports file rotation, concurrent writes via an asyncio queue,
    and can be used as an async context manager. The writer supports writing dictionaries,
    sequences, or polars DataFrames, and automatically handles file creation and format.

    Parameters:
        dest (str | Path): Destination file path. Supported extensions: .csv, .jsonl, .parquet.
        flush_rows (int, optional): Number of rows to buffer before flushing to disk. Default is 5,000.
        flush_secs (int, optional): Maximum seconds to wait before flushing buffer. Default is 30.
        rotation (int | None, optional): If set, rotates file after writing this many rows. Default is None.
        max_queue (int, optional): Maximum number of items in the write queue. Default is 50,000.

    Methods:
        write(row): Enqueue a row (dict, sequence, or DataFrame) for writing.
        close(): Stop the writer task and flush remaining data.
        force_flush(): Force a flush of the current buffer.
        write_async(rows): Asynchronously enqueue multiple rows for writing.

    Context Manager:
        Can be used with 'async with' statement to ensure proper resource cleanup.

    Raises:
        ValueError: If the file extension is not supported.
    """

    def __init__(
        self,
        dest: str | Path,
        *,
        flush_rows: int = 5_000,
        flush_secs: int = 30,
        rotation: int | None = None,
        max_queue: int = 50_000,
    ) -> None:
        raw_dest = Path(dest)
        self.dest = raw_dest if raw_dest.is_absolute() else settings.ROOT_DIR / raw_dest
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.ext = FileManager.assert_supported_path(
            self.dest, allowed_exts={".csv", ".jsonl", ".parquet"}
        )

        self.flush_rows = flush_rows
        self.flush_secs = flush_secs
        self.rotation = rotation
        self._rot_idx = 0
        self._file_manager = FileManager()

        self._last_flush = time.time()
        self._row_count = 0
        self._current_target = self.dest
        self._frames: list[pl.DataFrame] = []
        self._buffer: list[Any] = []

    async def write(self, row: dict[str, Any] | Sequence[Any] | pl.DataFrame) -> None:
        self._buffer.append(row)
        time_cond = (time.time() - self._last_flush) >= self.flush_secs
        if len(self._buffer) >= self.flush_rows or (self._buffer and time_cond):
            await self._flush(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

    async def close(self) -> None:
        if self._buffer:
            await self._flush(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
        await self._finalize_target()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        await self.close()

    async def _flush(self, rows: list[Any]) -> None:
        if not rows:
            return
        incoming_rows = len(rows)
        if self.rotation and (self._row_count + incoming_rows) > self.rotation:
            await self._finalize_target()
            self._rot_idx += 1
            self._row_count = 0
            self._current_target = _make_rotated_path(self.dest, self._rot_idx)

        df_rows: list[pl.DataFrame] = []
        for r in rows:
            if isinstance(r, pl.DataFrame):
                df_rows.append(r)
            elif isinstance(r, dict):
                df_rows.append(pl.DataFrame([r]))
            else:
                df_rows.append(pl.DataFrame([list(r)]))

        if df_rows:
            new_df = pl.concat(df_rows, how="diagonal")
            self._frames.append(new_df)

        self._row_count += incoming_rows
        logger.debug(f"Flushed {len(rows)} row(s) -> {self._current_target.name}")

    async def force_flush(self) -> None:
        """Force a flush of the current buffer."""
        if self._buffer:
            await self._flush(self._buffer)
            self._buffer.clear()
        self._last_flush = time.time()

    async def write_async(self, rows: Iterable[Any]) -> None:
        """Asynchronously enqueue multiple rows for writing."""
        tasks = [self.write(row) for row in rows]
        await asyncio.gather(*tasks)

    async def _finalize_target(self) -> None:
        if not self._frames:
            self._file_manager.save(pl.DataFrame([]), self._current_target)
            return

        combined = pl.concat(self._frames, how="vertical")
        self._file_manager.save(combined, self._current_target)
        self._frames.clear()


class ResultCollector:
    """
    ResultCollector is a utility class for collecting, deduplicating, and exporting result rows to CSV or XLSX files.

    Attributes:
        _writer (AsyncBufferedWriter): Internal writer for buffering and writing rows to a temporary CSV file.
        _seen (set[str]): Set of MSISDNs already processed to avoid duplicates.
        exec_id (str): Identifier for the current execution, used in output filenames.
        _tmp_path (Path): Path to the temporary file used for intermediate storage.

    Args:
        exec_id (str): Unique identifier for the execution session.
        flush_rows (int, optional): Number of rows to buffer before flushing to disk. Defaults to 2,000.
        rotation (int | None, optional): Optional file rotation parameter. Defaults to None.

    Methods:
        append_row(msisdn: str, request: str, result: str, obs: str | dict[str, list[str]]) -> None:
            Appends a result row to the collector, formatting observations as needed and deduplicating by MSISDN.
        has_row(msisdn: str) -> bool:
            Checks if a row with the given MSISDN has already been added.
        save(path: str | Path | None = None, *, fmt: str | None = None) -> Path:
            Saves the collected results to a file. If no path is provided, saves to 'outputs/{timestamp}_{exec_id}.xlsx'.
            Supports CSV or XLSX output based on file extension or 'fmt' parameter.
            Cleans up the temporary file after saving.
    """

    _writer: BufferedWriter
    _seen: set[str]

    def __init__(
        self,
        exec_id: str,
        *,
        flush_rows: int = 2_000,
        rotation: int | None = None,
    ) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_name = f"{ts}_{exec_id}.parquet"
        self._tmp_path = settings.OUTPUTS_DIR / "temp" / "result_collector" / tmp_name
        self._tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = BufferedWriter(self._tmp_path, flush_rows=flush_rows, rotation=rotation)
        self._seen = set()
        self.exec_id = exec_id
        self._file_manager = FileManager()

    async def append_row(
        self,
        msisdn: str,
        request: str,
        result: str,
        obs: str | dict[str, list[str]],
    ) -> None:
        if isinstance(obs, dict):
            partes: list[str] = []
            for chave, lista in obs.items():
                partes.append(f"{chave}: {', '.join(lista) if lista else 'nenhum'}")
            obs_txt = " | ".join(partes)
        else:
            obs_txt = str(obs)

        row = {
            "MSISDN": msisdn,
            "Solicitacao": request,
            "Resultado": result,
            "Observacoes": obs_txt,
        }
        await self._writer.write(row)
        self._seen.add(msisdn)

    def has_row(self, msisdn: str) -> bool:
        return msisdn in self._seen

    async def save(
        self,
        path: str | Path | None = None,
        *,
        fmt: str | None = None,
    ) -> Path:
        await self._writer.close()

        if not self._tmp_path.exists():
            logger.warning("No records captured; skipping save.")
            return Path()

        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = settings.OUTPUTS_DIR / "results" / f"{ts}_{self.exec_id}.xlsx"

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        df_to_save = self._file_manager.read(self._tmp_path)
        await self._file_manager.async_save(df_to_save, dest)

        logger.info(f"Resultado salvo em {dest}")

        try:
            self._tmp_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Failed to remove tmp file {self._tmp_path}: {exc}")

        return dest


def _get_engine() -> SequenceService:
    """
    Initializes and retrieves a context-local instance of SequenceService.

    Using contextvars ensures asyncio-safety and avoids raw threading primitives.
    """
    engine = _ENGINE_CTX.get()
    if engine is None:
        svc = LineService()
        validator = BusinessService()
        services = {"line": svc, "validator": validator}
        engine = SequenceService(services)
        _ENGINE_CTX.set(engine)
    return engine


async def _worker(task: Task, collector: ResultCollector) -> None:
    """
    Processes a single task by executing it with the engine and collecting the result.
    Args:
        task (Task): The task to be processed, expected to contain 'msisdn', 'sequence', and optionally 'payload'.
        collector (ResultCollector): The collector object used to store the results of the task execution.
    Behavior:
        - Validates that 'msisdn' and 'sequence' are present in the task.
        - Runs the engine with the provided task data.
        - Appends a result row to the collector indicating completion or failure, only if not already present.
        - Logs errors if the task is invalid or if an exception occurs during processing.
    """
    engine = _get_engine()
    msisdn = task.msisdn
    sequence = task.sequence
    payload = task.payload

    with logger.contextualize(
        msisdn=task.msisdn.split("/")[-1] if "/" in str(task.msisdn) else task.msisdn
    ):
        try:
            logger.info("Iniciando execução da sequência '{}'", sequence)
            await engine.run(msisdn, sequence, payload=payload, collector=collector)
            logger.success("Sequência '{}' concluída com sucesso.", sequence)
            if not collector.has_row(msisdn):
                await collector.append_row(msisdn, sequence, "Sucesso", "Executado")
        except Exception as e:
            logger.exception("Erro ao processar a tarefa para o MSISDN {}: {}", msisdn, e)
            if not collector.has_row(msisdn):
                await collector.append_row(msisdn, sequence, "Falha", str(e))


class Orchestrator:
    """Orchestrates concurrent task execution with result collection and logging.

    Design Patterns Applied:
        - **Command Pattern:** Each Task encapsulates an operation with its
          parameters, allowing queuing, logging, and concurrent execution.
        - **Facade Pattern:** Provides a unified interface for managing worker
          pools, result collection, and execution monitoring.
        - **Observer Pattern (ready):** Integrates with ResultCollector for
          event-driven result handling and progress tracking.

    Performance Optimizations:
        - Adaptive worker count via ResourceManager integration.
        - Async execution with asyncio for I/O-bound tasks.
        - Configurable resource_usage percentage for CPU/memory limits.

    Attributes:
        exec_id: Unique identifier for the execution batch.
        tasks: List of tasks to be executed.
        max_workers: Maximum number of concurrent workers.
        collector: Collector for storing task results.

    Methods:
        run(): Executes the batch of tasks with concurrency management.
    """

    def __init__(
        self,
        exec_id: str,
        tasks: Iterable[Task],
        max_workers: int | None = None,
        resource_usage: float | None = None,
    ):
        self.exec_id = exec_id
        self.tasks = list(tasks)

        from pff.shared.system.resource_manager import get_resource_manager

        if resource_usage is not None:
            resource_manager = get_resource_manager(
                cpu_usage_percent=resource_usage,
                memory_usage_percent=resource_usage,
            )
            limits = resource_manager.calculate_limits(
                task_count=len(self.tasks),
                estimated_task_size=5000,  # Assume ~5 KB per task
            )
            max_workers = limits.optimal_workers
            logger.info(
                f" Resource allocation: {resource_usage:.0f}% usage → "
                f"{max_workers} workers ({limits.profile_name})"
            )
        elif max_workers is not None:
            # LEGACY: Use fixed max_workers (deprecated, but supported)
            hardware_profile = HardwareDetector.detect()
            safe_max_workers = self._get_safe_max_workers(hardware_profile.profile_name)

            if max_workers > safe_max_workers:
                logger.warning(
                    f"  max_workers={max_workers} exceeds safe limit for {hardware_profile.profile_name}. "
                    f"Reducing to {safe_max_workers} (RAM: {hardware_profile.total_ram_gb:.1f} GB, "
                    f"CPU: {hardware_profile.cpu_threads} threads)"
                )
                max_workers = safe_max_workers
            elif max_workers <= 0:
                logger.warning(
                    f"  max_workers={max_workers} invalid. Using default {safe_max_workers}"
                )
                max_workers = safe_max_workers
        else:
            # DEFAULT: Use 90% resource allocation
            resource_manager = get_resource_manager(
                cpu_usage_percent=90.0,
                memory_usage_percent=90.0,
            )
            limits = resource_manager.calculate_limits(
                task_count=len(self.tasks),
                estimated_task_size=5000,
            )
            max_workers = limits.optimal_workers
            logger.info(
                f" Default resource allocation: 90% usage → "
                f"{max_workers} workers ({limits.profile_name})"
            )

        self.max_workers = max_workers
        self.collector = ResultCollector(exec_id=self.exec_id)
        logger.info(
            f"Orchestrator initialized: {len(self.tasks)} tasks, {self.max_workers} workers"
        )

    @staticmethod
    def _get_safe_max_workers(machine_name: str) -> int:
        """
        Returns safe max_workers based on machine profile.

        Conservative limits to prevent OOM:
        - low_spec (8GB RAM): 4 workers
        - mid_spec (12-16GB RAM): 8 workers
        - high_spec (32GB+ RAM): 16 workers

        Args:
            machine_name: Hardware profile ("low_spec", "mid_spec", "high_spec")

        Returns:
            Safe max_workers limit
        """
        limits = {
            "low_spec": 4,
            "mid_spec": 8,
            "high_spec": 16,
        }
        return limits.get(machine_name, 8)  # Default to mid_spec if unknown

    def _configure_file_logger(self) -> int:
        """
        Configures and adds a file logger with specific settings.

        Creates a log file in the directory specified by `settings.LOGS_DIR`, using a filename
        that includes the current UTC timestamp and the execution ID. The logger is set to:
            - Log at DEBUG level.
            - Rotate the log file when it reaches 10 MB.
            - Retain log files for 14 days.
            - Compress old log files as ZIP archives.
            - Serialize log records in JSON format.

        Returns:
            int: The identifier of the added logger sink.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_path = settings.LOGS_DIR / f"{ts}_{self.exec_id}.log"
        sink_id = logger.add(
            self._log_path,
            level="DEBUG",
            rotation="10 MB",
            retention="14 days",
            compression="zip",
            serialize=True,
        )
        return sink_id

    async def run(self, progress_hook: Callable[[int, int], None] | None = None):
        """
        Executes the batch of tasks end-to-end.
        """
        if not self.tasks:
            logger.warning("No tasks to execute.")
            return

        logger.info(f"Iniciando orquestrador para a execução: '{self.exec_id}'")
        logger.info(f"Total de tarefas: {len(self.tasks)}, Workers: {self.max_workers}")

        sink_id = self._configure_file_logger()

        try:
            worker_tasks = [(task, self.collector) for task in self.tasks]
            total = len(worker_tasks)
            done = 0

            async def _wrap_worker(task, collector):
                nonlocal done
                await _worker(task, collector)
                done += 1
                if progress_hook:
                    progress_hook(done, total)

            cm = ConcurrencyManager()
            await cm.execute(
                _wrap_worker,
                worker_tasks,
                task_type="io_async",
                max_workers=self.max_workers,
                desc=f"Executando '{self.exec_id}'",
            )

            output_path = await self.collector.save()
            logger.success(f"Execução concluída! Resultados salvos em: {output_path}")
        except Exception as exc:
            logger.critical(f"Erro catastrófico durante a orquestração: {exc}")
            raise
        finally:
            logger.remove(sink_id)
            try:
                reordered_path = LogReorderer.reorder(self._log_path)
                logger.success(f"Log reordenado por thread salvo em: {reordered_path}")
            except Exception as e:
                logger.warning(f"Failed to reorder log file: {e}")

            logger.info("Logger de arquivo finalizado.")

    async def shutdown(self) -> None:
        """
        Asynchronously shuts down the orchestrator, logging the shutdown process.
        If a collector with a 'save' method exists, saves the final results before shutdown.
        Returns:
            None
        """
        logger.info("Encerrando o orquestrador...")

        if hasattr(self, "collector") and self.collector and hasattr(self.collector, "save"):
            logger.info("Salvando resultados finais...")
            await self.collector.save()
