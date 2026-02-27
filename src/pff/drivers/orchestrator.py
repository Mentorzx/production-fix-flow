"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/drivers/orchestrator.py

"""

import asyncio
import datetime
import time
from collections.abc import Callable, Iterable, Sequence
from contextvars import ContextVar
from datetime import timezone
from pathlib import Path
from typing import Any

import polars as pl

from pff.application.services import BusinessService, LineService, SequenceService
from pff.domain.audit.manifest import TaskModel
from pff.shared import ConcurrencyManager, LogReorderer, logger
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager
from pff.shared.system.resource_manager import HardwareDetector

Task = TaskModel


_ENGINE_CTX: ContextVar[SequenceService | None] = ContextVar("engine_ctx", default=None)


def _make_rotated_path(path: Path, idx: int) -> Path:
    return path.with_name(f"{path.stem}_{idx:04d}{path.suffix}")


class BufferedWriter:
    """
    AsyncBufferedWriter is a fully asynchronous buffered writer for CSV, JSONL, and Parquet files.
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
        """Execute init.



        Args:

            dest: Input value used by this callable.

            flush_rows: Optional input value.

            flush_secs: Optional input value.

            rotation: Optional input value.

            max_queue: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
        """Execute write.



        Args:

            row: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._buffer.append(row)
        time_cond = (time.time() - self._last_flush) >= self.flush_secs
        if len(self._buffer) >= self.flush_rows or (self._buffer and time_cond):
            await self._flush(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

    async def close(self) -> None:
        """Execute close.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self._buffer:
            await self._flush(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
        await self._finalize_target()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _flush(self, rows: list[Any]) -> None:
        """Execute flush.



        Args:

            rows: Input value used by this callable.

        """

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
        logger.debug(
            f"component_name=BufferedWriter message='Flushed {len(rows)} row(s) -> {self._current_target.name}'"
        )

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
        """Execute finalize target."""

        if not self._frames:
            self._file_manager.save(pl.DataFrame([]), self._current_target)
            return

        combined = pl.concat(self._frames, how="vertical")
        self._file_manager.save(combined, self._current_target)
        self._frames.clear()


class ResultCollector:
    """
    ResultCollector is a utility class for collecting result rows.
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
        """Execute init.



        Args:

            exec_id: Input value used by this callable.

            flush_rows: Optional input value.

            rotation: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_name = f"{ts}_{exec_id}.parquet"
        self._tmp_path = settings.OUTPUTS_DIR / "temp" / "result_collector" / tmp_name
        self._tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = BufferedWriter(
            self._tmp_path, flush_rows=flush_rows, rotation=rotation
        )
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
        """Execute append row.



        Args:

            msisdn: Input value used by this callable.

            request: Input value used by this callable.

            result: Input value used by this callable.

            obs: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
        """Execute has row.



        Args:

            msisdn: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return msisdn in self._seen

    async def save(
        self,
        path: str | Path | None = None,
        *,
        fmt: str | None = None,
    ) -> Path:
        """Execute save.



        Args:

            path: Optional input value.

            fmt: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        await self._writer.close()

        if not self._tmp_path.exists():
            logger.warning(
                "component_name=ResultCollector stop_reason=no_records message='No records captured; skipping save.'"
            )
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
        except Exception as exc:
            logger.debug(
                f"component_name=ResultCollector message='Failed to remove tmp file {self._tmp_path}: {exc}'"
            )

        return dest


def _get_engine() -> SequenceService:
    """Execute get engine.



    Returns:

        Return value produced by the callable.

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
    """Execute worker.



    Args:

        task: Input value used by this callable.

        collector: Input value used by this callable.

    """

    engine = _get_engine()
    msisdn = task.msisdn
    sequence = task.sequence
    payload = task.payload

    with logger.contextualize(
        msisdn=task.msisdn.split("/")[-1] if "/" in str(task.msisdn) else task.msisdn
    ):
        try:
            logger.info(
                f"component_name=orchestrator message='Iniciando execução da sequência {sequence}'"
            )
            await engine.run(msisdn, sequence, payload=payload, collector=collector)
            logger.success(
                f"component_name=orchestrator stop_reason=step_completion message='Sequência {sequence} concluída com sucesso.'"
            )
            if not collector.has_row(msisdn):
                await collector.append_row(msisdn, sequence, "Sucesso", "Executado")
        except Exception as e:
            logger.exception(
                f"component_name=orchestrator stop_reason=error message='Error processing task for MSISDN {msisdn}: {e}'"
            )
            if not collector.has_row(msisdn):
                await collector.append_row(msisdn, sequence, "Falha", str(e))


class Orchestrator:
    """Orchestrates concurrent task execution."""

    def __init__(
        self,
        exec_id: str,
        tasks: Iterable[Task],
        max_workers: int | None = None,
        resource_usage: float | None = None,
    ):
        """Execute init.



        Args:

            exec_id: Input value used by this callable.

            tasks: Input value used by this callable.

            max_workers: Optional input value.

            resource_usage: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
                estimated_task_size=5000,
            )
            max_workers = limits.optimal_workers
            logger.info(
                f"component_name=orchestrator key_parameters={{'cpu_usage': {resource_usage}}} "
                f"message='Alocação de recursos: {resource_usage:.0f}% uso -> {max_workers} workers ({limits.profile_name})'"
            )
        elif max_workers is not None:
            hardware_profile = HardwareDetector.detect()
            safe_max_workers = self._get_safe_max_workers(hardware_profile.profile_name)

            if max_workers > safe_max_workers:
                logger.warning(
                    f"component_name=orchestrator key_parameters={{'requested': {max_workers}, 'safe': {safe_max_workers}}} "
                    f"message='max_workers exceeds safe limit for {hardware_profile.profile_name}. Reducing to {safe_max_workers}'"
                )
                max_workers = safe_max_workers
            elif max_workers <= 0:
                logger.warning(
                    f"component_name=orchestrator message='max_workers invalid ({max_workers}). Using default {safe_max_workers}'"
                )
                max_workers = safe_max_workers
        else:
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
                f"component_name=orchestrator message='Alocação de recursos padrão: 90% uso -> {max_workers} workers ({limits.profile_name})'"
            )

        self.max_workers = max_workers
        self.collector = ResultCollector(exec_id=self.exec_id)
        logger.info(
            f"component_name=orchestrator key_parameters={{'tasks': {len(self.tasks)}, 'workers': {self.max_workers}}} "
            "message='Orquestrador inicializado'"
        )

    @staticmethod
    def _get_safe_max_workers(machine_name: str) -> int:
        limits = {
            "low_spec": 4,
            "mid_spec": 8,
            "high_spec": 16,
        }
        return limits.get(machine_name, 8)

    def _configure_file_logger(self) -> int:
        """Execute configure file logger.



        Returns:

            Return value produced by the callable.

        """

        ts = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
        """Execute run.



        Args:

            progress_hook: Optional input value.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if not self.tasks:
            logger.warning(
                "component_name=orchestrator stop_reason=no_tasks message='No tasks to execute.'"
            )
            return

        logger.info(
            f"component_name=orchestrator message='Iniciando orquestrador para a execução: {self.exec_id}'"
        )
        logger.info(
            f"component_name=orchestrator message='Total de tarefas: {len(self.tasks)}, Workers: {self.max_workers}'"
        )

        sink_id = self._configure_file_logger()

        try:
            worker_tasks = [(task, self.collector) for task in self.tasks]
            total = len(worker_tasks)
            done = 0

            async def _wrap_worker(task, collector):
                """Execute wrap worker.



                Args:

                    task: Input value used by this callable.

                    collector: Input value used by this callable.

                """

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
            logger.success(
                f"component_name=orchestrator stop_reason=execution_complete message='Execução concluída! Resultados salvos em: {output_path}'"
            )
        except Exception as exc:
            logger.critical(
                f"component_name=orchestrator stop_reason=error message='Catastrophic error during orchestration: {exc}'"
            )
            raise
        finally:
            logger.remove(sink_id)
            try:
                reordered_path = LogReorderer.reorder(self._log_path)
                logger.success(
                    f"component_name=orchestrator stop_reason=log_reorder message='Log reordenado salvo em: {reordered_path}'"
                )
            except Exception as e:
                logger.warning(
                    f"component_name=orchestrator message='Failed to reorder log file: {e}'"
                )

            logger.info(
                "component_name=orchestrator message='Logger de arquivo finalizado.'"
            )

    async def shutdown(self) -> None:
        """Execute shutdown.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        logger.info(
            "component_name=orchestrator message='Encerrando o orquestrador...'"
        )

        if (
            hasattr(self, "collector")
            and self.collector
            and hasattr(self.collector, "save")
        ):
            logger.info(
                "component_name=orchestrator message='Salvando resultados finais...'"
            )
            await self.collector.save()
