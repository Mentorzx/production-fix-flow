import threading
from datetime import datetime, timezone
from typing import Callable, Iterable

from pff import TaskModel, settings
from pff.services import BusinessService, LineService, SequenceService
from pff.utils import (
    ConcurrencyManager,
    LogReorderer,
    ResultCollector,
    logger,
)
from pff.utils.hardware_detector import HardwareDetector

_THREAD_STATE = threading.local()
Task = TaskModel


def _get_engine() -> SequenceService:
    """
    Initializes and retrieves a thread-local instance of SequenceService.

    If the engine does not exist in the current thread's state, this function creates
    new instances of LineService and BusinessService, assigns them to the thread state,
    and then creates a SequenceService using these instances. The engine is then stored
    in the thread-local state for future retrievals.

    Returns:
        SequenceService: The thread-local instance of SequenceService.
    """
    if not hasattr(_THREAD_STATE, "engine"):
        svc = LineService()
        validator = BusinessService()
        services = {"line": svc, "validator": validator}
        _THREAD_STATE.services = services
        _THREAD_STATE.engine = SequenceService(services)
    return _THREAD_STATE.engine


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
        msisdn=task.msisdn.split('/')[-1] if '/' in str(task.msisdn) else task.msisdn
    ):
        try:
            logger.info("Iniciando execução da sequência '{}'", sequence)
            await engine.run(msisdn, sequence, payload=payload, collector=collector)
            logger.success("Sequência '{}' concluída com sucesso.", sequence)
            if not collector.has_row(msisdn):
                await collector.append_row(msisdn, sequence, "Sucesso", "Executado")
        except Exception as e:
            logger.exception(
                "Erro ao processar a tarefa para o MSISDN {}: {}", msisdn, e
            )
            if not collector.has_row(msisdn):
                await collector.append_row(msisdn, sequence, "Falha", str(e))


class Orchestrator:
    """
    Orchestrator is responsible for managing and executing a batch of tasks concurrently, collecting their results, and logging the execution process.
    Attributes:
        exec_id (str): Unique identifier for the execution batch.
        tasks (List[Task]): List of tasks to be executed.
        max_workers (int): Maximum number of concurrent workers.
        collector (ResultCollector): Collector for storing task results.
    Methods:
        __init__(exec_id: str, tasks: Iterable[Task], max_workers: int):
            Initializes the Orchestrator with the given execution ID, tasks, and worker count.
        _configure_file_logger() -> int:
            Configures and adds a file logger for the current execution, returning the logger sink ID.
        run():
            Executes the batch of tasks end-to-end, managing concurrency, collecting results, and handling logging.
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

        from pff.utils.resource_manager import get_resource_manager

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
                f"📊 Resource allocation: {resource_usage:.0f}% usage → "
                f"{max_workers} workers ({limits.profile_name})"
            )
        elif max_workers is not None:
            # LEGACY: Use fixed max_workers (deprecated, but supported)
            hardware_profile = HardwareDetector.detect()
            safe_max_workers = self._get_safe_max_workers(hardware_profile.profile_name)

            if max_workers > safe_max_workers:
                logger.warning(
                    f"⚠️  max_workers={max_workers} exceeds safe limit for {hardware_profile.profile_name}. "
                    f"Reducing to {safe_max_workers} (RAM: {hardware_profile.total_ram_gb:.1f} GB, "
                    f"CPU: {hardware_profile.cpu_threads} threads)"
                )
                max_workers = safe_max_workers
            elif max_workers <= 0:
                logger.warning(
                    f"⚠️  max_workers={max_workers} invalid. Using default {safe_max_workers}"
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
                f"📊 Default resource allocation: 90% usage → "
                f"{max_workers} workers ({limits.profile_name})"
            )

        self.max_workers = max_workers
        self.collector = ResultCollector(exec_id=self.exec_id)
        logger.info(
            f"Orchestrator initialized: {len(self.tasks)} tasks, "
            f"{self.max_workers} workers"
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
            logger.warning("Nenhuma tarefa para executar.")
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
                logger.warning(f"Falha ao tentar reordenar o arquivo de log: {e}")

            logger.info("Logger de arquivo finalizado.")

    async def shutdown(self) -> None:
        """
        Asynchronously shuts down the orchestrator, logging the shutdown process.
        If a collector with a 'save' method exists, saves the final results before shutdown.
        Returns:
            None
        """
        logger.info("Encerrando o orquestrador...")

        if (
            hasattr(self, "collector")
            and self.collector
            and hasattr(self.collector, "save")
        ):
            logger.info("Salvando resultados finais...")
            await self.collector.save()
