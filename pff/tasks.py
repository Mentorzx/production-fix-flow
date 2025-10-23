import redis
from pathlib import Path

from pff import celery_app, ManifestParser, Orchestrator, settings
from pff.utils import logger

rds = redis.Redis(
    host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
)


@celery_app.task(bind=True, name="pff.run")
async def run(self, manifest_path: str):
    """
    Execute a workflow defined in a manifest file asynchronously.
    This function is a Celery task that orchestrates the execution of tasks defined
    in a manifest file. It handles manifest parsing, execution tracking via Redis,
    progress reporting, and error handling.
    Args:
        manifest_path (str): The file system path to the manifest file that defines
            the workflow tasks to be executed.
    Returns:
        str: A message indicating the execution result. Returns a success message
            with the execution ID if completed successfully, or an error message
            if the manifest cannot be found or parsed.
    Raises:
        Exception: Re-raises any exception that occurs during orchestration after
            logging it and updating the execution status in Redis to 'failed'.
    Notes:
        - Updates execution status in Redis with keys in the format "exec:{exec_id}"
        - Tracks progress through Redis hash fields: status, progress, total, error
        - Updates Celery task state with 'PROGRESS' metadata during execution
        - Uses the ManifestParser to parse the manifest file
        - Uses the Orchestrator to execute tasks with configurable max_workers
        - Logs execution lifecycle events at various levels (info, error, critical, success)
    """

    logger.info(f"Tarefa Celery 'pff.run' iniciada com o manifesto: {manifest_path}")
    path = Path(manifest_path)

    try:
        parser = ManifestParser()
        manifest = parser.parse(path)
    except FileNotFoundError:
        logger.error(f"[Celery Task] Arquivo de manifesto não encontrado: {path}")
        return f"ERRO: Manifesto não encontrado em {path}"
    except Exception as e:
        logger.error(f"[Celery Task] Falha ao parsear o manifesto: {e}")
        return f"ERRO: Manifesto inválido: {e}"

    exec_id = manifest.execution_id
    rds.hset(
        f"exec:{exec_id}",
        mapping={"status": "running", "progress": 0, "total": len(manifest.tasks)},
    )

    def _redis_progress(done: int, total: int):
        progress_percent = int(done * 100 / total)
        rds.hset(f"exec:{exec_id}", mapping={"progress": progress_percent})
        self.update_state(
            state="PROGRESS",
            meta={"done": done, "total": total, "percent": progress_percent},
        )

    try:
        orchestrator = Orchestrator(
            exec_id=exec_id, tasks=manifest.tasks, max_workers=manifest.max_workers
        )
        await orchestrator.run(progress_hook=_redis_progress)
        rds.hset(f"exec:{exec_id}", mapping={"status": "done", "progress": 100})
        logger.success(f"[Celery Task {exec_id}] concluída com sucesso.")

        return f"Execução '{exec_id}' finalizada."

    except Exception as e:
        logger.critical(f"[Celery Task {exec_id}] falhou catastroficamente: {e}")
        rds.hset(f"exec:{exec_id}", mapping={"status": "failed", "error": str(e)})

        raise
