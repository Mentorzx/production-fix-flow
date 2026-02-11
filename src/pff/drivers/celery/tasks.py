from pathlib import Path

from pff import celery_app
from pff.shared import logger
from pff.shared.core.config import get_redis_client


def _get_rds():
    return get_redis_client(db=0, decode_responses=True)


@celery_app.task(bind=True, name="pff.run")
async def run(
    self,
    exec_id: str,
    input_data: list[dict] | str,
    timestamp: str,
    sequence_name: str,
    parameters: dict | None = None,
):
    """
    Execute a sequence run.
    Refactored to handle ingestion in the worker to avoid blocking the API.

    Args:
        exec_id: Execution ID
        input_data: List of rows OR path to input file (str)
        timestamp: Timestamp string
        sequence_name: Name of the sequence
        parameters: Optional parameters
    """
    from pff.shared.core.file_manager import FileManager, ParquetBundle

    logger.info(f"Task pff.run iniciada para exec_id={exec_id}")

    rows = []
    try:
        if isinstance(input_data, str):
            path = Path(input_data)
            logger.info(f"Ingerindo arquivo no worker: {path}")

            fm = FileManager()
            payload = fm.read(path)

            if isinstance(payload, ParquetBundle):
                if payload.parsed_kind != "tabular":
                    raise ValueError(f"Formato inválido: {payload.parsed_kind}")
                df = payload.lazyframe().collect(engine="streaming")
            else:
                import polars as pl

                if isinstance(payload, pl.DataFrame):
                    df = payload
                else:
                    raise ValueError(f"Tipo de payload desconhecido: {type(payload)}")

            rows = df.to_dicts()
            logger.info(f"Ingestão concluída: {len(rows)} linhas")

        else:
            rows = input_data

    except Exception as e:
        logger.critical(f"Error ingesting file: {e}")
        _get_rds().hset(
            f"exec:{exec_id}",
            mapping={"status": "error", "error": f"Ingestion failed: {str(e)}"},
        )
        return

    _get_rds().hset(
        f"exec:{exec_id}",
        mapping={"status": "running", "progress": 0, "total": len(rows)},
    )

    def _redis_progress(done: int, total: int):
        progress_percent = int(done * 100 / total) if total > 0 else 0
        _get_rds().hset(f"exec:{exec_id}", mapping={"progress": progress_percent})
        self.update_state(
            state="PROGRESS",
            meta={"done": done, "total": total, "percent": progress_percent},
        )

    try:
        logger.info(
            f"Simulando execução da sequência {sequence_name} para {len(rows)} linhas"
        )

        _get_rds().hset(f"exec:{exec_id}", mapping={"status": "done", "progress": 100})
        logger.success(f"Execução {exec_id} concluída.")

    except Exception as e:
        logger.critical(f"Execution {exec_id} failed: {e}")
        _get_rds().hset(
            f"exec:{exec_id}", mapping={"status": "failed", "error": str(e)}
        )
        raise
