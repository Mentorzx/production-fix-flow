from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import polars as pl
import redis
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from pff.drivers.api.models import ExecutionResponse, ExecutionStatus
from pff.drivers.celery.tasks import run
from pff.shared import CacheManager, ConcurrencyManager, FileManager, logger
from pff.shared.core.config import get_redis_client, settings
from pff.shared.core.file_manager import ParquetBundle

"""
Executions router for managing sequence executions.

This module provides endpoints for creating, monitoring, and retrieving
execution results. All executions are processed asynchronously using Celery.
"""

router = APIRouter(prefix="/executions", tags=["executions"])

file_manager = FileManager()
cache_manager = CacheManager()
concurrency_manager = ConcurrencyManager()
_rds: redis.Redis | None = None


def _get_rds() -> redis.Redis:
    """Lazy Redis client for execution tracking."""
    global _rds
    if _rds is None:
        db_idx = getattr(settings, "REDIS_DB_EXECUTIONS", 5)
        _rds = get_redis_client(db=db_idx, decode_responses=True)
    if _rds is None:
        raise RuntimeError("Failed to initialize Redis client")
    return cast(redis.Redis, _rds)


OUTPUT_DIR = Path(settings.OUTPUTS_DIR)
LOG_DIR = Path(settings.LOGS_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ExecutionRequest(BaseModel):
    """Request model for creating new execution"""

    sequence_name: str = Field(..., description="Name of sequence to execute")
    lines: list[dict[str, Any]] = Field(
        ..., description="List of lines/MSISDNs to process"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters"
    )


class ExecutionDetailResponse(ExecutionResponse):
    """Extended execution response with details"""

    progress: int = 0
    current_step: str | None = None
    total_steps: int | None = None
    start_time: str | None = None
    end_time: str | None = None
    error_message: str | None = None
    output_files: list[str] = Field(default_factory=list)


@router.post("/", response_model=ExecutionResponse, status_code=202)
async def run_sequence(
    file: UploadFile | None = File(default=None),
    sequence_name: str = Query(..., description="Sequence name to execute"),
):
    """
    Run a sequence by accepting an uploaded Excel file.

    Args:
        file: Excel file containing lines to process
        sequence_name: Name of the sequence to execute

    Returns:
        ExecutionResponse with execution_id and initial status

    Raises:
        HTTPException: If no input data provided
    """
    exec_id = uuid4().hex
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")

    logger.info(
        f"component_name=api_executions key_parameters={{'exec_id': '{exec_id}', 'sequence': '{sequence_name}'}} message='Criando nova execução'"
    )

    _get_rds().hset(
        f"exec:{exec_id}",
        mapping={
            "status": "queued",
            "progress": 0,
            "sequence_name": sequence_name,
            "start_time": ts,
        },
    )

    if file:
        content = await file.read()
        input_path = OUTPUT_DIR / f"{ts}-{exec_id}-input.xlsx"

        await file_manager.async_save(content, input_path)

        logger.success(
            f"component_name=api_executions stop_reason=file_saved message='Arquivo salvo para processamento: {input_path}'"
        )

        await run.delay(exec_id, str(input_path), ts, sequence_name)

    else:
        logger.error("No file provided")
        return JSONResponse(
            status_code=400, content={"detail": "No input data provided"}
        )

    return {"execution_id": exec_id, "status": ExecutionStatus.queued}


@router.post("/batch", response_model=ExecutionResponse, status_code=202)
async def run_batch_sequence(
    request: ExecutionRequest,
):
    """
    Run a sequence with JSON payload containing lines and sequence name.

    Args:
        request: ExecutionRequest containing sequence name, lines and parameters

    Returns:
        ExecutionResponse with execution_id and initial status
    """
    exec_id = uuid4().hex
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")

    logger.info(
        f"component_name=api_executions key_parameters={{'exec_id': '{exec_id}', 'sequence': '{request.sequence_name}'}} "
        f"message='Execução batch criada ({len(request.lines)} linhas)'"
    )

    _get_rds().hset(
        f"exec:{exec_id}",
        mapping={
            "status": "queued",
            "progress": 0,
            "sequence_name": request.sequence_name,
            "start_time": ts,
            "total_lines": len(request.lines),
        },
    )

    df = pl.DataFrame(request.lines)
    input_file = OUTPUT_DIR / f"{ts}-{exec_id}-input.parquet"
    await file_manager.async_save(df, input_file)

    if request.parameters:
        cache_manager.set(f"exec_params:{exec_id}", request.parameters, ttl=86400)

    await run.delay(
        exec_id, request.lines, ts, request.sequence_name, request.parameters
    )

    return {"execution_id": exec_id, "status": ExecutionStatus.queued}


@router.get("/{exec_id}", response_model=ExecutionDetailResponse)
async def get_status(
    exec_id: str,
):
    """
    Retrieve the detailed status and progress of an execution by its ID.

    Args:
        exec_id: Unique execution identifier

    Returns:
        ExecutionDetailResponse with full execution details

    Raises:
        HTTPException: If execution not found
    """
    cached = cache_manager.get(f"exec_detail:{exec_id}")
    if cached:
        return ExecutionDetailResponse(**cached)

    exec_data = cast(dict[str, str], _get_rds().hgetall(f"exec:{exec_id}"))
    if not exec_data:
        logger.warning(f"Execution not found: {exec_id}")
        raise HTTPException(status_code=404, detail="Execution not found")

    output_files = []
    allowed_exts = {".xlsx", ".parquet", ".json"}
    for file in OUTPUT_DIR.glob(f"*{exec_id}*"):
        try:
            FileManager.assert_supported_path(file, allowed_exts=allowed_exts)
        except ValueError:
            continue
        output_files.append(file.name)

    response = ExecutionDetailResponse(
        execution_id=exec_id,
        status=ExecutionStatus(exec_data.get("status", "unknown")),
        progress=int(exec_data.get("progress", 0)),
        current_step=exec_data.get("current_step"),
        total_steps=(
            int(exec_data.get("total_steps", 0)) if "total_steps" in exec_data else None
        ),
        start_time=exec_data.get("start_time"),
        end_time=exec_data.get("end_time"),
        error_message=exec_data.get("error"),
        output_files=output_files,
    )

    if response.status in [ExecutionStatus.done, ExecutionStatus.error]:
        cache_manager.set(f"exec_detail:{exec_id}", response.model_dump(), ttl=3600)

    return response


@router.get("/{exec_id}/status")
def get_simple_status(
    exec_id: str,
):
    """
    Get simple status of execution (running, completed, failed).

    Args:
        exec_id: Unique execution identifier

    Returns:
        Simple status dict with execution_id, status, progress and is_running flag

    Raises:
        HTTPException: If execution not found
    """
    status_data = cast(str | None, _get_rds().hget(f"exec:{exec_id}", "status"))
    if not status_data:
        raise HTTPException(status_code=404, detail="Execution not found")

    progress_data = cast(str | None, _get_rds().hget(f"exec:{exec_id}", "progress"))

    return {
        "execution_id": exec_id,
        "status": status_data,
        "progress": int(progress_data) if progress_data else 0,
        "is_running": status_data in ["queued", "running"],
    }


@router.get("/{exec_id}/log", response_class=StreamingResponse)
async def download_log(
    exec_id: str,
):
    """
    Download execution log file.

    Args:
        exec_id: Unique execution identifier

    Returns:
        StreamingResponse with log file content

    Raises:
        HTTPException: If log file not found
    """
    log_files = list(LOG_DIR.glob(f"*{exec_id}*.log"))
    if not log_files:
        logger.error(f"Log file not found for execution {exec_id}")
        raise HTTPException(status_code=404, detail="Log file not found")

    log_file = log_files[0]

    async def iterfile():
        content = await file_manager.async_read(log_file)
        yield content

    return StreamingResponse(
        iterfile(),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={log_file.name}"},
    )


@router.get("/{exec_id}/excel", response_class=FileResponse)
async def download_excel(
    exec_id: str,
):
    """
    Download execution result as Excel file.

    Args:
        exec_id: Unique execution identifier

    Returns:
        FileResponse with Excel file

    Raises:
        HTTPException: If Excel file cannot be found or generated
    """
    cached_path = cache_manager.get(f"excel_path:{exec_id}")
    if cached_path and Path(cached_path).exists():
        return FileResponse(
            path=cached_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"execution_{exec_id}.xlsx",
        )

    excel_files = list(OUTPUT_DIR.glob(f"*{exec_id}*.xlsx"))
    if not excel_files:
        parquet_files = list(OUTPUT_DIR.glob(f"*{exec_id}*output.parquet"))
        if parquet_files:
            logger.info(
                f"component_name=api_executions message='Gerando Excel a partir do Parquet para execução {exec_id}'"
            )
            payload = file_manager.read(parquet_files[0])
            if isinstance(payload, ParquetBundle):
                if payload.parsed_kind != "tabular":
                    logger.error(
                        f"Unsupported output format for execution: {parquet_files[0]}"
                    )
                    raise HTTPException(
                        status_code=400, detail="Unsupported output format"
                    )
                df = payload.lazyframe().collect(engine="streaming")
            else:
                df = payload
            excel_path = OUTPUT_DIR / f"{exec_id}_result.xlsx"
            file_manager.save(df, excel_path)
            excel_files = [excel_path]
        else:
            logger.error(f"Excel file not found for execution {exec_id}")
            raise HTTPException(status_code=404, detail="Excel file not found")

    cache_manager.set(f"excel_path:{exec_id}", str(excel_files[0]), ttl=3600)

    return FileResponse(
        path=excel_files[0],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"execution_{exec_id}.xlsx",
    )


@router.get("/{exec_id}/output", response_class=FileResponse)
async def download_output(
    exec_id: str,
    fmt: str = Query("xlsx", pattern="^(xlsx|json|parquet)$"),
):
    """
    Download execution output in specified format.

    Args:
        exec_id: Unique execution identifier
        fmt: Output format (xlsx, json, parquet)

    Returns:
        FileResponse with output file in requested format

    Raises:
        HTTPException: If output file not found
    """
    output_files = list(OUTPUT_DIR.glob(f"*{exec_id}*output*"))
    if not output_files:
        logger.error(f"Output not found for execution {exec_id}")
        raise HTTPException(status_code=404, detail="Output file not found")

    source_file = output_files[0]
    output_path = OUTPUT_DIR / f"{exec_id}_output.{fmt}"

    if not FileManager.same_extension(source_file, output_path):
        bundle = file_manager.read(source_file)
        file_manager.export(bundle, output_path)
    else:
        output_path = source_file

    media_types = {
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "json": "application/json",
        "parquet": "application/octet-stream",
    }

    return FileResponse(
        path=output_path,
        media_type=media_types[fmt],
        filename=f"execution_{exec_id}.{fmt}",
    )


@router.delete("/{exec_id}")
async def cancel_execution(
    exec_id: str,
):
    """
    Cancel a running execution.

    Args:
        exec_id: Unique execution identifier

    Returns:
        Success message

    Raises:
        HTTPException: If execution not found or cannot be cancelled
    """
    status_data = cast(str | None, _get_rds().hget(f"exec:{exec_id}", "status"))
    if not status_data:
        raise HTTPException(status_code=404, detail="Execution not found")

    current_status = status_data
    if current_status not in ["queued", "running"]:
        logger.warning(
            f"Attempt to cancel execution {exec_id} with status {current_status}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel execution with status: {current_status}",
        )

    _get_rds().hset(
        f"exec:{exec_id}",
        mapping={
            "status": "cancelled",
            "end_time": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%S"
            ),
        },
    )

    try:
        from celery.result import AsyncResult

        task = AsyncResult(exec_id)
        task.revoke(terminate=True)
    except Exception as e:
        logger.warning(f"Failed to revoke Celery task: {e}")

    logger.info(f"Execucao {exec_id} cancelada")

    return {"message": f"Execution {exec_id} cancelled successfully"}


@router.get("/{exec_id}/events")
async def stream_events(
    exec_id: str,
):
    """
    Stream execution progress updates via Server-Sent Events.

    Args:
        exec_id: Unique execution identifier

    Returns:
        StreamingResponse with Server-Sent Events
    """

    async def event_generator():
        last_progress = -1
        while True:
            exec_data = cast(dict[str, str], _get_rds().hgetall(f"exec:{exec_id}"))
            if not exec_data:
                yield f"data: {FileManager.json_dumps({'error': 'Execution not found'})}\n\n"
                break

            status = exec_data.get("status", "unknown")
            progress = int(exec_data.get("progress", 0))

            if progress != last_progress:
                event_data = {
                    "execution_id": exec_id,
                    "status": status,
                    "progress": progress,
                    "current_step": exec_data.get("current_step"),
                }
                yield f"data: {FileManager.json_dumps(event_data)}\n\n"
                last_progress = progress

            if status in ["done", "error", "cancelled"]:
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
