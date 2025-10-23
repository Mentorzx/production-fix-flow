from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import orjson
import polars as pl
import redis
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from pff.api.models import ExecutionResponse, ExecutionStatus
from pff.config import settings
from pff.tasks import run
from pff.utils import CacheManager, ConcurrencyManager, FileManager, logger

# from pff.api.auth import get_current_user

"""
Executions router for managing sequence executions.

This module provides endpoints for creating, monitoring, and retrieving
execution results. All executions are processed asynchronously using Celery.
"""

router = APIRouter(prefix="/executions", tags=["executions"])

# Initialize utilities
file_manager = FileManager()
cache_manager = CacheManager()
concurrency_manager = ConcurrencyManager()
rds = redis.Redis(
    host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=5, decode_responses=True
)

# Get paths from settings
OUTPUT_DIR = Path(settings.OUTPUTS_DIR)
LOG_DIR = Path(settings.LOGS_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Models ─────────────────────────────────────────────────────────────────
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


# ──────────────── POST ─────────────────────────────────────────────────────
@router.post("/", response_model=ExecutionResponse, status_code=202)
async def run_sequence(
    file: UploadFile | None = File(default=None),
    sequence_name: str = Query(..., description="Sequence name to execute"),
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
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

    logger.info(f"Criando nova execução {exec_id} para sequência {sequence_name}")

    # Store initial status in Redis
    rds.hset(
        f"exec:{exec_id}",
        mapping={
            "status": "queued",
            "progress": 0,
            "sequence_name": sequence_name,
            "start_time": ts,
        },
    )

    if file:
        # Read Excel file content
        content = await file.read()
        input_path = OUTPUT_DIR / f"{ts}-{exec_id}-input.xlsx"

        # Save raw bytes to file
        input_path.write_bytes(content)

        # Read Excel with polars
        df = pl.read_excel(input_path)
        logger.success(f"Arquivo Excel carregado: {len(df)} linhas")
    else:
        logger.error("Nenhum arquivo fornecido")
        return JSONResponse(
            status_code=400, content={"detail": "No input data provided"}
        )

    # Save input as parquet for efficiency
    parquet_path = OUTPUT_DIR / f"{ts}-{exec_id}-input.parquet"
    file_manager.save(df, parquet_path)

    # Dispatch async task
    rows = df.to_dicts()
    await run.delay(exec_id, rows, ts, sequence_name)  # type: ignore[attr-defined]

    return {"execution_id": exec_id, "status": ExecutionStatus.queued}


@router.post("/batch", response_model=ExecutionResponse, status_code=202)
async def run_batch_sequence(
    request: ExecutionRequest,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Run a sequence with JSON payload containing lines and sequence name.

    This is the main endpoint for programmatic access, allowing batch
    processing of multiple lines with a specified sequence.

    Args:
        request: ExecutionRequest containing sequence name, lines and parameters

    Returns:
        ExecutionResponse with execution_id and initial status
    """
    exec_id = uuid4().hex
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")

    logger.info(
        f"Execução batch criada: {exec_id}, sequência: {request.sequence_name}, {len(request.lines)} linhas"
    )

    # Store execution metadata in Redis
    rds.hset(
        f"exec:{exec_id}",
        mapping={
            "status": "queued",
            "progress": 0,
            "sequence_name": request.sequence_name,
            "start_time": ts,
            "total_lines": len(request.lines),
        },
    )

    # Save input data
    df = pl.DataFrame(request.lines)
    input_file = OUTPUT_DIR / f"{ts}-{exec_id}-input.parquet"
    file_manager.save(df, input_file)

    # Cache execution parameters if provided
    if request.parameters:
        cache_manager.set(f"exec_params:{exec_id}", request.parameters, ttl=86400)

    # Dispatch task
    await run.delay(exec_id, request.lines, ts, request.sequence_name, request.parameters)  # type: ignore[attr-defined]

    return {"execution_id": exec_id, "status": ExecutionStatus.queued}


# ──────────────── GET ───────────────────────────────────────────────────────
@router.get("/{exec_id}", response_model=ExecutionDetailResponse)
async def get_status(
    exec_id: str,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Retrieve the detailed status and progress of an execution by its ID.

    Provides comprehensive information about the execution including
    current step, progress, timing, and output files.

    Args:
        exec_id: Unique execution identifier

    Returns:
        ExecutionDetailResponse with full execution details

    Raises:
        HTTPException: If execution not found
    """
    # Try cache first
    cached = cache_manager.get(f"exec_detail:{exec_id}")
    if cached:
        return ExecutionDetailResponse(**cached)

    exec_data = cast(dict[str, str], rds.hgetall(f"exec:{exec_id}"))
    if not exec_data:
        logger.warning(f"Execução não encontrada: {exec_id}")
        raise HTTPException(status_code=404, detail="Execution not found")

    # Find output files
    output_files = []
    for file in OUTPUT_DIR.glob(f"*{exec_id}*"):
        if file.suffix in [".xlsx", ".parquet", ".json"]:
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

    # Cache if completed
    if response.status in [ExecutionStatus.done, ExecutionStatus.error]:
        cache_manager.set(f"exec_detail:{exec_id}", response.model_dump(), ttl=3600)

    return response


@router.get("/{exec_id}/status")
def get_simple_status(
    exec_id: str,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Get simple status of execution (running, completed, failed).

    Lightweight endpoint for polling execution status without
    retrieving full details.

    Args:
        exec_id: Unique execution identifier

    Returns:
        Simple status dict with execution_id, status, progress and is_running flag

    Raises:
        HTTPException: If execution not found
    """
    status_data = cast(str | None, rds.hget(f"exec:{exec_id}", "status"))
    if not status_data:
        raise HTTPException(status_code=404, detail="Execution not found")

    progress_data = cast(str | None, rds.hget(f"exec:{exec_id}", "progress"))

    return {
        "execution_id": exec_id,
        "status": status_data,
        "progress": int(progress_data) if progress_data else 0,
        "is_running": status_data in ["queued", "running"],
    }


@router.get("/{exec_id}/log", response_class=StreamingResponse)
async def download_log(
    exec_id: str,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Download execution log file.

    Streams the log file for the specified execution.

    Args:
        exec_id: Unique execution identifier

    Returns:
        StreamingResponse with log file content

    Raises:
        HTTPException: If log file not found
    """
    # Find log file
    log_files = list(LOG_DIR.glob(f"*{exec_id}*.log"))
    if not log_files:
        logger.error(f"Log não encontrado para execução {exec_id}")
        raise HTTPException(status_code=404, detail="Log file not found")

    log_file = log_files[0]

    async def iterfile():
        """Stream file content"""
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
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Download execution result as Excel file.

    If Excel file doesn't exist, attempts to generate it from
    parquet output file.

    Args:
        exec_id: Unique execution identifier

    Returns:
        FileResponse with Excel file

    Raises:
        HTTPException: If Excel file cannot be found or generated
    """
    # Check cache for file path
    cached_path = cache_manager.get(f"excel_path:{exec_id}")
    if cached_path and Path(cached_path).exists():
        return FileResponse(
            path=cached_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"execution_{exec_id}.xlsx",
        )

    # Find Excel file
    excel_files = list(OUTPUT_DIR.glob(f"*{exec_id}*.xlsx"))
    if not excel_files:
        # Try to generate from parquet
        parquet_files = list(OUTPUT_DIR.glob(f"*{exec_id}*output.parquet"))
        if parquet_files:
            logger.info(f"Gerando Excel a partir do Parquet para execução {exec_id}")
            df = file_manager.read(parquet_files[0])
            excel_path = OUTPUT_DIR / f"{exec_id}_result.xlsx"
            file_manager.save(df, excel_path)
            excel_files = [excel_path]
        else:
            logger.error(f"Arquivo Excel não encontrado para execução {exec_id}")
            raise HTTPException(status_code=404, detail="Excel file not found")

    # Cache the path
    cache_manager.set(f"excel_path:{exec_id}", str(excel_files[0]), ttl=3600)

    return FileResponse(
        path=excel_files[0],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"execution_{exec_id}.xlsx",
    )


@router.get("/{exec_id}/output", response_class=FileResponse)
async def download_output(
    exec_id: str,
    fmt: str = Query("xlsx", regex="^(xlsx|json|parquet)$"),
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Download execution output in specified format.

    Supports xlsx, json and parquet formats. Converts between
    formats as needed.

    Args:
        exec_id: Unique execution identifier
        fmt: Output format (xlsx, json, parquet)

    Returns:
        FileResponse with output file in requested format

    Raises:
        HTTPException: If output file not found
    """
    # Find output file
    output_files = list(OUTPUT_DIR.glob(f"*{exec_id}*output*"))
    if not output_files:
        logger.error(f"Output não encontrado para execução {exec_id}")
        raise HTTPException(status_code=404, detail="Output file not found")

    source_file = output_files[0]
    output_path = OUTPUT_DIR / f"{exec_id}_output.{fmt}"

    # Convert format if needed
    if output_path.suffix != source_file.suffix:
        df = file_manager.read(source_file)
        file_manager.save(df, output_path)
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
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Cancel a running execution.

    Only executions with status 'queued' or 'running' can be cancelled.

    Args:
        exec_id: Unique execution identifier

    Returns:
        Success message

    Raises:
        HTTPException: If execution not found or cannot be cancelled
    """
    status_data = cast(str | None, rds.hget(f"exec:{exec_id}", "status"))
    if not status_data:
        raise HTTPException(status_code=404, detail="Execution not found")

    current_status = status_data
    if current_status not in ["queued", "running"]:
        logger.warning(
            f"Tentativa de cancelar execução {exec_id} com status {current_status}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel execution with status: {current_status}",
        )

    # Update status
    rds.hset(
        f"exec:{exec_id}",
        mapping={
            "status": "cancelled",
            "end_time": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%S"
            ),
        },
    )

    # Revoke Celery task if possible
    try:
        from celery.result import AsyncResult

        task = AsyncResult(exec_id)
        task.revoke(terminate=True)
    except Exception as e:
        logger.warning(f"Não foi possível revogar task Celery: {e}")

    logger.info(f"Execução {exec_id} cancelada")

    return {"message": f"Execution {exec_id} cancelled successfully"}


# ──────────────── Server-Sent Events ────────────────────────────────────────
@router.get("/{exec_id}/events")
async def stream_events(
    exec_id: str,
    # user: Annotated[dict, Depends(get_current_user)]  # Temporariamente comentado
):
    """
    Stream execution progress updates via Server-Sent Events.

    Provides real-time updates on execution progress. Client should
    handle connection errors and reconnect as needed.

    Args:
        exec_id: Unique execution identifier

    Returns:
        StreamingResponse with Server-Sent Events
    """

    async def event_generator():
        """Generate SSE events for execution progress"""
        last_progress = -1
        while True:
            exec_data = cast(dict[str, str], rds.hgetall(f"exec:{exec_id}"))
            if not exec_data:
                yield f"data: {orjson.dumps({'error': 'Execution not found'}).decode()}\n\n"
                break

            status = exec_data.get("status", "unknown")
            progress = int(exec_data.get("progress", 0))

            # Send update only if changed
            if progress != last_progress:
                event_data = {
                    "execution_id": exec_id,
                    "status": status,
                    "progress": progress,
                    "current_step": exec_data.get("current_step"),
                }
                yield f"data: {orjson.dumps(event_data).decode()}\n\n"
                last_progress = progress

            # Stop if execution finished
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
