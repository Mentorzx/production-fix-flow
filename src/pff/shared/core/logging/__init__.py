from __future__ import annotations

from pff.shared.core.logging.config import LOG_DIR, create_isolated_logger, logger
from pff.shared.core.logging.context import TraceContext, bind_trace_id, start_span
from pff.shared.core.logging.reorderer import LogReorderer
from pff.shared.core.logging.utils import (
    catch,
    local_timestamp,
    silence_libs,
    suppress_output,
    timeit,
)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
exception = logger.exception
critical = logger.critical


FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<cyan>[{extra[task_id]:^11}]</cyan> - "
    "<level>{message}</level>"
)

__all__ = [
    "logger",
    "LOG_DIR",
    "timeit",
    "catch",
    "suppress_output",
    "silence_libs",
    "local_timestamp",
    "create_isolated_logger",
    "LogReorderer",
    "start_span",
    "bind_trace_id",
    "TraceContext",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
    "FORMAT",
]
