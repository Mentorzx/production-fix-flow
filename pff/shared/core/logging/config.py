from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from loguru import logger as _loguru_logger
from rich.logging import RichHandler
from rich.traceback import install as rich_tb_install

from pff.shared.core.config import settings
from pff.shared.core.logging.context import TraceContext
from pff.shared.core.logging.masking import mask_secrets

_loguru_logger.remove()

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def patcher(record):
    """
    1. Inject Trace ID / Span ID from ContextVars
    2. Ensure task_id exists
    3. Mask secrets in message
    """

    ctx = TraceContext.get()
    if ctx["trace_id"]:
        record["extra"]["trace_id"] = ctx["trace_id"]
    if ctx["span_id"]:
        record["extra"]["span_id"] = ctx["span_id"]

    if "task_id" not in record["extra"]:
        record["extra"]["task_id"] = "MAIN"

    record["message"] = mask_secrets(record["message"])


_loguru_logger.configure(patcher=patcher)


_IS_TTY = "DISABLE_RICH" not in os.environ and sys.stderr.isatty()
if _IS_TTY:
    rich_tb_install(show_locals=False, theme=os.getenv("RICH_THEME", "monokai"))
    _loguru_logger.add(
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            highlighter=None,
        ),
        level=_LEVEL,
        format="{message}",
    )

elif os.getenv("LOG_TO_STDOUT") == "1":
    _loguru_logger.add(sys.stdout, level=_LEVEL, serialize=True)

else:
    FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<cyan>[{extra[task_id]:^11}]</cyan> - "
        "<level>{message}</level>"
    )
    _loguru_logger.add(sys.stderr, level=_LEVEL, format=FORMAT)


LOG_DIR = Path(os.getenv("LOG_DIR", settings.LOGS_DIR)).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    _loguru_logger.add(
        LOG_DIR / "{time:YYYY-MM-DD}.log",
        level=os.getenv("FILE_LOG_LEVEL", "DEBUG"),
        rotation=os.getenv("LOG_ROTATION", "100 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        enqueue=True,
        backtrace=False,
        serialize=True,
    )
except PermissionError:
    pass

logger = _loguru_logger


def create_isolated_logger(name: str, log_dir: Path | None = None):
    """Create an isolated logger with its own file sink."""
    from datetime import datetime, timezone

    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.log"

    isolated = logger.bind(component=name)
    logger.add(
        log_path,
        filter=lambda record: record["extra"].get("component") == name,
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
        serialize=True,
    )
    return isolated
