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
_EXCLUDED_COMPONENTS = {"hpo_dashboard"}


def _human_formatter(record: dict) -> str:
    """Compact human-readable format that omits empty/default fields."""
    extra = record["extra"]
    ts = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    level = f"{record['level'].name:8}"
    comp = extra.get("component_name") or record["name"]
    msg = record["message"]

    parts = [f"{ts} | {level} | {comp} | {msg}"]

    stop = extra.get("stop_reason")
    if stop and stop != "unspecified":
        parts.append(f" | stop={stop}")
    kp = extra.get("key_parameters")
    if kp:
        parts.append(f" | params={kp}")
    parts.append("\n")
    return "".join(parts)


_HUMAN_FORMAT = _human_formatter


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

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


if os.environ.get("PFF_CLEAN_MODE") == "1":
    logging.basicConfig(
        handlers=[logging.NullHandler()], level=logging.CRITICAL, force=True
    )
else:
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def patcher(record):
    """
    1. Inject Trace ID / Span ID from ContextVars
    2. Ensure task_id exists
    3. Mask secrets in message
    """

    extra = record["extra"]
    ctx = TraceContext.get()
    extra.setdefault("trace_id", ctx["trace_id"])
    extra.setdefault("span_id", ctx["span_id"])
    if "task_id" not in extra:
        extra["task_id"] = "MAIN"
    extra.setdefault("timestamp", record["time"].isoformat())
    extra.setdefault("component_name", extra.get("component") or record["name"])
    extra.setdefault("key_parameters", {})
    extra.setdefault("stop_reason", "unspecified")

    record["message"] = mask_secrets(record["message"])


_loguru_logger.configure(patcher=patcher)


_IS_TTY = "DISABLE_RICH" not in os.environ and sys.stderr.isatty()
if os.environ.get("PFF_CLEAN_MODE") == "1":
    _loguru_logger.add(sys.stderr, level=_LEVEL, format="{message}")
elif _IS_TTY:
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
    _loguru_logger.add(
        sys.stdout,
        level=_LEVEL,
        format=_HUMAN_FORMAT,  # type: ignore[arg-type]
        colorize=False,
        serialize=False,
    )

else:
    FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<cyan>[{extra[task_id]:^11}]</cyan> - "
        "<level>{message}</level>"
    )
    if os.environ.get("PFF_CLEAN_MODE") != "1":
        _loguru_logger.add(sys.stderr, level=_LEVEL, format=FORMAT)


LOG_DIR = Path(os.getenv("LOG_DIR", settings.LOGS_DIR)).expanduser()


def _exclude_component(record) -> bool:
    component = record["extra"].get("component")
    return component not in _EXCLUDED_COMPONENTS


if os.environ.get("PFF_CLEAN_MODE") != "1":
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
            filter=_exclude_component,
        )
    except PermissionError:
        pass
    _HUMAN_DIR = LOG_DIR / "readable"
    _HUMAN_DIR.mkdir(parents=True, exist_ok=True)

    def _level_filter(levels: set[str]):
        return (
            lambda record, allowed=levels: _exclude_component(record)
            and record["level"].name in allowed
        )

    _LEVEL_SINKS = {
        "debug": ({"DEBUG"}, "DEBUG"),
        "info": ({"INFO"}, "INFO"),
        "success": ({"SUCCESS"}, "SUCCESS"),
        "warning": ({"WARNING"}, "WARNING"),
        "error": ({"ERROR", "CRITICAL"}, "ERROR"),
    }
    for suffix, (levels, min_level) in _LEVEL_SINKS.items():
        _loguru_logger.add(
            _HUMAN_DIR / f"{{time:YYYY-MM-DD}}.{suffix}.log",
            level=min_level,
            format=_HUMAN_FORMAT,  # type: ignore[arg-type]
            filter=_level_filter(levels),
            colorize=False,
            rotation=os.getenv("LOG_ROTATION", "100 MB"),
            retention=os.getenv("LOG_RETENTION", "30 days"),
            compression=os.getenv("LOG_COMPRESSION", "zip"),
            enqueue=True,
            backtrace=False,
            serialize=False,
        )

    _loguru_logger.add(
        _HUMAN_DIR / "{time:YYYY-MM-DD}.combined.log",
        level="INFO",  # type: ignore[arg-type]
        format=_HUMAN_FORMAT,  # type: ignore[arg-type]
        filter=_exclude_component,
        colorize=False,
        rotation=os.getenv("LOG_ROTATION", "100 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        enqueue=True,
        backtrace=False,
        serialize=False,
    )

logger = _loguru_logger


def create_isolated_logger(name: str, log_dir: Path | None = None):
    """Create an isolated logger with its own file sink."""
    from datetime import datetime, timezone

    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    def component_filter(record, component=name):
        return record["extra"].get("component") == component

    isolated = logger.bind(component=name)
    logger.add(
        target_dir / f"{name}-{timestamp}.log",
        filter=component_filter,
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
        serialize=True,
    )

    readable_dir = target_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)
    human_format = globals().get(
        "_HUMAN_FORMAT",
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {extra[component_name]} | {message}",
    )

    for suffix, (levels, min_level) in globals().get("_LEVEL_SINKS", {}).items():
        logger.add(
            readable_dir / f"{name}-{timestamp}.{suffix}.log",
            level=min_level,
            format=human_format,
            filter=lambda record, allowed=levels: component_filter(record)  # type: ignore[misc]
            and record["level"].name in allowed,
            colorize=False,
            rotation=os.getenv("LOG_ROTATION", "100 MB"),
            retention=os.getenv("LOG_RETENTION", "30 days"),
            compression=os.getenv("LOG_COMPRESSION", "zip"),
            enqueue=True,
            backtrace=False,
            serialize=False,
        )

    return isolated
