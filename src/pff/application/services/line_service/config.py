"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/application/services/line_service/config.py

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pff.application.ports.file_manager import FileManagerPort
from pff.shared.core.config import LINE_SERVICE_CONFIG_PATH
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger


@dataclass
class CircuitBreakerConfig:
    """Represent CircuitBreakerConfig."""

    fail_max: int
    timeout_duration_s: float


@dataclass
class LineServiceConfig:
    """Represent LineServiceConfig.



    Notes:

        Encapsulates behavior while preserving architecture boundaries.

    """

    read_breaker: CircuitBreakerConfig = field(
        default_factory=lambda: CircuitBreakerConfig(
            fail_max=5, timeout_duration_s=60.0
        )
    )
    write_breaker: CircuitBreakerConfig = field(
        default_factory=lambda: CircuitBreakerConfig(
            fail_max=3, timeout_duration_s=30.0
        )
    )
    coalescing_delay_s: int = 10


def load_line_service_config(
    path: Path | None = None,
    *,
    file_manager: FileManagerPort | None = None,
) -> LineServiceConfig:
    """Execute load line service config.



    Args:

        path: Optional input value.

        file_manager: Optional input value.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    cfg_path = path or LINE_SERVICE_CONFIG_PATH
    fm = file_manager or FileManager()
    raw: dict[str, Any] = {}

    if fm.exists(cfg_path):
        try:
            raw = fm.read(cfg_path, return_native=True) or {}
            logger.debug(f"LineService config loaded from {cfg_path}")
        except Exception as exc:
            logger.warning(f"Failed to load LineService config: {exc}")
    else:
        logger.debug(f"LineService config not found at {cfg_path}; using defaults")

    cb_raw = raw.get("circuit_breaker", {})
    read_raw = cb_raw.get("read", {})
    write_raw = cb_raw.get("write", {})

    read_breaker = CircuitBreakerConfig(
        fail_max=int(read_raw.get("fail_max", 5)),
        timeout_duration_s=float(read_raw.get("timeout_duration_s", 60.0)),
    )

    write_breaker = CircuitBreakerConfig(
        fail_max=int(write_raw.get("fail_max", 3)),
        timeout_duration_s=float(write_raw.get("timeout_duration_s", 30.0)),
    )

    return LineServiceConfig(
        read_breaker=read_breaker,
        write_breaker=write_breaker,
        coalescing_delay_s=int(raw.get("coalescing_delay_s", 10)),
    )
