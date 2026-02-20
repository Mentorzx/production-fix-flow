"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/core/test_logging_structured_fields.py

"""

from __future__ import annotations

import logging
from typing import Any

from pff.shared.core.logging import logger
from pff.shared.core.logging.config import InterceptHandler


def test_loguru_injects_mandatory_fields() -> None:
    """Execute test loguru injects mandatory fields.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    records: list[dict[str, Any]] = []

    def sink(message):
        """Execute sink.



        Args:

            message: Input value used by this callable.

        """

        records.append(message.record)

    sink_id = logger.add(sink, level="INFO")
    try:
        logger.info("Teste de log estruturado")
    finally:
        logger.remove(sink_id)

    assert records, "Expected at least one log record"
    extra = records[0]["extra"]
    assert "timestamp" in extra
    assert "component_name" in extra
    assert "key_parameters" in extra
    assert "stop_reason" in extra


def test_intercept_handler_escapes_angle_brackets() -> None:
    """Intercept handler must not fail on messages with '<...>' URLs."""
    record = logging.LogRecord(
        name="logging",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=(
            'Link header: <https://huggingface.co/api/models/foo>; rel="xet-auth", '
            '<https://cas-server.xethub.hf.co/v1/reconstructions/bar>; rel="xet-reconstruction-info"'
        ),
        args=(),
        exc_info=None,
    )
    handler = InterceptHandler()
    handler.emit(record)
