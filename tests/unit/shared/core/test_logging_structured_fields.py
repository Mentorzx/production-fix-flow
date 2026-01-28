from __future__ import annotations

from typing import Any

from pff.shared.core.logging import logger


def test_loguru_injects_mandatory_fields() -> None:
    records: list[dict[str, Any]] = []

    def sink(message):
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
