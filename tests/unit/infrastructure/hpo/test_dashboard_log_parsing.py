"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/hpo/test_dashboard_log_parsing.py

"""

from __future__ import annotations

import orjson

from pff.infrastructure.hpo.dashboard import server as dashboard_server
from pff.infrastructure.hpo.dashboard.server import (
    _has_usable_search_space_advice,
    _load_raw_dashboard_data,
    _parse_log_line,
)


def test_parse_log_line_loguru_pipe_format() -> None:
    """Execute test parse log line loguru pipe format.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    line = (
        "2026-02-11 10:11:12.123 | WARNING | pff.infrastructure.hpo.runner | "
        "task=abc | component_name=hpo_runner stop_reason=ok key_parameters={} "
        "message='fallback activated'"
    )
    parsed = _parse_log_line(line)

    assert parsed is not None
    assert parsed["timestamp"] == "2026-02-11 10:11:12.123"
    assert parsed["level"] == "WARNING"
    assert parsed["module"] == "runner"
    assert parsed["message"] == "fallback activated"


def test_parse_log_line_pipe_fallback_with_human_text() -> None:
    """Execute test parse log line pipe fallback with human text."""

    line = (
        "2026-02-11 10:11:12.123 | ERROR | pff.shared.ops.global_interrupt_manager | "
        "task=x | trace=y | stop=z | worker interrupted | params={}"
    )
    parsed = _parse_log_line(line)

    assert parsed is not None
    assert parsed["level"] == "ERROR"
    assert parsed["module"] == "global_interrupt_manager"
    assert parsed["message"] == "worker interrupted"


def test_parse_log_line_legacy_json_payload() -> None:
    """Execute test parse log line legacy json payload."""

    line = orjson.dumps({"text": "legacy warning"}).decode("utf-8")
    parsed = _parse_log_line(line)

    assert parsed is not None
    assert parsed["level"] == "WARNING"
    assert parsed["message"] == "legacy warning"


def test_parse_log_line_suppresses_known_noise() -> None:
    """Execute test parse log line suppresses known noise."""

    line = "2026-02-11 10:11:12.123 | WARNING | module | NumPy compatibility shim active"
    assert _parse_log_line(line) is None


def test_parse_log_line_raw_fallback() -> None:
    """Execute test parse log line raw fallback."""

    line = "unstructured message"
    parsed = _parse_log_line(line)

    assert parsed is not None
    assert parsed["level"] == "WARNING"
    assert parsed["message"] == "unstructured message"


def test_has_usable_search_space_advice_with_recommendations() -> None:
    """Execute usable search space advice with populated recommendations."""
    assert _has_usable_search_space_advice(
        {"recommendations": [{"param_name": "lr"}], "metadata": {}}
    )


def test_has_usable_search_space_advice_with_insufficient_evidence() -> None:
    """Execute usable search space advice with insufficient evidence metadata."""
    assert _has_usable_search_space_advice(
        {"recommendations": [], "metadata": {"insufficient_evidence": True}}
    )


def test_has_usable_search_space_advice_rejects_empty_recommendations() -> None:
    """Execute unusable search space advice when recommendations are empty."""
    assert not _has_usable_search_space_advice(
        {"recommendations": [], "metadata": {"insufficient_evidence": False}}
    )


def test_load_raw_dashboard_data_invalid_payload_returns_empty(monkeypatch, tmp_path) -> None:
    """Execute invalid dashboard payload fallback to empty dict."""

    fake_path = tmp_path / "fake_dashboard_data.json"
    fake_path.write_text("null")

    monkeypatch.setattr(dashboard_server, "_collect_dashboard_data_paths", lambda: [fake_path])
    monkeypatch.setattr(dashboard_server.FileManager, "exists", lambda _path: True)
    monkeypatch.setattr(dashboard_server.FileManager, "read", lambda *_args, **_kwargs: None)

    payload = _load_raw_dashboard_data()
    assert payload == {}
