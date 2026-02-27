"""Log parsing helpers for dashboard readable log streams."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from pff.shared.core.file_manager import FileManager

MAX_LOG_ENTRIES = 200
MAX_TAIL_BYTES = 65536
MAX_TAIL_LINES = 150

_LOG_SUPPRESSION_PATTERNS: list[str] = [
    "numpy_compat_shim",
    "NumPy compatibility shim",
]

_LOGURU_PIPE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*"
    r"(?P<level>\w+)\s*\|\s*"
    r"(?P<module>[^\|]+?)\s*\|"
    r".*?\|\s*"
    r"(?:component_name=\S+\s+)?(?:stop_reason=\S+\s+)?(?:key_parameters=\{[^}]*\}\s+)?"
    r"message='(?P<message>[^']+)'"
)


def read_tail_lines(
    path: Path, *, max_bytes: int = 65536, max_lines: int = 200
) -> list[str]:
    """Return decoded tail lines from a log file path."""
    if max_bytes <= 0 or max_lines <= 0:
        return []
    raw = FileManager.read_tail_bytes(path, max_bytes=max_bytes)
    if not raw:
        return []
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []
    return lines[-max_lines:]


def load_json_payload(raw: str | bytes) -> dict[str, Any]:
    """Parse JSON payload and enforce mapping root type."""
    payload = FileManager.json_loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return cast(dict[str, Any], payload)


def _is_suppressed_log_line(stripped: str) -> bool:
    for pattern in _LOG_SUPPRESSION_PATTERNS:
        if pattern in stripped:
            return True
    return False


def _parse_loguru_pipe_line(stripped: str) -> dict[str, str] | None:
    match = _LOGURU_PIPE_RE.match(stripped)
    if not match:
        return None
    return {
        "timestamp": match.group("ts"),
        "level": match.group("level").strip().upper(),
        "module": match.group("module").strip().rsplit(".", 1)[-1],
        "message": match.group("message").strip(),
    }


def _extract_pipe_fallback_message(rest: str) -> str:
    msg_match = re.search(r"message='([^']+)'", rest)
    if msg_match:
        return msg_match.group(1).strip()

    segments = [segment.strip() for segment in rest.split("|")]
    human_parts = [
        segment
        for segment in segments
        if segment
        and not re.match(
            r"^(task=|trace=|span=|stop=|params=|component_name=|"
            r"stop_reason=|key_parameters=)",
            segment,
        )
    ]
    return " ".join(human_parts).strip() if human_parts else rest.strip()


def _parse_pipe_fallback_line(stripped: str) -> dict[str, str] | None:
    parts = stripped.split("|")
    if len(parts) < 3:
        return None
    timestamp = parts[0].strip()
    level = parts[1].strip().upper()
    module = parts[2].strip().rsplit(".", 1)[-1]
    rest = "|".join(parts[3:])
    message = _extract_pipe_fallback_message(rest)
    if not timestamp or level not in ("ERROR", "WARNING", "CRITICAL"):
        return None
    return {
        "timestamp": timestamp,
        "level": level,
        "module": module,
        "message": message,
    }


def _parse_json_legacy_line(stripped: str) -> dict[str, str] | None:
    if not stripped.startswith("{"):
        return None
    try:
        payload = load_json_payload(stripped)
    except Exception:
        return None
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "timestamp": "",
        "level": "WARNING",
        "module": "",
        "message": text.strip(),
    }


def parse_log_line(raw: str) -> dict[str, str] | None:
    """Parse one raw dashboard log line into normalized fields."""
    stripped = raw.strip()
    if not stripped:
        return None
    if _is_suppressed_log_line(stripped):
        return None
    parsed = _parse_loguru_pipe_line(stripped)
    if parsed:
        return parsed
    parsed = _parse_pipe_fallback_line(stripped)
    if parsed:
        return parsed
    parsed = _parse_json_legacy_line(stripped)
    if parsed:
        return parsed
    return {
        "timestamp": "",
        "level": "WARNING",
        "module": "",
        "message": stripped,
    }


def normalize_log_entries(lines: list[str]) -> list[dict[str, str]]:
    """Normalize raw lines into dashboard log-entry payload."""
    entries: list[dict[str, str]] = []
    for line in lines:
        entry = parse_log_line(line)
        if entry is not None:
            entries.append(entry)
    return entries


__all__ = [
    "MAX_LOG_ENTRIES",
    "MAX_TAIL_BYTES",
    "MAX_TAIL_LINES",
    "load_json_payload",
    "normalize_log_entries",
    "parse_log_line",
    "read_tail_lines",
]
