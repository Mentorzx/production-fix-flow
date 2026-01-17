"""Audit report JSON Schema validation.

This module enforces the audit report contract at runtime by validating
payloads against `config/audit/audit_report.schema.v1.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from pff.shared.core.config import AUDIT_REPORT_SCHEMA_V1_PATH
from pff.shared import logger
from pff.shared.core.file_manager import FileManager


def _default_schema_path() -> Path:
    return AUDIT_REPORT_SCHEMA_V1_PATH


def _format_error(error: Any) -> str:
    path = "/".join(str(part) for part in getattr(error, "absolute_path", []))
    message = getattr(error, "message", str(error))
    validator = getattr(error, "validator", None)
    if path:
        return f"{message} path=/{path} validator={validator}"
    return f"{message} validator={validator}"


@dataclass
class AuditReportSchemaValidator:
    """JSON Schema validator for audit reports.

    Args:
        schema_path: Path to the JSON Schema file.
        file_manager: Optional FileManager.
    """

    schema_path: Path = _default_schema_path()
    file_manager: FileManager | None = None

    def validate(self, report: Mapping[str, Any]) -> None:
        """Validate an audit report payload.

        Args:
            report: Audit report payload.

        Raises:
            RuntimeError: If validation fails or jsonschema is unavailable.
        """
        fm = self.file_manager or FileManager()
        schema_obj = fm.read(self.schema_path, return_native=True)
        if not isinstance(schema_obj, dict):
            raise RuntimeError(
                f"Audit report schema not a dict: path={self.schema_path}"
            )

        try:
            import jsonschema
        except Exception as exc:  # noqa: BLE001
            logger.error(f"jsonschema unavailable for audit report validation: {exc}")
            raise RuntimeError(
                "jsonschema unavailable for audit report validation"
            ) from exc

        validator = jsonschema.Draft202012Validator(schema_obj)
        errors = sorted(
            validator.iter_errors(report), key=lambda e: list(e.absolute_path)
        )
        if not errors:
            return

        formatted = [_format_error(err) for err in errors[:10]]
        logger.error(
            "Audit report schema validation failed: "
            f"errors={len(errors)} schema_path={self.schema_path} "
            f"sample_errors={formatted}"
        )
        raise RuntimeError(
            f"Audit report schema validation failed: errors={len(errors)}"
        )
