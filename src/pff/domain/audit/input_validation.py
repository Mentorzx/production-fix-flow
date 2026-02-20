"""Input JSON validation helpers (JSON Schema).

Design patterns:
    - Adapter: converts jsonschema error paths to RFC 6901 JSON Pointer strings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pff.shared.core.logging import logger


def _escape_json_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _path_to_json_pointer(path: Iterable[Any]) -> str:
    """Execute path to json pointer.



    Args:

        path: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    parts: list[str] = []
    for segment in path:
        parts.append(_escape_json_pointer_token(str(segment)))
    if not parts:
        return ""
    return "/" + "/".join(parts)


@dataclass(frozen=True)
class SchemaViolation:
    """A single JSON Schema violation mapped to an actionable JSON Pointer."""

    error_code: str
    message: str
    json_pointer: str
    validator: str | None
    validator_value: Any
    instance_snippet: Any

    def to_dict(self) -> dict[str, Any]:
        """Execute to dict.



        Returns:

            Return value produced by the callable.

        """

        return {
            "error_code": self.error_code,
            "message": self.message,
            "json_pointer": self.json_pointer,
            "validator": self.validator,
            "validator_value": self.validator_value,
            "instance_snippet": self.instance_snippet,
        }


class AuditInputSchemaValidator:
    """Validate an input document against a JSON Schema and return a report."""

    def __init__(self, *, schema: Mapping[str, Any]) -> None:
        """Execute init.



        Args:

            schema: Input value used by this callable.

        """

        self._schema = dict(schema)

    def validate(self, document: Any) -> list[dict[str, Any]]:
        """Validate a JSON document.

        Args:
            document: JSON-like instance to validate.

        Returns:
            List of schema violations as dicts, sorted by json_pointer.

        Raises:
            RuntimeError: If jsonschema is unavailable.
        """

        try:
            import jsonschema
        except Exception as exc:
            logger.error(f"jsonschema unavailable for input schema validation: {exc}")
            raise RuntimeError(
                "jsonschema indisponível para validação de esquema de entrada"
            ) from exc

        validator = jsonschema.Draft202012Validator(self._schema)
        violations: list[SchemaViolation] = []
        for err in validator.iter_errors(document):
            ptr = _path_to_json_pointer(getattr(err, "absolute_path", []))
            violations.append(
                SchemaViolation(
                    error_code="json_schema_validation_error",
                    message=getattr(err, "message", str(err)),
                    json_pointer=ptr,
                    validator=getattr(err, "validator", None),
                    validator_value=getattr(err, "validator_value", None),
                    instance_snippet=getattr(err, "instance", None),
                )
            )
        violations.sort(key=lambda v: v.json_pointer)
        return [v.to_dict() for v in violations]
