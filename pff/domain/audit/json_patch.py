"""RFC 6902 JSON Patch utilities for audit repair suggestions.

Design patterns:
    - Command: each patch op is treated as a command applied to a document.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from pff.domain.audit.input_validation import AuditInputSchemaValidator


_RE_REQUIRED = re.compile(r"^'(?P<key>[^']+)' is a required property$")
_RE_ADDITIONAL = re.compile(
    r"^Additional properties are not allowed \\('(?P<key>[^']+)' was unexpected\\)$"
)


def _unescape_json_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _escape_json_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _parse_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must start with '/' or be empty")
    parts = pointer.split("/")[1:]
    return [_unescape_json_pointer_token(p) for p in parts]


def _resolve_parent(document: Any, pointer: str) -> tuple[Any, str]:
    parts = _parse_pointer(pointer)
    if not parts:
        raise ValueError("Root pointer has no parent")
    parent_parts = parts[:-1]
    key = parts[-1]
    node = document
    for part in parent_parts:
        if isinstance(node, dict):
            node = node[part]
            continue
        if isinstance(node, list):
            node = node[int(part)]
            continue
        raise KeyError(f"Invalid pointer traversal at token={part!r}")
    return node, key


def apply_json_patch(  # noqa: PLR0912
    document: Any,
    ops: list[dict[str, Any]],
) -> Any:
    """Apply a subset of RFC 6902 ops (add/remove/replace) to a JSON document."""

    doc = copy.deepcopy(document)
    for op in ops:
        if not isinstance(op, dict):
            raise ValueError("op must be an object")
        kind = str(op.get("op", "")).lower()
        path = op.get("path")
        if not isinstance(path, str):
            raise ValueError("op.path must be a string")

        if kind in ("add", "replace"):
            value = op.get("value")
            if path == "":
                doc = value
                continue
            parent, key = _resolve_parent(doc, path)
            if isinstance(parent, dict):
                parent[key] = value
                continue
            if isinstance(parent, list):
                if key == "-":
                    parent.append(value)
                    continue
                idx = int(key)
                if kind == "add":
                    parent.insert(idx, value)
                else:
                    parent[idx] = value
                continue
            raise KeyError(f"Invalid patch target at path={path}")

        if kind == "remove":
            if path == "":
                raise ValueError("remove at root is not supported")
            parent, key = _resolve_parent(doc, path)
            if isinstance(parent, dict):
                parent.pop(key, None)
                continue
            if isinstance(parent, list):
                parent.pop(int(key))
                continue
            raise KeyError(f"Invalid patch target at path={path}")

        raise ValueError(f"Unsupported JSON Patch op: {kind!r}")

    return doc


def validate_patch_against_schema(
    *,
    document: Any,
    ops: list[dict[str, Any]],
    schema: dict[str, Any],
) -> bool:
    """Return True if the patched document is valid under the provided JSON Schema."""

    patched = apply_json_patch(document, ops)
    report = AuditInputSchemaValidator(schema=schema).validate(patched)
    return len(report) == 0


def suggest_repairs_from_schema_report(
    *,
    document: Any,
    schema: dict[str, Any],
    schema_report: list[dict[str, Any]],
    max_repairs: int = 50,
) -> list[dict[str, Any]]:
    """Suggest JSON Patch repairs from a schema validation report.

    Each suggestion is validated by applying the patch and re-validating the
    document against the same schema. Only patches that reduce violations to
    zero are returned.
    """

    suggestions: list[dict[str, Any]] = []
    for item in schema_report:
        if len(suggestions) >= max(0, int(max_repairs)):
            break
        if not isinstance(item, dict):
            continue
        validator = str(item.get("validator", "")).lower()
        pointer = str(item.get("json_pointer", ""))
        message = str(item.get("message", ""))

        ops: list[dict[str, Any]] = []
        rationale = ""

        if validator == "required":
            m = _RE_REQUIRED.match(message)
            if m:
                missing = m.group("key")
                path = (
                    f"{pointer}/{_escape_json_pointer_token(missing)}"
                    if pointer
                    else f"/{_escape_json_pointer_token(missing)}"
                )
                ops = [{"op": "add", "path": path, "value": None}]
                rationale = f"Add missing required property: path={path}"

        elif validator == "additionalproperties":
            m = _RE_ADDITIONAL.match(message)
            if m:
                key = m.group("key")
                path = (
                    f"{pointer}/{_escape_json_pointer_token(key)}"
                    if pointer
                    else f"/{_escape_json_pointer_token(key)}"
                )
                ops = [{"op": "remove", "path": path}]
                rationale = f"Remove unexpected property: path={path}"

        elif validator == "type":
            if pointer:
                ops = [{"op": "replace", "path": pointer, "value": None}]
                rationale = f"Replace value with null to satisfy type at path={pointer}"

        if not ops:
            continue

        if validate_patch_against_schema(document=document, ops=ops, schema=schema):
            suggestions.append(
                {
                    "ops": ops,
                    "rationale": rationale,
                    "expected_impact": {"schema_errors": "resolved"},
                }
            )

    return suggestions
