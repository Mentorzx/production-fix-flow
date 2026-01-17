"""Deterministic JSON canonicalization for JSON→Graph→JSON audit workflows.

Design patterns:
    - Builder: functions build canonical "record" and "triple" payloads.
    - Adapter: JSON Pointer (RFC 6901) is adapted into a stable `field_path`
      template with list indices replaced by `*` for relation-level grouping.

This module is intentionally pure (no I/O). Persistence is handled by the
PostgreSQL repositories under `pff/db/repositories/**` and higher-level
pipeline orchestration under `pff/utils/audit/**`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pff.shared.core.file_manager import FileManager
from pff.shared.hash import hash_bytes


@dataclass(frozen=True)
class CanonicalRecord:
    """Canonical leaf record extracted from an input JSON document.

    Args:
        document_id: Stable identifier for the full JSON document.
        json_pointer: RFC 6901 JSON Pointer locating the leaf value.
        field_path: Pointer-like template with indices replaced by `*`.
        key: Last path segment (unescaped).
        value_type: One of: null|bool|int|float|str.
        normalized_value: Stable string representation used for hashing/graph edges.
        record_hash: Deterministic hash (hex string) for record identity.
        raw_value: Original leaf value (JSON-serializable scalar).
    """

    document_id: str
    json_pointer: str
    field_path: str
    key: str
    value_type: str
    normalized_value: str
    record_hash: str
    raw_value: Any


@dataclass(frozen=True)
class CanonicalTriple:
    """Canonical KG triple with provenance back to a JSON Pointer.

    Args:
        run_id: Audit run identifier (stable per document/baseline/seed tuple).
        s: Subject identifier (default: document_id).
        p: Predicate identifier (default: field_path).
        o: Object identifier (default: normalized_value).
        json_pointer: Source pointer for provenance.
        record_hash: Record identity hash for provenance joins.
        triple_hash: Deterministic hash (hex string) for triple identity.
    """

    run_id: str
    s: str
    p: str
    o: str
    json_pointer: str
    record_hash: str
    triple_hash: str


def _escape_json_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _infer_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _stable_record_hash(payload: dict[str, Any]) -> str:
    dumped = FileManager.json_dumps(payload, sort_keys=True)
    return f"{hash_bytes(dumped):x}"


def canonicalize_json_document(
    document: Any,
    *,
    document_id: str,
) -> list[CanonicalRecord]:
    """Canonicalize a JSON document into deterministic leaf records.

    This function traverses `dict`/`list` containers and emits one record per
    scalar leaf (null/bool/int/float/str). Dict keys are processed in sorted
    order to guarantee determinism independent of input key ordering.

    Args:
        document: JSON-like object (dict/list/scalars).
        document_id: Stable identifier for the full document.

    Returns:
        List of `CanonicalRecord` in deterministic order.
    """

    records: list[CanonicalRecord] = []

    def _walk(value: Any, *, pointer: str, field_path: str, key: str) -> None:
        if isinstance(value, dict):
            for child_key in sorted(value.keys(), key=lambda k: str(k)):
                child_val = value[child_key]
                token = _escape_json_pointer_token(str(child_key))
                next_pointer = f"{pointer}/{token}" if pointer != "" else f"/{token}"
                next_field_path = (
                    f"{field_path}/{token}" if field_path != "" else f"/{token}"
                )
                _walk(
                    child_val,
                    pointer=next_pointer,
                    field_path=next_field_path,
                    key=str(child_key),
                )
            return

        if isinstance(value, list):
            for idx, child_val in enumerate(value):
                next_pointer = f"{pointer}/{idx}" if pointer != "" else f"/{idx}"
                next_field_path = f"{field_path}/*" if field_path != "" else "/*"
                _walk(
                    child_val, pointer=next_pointer, field_path=next_field_path, key=key
                )
            return

        value_type = _infer_value_type(value)
        normalized = _normalize_scalar(value)
        record_payload = {
            "document_id": document_id,
            "json_pointer": pointer,
            "field_path": field_path,
            "key": key,
            "value_type": value_type,
            "normalized_value": normalized,
        }
        record_hash = _stable_record_hash(record_payload)
        records.append(
            CanonicalRecord(
                document_id=document_id,
                json_pointer=pointer,
                field_path=field_path,
                key=key,
                value_type=value_type,
                normalized_value=normalized,
                record_hash=record_hash,
                raw_value=value,
            )
        )

    _walk(document, pointer="", field_path="", key="")
    return records


def records_to_triples(
    records: Iterable[CanonicalRecord],
    *,
    run_id: str,
    subject: str | None = None,
) -> list[CanonicalTriple]:
    """Convert canonical records into canonical triples with provenance.

    Args:
        records: Canonical leaf records.
        run_id: Audit run identifier for persistence grouping.
        subject: Optional subject override. When not provided, uses the record's
            `document_id` per record.

    Returns:
        List of `CanonicalTriple` in the same order as input records.
    """

    triples: list[CanonicalTriple] = []
    for record in records:
        s = subject or record.document_id
        p = record.field_path
        o = record.normalized_value
        triple_payload = {
            "run_id": run_id,
            "s": s,
            "p": p,
            "o": o,
            "json_pointer": record.json_pointer,
            "record_hash": record.record_hash,
        }
        triple_hash = _stable_record_hash(triple_payload)
        triples.append(
            CanonicalTriple(
                run_id=run_id,
                s=s,
                p=p,
                o=o,
                json_pointer=record.json_pointer,
                record_hash=record.record_hash,
                triple_hash=triple_hash,
            )
        )
    return triples
