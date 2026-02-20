"""Deterministic identifiers for audit runs.

The audit pipeline needs stable IDs to enable reproducibility and to persist
artifacts under a predictable directory layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import orjson
from pff_rust import stable_hash


def _hash_hexdigest(data: bytes) -> str:
    """Return a hex digest for the given bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Hex digest string.
    """
    digest_int = stable_hash(data, truncate=64)
    return f"{digest_int:x}"


def _canonicalize_for_hash(value: Any) -> bytes:
    """Canonicalize an object for deterministic hashing.

    Args:
        value: Any JSON-serializable object.

    Returns:
        Canonical bytes representation.
    """
    if value is None:
        return b"null"
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    except Exception:
        return repr(value).encode("utf-8")


def _truncate_hex(hex_digest: str, *, truncate: int) -> str:
    """Truncate a hex digest to a fixed number of characters.

    Args:
        hex_digest: Full hex digest.
        truncate: Number of characters to keep.

    Returns:
        Truncated hex digest.
    """
    if truncate <= 0:
        raise ValueError("truncate must be > 0")
    return hex_digest[:truncate]


@dataclass(frozen=True)
class AuditRunIds:
    """Deterministic identifiers for an audit run.

    Attributes:
        document_id: Stable identifier for the input document.
        baseline_id: Stable identifier for the baseline/profile used.
        run_id: Stable identifier for the run, derived from document_id and baseline_id.
    """

    document_id: str
    baseline_id: str
    run_id: str


def compute_document_id(
    document: Any,
    *,
    truncate: int = 16,
) -> str:
    """Compute a stable identifier for an input document.

    Args:
        document: Input JSON-like object (dict/list/str/bytes).
        truncate: Number of hex characters to keep.

    Returns:
        Stable document identifier as a hex string.
    """
    payload = _canonicalize_for_hash(document)
    return _truncate_hex(_hash_hexdigest(payload), truncate=truncate)


def compute_baseline_id(
    baseline_key: Any,
    *,
    truncate: int = 16,
) -> str:
    """Compute a stable identifier for a baseline/profile reference.

    Args:
        baseline_key: Any stable key describing the baseline (name, window, digest).
        truncate: Number of hex characters to keep.

    Returns:
        Stable baseline identifier as a hex string.
    """
    payload = _canonicalize_for_hash(baseline_key)
    return _truncate_hex(_hash_hexdigest(payload), truncate=truncate)


def compute_run_id(
    *,
    document_id: str,
    baseline_id: str,
    schema_version: str | int,
    truncate: int = 16,
) -> str:
    """Compute a stable run identifier.

    The run_id must be stable for the same (document, baseline, schema_version)
    triple to support reproducible artifact paths.

    Args:
        document_id: Stable document id.
        baseline_id: Stable baseline id.
        schema_version: Input schema version for the audited document.
        truncate: Number of hex characters to keep.

    Returns:
        Stable run identifier as a hex string.
    """
    key = (document_id, baseline_id, str(schema_version))
    digest_int = stable_hash(key, truncate=truncate)
    width = truncate
    return f"{digest_int:0{width}x}"[-width:]


def build_audit_run_ids(
    *,
    document: Any,
    baseline_key: Any,
    schema_version: str | int,
    truncate: int = 16,
) -> AuditRunIds:
    """Build the deterministic identifiers required by the audit artifact layout.

    Args:
        document: Input JSON-like object.
        baseline_key: Baseline reference key.
        schema_version: Input document schema version.
        truncate: Number of hex characters to keep for ids.

    Returns:
        AuditRunIds instance with document_id, baseline_id, run_id.
    """
    document_id = compute_document_id(document, truncate=truncate)
    baseline_id = compute_baseline_id(baseline_key, truncate=truncate)
    run_id = compute_run_id(
        document_id=document_id,
        baseline_id=baseline_id,
        schema_version=schema_version,
        truncate=truncate,
    )
    return AuditRunIds(document_id=document_id, baseline_id=baseline_id, run_id=run_id)
