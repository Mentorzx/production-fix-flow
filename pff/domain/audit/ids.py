"""Deterministic identifiers for audit runs.

The audit pipeline needs stable IDs to enable reproducibility and to persist
artifacts under a predictable directory layout.
"""

from __future__ import annotations

import orjson

from pff.shared.hash import stable_hash


def _hash_hexdigest(data: bytes, *, algorithm: str) -> str:
    """Return a hex digest for the given bytes.

    Args:
        data: Bytes to hash.
        algorithm: Hash algorithm supported by stable_hash (e.g., sha1, sha256).

    Returns:
        Hex digest string.
    """
    digest_int = stable_hash(data, algorithm=algorithm, truncate=64)
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
    except Exception:  # noqa: BLE001
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
    algorithm: str = "sha1",
    truncate: int = 16,
) -> str:
    """Compute a stable identifier for an input document.

    Args:
        document: Input JSON-like object (dict/list/str/bytes).
        algorithm: Hash algorithm (sha1 by default).
        truncate: Number of hex characters to keep.

    Returns:
        Stable document identifier as a hex string.
    """
    payload = _canonicalize_for_hash(document)
    return _truncate_hex(_hash_hexdigest(payload, algorithm=algorithm), truncate=truncate)


def compute_baseline_id(
    baseline_key: Any,
    *,
    algorithm: str = "sha1",
    truncate: int = 16,
) -> str:
    """Compute a stable identifier for a baseline/profile reference.

    Args:
        baseline_key: Any stable key describing the baseline (name, window, digest).
        algorithm: Hash algorithm (sha1 by default).
        truncate: Number of hex characters to keep.

    Returns:
        Stable baseline identifier as a hex string.
    """
    payload = _canonicalize_for_hash(baseline_key)
    return _truncate_hex(_hash_hexdigest(payload, algorithm=algorithm), truncate=truncate)


def compute_run_id(
    *,
    document_id: str,
    baseline_id: str,
    schema_version: str | int,
    algorithm: str = "sha1",
    truncate: int = 16,
) -> str:
    """Compute a stable run identifier.

    The run_id must be stable for the same (document, baseline, schema_version)
    triple to support reproducible artifact paths.

    Args:
        document_id: Stable document id.
        baseline_id: Stable baseline id.
        schema_version: Input schema version for the audited document.
        algorithm: Hash algorithm (sha1 by default).
        truncate: Number of hex characters to keep.

    Returns:
        Stable run identifier as a hex string.
    """
    key = (document_id, baseline_id, str(schema_version))
    digest_int = stable_hash(key, algorithm=algorithm, truncate=truncate)
    width = truncate
    return f"{digest_int:0{width}x}"[-width:]


def build_audit_run_ids(
    *,
    document: Any,
    baseline_key: Any,
    schema_version: str | int,
    algorithm: str = "sha1",
    truncate: int = 16,
) -> AuditRunIds:
    """Build the deterministic identifiers required by the audit artifact layout.

    Args:
        document: Input JSON-like object.
        baseline_key: Baseline reference key.
        schema_version: Input document schema version.
        algorithm: Hash algorithm (sha1 by default).
        truncate: Number of hex characters to keep for ids.

    Returns:
        AuditRunIds instance with document_id, baseline_id, run_id.
    """
    document_id = compute_document_id(document, algorithm=algorithm, truncate=truncate)
    baseline_id = compute_baseline_id(baseline_key, algorithm=algorithm, truncate=truncate)
    run_id = compute_run_id(
        document_id=document_id,
        baseline_id=baseline_id,
        schema_version=schema_version,
        algorithm=algorithm,
        truncate=truncate,
    )
    return AuditRunIds(document_id=document_id, baseline_id=baseline_id, run_id=run_id)
