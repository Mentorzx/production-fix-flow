"""Dataset fingerprint and profile helpers for Search Space Advisor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pff.shared import stable_hash
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager


def resolve_dataset_parquet_candidates() -> list[Path]:
    """Return preferred processed KG parquet candidates for fingerprinting."""
    mappings_dir = settings.OUTPUTS_DIR / "kg" / "mappings"
    outputs_kg_dir = settings.OUTPUTS_DIR / "kg"
    return [
        mappings_dir / "train.preprocessed.parquet",
        mappings_dir / "valid.preprocessed.parquet",
        mappings_dir / "test.preprocessed.parquet",
        outputs_kg_dir / "train.parquet",
        outputs_kg_dir / "valid.parquet",
        outputs_kg_dir / "test.parquet",
    ]


def resolve_split_columns(schema_names: set[str]) -> tuple[str, str, str] | None:
    """Resolve canonical triple columns based on schema names."""
    if {"s", "p", "o"}.issubset(schema_names):
        return ("s", "p", "o")
    if {"head", "relation", "tail"}.issubset(schema_names):
        return ("head", "relation", "tail")
    return None


def _schema_entries_for_path(path: Path) -> tuple[list[str], set[str]]:
    """Read schema entries for parquet path via shared file manager."""
    lazy = FileManager.scan_parquet(str(path))
    try:
        schema = lazy.collect_schema()
        entries = [f"{name}:{dtype}" for name, dtype in schema.items()]
        return entries, set(schema.names())
    except Exception:
        frame = FileManager.read(path)
        columns = list(getattr(frame, "columns", []))
        dtypes = list(getattr(frame, "dtypes", []))
        entries = [
            f"{name}:{dtype}" for name, dtype in zip(columns, dtypes, strict=False)
        ]
        return entries, set(columns)


def compute_dataset_profile_fingerprint(
    candidate_paths: list[Path] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Compute deterministic dataset fingerprint and lightweight profile."""
    candidates = candidate_paths or resolve_dataset_parquet_candidates()
    existing = [p for p in candidates if FileManager.exists(p)]
    if not existing:
        return None, None

    split_signatures: list[dict[str, Any]] = []
    total_triples = 0
    max_entity_id = -1
    max_relation_id = -1

    for path in sorted(existing, key=lambda item: str(item)):
        lazy = FileManager.scan_parquet(str(path))
        row_count = (
            lazy.select(pl.len().alias("n_rows"))
            .collect(engine="streaming")
            .item(0, "n_rows")
        )
        total_triples += int(row_count)
        schema_entries, schema_names = _schema_entries_for_path(path)

        split_cols = resolve_split_columns(schema_names)
        if split_cols:
            h_col, r_col, t_col = split_cols
            extrema = lazy.select(
                [
                    pl.max(h_col).alias("h_max"),
                    pl.max(t_col).alias("t_max"),
                    pl.max(r_col).alias("r_max"),
                ]
            ).collect(engine="streaming")
            h_max = extrema.item(0, "h_max")
            t_max = extrema.item(0, "t_max")
            r_max = extrema.item(0, "r_max")
            if isinstance(h_max, (int, float)) and isinstance(t_max, (int, float)):
                max_entity_id = max(max_entity_id, int(h_max), int(t_max))
            if isinstance(r_max, (int, float)):
                max_relation_id = max(max_relation_id, int(r_max))

        split_signatures.append(
            {
                "path": str(path.resolve()),
                "rows": int(row_count),
                "schema": schema_entries,
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
        )

    n_entities = max_entity_id + 1 if max_entity_id >= 0 else 0
    n_relations = max_relation_id + 1 if max_relation_id >= 0 else 0
    density = (
        float(total_triples) / float(max(1, n_entities * max(1, n_relations)))
        if total_triples > 0
        else 0.0
    )
    profile = {
        "n_entities": int(n_entities),
        "n_relations": int(n_relations),
        "n_triples": int(total_triples),
        "density": float(density),
        "sources": [entry["path"] for entry in split_signatures],
    }
    payload = {"profile": profile, "splits": split_signatures}
    encoded = FileManager.json_dumps(payload, sort_keys=True)
    fingerprint = f"{int(stable_hash(encoded, truncate=64)) & ((1 << 64) - 1):016x}"[
        :24
    ]
    return fingerprint, profile


__all__ = [
    "compute_dataset_profile_fingerprint",
    "resolve_dataset_parquet_candidates",
    "resolve_split_columns",
]
