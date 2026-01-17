"""Mapping utilities for DSLFM model.

This module provides utilities for loading and managing entity/relation
mappings used by the DSLFM model.

Design Patterns Applied:
    - **Facade:** Simplified interface for loading parquet-based mappings.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.shared import FileManager, logger
from pff.shared.core.file_manager import ParquetBundle


def load_mappings(
    entity_map_path: Path,
    relation_map_path: Path,
) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    """Load entity and relation mappings from parquet files.

    Args:
        entity_map_path: Path to entity mapping parquet file.
        relation_map_path: Path to relation mapping parquet file.

    Returns:
        Tuple of (entity_to_idx, idx_to_entity, relation_to_idx, idx_to_relation).

    Raises:
        FileNotFoundError: If mapping files don't exist.
        ValueError: If mapping files have invalid format.
    """
    fm = FileManager()

    if not fm.exists(entity_map_path):
        raise FileNotFoundError(f"Entity mapping not found: {entity_map_path}")
    if not fm.exists(relation_map_path):
        raise FileNotFoundError(f"Relation mapping not found: {relation_map_path}")

    entity_bundle = fm.read(entity_map_path)
    relation_bundle = fm.read(relation_map_path)
    entity_df = (
        entity_bundle.lazyframe().collect(engine="streaming")
        if isinstance(entity_bundle, ParquetBundle)
        else entity_bundle
    )
    relation_df = (
        relation_bundle.lazyframe().collect(engine="streaming")
        if isinstance(relation_bundle, ParquetBundle)
        else relation_bundle
    )

    entity_to_idx, idx_to_entity = _parse_mapping_df(entity_df, "entity")
    relation_to_idx, idx_to_relation = _parse_mapping_df(relation_df, "relation")

    logger.debug(
        f"Mappings loaded: entities={len(entity_to_idx):,} relations={len(relation_to_idx):,}"
    )

    return entity_to_idx, idx_to_entity, relation_to_idx, idx_to_relation


def _detect_mapping_columns(columns: list[str], mapping_type: str) -> tuple[str, str]:
    """Infer index/name columns from common mapping schemas.

    Supported schemas:
        - (id, label)
        - (idx, <mapping_type>) where mapping_type in {"entity", "relation"}
        - (index, name)
        - fallback: first two columns

    Args:
        columns: DataFrame columns.
        mapping_type: Mapping type to detect the name column.

    Returns:
        Tuple of (idx_col, name_col).
    """
    if "id" in columns and "label" in columns:
        return "id", "label"
    if "idx" in columns and mapping_type in columns:
        return "idx", mapping_type
    if "index" in columns and "name" in columns:
        return "index", "name"
    return columns[0], columns[1]


def _parse_mapping_df(
    df: Any,
    mapping_type: str,
) -> tuple[dict[str, int], dict[int, str]]:
    """Parse a mapping DataFrame into dictionaries.

    Optimized: Uses Polars iterators to avoid full materialization.

    Args:
        df: Polars or Pandas DataFrame with mapping data.
        mapping_type: Type of mapping ('entity' or 'relation') for column detection.

    Returns:
        Tuple of (name_to_idx, idx_to_name) dictionaries.
    """
    columns = list(df.columns)
    idx_col, name_col = _detect_mapping_columns(columns, mapping_type)

    try:
        name_to_idx = dict(zip(df[name_col], df[idx_col]))
        idx_to_name = dict(zip(df[idx_col], df[name_col]))
    except (AttributeError, TypeError):
        idx_values = df[idx_col].tolist()
        name_values = df[name_col].tolist()
        name_to_idx = dict(zip(name_values, idx_values))
        idx_to_name = dict(zip(idx_values, name_values))

    return name_to_idx, idx_to_name
