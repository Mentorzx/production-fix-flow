"""Mapping utilities for RotatE model.

This module provides utilities for loading and managing entity/relation
mappings used by the RotatE model.

Design Patterns Applied:
    - **Facade:** Simplified interface for loading parquet-based mappings.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.utils import FileManager, logger


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

    if not entity_map_path.exists():
        raise FileNotFoundError(f"Entity mapping not found: {entity_map_path}")
    if not relation_map_path.exists():
        raise FileNotFoundError(f"Relation mapping not found: {relation_map_path}")

    entity_df = fm.read(entity_map_path)
    relation_df = fm.read(relation_map_path)

    # Detect column naming convention
    entity_to_idx, idx_to_entity = _parse_mapping_df(entity_df, "entity")
    relation_to_idx, idx_to_relation = _parse_mapping_df(relation_df, "relation")

    logger.debug(
        f"Mapeamentos carregados: {len(entity_to_idx)} entidades, "
        f"{len(relation_to_idx)} relacoes"
    )

    return entity_to_idx, idx_to_entity, relation_to_idx, idx_to_relation


def _parse_mapping_df(
    df: Any,
    mapping_type: str,
) -> tuple[dict[str, int], dict[int, str]]:
    """Parse a mapping DataFrame into dictionaries.

    Args:
        df: Polars or Pandas DataFrame with mapping data.
        mapping_type: Type of mapping ('entity' or 'relation') for column detection.

    Returns:
        Tuple of (name_to_idx, idx_to_name) dictionaries.
    """
    cols = df.columns

    # Try different column naming conventions
    idx_col = None
    name_col = None

    # Convention 1: (id, label)
    if "id" in cols and "label" in cols:
        idx_col, name_col = "id", "label"
    # Convention 2: (idx, entity/relation)
    elif "idx" in cols and mapping_type in cols:
        idx_col, name_col = "idx", mapping_type
    # Convention 3: (index, name)
    elif "index" in cols and "name" in cols:
        idx_col, name_col = "index", "name"
    # Convention 4: Assume first two columns are (idx, name)
    else:
        idx_col, name_col = cols[0], cols[1]

    # Handle both Polars and Pandas
    try:
        # Polars API
        idx_values = df[idx_col].to_list()
        name_values = df[name_col].to_list()
    except AttributeError:
        # Pandas API
        idx_values = df[idx_col].tolist()
        name_values = df[name_col].tolist()

    name_to_idx = dict(zip(name_values, idx_values))
    idx_to_name = dict(zip(idx_values, name_values))

    return name_to_idx, idx_to_name
