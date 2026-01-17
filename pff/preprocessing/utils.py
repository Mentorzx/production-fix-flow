"""
Utility helpers for preprocessing outputs.

Design Pattern: Helper/Adapter
- Adapts existing preprocessing configuration to post-load filtering so that
  consumers (HPO, training CLI) can enforce attribute removal even when
  using persisted/pre-split data from PostgreSQL.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pff.shared import logger

from .config import (
    ATTRIBUTE_HANDLING_REMOVE,
    ATTRIBUTE_HANDLING_SEPARATE,
    PreprocessingConfig,
)


def filter_attribute_relations(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame | None,
    test_df: pl.DataFrame | None,
    config: PreprocessingConfig,
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None, dict[str, Any]]:
    """Remove attribute relations (and inverses) from preprocessed splits.

    This guard is applied post-load to ensure stale PostgreSQL snapshots that
    still contain attribute triples do not reach training/HPO. It respects the
    attribute handling mode configured via ``PreprocessingConfig``.

    Args:
        train_df: Training split DataFrame.
        valid_df: Validation split DataFrame or None.
        test_df: Test split DataFrame or None.
        config: Preprocessing configuration loaded from YAML.

    Returns:
        Tuple with filtered (train, valid, test) DataFrames and a stats dict
        containing removal counts and blocked relations observed.
    """
    remove_attrs = config.attribute_handling in {
        ATTRIBUTE_HANDLING_REMOVE,
        ATTRIBUTE_HANDLING_SEPARATE,
    }
    has_explicit = bool(config.attribute_relations)
    has_patterns = bool(config.attribute_patterns)
    if not remove_attrs or (not has_explicit and not has_patterns):
        return (
            train_df,
            valid_df,
            test_df,
            {
                "removed": 0,
                "removed_by_split": {},
                "blocked_relations": [],
            },
        )

    attribute_relations = set(config.attribute_relations)
    blocked_relations = attribute_relations | {
        f"{rel}{config.inverse_suffix}" for rel in attribute_relations
    }
    if has_patterns:
        import re

        compiled = [
            re.compile(p, flags=re.IGNORECASE) for p in config.attribute_patterns
        ]
        unique_relations = train_df["p"].unique().to_list()
        for rel in unique_relations:
            if any(p.search(rel) for p in compiled):
                blocked_relations.add(rel)
                if rel.endswith(config.inverse_suffix):
                    continue
                blocked_relations.add(f"{rel}{config.inverse_suffix}")

    def _filter_split(
        df: pl.DataFrame | None, split_name: str
    ) -> tuple[pl.DataFrame | None, int, set[str]]:
        if df is None:
            return None, 0, set()
        relation_expr = pl.col("p")
        df_aligned = (
            df
            if df.schema.get("p") == pl.Utf8
            else df.with_columns(relation_expr.cast(pl.Utf8))
        )
        mask = ~relation_expr.is_in(list(blocked_relations))
        filtered = df_aligned.filter(mask)
        removed = len(df) - len(filtered)
        present_blocked = set(df_aligned["p"].unique()) & blocked_relations
        if removed > 0:
            logger.info(
                f"[ATRIBUTOS] Removidas {removed:,} triplas de atributo do split {split_name}"
            )
        return filtered, removed, present_blocked

    train_filtered, train_removed, train_blocked = _filter_split(train_df, "train")
    valid_filtered, valid_removed, valid_blocked = _filter_split(valid_df, "valid")
    test_filtered, test_removed, test_blocked = _filter_split(test_df, "test")

    total_removed = train_removed + valid_removed + test_removed
    blocked_seen = sorted(train_blocked | valid_blocked | test_blocked)

    if total_removed == 0:
        logger.info("Nenhuma tripla de atributo encontrada nos splits preprocessados")
    else:
        logger.info(
            "Remocao de atributos concluida: "
            f"total={total_removed:,}, train={train_removed:,}, "
            f"valid={valid_removed:,}, test={test_removed:,}"
        )

    stats = {
        "removed": total_removed,
        "removed_by_split": {
            "train": train_removed,
            "valid": valid_removed,
            "test": test_removed,
        },
        "blocked_relations": blocked_seen,
    }

    return train_filtered, valid_filtered, test_filtered, stats
