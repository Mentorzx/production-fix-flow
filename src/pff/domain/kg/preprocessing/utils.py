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


def _empty_filter_stats() -> dict[str, Any]:
    return {
        "removed": 0,
        "removed_by_split": {},
        "blocked_relations": [],
    }


def _should_filter_attributes(config: PreprocessingConfig) -> bool:
    """Execute should filter attributes.



    Args:

        config: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    remove_attrs = config.attribute_handling in {
        ATTRIBUTE_HANDLING_REMOVE,
        ATTRIBUTE_HANDLING_SEPARATE,
    }
    has_explicit = bool(config.attribute_relations)
    has_patterns = bool(config.attribute_patterns)
    return remove_attrs and (has_explicit or has_patterns)


def _build_blocked_relations(
    train_df: pl.DataFrame,
    config: PreprocessingConfig,
) -> set[str]:
    """Execute build blocked relations.



    Args:

        train_df: Input value used by this callable.

        config: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    blocked_relations = set(config.attribute_relations) | {
        f"{rel}{config.inverse_suffix}" for rel in config.attribute_relations
    }
    if not config.attribute_patterns:
        return blocked_relations

    import re

    compiled = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in config.attribute_patterns
    ]
    for relation in train_df["p"].unique().to_list():
        if any(pattern.search(relation) for pattern in compiled):
            blocked_relations.add(relation)
            if not relation.endswith(config.inverse_suffix):
                blocked_relations.add(f"{relation}{config.inverse_suffix}")
    return blocked_relations


def _filter_split_relations(
    df: pl.DataFrame | None,
    split_name: str,
    blocked_relations: set[str],
) -> tuple[pl.DataFrame | None, int, set[str]]:
    """Execute filter split relations.



    Args:

        df: Input value used by this callable.

        split_name: Input value used by this callable.

        blocked_relations: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if df is None:
        return None, 0, set()

    relation_expr = pl.col("p")
    if df.schema.get("p") != pl.Utf8:
        relation_expr = relation_expr.cast(pl.Utf8)

    mask = ~relation_expr.is_in(list(blocked_relations))
    filtered = df.filter(mask)
    removed = len(df) - len(filtered)
    present_blocked = (
        set(df.select(relation_expr).unique().to_series().to_list()) & blocked_relations
    )

    if removed > 0:
        logger.info(
            f"[ATRIBUTOS] Removidas {removed:,} triplas de atributo do split {split_name}"
        )
    return filtered, removed, present_blocked


def _log_filter_summary(
    *,
    total_removed: int,
    train_removed: int,
    valid_removed: int,
    test_removed: int,
) -> None:
    """Execute log filter summary.



    Args:

        total_removed: Input value used by this callable.

        train_removed: Input value used by this callable.

        valid_removed: Input value used by this callable.

        test_removed: Input value used by this callable.

    """

    if total_removed == 0:
        logger.info("Nenhuma tripla de atributo encontrada nos splits preprocessados")
        return
    logger.info(
        "Remocao de atributos concluida: "
        f"total={total_removed:,}, train={train_removed:,}, "
        f"valid={valid_removed:,}, test={test_removed:,}"
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
    if not _should_filter_attributes(config):
        return train_df, valid_df, test_df, _empty_filter_stats()

    blocked_relations = _build_blocked_relations(train_df, config)
    train_filtered, train_removed, train_blocked = _filter_split_relations(
        train_df, "train", blocked_relations
    )
    valid_filtered, valid_removed, valid_blocked = _filter_split_relations(
        valid_df, "valid", blocked_relations
    )
    test_filtered, test_removed, test_blocked = _filter_split_relations(
        test_df, "test", blocked_relations
    )

    total_removed = train_removed + valid_removed + test_removed
    blocked_seen = sorted(train_blocked | valid_blocked | test_blocked)
    _log_filter_summary(
        total_removed=total_removed,
        train_removed=train_removed,
        valid_removed=valid_removed,
        test_removed=test_removed,
    )

    stats: dict[str, Any] = {
        "removed": total_removed,
        "removed_by_split": {
            "train": train_removed,
            "valid": valid_removed,
            "test": test_removed,
        },
        "blocked_relations": blocked_seen,
    }

    assert train_filtered is not None
    return train_filtered, valid_filtered, test_filtered, stats
