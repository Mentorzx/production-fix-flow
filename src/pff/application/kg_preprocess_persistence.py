"""Persistence bridge for KG preprocessing side effects.

Keeps async-to-sync execution outside domain orchestration.
"""

from __future__ import annotations

import polars as pl

from pff.shared.acceleration.asyncio_runner import run_coroutine_sync


def persist_preprocessed_splits_sync(
    splits_repo, splits: dict[str, pl.DataFrame]
) -> None:
    """Persist preprocessed splits into repository using synchronous bridge."""

    async def _save() -> None:
        await splits_repo.delete_preprocessed()
        train_df = splits.get("train")
        valid_df = splits.get("valid")
        test_df = splits.get("test")
        await splits_repo.save_preprocessed_splits(
            train_df=train_df if train_df is not None else pl.DataFrame(),
            valid_df=valid_df if valid_df is not None else pl.DataFrame(),
            test_df=test_df if test_df is not None else pl.DataFrame(),
            source="pff_learn_preprocessing",
        )

    run_coroutine_sync(_save(), timeout_s=60.0)


def persist_mappings_sync(
    mappings_repo,
    entity_map: pl.DataFrame,
    relation_map: pl.DataFrame,
) -> None:
    """Persist entity/relation mappings into repository using synchronous bridge."""

    async def _persist() -> None:
        await mappings_repo.save_mappings_from_dataframe(
            "entity", entity_map, source="preprocess"
        )
        await mappings_repo.save_mappings_from_dataframe(
            "relation", relation_map, source="preprocess"
        )

    run_coroutine_sync(_persist(), timeout_s=60.0)
