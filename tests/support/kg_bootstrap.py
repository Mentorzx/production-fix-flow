"""Test helpers for bootstrapping KG artifacts and preprocessed splits."""

from __future__ import annotations

import polars as pl

from pff.infrastructure.hpo.trials.data_loader import (
    HAS_PREPROCESSING_MODULE,
    _materialize_raw_splits_from_correct_parquet,
    _load_from_parquet_and_push,
    PreprocessingConfig,
    load_preprocessed_from_postgres,
)
from pff.infrastructure.persistence.db.repositories import KGSplitsRepository
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager

KG_OUTPUT_DIR = settings.OUTPUTS_DIR / "kg" / "graph"
KG_PREPROCESSED_DIR = settings.OUTPUTS_DIR / "kg" / "mappings"
PREPROCESSING_OUTPUT_DIR = settings.OUTPUTS_DIR / "preprocessing"


def _preprocessed_exists_in_postgres() -> bool:
    """Return whether canonical preprocessed train/valid splits exist in PostgreSQL."""
    repo = KGSplitsRepository()
    return bool(run_coroutine_sync(repo.preprocessed_exists(), timeout_s=20.0))


def load_existing_split(split_name: str) -> pl.DataFrame | None:
    """Load a split from the preferred preprocessed or raw artifact locations."""
    preprocessed_candidates = [
        KG_PREPROCESSED_DIR / f"{split_name}.homogenized.parquet",
        KG_PREPROCESSED_DIR / f"{split_name}.preprocessed.parquet",
        PREPROCESSING_OUTPUT_DIR / f"{split_name}_preprocessed.parquet",
    ]
    for path in preprocessed_candidates:
        if path.exists():
            return pl.read_parquet(path)

    path = KG_OUTPUT_DIR / f"{split_name}.parquet"
    if not path.exists():
        path = settings.OUTPUTS_DIR / "kg" / f"{split_name}.parquet"

    if path.exists():
        return pl.read_parquet(path)
    return None


def ensure_preprocessed_kg_in_postgres() -> None:
    """Populate PostgreSQL preprocessed splits using the canonical loader flow."""
    file_manager = FileManager()

    try:
        load_preprocessed_from_postgres(
            file_manager=file_manager,
            require_preprocessed=True,
            auto_populate_if_missing=True,
            allow_fallback=True,
        )
    except Exception:
        # The tests only care that PostgreSQL ends up populated; fall through to the explicit persistence check.
        pass

    if _preprocessed_exists_in_postgres():
        return

    preprocessing_config = (
        PreprocessingConfig.from_yaml() if HAS_PREPROCESSING_MODULE else None
    )
    loaded = _load_from_parquet_and_push(
        preprocessing_config=preprocessing_config,
        file_manager=file_manager,
        persist_to_postgres=True,
    )
    if loaded is None or not _preprocessed_exists_in_postgres():
        raise RuntimeError(
            "KG preprocessed splits could not be materialized into PostgreSQL for tests."
        )


def load_bootstrapped_kg_splits() -> dict[str, pl.DataFrame]:
    """Ensure KG artifacts exist locally and return normalized train/valid/test splits."""
    ensure_preprocessed_kg_in_postgres()

    if load_existing_split("test") is None:
        _materialize_raw_splits_from_correct_parquet()

    train = load_existing_split("train")
    valid = load_existing_split("valid")
    test = load_existing_split("test")

    if train is None or valid is None:
        raise RuntimeError("KG splits could not be bootstrapped from canonical sources.")

    return {
        "train": train,
        "valid": valid,
        "test": test if test is not None else pl.DataFrame(schema=train.schema),
    }