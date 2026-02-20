"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/preprocessing/test_id_mapping.py

"""

from __future__ import annotations

import polars as pl

from pff.domain.kg.preprocessing.config import PreprocessingConfig
from pff.domain.kg.preprocessing.pipeline import KGPreprocessingPipeline


def test_map_ids_converts_text_to_ints_and_persists_paths(tmp_path) -> None:
    """Execute test map ids converts text to ints and persists paths.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = PreprocessingConfig(output_dir=str(tmp_path))
    pipeline = KGPreprocessingPipeline(config)
    df = pl.DataFrame({"s": ["a", "b"], "p": ["r1", "r2"], "o": ["c", "a"]})

    mapped, meta = pipeline._map_ids(df, source="test")

    assert mapped["s"].dtype.is_integer()
    assert mapped["p"].dtype.is_integer()
    assert mapped["o"].dtype.is_integer()
    assert "entity_map_path" in meta and "relation_map_path" in meta


def test_map_ids_for_splits_uses_shared_mapping() -> None:
    """Execute test map ids for splits uses shared mapping.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = PreprocessingConfig()
    pipeline = KGPreprocessingPipeline(config)
    train = pl.DataFrame({"s": ["a"], "p": ["r1"], "o": ["b"]})
    valid = pl.DataFrame({"s": ["b"], "p": ["r2"], "o": ["a"]})
    test = pl.DataFrame({"s": ["c"], "p": ["r2"], "o": ["b"]})

    mapped_train, mapped_valid, mapped_test = pipeline._map_ids_for_splits(train, valid, test)
    assert mapped_valid is not None
    assert mapped_test is not None

    # All splits should reuse the same mapping; check consistent max entity id (>=2)
    entity_ids = (
        mapped_train["s"].to_list()
        + mapped_train["o"].to_list()
        + mapped_valid["s"].to_list()
        + mapped_valid["o"].to_list()
        + mapped_test["s"].to_list()
        + mapped_test["o"].to_list()
    )
    max_entity_id = max(entity_ids)
    assert max_entity_id >= 2
    assert mapped_train["p"].dtype.is_integer()
