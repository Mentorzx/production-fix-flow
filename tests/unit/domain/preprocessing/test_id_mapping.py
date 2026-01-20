from __future__ import annotations

import polars as pl

from pff.domain.kg.preprocessing.config import PreprocessingConfig
from pff.domain.kg.preprocessing.pipeline import KGPreprocessingPipeline


def test_map_ids_converts_text_to_ints_and_persists_paths(tmp_path) -> None:
    config = PreprocessingConfig(output_dir=str(tmp_path))
    pipeline = KGPreprocessingPipeline(config)
    df = pl.DataFrame({"s": ["a", "b"], "p": ["r1", "r2"], "o": ["c", "a"]})

    mapped, meta = pipeline._map_ids(df, source="test")

    assert mapped["s"].dtype.is_integer()
    assert mapped["p"].dtype.is_integer()
    assert mapped["o"].dtype.is_integer()
    assert "entity_map_path" in meta and "relation_map_path" in meta


def test_map_ids_for_splits_uses_shared_mapping() -> None:
    config = PreprocessingConfig()
    pipeline = KGPreprocessingPipeline(config)
    train = pl.DataFrame({"s": ["a"], "p": ["r1"], "o": ["b"]})
    valid = pl.DataFrame({"s": ["b"], "p": ["r2"], "o": ["a"]})
    test = pl.DataFrame({"s": ["c"], "p": ["r2"], "o": ["b"]})

    mapped_train, mapped_valid, mapped_test = pipeline._map_ids_for_splits(train, valid, test)

    # All splits should reuse the same mapping; check consistent max entity id (>=2)
    max_entity_id = max(
        mapped_train["s"].max(),
        mapped_train["o"].max(),
        mapped_valid["s"].max(),
        mapped_valid["o"].max(),
        mapped_test["s"].max(),
        mapped_test["o"].max(),
    )
    assert max_entity_id >= 2
    assert mapped_train["p"].dtype.is_integer()
