"""Regression tests for KGPreprocessor persistence hook seams."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pff.domain.kg.config import ConfigurationInterface
from pff.domain.kg.preprocess import KGPreprocessor


class _StubConfig(ConfigurationInterface):
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def validate(self) -> bool:
        return True

    def get_split_path(self, split_name: str) -> Path:
        return self._base_dir / f"{split_name}.parquet"

    def get_preprocessing_parameters(self) -> dict[str, float | int | bool]:
        return {"use_centralized_preprocessing": True}

    def get_entity_map_path(self) -> Path:
        return self._base_dir / "entity_map.json"

    def get_relation_map_path(self) -> Path:
        return self._base_dir / "relation_map.json"

    def get_max_chunk_size(self) -> int:
        return 1

    def get_mappings_directory(self) -> Path:
        return self._base_dir

    def get_calibration_config(self) -> dict:
        return {}

    def get_dask_configuration(self) -> dict:
        return {}


def test_save_preprocessed_uses_injected_hook(tmp_path: Path) -> None:
    config = _StubConfig(tmp_path)
    captured: dict[str, object] = {}

    def _save_hook(repo, splits: dict[str, pl.DataFrame]) -> None:
        captured["repo"] = repo
        captured["train_rows"] = len(splits["train"])

    preprocessor = KGPreprocessor(
        config,
        splits_repo=object(),
        save_splits_hook=_save_hook,
    )

    preprocessor._save_preprocessed_to_postgres(
        {
            "train": pl.DataFrame({"s": ["a"], "p": ["r"], "o": ["b"]}),
            "valid": pl.DataFrame(),
            "test": pl.DataFrame(),
        }
    )

    assert captured["repo"] is preprocessor.splits_repo
    assert captured["train_rows"] == 1


def test_save_mappings_uses_injected_hook(tmp_path: Path) -> None:
    config = _StubConfig(tmp_path)
    captured: dict[str, object] = {}

    def _mappings_hook(repo, entity_map: pl.DataFrame, relation_map: pl.DataFrame) -> None:
        captured["repo"] = repo
        captured["entity_rows"] = len(entity_map)
        captured["relation_rows"] = len(relation_map)

    preprocessor = KGPreprocessor(
        config,
        mappings_repo=object(),
        save_mappings_hook=_mappings_hook,
    )

    preprocessor._persist_mappings_to_database(
        pl.DataFrame({"label": ["x"], "id": [0]}),
        pl.DataFrame({"label": ["rel"], "id": [0]}),
    )

    assert captured["repo"] is preprocessor.mappings_repo
    assert captured["entity_rows"] == 1
    assert captured["relation_rows"] == 1


class _SplitsRepoStub:
    def __init__(self) -> None:
        self.saved = False

    async def delete_preprocessed(self) -> None:
        return None

    async def save_preprocessed_splits(self, **_kwargs) -> None:
        self.saved = True


class _MappingsRepoStub:
    def __init__(self) -> None:
        self.saved = False

    async def save_mappings_from_dataframe(self, *_args, **_kwargs) -> None:
        self.saved = True


def test_save_preprocessed_without_hook_uses_legacy_fallback(tmp_path: Path) -> None:
    config = _StubConfig(tmp_path)
    repo = _SplitsRepoStub()
    preprocessor = KGPreprocessor(config, splits_repo=repo)

    preprocessor._save_preprocessed_to_postgres(
        {
            "train": pl.DataFrame({"s": ["a"], "p": ["r"], "o": ["b"]}),
            "valid": pl.DataFrame(),
            "test": pl.DataFrame(),
        }
    )

    assert repo.saved is True


def test_save_mappings_without_hook_uses_legacy_fallback(tmp_path: Path) -> None:
    config = _StubConfig(tmp_path)
    repo = _MappingsRepoStub()
    preprocessor = KGPreprocessor(config, mappings_repo=repo)

    preprocessor._persist_mappings_to_database(
        pl.DataFrame({"label": ["x"], "id": [0]}),
        pl.DataFrame({"label": ["rel"], "id": [0]}),
    )

    assert repo.saved is True
