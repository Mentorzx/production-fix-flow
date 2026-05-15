"""Regression tests for HPO data loader fallback paths."""

from __future__ import annotations

import polars as pl

from pff.infrastructure.hpo.trials import data_loader
from pff.infrastructure.persistence.db import repositories


class _InMemoryFileManager:
    def __init__(self, state: dict[str, bool]) -> None:
        self._state = state
        self._train = pl.DataFrame({"s": [0, 1], "p": [0, 1], "o": [1, 0]})
        self._valid = pl.DataFrame({"s": [1], "p": [1], "o": [0]})

    def exists(self, path) -> bool:
        path_str = str(path)
        if (
            "train.preprocessed.parquet" in path_str
            or "valid.preprocessed.parquet" in path_str
        ):
            return False
        if "train.parquet" in path_str or "valid.parquet" in path_str:
            return bool(self._state["materialized"])
        if "test.parquet" in path_str:
            return False
        return False

    def read(self, path):
        path_str = str(path)
        if "train.parquet" in path_str:
            return self._train
        if "valid.parquet" in path_str:
            return self._valid
        raise FileNotFoundError(path_str)


class _FakeSplitsRepository:
    async def delete_preprocessed(self) -> None:
        return None

    async def save_preprocessed_splits(self, train_df, valid_df, test_df=None) -> None:
        return None


def test_load_from_parquet_and_push_materializes_from_correct_when_missing(monkeypatch):
    """Should materialize raw splits before parquet fallback when no local splits exist."""
    state = {"materialized": False}
    fm = _InMemoryFileManager(state)
    calls = {"materialize": 0}

    def _materialize(config_path=None) -> bool:
        calls["materialize"] += 1
        state["materialized"] = True
        return True

    monkeypatch.setattr(
        data_loader, "_materialize_raw_splits_from_correct_parquet", _materialize
    )
    monkeypatch.setattr(
        data_loader, "_mirror_preprocessed_to_lance", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(repositories, "KGSplitsRepository", _FakeSplitsRepository)

    loaded = data_loader._load_from_parquet_and_push(
        preprocessing_config=None,
        file_manager=fm,  # type: ignore[arg-type]
    )

    assert calls["materialize"] == 1
    assert loaded is not None
    train_df, valid_df, test_df = loaded
    assert len(train_df) == 2
    assert len(valid_df) == 1
    assert test_df is None


def test_load_preprocessed_from_postgres_uses_parquet_fallback(monkeypatch):
    """Should return parquet fallback payload when postgres/preprocess paths are unavailable."""
    train_df = pl.DataFrame({"s": [0], "p": [0], "o": [1]})
    valid_df = pl.DataFrame({"s": [1], "p": [0], "o": [0]})

    monkeypatch.setattr(data_loader, "HAS_PREPROCESSING_MODULE", False)
    monkeypatch.setattr(data_loader, "_get_local_baseline_counts", lambda _fm: None)
    monkeypatch.setattr(data_loader, "_get_postgres_raw_baseline", lambda: None)
    monkeypatch.setattr(data_loader, "_is_postgres_storage_enabled", lambda _fm: True)
    monkeypatch.setattr(data_loader, "run_coroutine_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        data_loader, "_get_preprocessed_parquet_baseline", lambda _fm: None
    )
    monkeypatch.setattr(
        data_loader, "_try_load_existing_preprocessed", lambda **kwargs: None
    )
    monkeypatch.setattr(
        data_loader, "_try_populate_and_reload_preprocessed", lambda **kwargs: None
    )
    monkeypatch.setattr(
        data_loader,
        "_load_from_parquet_and_push",
        lambda preprocessing_config, file_manager, config_path=None: (
            train_df,
            valid_df,
            None,
        ),
    )

    loaded_train, loaded_valid, info = data_loader.load_preprocessed_from_postgres(
        file_manager=_InMemoryFileManager({"materialized": True}),  # type: ignore[arg-type]
        require_preprocessed=True,
        auto_populate_if_missing=True,
        allow_fallback=True,
    )

    assert len(loaded_train) == 1
    assert len(loaded_valid) == 1
    assert info["source"] == "parquet_fallback"
    assert info["populated_by"] == "parquet_fallback"


def test_try_populate_and_reload_preprocessed_falls_back_to_correct_parquet(
    monkeypatch,
):
    """When KG pipeline population fails, loader should fallback to correct.parquet path."""
    train_df = pl.DataFrame({"s": [0, 1], "p": [0, 0], "o": [1, 0]})
    valid_df = pl.DataFrame({"s": [1], "p": [0], "o": [0]})

    monkeypatch.setattr(
        data_loader, "_populate_preprocessed_splits", lambda **kwargs: False
    )
    monkeypatch.setattr(
        data_loader,
        "_load_from_parquet_and_push",
        lambda preprocessing_config, file_manager, config_path=None: (
            train_df,
            valid_df,
            None,
        ),
    )

    loaded = data_loader._try_populate_and_reload_preprocessed(
        preprocessing_config=None,
        baseline_counts=None,
        config_path=None,
    )

    assert loaded is not None
    loaded_train, loaded_valid, info = loaded
    assert len(loaded_train) == 2
    assert len(loaded_valid) == 1
    assert info["source"] == "correct_parquet"
    assert info["populated_by"] == "correct_parquet"


def test_materialize_raw_splits_uses_builder_config(monkeypatch, tmp_path):
    """Should respect max_members and other builder configs from kg.yaml."""
    import yaml

    builder_calls = {}

    class _FakeKGBuilder:
        def __init__(self, **kwargs):
            builder_calls["init_kwargs"] = kwargs

        async def run(self):
            return None

    # Mock imports inside the function
    monkeypatch.setattr("pff.domain.kg.builder.KGBuilder", _FakeKGBuilder)
    monkeypatch.setattr(
        "pff.infrastructure.persistence.db.repositories.KGSplitsRepository",
        _FakeSplitsRepository,
    )

    def _consume_coroutine(coro, **_kwargs):
        coro.close()
        return None

    monkeypatch.setattr(data_loader, "run_coroutine_in_new_loop", _consume_coroutine)

    config_path = tmp_path / "kg.yaml"
    config_data = {
        "paths": {"data_dir": "./data", "output_dir": "kg"},
        "builder": {
            "source_path": "data/models/correct.parquet",
            "parallel": False,
            "disk_cache": True,
            "max_members": 300,
        },
    }
    config_path.write_text(yaml.dump(config_data))

    data_loader._materialize_raw_splits_from_correct_parquet(config_path=config_path)

    assert builder_calls["init_kwargs"]["max_members"] == 300
    assert builder_calls["init_kwargs"]["parallel"] is False
    assert builder_calls["init_kwargs"]["disk_cache"] is True


def test_cast_preindexed_string_ids_preserves_existing_ids() -> None:
    """Pre-indexed numeric string IDs should be cast, not remapped."""
    train_df = pl.DataFrame(
        {
            "s": ["10", "11", "12"],
            "p": ["5", "7", "5"],
            "o": ["11", "12", "10"],
        }
    )
    valid_df = pl.DataFrame(
        {
            "s": ["12"],
            "p": ["7"],
            "o": ["10"],
        }
    )

    cast_train, cast_valid = data_loader._cast_preindexed_string_ids(train_df, valid_df)

    assert cast_train is not None
    assert cast_valid is not None
    assert cast_train["s"].to_list() == [10, 11, 12]
    assert cast_train["p"].to_list() == [5, 7, 5]
    assert cast_valid["p"].to_list() == [7]


def test_cast_preindexed_string_ids_ignores_semantic_labels() -> None:
    """Semantic string triples must still go through canonical ID mapping."""
    train_df = pl.DataFrame(
        {
            "s": ["account_1", "account_2"],
            "p": ["billCycleChangeType", "status"],
            "o": ["x", "y"],
        }
    )
    valid_df = pl.DataFrame(
        {
            "s": ["account_2"],
            "p": ["status"],
            "o": ["x"],
        }
    )

    cast_train, cast_valid = data_loader._cast_preindexed_string_ids(train_df, valid_df)

    assert cast_train is None
    assert cast_valid is None


def test_remap_preindexed_token_ids_maps_hex_tokens_deterministically() -> None:
    """Hex/UUID-like IDs should be remapped to stable int64 IDs per column."""
    train_df = pl.DataFrame(
        {
            "s": ["94B2AE7D1E714C008C59CBFA", "A02F1069DE0D4F5DBD8E5B12"],
            "p": ["REL_A", "REL_B"],
            "o": ["A02F1069DE0D4F5DBD8E5B12", "94B2AE7D1E714C008C59CBFA"],
        }
    )
    valid_df = pl.DataFrame(
        {
            "s": ["A02F1069DE0D4F5DBD8E5B12"],
            "p": ["REL_A"],
            "o": ["94B2AE7D1E714C008C59CBFA"],
        }
    )

    mapped_train, mapped_valid = data_loader._remap_preindexed_token_ids(
        train_df, valid_df
    )

    assert mapped_train is not None
    assert mapped_valid is not None
    assert mapped_train["s"].dtype == pl.Int64
    assert mapped_train["p"].dtype == pl.Int64
    assert mapped_train["o"].dtype == pl.Int64
    assert mapped_train["s"].n_unique() == 2
    assert mapped_train["p"].n_unique() == 2
    assert mapped_train["o"].n_unique() == 2
    assert mapped_train["s"][0] == mapped_train["o"][1]
    assert mapped_train["s"][1] == mapped_train["o"][0]
    assert mapped_valid["s"][0] == mapped_train["s"][1]
    assert mapped_valid["o"][0] == mapped_train["s"][0]


def test_load_inverse_filter_settings_reads_defaults() -> None:
    """Inverse filter settings should be sourced from optimization defaults."""

    class _FakeFileManager:
        def read(self, _path):
            return {
                "defaults": {
                    "inverse_relation_policy": "drop_suffix",
                    "inverse_suffix": "_reverse",
                }
            }

    policy, suffix = data_loader._load_inverse_filter_settings(_FakeFileManager())  # type: ignore[arg-type]

    assert policy == "drop_suffix"
    assert suffix == "_reverse"


def test_apply_inverse_relation_policy_drops_suffix_for_string_relations() -> None:
    """String relation labels ending with suffix should be removed."""
    train_df = pl.DataFrame({"s": ["A", "B"], "p": ["rel", "rel_inv"], "o": ["B", "A"]})
    valid_df = pl.DataFrame({"s": ["C"], "p": ["rel_inv"], "o": ["A"]})

    filtered_train, filtered_valid, _filtered_test, stats = (
        data_loader._apply_inverse_relation_policy(
            train_df,
            valid_df,
            None,
            policy=data_loader.INVERSE_POLICY_DROP_SUFFIX,
            inverse_suffix="_inv",
            file_manager=_InMemoryFileManager({"materialized": True}),  # type: ignore[arg-type]
            preprocessing_config=None,
        )
    )

    assert filtered_train["p"].to_list() == ["rel"]
    assert filtered_valid["p"].to_list() == []
    assert stats["removed"] == 2
    assert stats["removed_by_split"] == {"train": 1, "valid": 1, "test": 0}


def test_apply_inverse_relation_policy_drops_integer_ids_from_relation_map(
    monkeypatch,
) -> None:
    """Integer relation IDs should be filtered using relation-map inverse IDs."""
    train_df = pl.DataFrame({"s": [0, 1, 2], "p": [1, 3, 5], "o": [1, 2, 0]})
    valid_df = pl.DataFrame({"s": [2, 3], "p": [5, 1], "o": [0, 1]})

    monkeypatch.setattr(
        data_loader,
        "_resolve_inverse_relation_ids",
        lambda **_kwargs: ({3, 5}, ["r3_inv", "r5_inv"]),
    )

    filtered_train, filtered_valid, _filtered_test, stats = (
        data_loader._apply_inverse_relation_policy(
            train_df,
            valid_df,
            None,
            policy=data_loader.INVERSE_POLICY_DROP_SUFFIX,
            inverse_suffix="_inv",
            file_manager=_InMemoryFileManager({"materialized": True}),  # type: ignore[arg-type]
            preprocessing_config=None,
        )
    )

    assert filtered_train["p"].to_list() == [1]
    assert filtered_valid["p"].to_list() == [1]
    assert stats["removed"] == 3
    assert stats["filtered_relation_ids"] == [3, 5]


def test_load_preprocessed_from_postgres_applies_inverse_policy(monkeypatch) -> None:
    """Top-level loader should apply inverse filtering before returning data."""
    train_df = pl.DataFrame({"s": ["A", "B"], "p": ["rel", "rel_inv"], "o": ["B", "A"]})
    valid_df = pl.DataFrame({"s": ["B"], "p": ["rel_inv"], "o": ["A"]})
    base_info = {"source": "postgresql_preprocessed", "attribute_filter": {}}

    monkeypatch.setattr(data_loader, "HAS_PREPROCESSING_MODULE", False)
    monkeypatch.setattr(data_loader, "_get_local_baseline_counts", lambda _fm: None)
    monkeypatch.setattr(data_loader, "_get_postgres_raw_baseline", lambda: None)
    monkeypatch.setattr(data_loader, "_is_postgres_storage_enabled", lambda _fm: True)
    monkeypatch.setattr(data_loader, "run_coroutine_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        data_loader, "_get_preprocessed_parquet_baseline", lambda _fm: None
    )
    monkeypatch.setattr(
        data_loader,
        "_try_load_existing_preprocessed",
        lambda **kwargs: (train_df, valid_df, dict(base_info)),
    )
    monkeypatch.setattr(
        data_loader,
        "_load_inverse_filter_settings",
        lambda _fm: ("drop_suffix", "_inv"),
    )

    loaded_train, loaded_valid, info = data_loader.load_preprocessed_from_postgres(
        file_manager=_InMemoryFileManager({"materialized": True}),  # type: ignore[arg-type]
        require_preprocessed=True,
        auto_populate_if_missing=False,
        allow_fallback=False,
    )

    assert loaded_train["p"].to_list() == ["rel"]
    assert loaded_valid["p"].to_list() == []
    assert info["inverse_filter"]["removed"] == 2
