import polars as pl

from pff.domain.kg.preprocessing import PreprocessingConfig, filter_attribute_relations


def test_filter_attribute_relations_removes_attributes_and_inverses():
    config = PreprocessingConfig.from_mapping(
        {
            "attribute_relations": ["id", "status"],
            "attribute_handling": "remove",
            "inverse_suffix": "_inv",
        }
    )
    train_df = pl.DataFrame(
        {
            "s": ["e1", "e2", "e3", "e4"],
            "p": ["r1", "id", "status_inv", "r2"],
            "o": ["e2", "lit1", "lit2", "e1"],
        }
    )
    valid_df = pl.DataFrame(
        {
            "s": ["e5", "e6"],
            "p": ["status", "r3"],
            "o": ["lit3", "e7"],
        }
    )

    filtered_train, filtered_valid, filtered_test, stats = filter_attribute_relations(
        train_df, valid_df, None, config
    )

    assert filtered_test is None
    assert stats["removed"] == 3
    assert stats["removed_by_split"] == {"train": 2, "valid": 1, "test": 0}
    assert stats["blocked_relations"] == ["id", "status", "status_inv"]
    assert set(filtered_train["p"].to_list()) == {"r1", "r2"}
    assert set(filtered_valid["p"].to_list()) == {"r3"}


def test_filter_attribute_relations_noop_when_mark_only():
    config = PreprocessingConfig.from_mapping(
        {
            "attribute_relations": ["id"],
            "attribute_handling": "mark",
            "inverse_suffix": "_inv",
        }
    )
    train_df = pl.DataFrame(
        {
            "s": ["e1", "e2"],
            "p": ["id", "r1"],
            "o": ["lit1", "e3"],
        }
    )

    filtered_train, filtered_valid, filtered_test, stats = filter_attribute_relations(
        train_df, None, None, config
    )

    assert stats["removed"] == 0
    assert filtered_valid is None
    assert filtered_test is None
    assert filtered_train.equals(train_df)


def test_filter_attribute_relations_patterns_match():
    config = PreprocessingConfig.from_mapping(
        {
            "attribute_relations": [],
            "attribute_patterns": [".*ExternalId", "(?i)status"],
            "attribute_handling": "remove",
            "inverse_suffix": "_inv",
        }
    )
    train_df = pl.DataFrame(
        {
            "s": ["e1", "e2", "e3"],
            "p": ["consumerExternalId", "Status_inv", "r2"],
            "o": ["lit1", "lit2", "e1"],
        }
    )

    filtered_train, filtered_valid, filtered_test, stats = filter_attribute_relations(
        train_df, None, None, config
    )

    assert filtered_valid is None
    assert filtered_test is None
    assert stats["removed"] == 2
    assert "consumerExternalId" in stats["blocked_relations"]
    assert "Status_inv" in stats["blocked_relations"]
    assert set(filtered_train["p"].to_list()) == {"r2"}


def test_filter_attribute_relations_handles_int_relations():
    config = PreprocessingConfig.from_mapping(
        {
            "attribute_relations": ["id"],
            "attribute_handling": "remove",
            "inverse_suffix": "_inv",
        }
    )
    train_df = pl.DataFrame(
        {
            "s": ["e1", "e2"],
            "p": [101, 202],
            "o": ["e3", "e4"],
        }
    )

    filtered_train, _, _, stats = filter_attribute_relations(
        train_df, None, None, config
    )

    assert stats["removed"] == 0
    assert filtered_train.schema["p"] == pl.Utf8
    assert set(filtered_train["p"].to_list()) == {"101", "202"}
