"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/preprocessing/test_homogenizer_dtype.py

"""

import polars as pl

from pff.domain.kg.preprocess import DataHomogenizer


def test_homogenize_handles_int_columns() -> None:
    """Execute test homogenize handles int columns.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    homogenizer = DataHomogenizer()
    df = pl.DataFrame({"s": [1, 2], "p": [10, 20], "o": [100, 200]})
    stats = pl.DataFrame({"p": ["10", "20"], "support": [5, 3]})

    result = homogenizer.homogenize_dataframe(
        df, stats, homogeneity_level=0.01, total_training_triples=10
    )

    assert result.schema == {"s": pl.Utf8, "p": pl.Utf8, "o": pl.Utf8}
    assert set(result["p"].to_list()) == {"10", "20"}


def test_homogenize_handles_relation_stats_with_int_relation_ids() -> None:
    """Ensure join keys are normalized even when relation stats use integer IDs."""
    homogenizer = DataHomogenizer()
    dataframe = pl.DataFrame({"s": ["1", "2"], "p": ["10", "20"], "o": ["100", "200"]})
    relation_statistics = pl.DataFrame({"p": [10, 20], "support": [5, 3]})

    result = homogenizer.homogenize_dataframe(
        dataframe, relation_statistics, homogeneity_level=0.01, total_training_triples=10
    )

    assert result.schema == {"s": pl.Utf8, "p": pl.Utf8, "o": pl.Utf8}
    assert set(result["p"].to_list()) == {"10", "20"}
