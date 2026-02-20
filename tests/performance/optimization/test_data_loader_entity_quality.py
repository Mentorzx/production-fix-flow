"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_data_loader_entity_quality.py

"""

import polars as pl
import pytest

from pff.infrastructure.hpo.trials.data_loader import compute_entity_quality_scores


def test_compute_entity_quality_scores_handles_series_concat():
    """Execute test compute entity quality scores handles series concat.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    train_df = pl.DataFrame(
        {
            "s": ["a", "b", "a"],
            "p": ["r1", "r1", "r2"],
            "o": ["x", "y", "y"],
        }
    )
    valid_df = pl.DataFrame(
        {
            "s": ["c", "b"],
            "p": ["r1", "r1"],
            "o": ["x", "x"],
        }
    )

    result = compute_entity_quality_scores(train_df, valid_df)
    degree_df = result["degree"]

    assert set(degree_df.columns) == {"entity", "degree", "degree_norm"}
    assert result["max_degree"] == 3
    assert result["n_entities_with_degree"] == degree_df.height

    x_stats = (
        degree_df.filter(pl.col("entity") == "x").select("degree", "degree_norm").to_dicts()[0]
    )
    assert x_stats["degree"] == 3
    assert x_stats["degree_norm"] == pytest.approx(1.0)
