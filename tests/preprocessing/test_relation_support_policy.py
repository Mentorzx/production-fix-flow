from __future__ import annotations

import polars as pl

from pff.preprocessing.strategies import RelationSupportFilter


def test_relation_support_warn_keeps_all() -> None:
    df = pl.DataFrame({"s": [0, 1, 2], "p": [0, 1, 1], "o": [1, 2, 0]})
    filt = RelationSupportFilter(min_support=2, policy="warn")

    result = filt.process(df)

    assert result.data.shape[0] == 3
    assert result.stats["rare_relations"] == 1


def test_relation_support_drop_filters_rare() -> None:
    df = pl.DataFrame({"s": [0, 1, 2], "p": [0, 1, 1], "o": [1, 2, 0]})
    filt = RelationSupportFilter(min_support=2, policy="drop")

    result = filt.process(df)

    assert result.data.shape[0] == 2
    assert result.stats["relations_removed"] == 1
