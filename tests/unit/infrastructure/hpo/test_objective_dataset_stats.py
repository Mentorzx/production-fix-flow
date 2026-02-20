"""Dataset stats regressions for HPO objective."""

from __future__ import annotations

import polars as pl

from pff.infrastructure.hpo.trials.objective import _infer_dataset_stats


def test_infer_dataset_stats_uses_upper_bound_for_sparse_relation_ids() -> None:
    """num_relations must honor max_id+1 for sparse integer relation IDs."""
    train_df = pl.DataFrame({"s": [0, 1], "p": [5, 7], "o": [1, 2]})
    valid_df = pl.DataFrame({"s": [2], "p": [7], "o": [0]})

    num_entities, num_relations = _infer_dataset_stats(train_df, valid_df)

    assert num_entities == 3
    assert num_relations == 8
