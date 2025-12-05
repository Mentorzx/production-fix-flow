from __future__ import annotations

import pytest
import polars as pl

from pff.utils import FileManager
from scripts.optimization.trials.pipeline import _compute_relation_metrics


def test_relation_coverage_and_rules_per_relation(tmp_path) -> None:
    """_compute_relation_metrics deve calcular cobertura e densidade de regras por relação."""
    rel_df = pl.DataFrame({"id": [0, 1, 2], "label": ["r1", "r2", "r3"]})
    rel_path = tmp_path / "relations.parquet"
    FileManager().save(rel_df, rel_path)

    filtered_metadata = [
        {"rule": "r1(X,Y) <= p(X,Z)"},
        {"rule": "r2(X,Y) <= q(X,Z)"},
        {"rule": "r2(X,Y) <= q(Z,Y)"},
    ]

    metrics = _compute_relation_metrics(filtered_metadata, rel_path)

    assert metrics["relation_coverage"] == pytest.approx(2 / 3)
    assert metrics["rules_per_relation"] == pytest.approx(3 / 3)
