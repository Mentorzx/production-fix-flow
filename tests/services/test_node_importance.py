import polars as pl

from pff.validators.kg.optimizer import NodeImportanceService


def _sample_df():
    return pl.DataFrame({"s": ["a", "b", "a"], "o": ["b", "c", "c"]})


def test_node_importance_fallback_to_degree_on_timeout(caplog):
    svc = NodeImportanceService()
    df = _sample_df()
    result = svc.compute(df, method="pagerank", timeout_seconds=0)
    assert result
    assert result.get("a", 0.0) > 0.0
    # Pagerank with timeout falls back to degree, so b and c exist
    assert all(node in result for node in ("b", "c"))


def test_node_importance_uses_cache_hits():
    svc = NodeImportanceService()
    df = _sample_df()
    first = svc.compute(df, method="degree")
    second = svc.compute(df, method="degree")
    assert first == second
    assert svc.cache_hits == 1
