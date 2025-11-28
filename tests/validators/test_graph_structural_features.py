import polars as pl

from pff.validators.ensembles.ensemble_wrappers.transformers import (
    GraphStructuralFeatureExtractor,
)


def test_graph_structural_feature_extractor(tmp_path):
    data = pl.DataFrame(
        {
            "head": ["a", "a", "b", "c"],
            "relation": ["r1", "r2", "r1", "r2"],
            "tail": ["b", "c", "c", "a"],
        }
    )
    kg_path = tmp_path / "kg.parquet"
    data.write_parquet(kg_path)
    cache_path = tmp_path / "stats.pkl"

    extractor = GraphStructuralFeatureExtractor(kg_path=kg_path, cache_path=cache_path)
    extractor.fit([])

    sample = [[("a", "r1", "b")]]
    features = extractor.transform(sample)

    assert features.shape == (1, extractor.n_features_)
    assert (features >= 0).all()
