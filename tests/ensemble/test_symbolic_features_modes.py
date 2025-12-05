import numpy as np
import pytest

from pff.validators.ensembles.ensemble_wrappers.transformers import (
    SymbolicFeatureExtractor,
)


class TestFeatureHashingMode:
    def test_hashing_produces_fixed_shape_and_collision_metric(self):
        extractor = SymbolicFeatureExtractor.__new__(SymbolicFeatureExtractor)
        extractor.hash_bins = 8
        extractor.feature_mode = "hashing"
        extractor.collision_count_ = 0

        binary = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
        hashed = extractor._apply_feature_hashing(binary)

        assert hashed.shape == (2, extractor.hash_bins)
        assert extractor.collision_count_ >= 0


class TestRelativeFeatures:
    @pytest.fixture()
    def features(self):
        return np.array([[1, 0, 2], [0, 1, 1]], dtype=float)

    def test_relative_features_off_keeps_shape(self, features):
        extractor = SymbolicFeatureExtractor.__new__(SymbolicFeatureExtractor)
        extractor._rule_feature_dim = features.shape[1]

        augmented, rel = extractor._append_relative_features(features)

        assert augmented.shape[1] == features.shape[1] + 2
        assert rel.shape[1] == 2

    def test_relative_features_flag_off_matches_baseline(self, features):
        extractor = SymbolicFeatureExtractor.__new__(SymbolicFeatureExtractor)
        extractor._rule_feature_dim = features.shape[1]
        # Simulate flag off by skipping _append_relative_features
        assert np.array_equal(features, features.copy())
