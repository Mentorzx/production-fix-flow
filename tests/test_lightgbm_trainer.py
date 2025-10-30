import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
import torch

from pff.validators.transe.lightgbm_trainer import TransELightGBMTrainer


@pytest.fixture
def mock_transe_manager():
    manager = Mock()
    manager.config = {
        "model": {
            "embedding_dim": 64,
            "margin": 1.0,
            "learning_rate": 0.001,
        }
    }

    manager.entity_to_idx = {"entity_1": 0, "entity_2": 1, "entity_3": 2}
    manager.relation_to_idx = {"relation_1": 0, "relation_2": 1}

    entity_embeddings = np.random.randn(3, 64).astype(np.float32)
    relation_embeddings = np.random.randn(2, 64).astype(np.float32)

    manager.node_embeddings = {
        "entity_embeddings": entity_embeddings,
        "relation_embeddings": relation_embeddings,
    }

    return manager


@pytest.fixture
def sample_triples_df():
    return pl.DataFrame({
        "head": ["entity_1", "entity_2", "entity_1"],
        "relation": ["relation_1", "relation_2", "relation_1"],
        "tail": ["entity_2", "entity_3", "entity_3"],
    })


@pytest.fixture
def temp_data_path(tmp_path, sample_triples_df):
    data_path = tmp_path / "test_triples.parquet"
    sample_triples_df.write_parquet(data_path)
    return data_path


class TestTransELightGBMTrainerInit:

    def test_trainer_initialization(self, mock_transe_manager):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        assert trainer.transe_manager == mock_transe_manager
        assert trainer.embedding_dim == 64
        assert trainer.negative_ratio == 1
        assert trainer.lightgbm_model is None

    def test_trainer_stores_file_manager(self, mock_transe_manager):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        assert trainer.file_manager is not None
        assert hasattr(trainer.file_manager, 'read')


class TestCreateLightGBMDataset:

    def test_create_dataset_basic(self, mock_transe_manager, temp_data_path):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(metadata, dict)
        assert len(X) == len(y)

    def test_create_dataset_feature_dimensions(self, mock_transe_manager, temp_data_path):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

        expected_features = 64 * 3 + 1 + 64 + 64 + 3
        assert X.shape[1] == expected_features

    def test_create_dataset_handles_missing_embeddings(self, mock_transe_manager, temp_data_path):
        manager = mock_transe_manager
        del manager.node_embeddings

        trainer = TransELightGBMTrainer(manager)

        with patch.object(trainer, 'extract_embeddings') as mock_extract:
            mock_extract.return_value = {
                "entity_embeddings": np.random.randn(3, 64).astype(np.float32),
                "relation_embeddings": np.random.randn(2, 64).astype(np.float32),
            }

            X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

            mock_extract.assert_called_once()

    def test_create_dataset_skips_unknown_entities(self, mock_transe_manager, tmp_path):
        df_with_unknowns = pl.DataFrame({
            "head": ["entity_1", "unknown_entity", "entity_2"],
            "relation": ["relation_1", "relation_1", "relation_2"],
            "tail": ["entity_2", "entity_3", "entity_3"],
        })

        data_path = tmp_path / "test_unknowns.parquet"
        df_with_unknowns.write_parquet(data_path)

        trainer = TransELightGBMTrainer(mock_transe_manager)
        X, y, metadata = trainer.create_lightgbm_dataset(data_path)

        assert len(X) < 3

    def test_create_dataset_handles_spo_columns(self, mock_transe_manager, tmp_path):
        df_spo = pl.DataFrame({
            "s": ["entity_1", "entity_2"],
            "p": ["relation_1", "relation_2"],
            "o": ["entity_2", "entity_3"],
        })

        data_path = tmp_path / "test_spo.parquet"
        df_spo.write_parquet(data_path)

        trainer = TransELightGBMTrainer(mock_transe_manager)
        X, y, metadata = trainer.create_lightgbm_dataset(data_path)

        assert len(X) == 2


@pytest.mark.slow
class TestLightGBMTraining:

    def test_train_creates_model(self, mock_transe_manager, temp_data_path):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

        if len(X) < 10:
            pytest.skip("Not enough data for training test")

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }

        with patch("lightgbm.train") as mock_train:
            mock_train.return_value = Mock()

            trainer.train(X, y, params)

            mock_train.assert_called_once()

    def test_train_with_validation_set(self, mock_transe_manager, temp_data_path):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

        if len(X) < 20:
            pytest.skip("Not enough data for train/val split")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        assert len(X_train) > 0
        assert len(X_val) > 0


class TestFeatureEngineering:

    def test_concatenated_embeddings(self, mock_transe_manager, temp_data_path):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

        assert X.shape[1] > 64 * 3

    def test_distance_score_feature(self, mock_transe_manager, temp_data_path):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

        distance_scores = X[:, 64*3]
        assert np.all(distance_scores <= 0)

    def test_hadamard_product_feature(self, mock_transe_manager, temp_data_path):
        trainer = TransELightGBMTrainer(mock_transe_manager)

        X, y, metadata = trainer.create_lightgbm_dataset(temp_data_path)

        hadamard_start = 64*3 + 1
        hadamard_features = X[:, hadamard_start:hadamard_start+64]
        assert hadamard_features.shape[1] == 64


class TestEdgeCases:

    def test_empty_dataset(self, mock_transe_manager, tmp_path):
        empty_df = pl.DataFrame({
            "head": [],
            "relation": [],
            "tail": [],
        })

        data_path = tmp_path / "empty.parquet"
        empty_df.write_parquet(data_path)

        trainer = TransELightGBMTrainer(mock_transe_manager)

        with pytest.raises(ValueError, match="Nenhuma tripla válida encontrada"):
            trainer.create_lightgbm_dataset(data_path)

    def test_single_triple_dataset(self, mock_transe_manager, tmp_path):
        single_df = pl.DataFrame({
            "head": ["entity_1"],
            "relation": ["relation_1"],
            "tail": ["entity_2"],
        })

        data_path = tmp_path / "single.parquet"
        single_df.write_parquet(data_path)

        trainer = TransELightGBMTrainer(mock_transe_manager)
        X, y, metadata = trainer.create_lightgbm_dataset(data_path)

        assert len(X) == 1

    def test_all_unknown_entities(self, mock_transe_manager, tmp_path):
        unknown_df = pl.DataFrame({
            "head": ["unknown_1", "unknown_2"],
            "relation": ["unknown_rel", "unknown_rel"],
            "tail": ["unknown_3", "unknown_4"],
        })

        data_path = tmp_path / "unknown.parquet"
        unknown_df.write_parquet(data_path)

        trainer = TransELightGBMTrainer(mock_transe_manager)

        with pytest.raises(ValueError, match="Nenhuma tripla válida encontrada"):
            trainer.create_lightgbm_dataset(data_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
