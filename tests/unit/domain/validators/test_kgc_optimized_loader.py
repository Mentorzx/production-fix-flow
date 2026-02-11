from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any
import polars as pl

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistencePort:
    def save_checkpoint(self, checkpoint_data: dict[str, Any], filename: str) -> None:
        pass

    def load_checkpoint(
        self, filename: str, map_location=None
    ) -> dict[str, Any] | None:
        return None


def test_optimized_triple_loader(tmp_path: Path) -> None:
    """Regression test: Ensures optimized Arrow loader returns correct data."""
    device = torch.device("cpu")
    model_config = DSLFMKGCConfig(num_entities=100, num_relations=10)
    train_config = KGCTrainingConfig(epochs=1)

    manager = DSLFMKGCManager(
        model_config,
        train_config,
        persistence_port=MockPersistencePort(),
        device=device,
    )

    # 1. Create Arrow file
    triples = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    df = pl.DataFrame(triples, schema=["h", "r", "t"])
    arrow_path = tmp_path / "test.arrow"
    df.write_ipc(arrow_path, compression="uncompressed")

    # 2. Load via optimized loader
    loaded = manager._load_triples_optimized(arrow_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (2, 3)
    assert np.array_equal(loaded, triples)


def test_train_accepts_paths(tmp_path: Path) -> None:
    """Regression test: Ensures train method accepts and loads paths."""
    device = torch.device("cpu")
    model_config = DSLFMKGCConfig(num_entities=100, num_relations=10)
    # Fast config
    train_config = KGCTrainingConfig(epochs=0)  # 0 epochs to avoid actual loop

    manager = DSLFMKGCManager(
        model_config,
        train_config,
        persistence_port=MockPersistencePort(),
        device=device,
    )

    triples = np.array([[1, 2, 3]], dtype=np.int64)
    df = pl.DataFrame(triples, schema=["h", "r", "t"])
    path = tmp_path / "train.arrow"
    df.write_ipc(path)

    # This should call _load_triples_optimized internally
    # and not crash
    stats = manager.train(path, path)
    assert "best_val_mrr" in stats
