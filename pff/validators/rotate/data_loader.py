"""RotatE Data Loader Component.

Handles data loading, preprocessing, and DataLoader creation for RotatE training.
Extracted from RotatEManager for Single Responsibility Principle (SRP).

Design Patterns Applied:
    - **Factory Pattern:** Creates configured DataLoader instances.
    - **Strategy Pattern:** Supports different negative sampling strategies.

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pff import settings
from pff.utils import FileManager, logger
from pff.utils.system.hardware_detector import HardwareDetector
from pff.validators.rotate.core import RotatEDataset
from pff.validators.rotate.negative_sampling import (
    NegativeSamplerFactory,
    NegativeSamplingStrategy,
)


def _get_optimal_num_workers() -> int:
    """Get optimal number of DataLoader workers based on hardware detection.

    Returns:
        Optimal number of workers (1-4 range).
    """
    try:
        profile = HardwareDetector.detect()
        return max(1, min(4, profile.cpu_cores - 1))
    except Exception as exc:  # noqa: BLE001 - defensive fallback
        logger.debug(f"Hardware detection fallback for workers: {exc}")
        return 2


class RotatEDataLoader:
    """Factory for creating RotatE DataLoaders with proper configuration.

    Handles data loading from multiple path conventions and creates
    optimized DataLoaders with negative sampling strategies.

    Attributes:
        file_manager: FileManager instance for I/O operations.
        entity_to_idx: Entity name to index mapping.
        idx_to_entity: Index to entity name mapping.
        relation_to_idx: Relation name to index mapping.
        idx_to_relation: Index to relation name mapping.
    """

    def __init__(self, file_manager: FileManager | None = None) -> None:
        """Initialize data loader component.

        Args:
            file_manager: FileManager instance for I/O operations.
        """
        self.file_manager = file_manager or FileManager()
        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.relation_to_idx: dict[str, int] = {}
        self.idx_to_relation: dict[int, str] = {}
        self._train_triples: np.ndarray | None = None
        self._val_triples: np.ndarray | None = None
        self._test_triples: np.ndarray | None = None

    @property
    def num_entities(self) -> int:
        """Return number of entities."""
        return len(self.entity_to_idx)

    @property
    def num_relations(self) -> int:
        """Return number of relations."""
        return len(self.relation_to_idx)

    @property
    def train_triples(self) -> np.ndarray | None:
        """Return training triples."""
        return self._train_triples

    @property
    def val_triples(self) -> np.ndarray | None:
        """Return validation triples."""
        return self._val_triples

    @property
    def test_triples(self) -> np.ndarray | None:
        """Return test triples."""
        return self._test_triples

    def load_data(self, base_path: Path | None = None) -> None:
        """Load and prepare data for training.

        Args:
            base_path: Optional base path to search for data.

        Raises:
            FileNotFoundError: If mappings or training data not found.
        """
        possible_paths = [
            settings.OUTPUTS_DIR / "rotate",
            settings.OUTPUTS_DIR / "pyclause",
            settings.OUTPUTS_DIR / "kg",
        ]

        if base_path:
            possible_paths.insert(0, base_path)

        maps_path = self._find_mappings(possible_paths)
        self._load_mappings(maps_path)
        self._load_triples(maps_path)

        logger.info(
            f"Dados carregados: entities={self.num_entities:,}, "
            f"relations={self.num_relations:,}, "
            f"train={len(self._train_triples) if self._train_triples is not None else 0:,}"
        )

    def _find_mappings(self, paths: list[Path]) -> Path:
        """Find directory containing entity/relation mappings.

        Args:
            paths: List of paths to search.

        Returns:
            Path to directory with mappings.

        Raises:
            FileNotFoundError: If mappings not found in any path.
        """
        for path in paths:
            if not path.exists():
                continue

            entity_candidates = [
                path / "rotate_entity_map.parquet",
                path / "entity_map.parquet",
            ]
            relation_candidates = [
                path / "rotate_relation_map.parquet",
                path / "relation_map.parquet",
            ]

            entity_found = any(p.exists() for p in entity_candidates)
            relation_found = any(p.exists() for p in relation_candidates)

            if entity_found and relation_found:
                return path

        raise FileNotFoundError(
            f"Entity/relation mappings not found in: {[str(p) for p in paths]}"
        )

    def _load_mappings(self, maps_path: Path) -> None:
        """Load entity and relation mappings from path.

        Args:
            maps_path: Directory containing mapping files.
        """
        # Find entity mapping file
        entity_path = None
        for name in ["rotate_entity_map.parquet", "entity_map.parquet"]:
            candidate = maps_path / name
            if candidate.exists():
                entity_path = candidate
                break

        # Find relation mapping file
        relation_path = None
        for name in ["rotate_relation_map.parquet", "relation_map.parquet"]:
            candidate = maps_path / name
            if candidate.exists():
                relation_path = candidate
                break

        if not entity_path or not relation_path:
            raise FileNotFoundError(f"Mapping files not found in {maps_path}")

        entity_df = self.file_manager.read(entity_path)
        relation_df = self.file_manager.read(relation_path)

        # Detect column naming and build mappings
        self.entity_to_idx, self.idx_to_entity = self._parse_mapping_df(entity_df)
        self.relation_to_idx, self.idx_to_relation = self._parse_mapping_df(relation_df)

        logger.info(f"Mapeamentos carregados de {maps_path}")

    def _parse_mapping_df(self, df: Any) -> tuple[dict[str, int], dict[int, str]]:
        """Parse mapping DataFrame with flexible column detection.

        Args:
            df: DataFrame with index and label columns.

        Returns:
            Tuple of (name_to_idx, idx_to_name) dictionaries.
        """
        cols = list(df.columns)

        # Common conventions: (id, label), (idx, entity/relation), (index, name)
        if "label" in cols and "id" in cols:
            name_to_idx = dict(zip(df["label"], df["id"]))
            idx_to_name = dict(zip(df["id"], df["label"]))
        elif "entity" in cols and "idx" in cols:
            name_to_idx = dict(zip(df["entity"], df["idx"]))
            idx_to_name = dict(zip(df["idx"], df["entity"]))
        elif "relation" in cols and "idx" in cols:
            name_to_idx = dict(zip(df["relation"], df["idx"]))
            idx_to_name = dict(zip(df["idx"], df["relation"]))
        else:
            # Assume first two columns are (idx, name)
            name_to_idx = dict(zip(df[cols[1]], df[cols[0]]))
            idx_to_name = dict(zip(df[cols[0]], df[cols[1]]))

        return name_to_idx, idx_to_name

    def _load_triples(self, maps_path: Path) -> None:
        """Load training, validation, and test triples.

        Args:
            maps_path: Directory containing triple files.
        """
        # Training data
        train_path = maps_path / "train_indexed.npy"
        if train_path.exists():
            self._train_triples = self.file_manager.read(train_path)
        else:
            train_parquet = maps_path / "train.homogenized.parquet"
            if train_parquet.exists():
                self._train_triples = self._convert_parquet_to_indexed(train_parquet)
            else:
                raise FileNotFoundError(f"Training data not found in {maps_path}")

        # Validation data (optional)
        val_path = maps_path / "valid_indexed.npy"
        if val_path.exists():
            self._val_triples = self.file_manager.read(val_path)
        else:
            val_parquet = maps_path / "valid.homogenized.parquet"
            if val_parquet.exists():
                self._val_triples = self._convert_parquet_to_indexed(val_parquet)

        # Test data (optional)
        test_path = maps_path / "test_indexed.npy"
        if test_path.exists():
            self._test_triples = self.file_manager.read(test_path)
        else:
            test_parquet = maps_path / "test.homogenized.parquet"
            if test_parquet.exists():
                self._test_triples = self._convert_parquet_to_indexed(test_parquet)

    def _convert_parquet_to_indexed(self, parquet_path: Path) -> np.ndarray:
        """Convert parquet triples to indexed numpy array.

        Args:
            parquet_path: Path to parquet file.

        Returns:
            Indexed triples array of shape [n_triples, 3].
        """
        df = self.file_manager.read(parquet_path)
        cols = list(df.columns)

        # Detect column names
        if "s" in cols:
            head_col, rel_col, tail_col = "s", "p", "o"
        elif "head" in cols:
            head_col, rel_col, tail_col = "head", "relation", "tail"
        elif "subject" in cols:
            head_col, rel_col, tail_col = "subject", "predicate", "object"
        else:
            head_col, rel_col, tail_col = cols[0], cols[1], cols[2]

        try:
            heads = df[head_col].to_list()
            rels = df[rel_col].to_list()
            tails = df[tail_col].to_list()
        except AttributeError:
            heads = df[head_col].tolist()
            rels = df[rel_col].tolist()
            tails = df[tail_col].tolist()

        indexed = []
        for h, r, t in zip(heads, rels, tails):
            h_idx = self.entity_to_idx.get(str(h), 0)
            r_idx = self.relation_to_idx.get(str(r), 0)
            t_idx = self.entity_to_idx.get(str(t), 0)
            indexed.append([h_idx, r_idx, t_idx])

        return np.array(indexed, dtype=np.int64)

    def create_dataloader(
        self,
        triples: np.ndarray | None = None,
        batch_size: int = 1024,
        num_negatives: int = 256,
        seed: int = 42,
        num_workers: int | None = None,
        device: torch.device | None = None,
        sota_config: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Create a configured DataLoader for training.

        Args:
            triples: Training triples (uses loaded train_triples if None).
            batch_size: Batch size for DataLoader.
            num_negatives: Number of negative samples per positive.
            seed: Random seed for reproducibility.
            num_workers: Number of DataLoader workers (auto-detected if None).
            device: Device for pin_memory optimization.
            sota_config: SOTA configuration from rotate.yaml.

        Returns:
            Configured DataLoader instance.

        Raises:
            ValueError: If no training triples available.
        """
        if triples is None:
            triples = self._train_triples

        if triples is None:
            raise ValueError("No training triples available")

        # Create dataset
        dataset = RotatEDataset(
            triples,
            num_entities=self.num_entities,
            num_negatives=num_negatives,
            seed=seed,
        )

        # Determine num_workers
        if num_workers is None:
            num_workers = _get_optimal_num_workers()

        # Determine pin_memory
        pin_memory = device is not None and device.type == "cuda"

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataloader

    def create_negative_sampler(
        self,
        device: torch.device,
        sota_config: dict[str, Any] | None = None,
    ) -> Any:
        """Create a negative sampler based on SOTA configuration.

        Args:
            device: Device for sampler tensors.
            sota_config: SOTA configuration section from rotate.yaml.

        Returns:
            Configured NegativeSampler instance.
        """
        if sota_config is None:
            sota_config = {}

        type_constraints = sota_config.get("type_constraints", {})
        use_type_constraints = type_constraints.get("enabled", False)
        use_relation_aware = sota_config.get("relation_aware_negatives", False)

        if use_type_constraints:
            strategy = NegativeSamplingStrategy.TYPE_CONSTRAINED
            logger.info("Usando amostragem negativa com restricoes de tipo")
        elif use_relation_aware:
            strategy = NegativeSamplingStrategy.RELATION_AWARE
            logger.info("Usando amostragem negativa relation-aware")
        else:
            strategy = NegativeSamplingStrategy.UNIFORM
            logger.info("Usando amostragem negativa uniforme")

        sampler = NegativeSamplerFactory.create(
            strategy=strategy,
            num_entities=self.num_entities,
            device=device,
            num_relations=self.num_relations,
        )

        # Build frequency tables for relation-aware sampling
        if use_relation_aware and self._train_triples is not None:
            train_tensor = torch.tensor(self._train_triples, device=device)
            sampler.build_frequency_tables(train_tensor)

        return sampler
