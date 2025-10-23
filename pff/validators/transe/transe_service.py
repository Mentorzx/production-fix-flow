from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pff import settings
from pff.utils import FileManager, logger
from pff.validators.kg.config import KGConfig
from pff.validators.transe.core import TransEManager
from pff.validators.transe.mapping_utils import load_mappings

"""
TransE Scorer Service

This module provides a high-level service interface for TransE model scoring
and inference. It handles model loading, triple scoring, and ranking operations.
"""


class TransEScorerService:
    """
    Service class for TransE model scoring and inference.

    This class provides a high-level interface for scoring triples,
    ranking entities, and performing link prediction tasks using a
    trained TransE model.
    """

    def __init__(
        self,
        kg_config: KGConfig,
        transe_config_path: Path,
        load_best_model: bool = True,
    ):
        """
        Initialize the TransE scorer service.

        Args:
            kg_config: Knowledge graph configuration
            transe_config_path: Path to TransE configuration file
            load_best_model: Whether to load the best model on initialization
        """
        self.file_manager = FileManager()
        self.kg_config = kg_config
        self.transe_config_path = transe_config_path
        self.transe_config = self.file_manager.read(transe_config_path)

        # Initialize mappings
        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.relation_to_idx: dict[str, int] = {}
        self.idx_to_relation: dict[int, str] = {}

        # Initialize manager
        self.transe_manager: TransEManager | None = None

        # Try to load from checkpoint first
        self._initialized = False
        if load_best_model:
            self._initialize_from_checkpoint()
            
    @property
    def device(self):
        """Get device from config or default to CPU."""
        return self.transe_config.get("training", {}).get("device", "cpu")

    def _initialize_from_checkpoint(self) -> None:
        """Initialize service from checkpoint if available."""
        checkpoint_path = Path("checkpoints") / "transe" / "best_model.pt"

        if checkpoint_path.exists():
            logger.info("🔄 Inicializando TransEScorerService do checkpoint...")

            try:
                # Load checkpoint
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )

                # Extract mappings
                if "entity_to_idx" in checkpoint and "relation_to_idx" in checkpoint:
                    self.entity_to_idx = checkpoint["entity_to_idx"]
                    self.relation_to_idx = checkpoint["relation_to_idx"]
                    self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
                    self.idx_to_relation = {
                        v: k for k, v in self.relation_to_idx.items()
                    }

                    logger.info(
                        f"✅ Mapeamentos carregados do checkpoint: "
                        f"{len(self.entity_to_idx)} entidades, "
                        f"{len(self.relation_to_idx)} relações"
                    )

                    # Initialize manager with mappings
                    self._initialize_manager_with_checkpoint()
                    self._initialized = True

                else:
                    logger.warning("⚠️ Checkpoint não contém mapeamentos")
                    self._initialize_from_files()

            except Exception as e:
                logger.error(f"❌ Erro ao carregar checkpoint: {e}")
                self._initialize_from_files()
        else:
            logger.warning(f"⚠️ Checkpoint não encontrado: {checkpoint_path}")
            self._initialize_from_files()

    def _initialize_from_files(self) -> None:
        """Initialize service from mapping files."""
        logger.info("🔄 Inicializando TransEScorerService dos arquivos...")

        # Load mappings
        maps_path = settings.OUTPUTS_DIR / "transe"
        entity_map_path = maps_path / "transe_entity_map.parquet"
        relation_map_path = maps_path / "transe_relation_map.parquet"

        # Try raw mappings if processed ones don't exist
        if not entity_map_path.exists():
            entity_map_path = maps_path / "transe_entity_map_raw.parquet"
            relation_map_path = maps_path / "transe_relation_map_raw.parquet"

        if not entity_map_path.exists() or not relation_map_path.exists():
            raise FileNotFoundError(
                f"Mapeamentos não encontrados em {maps_path}. "
                "Execute a pipeline de treinamento primeiro."
            )

        # Load mappings
        (
            self.entity_to_idx,
            self.idx_to_entity,
            self.relation_to_idx,
            self.idx_to_relation,
        ) = load_mappings(entity_map_path, relation_map_path)

        logger.info(
            f"✅ Mapeamentos carregados dos arquivos: "
            f"{len(self.entity_to_idx)} entidades, "
            f"{len(self.relation_to_idx)} relações"
        )

        # Initialize manager
        self._initialize_manager()
        self._initialized = True

    def _initialize_manager_with_checkpoint(self) -> None:
        """Initialize TransE manager with loaded checkpoint data."""
        self.transe_manager = TransEManager(
            transe_config_path=self.transe_config_path,
            kg_config_path=self.kg_config.configuration_path,
        )

        # Set mappings
        self.transe_manager.entity_to_idx = self.entity_to_idx
        self.transe_manager.relation_to_idx = self.relation_to_idx
        self.transe_manager.idx_to_entity = self.idx_to_entity
        self.transe_manager.idx_to_relation = self.idx_to_relation

        # Setup and load model
        self.transe_manager._setup_model()
        loaded = self.transe_manager._load_best_model()

        if not loaded:
            logger.warning("⚠️ Falha ao carregar melhor modelo")
            # Try any checkpoint
            checkpoints = list(self.transe_manager.checkpoint_dir.glob("*.pt"))
            if checkpoints:
                loaded = self.transe_manager._load_checkpoint(checkpoints[0])

        if loaded:
            logger.success("✅ TransEScorerService inicializado com checkpoint")
        else:
            raise RuntimeError("Falha ao carregar modelo TransE")

    def _initialize_manager(self) -> None:
        """Initialize TransE manager from scratch."""
        self.transe_manager = TransEManager(
            transe_config_path=self.transe_config_path,
            kg_config_path=self.kg_config.configuration_path,
        )

        # Manager will load its own data and model
        self.transe_manager._setup_data()
        self.transe_manager._setup_model()

        # Try to load trained model
        if not self.transe_manager._load_best_model():
            logger.warning("⚠️ Modelo treinado não encontrado")

    def _ensure_initialized(self) -> None:
        """Ensure the service is properly initialized."""
        if not self._initialized:
            self._initialize_from_files()

        if self.transe_manager is None:
            raise RuntimeError("TransE manager não inicializado")

        if self.transe_manager.model is None:
            raise RuntimeError("Modelo TransE não carregado")

    def score_triple(self, head: str, relation: str, tail: str) -> float:
        """
        Score a single triple.

        Args:
            head: Head entity label
            relation: Relation label
            tail: Tail entity label

        Returns:
            Score for the triple (higher is better)

        Raises:
            KeyError: If any entity/relation is not in vocabulary
            RuntimeError: If model is not loaded
        """
        self._ensure_initialized()

        # Check if the manager and model are initialized
        if self.transe_manager is None or self.transe_manager.model is None:
            raise RuntimeError("TransE manager ou modelo não inicializado")

        # Check if entities and relation exist
        if head not in self.entity_to_idx:
            raise KeyError(f"Entidade head não encontrada: {head}")
        if tail not in self.entity_to_idx:
            raise KeyError(f"Entidade tail não encontrada: {tail}")
        if relation not in self.relation_to_idx:
            raise KeyError(f"Relação não encontrada: {relation}")

        # Get indices
        head_idx = self.entity_to_idx[head]
        rel_idx = self.relation_to_idx[relation]
        tail_idx = self.entity_to_idx[tail]

        # Score using model
        score = self.transe_manager.model.score_triple(head_idx, rel_idx, tail_idx)

        return float(score)

    def predict_tail(
        self,
        head: str,
        relation: str,
        k: int = 10,
        filter_entities: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Predict top-k tail entities for a given (head, relation, ?).

        Args:
            head: Head entity label
            relation: Relation label
            k: Number of top entities to return
            filter_entities: Optional set of entities to filter from results

        Returns:
            List of (entity, score) tuples sorted by score (descending)
        """
        self._ensure_initialized()

        if head not in self.entity_to_idx:
            raise KeyError(f"Entidade head não encontrada: {head}")
        if relation not in self.relation_to_idx:
            raise KeyError(f"Relação não encontrada: {relation}")

        if self.transe_manager is None or self.transe_manager.model is None:
            raise RuntimeError("TransE manager ou modelo não inicializado")

        head_idx = self.entity_to_idx[head]
        rel_idx = self.relation_to_idx[relation]

        # Score all entities as potential tails
        scores = []
        with torch.no_grad():
            # Batch processing for efficiency
            batch_size = 1000
            num_entities = len(self.entity_to_idx)

            for start_idx in range(0, num_entities, batch_size):
                end_idx = min(start_idx + batch_size, num_entities)
                batch_indices = list(range(start_idx, end_idx))

                # Create batch tensors
                heads = torch.full((len(batch_indices),), head_idx, dtype=torch.long)
                relations = torch.full((len(batch_indices),), rel_idx, dtype=torch.long)
                tails = torch.tensor(batch_indices, dtype=torch.long)

                # Move to device
                device = next(self.transe_manager.model.parameters()).device
                heads = heads.to(device)
                relations = relations.to(device)
                tails = tails.to(device)

                # Get scores
                batch_scores = self.transe_manager.model(heads, relations, tails)
                scores.extend(batch_scores.cpu().numpy())

        # Convert to numpy array
        scores = np.array(scores)

        # Filter entities if requested
        if filter_entities:
            filter_indices = {
                self.entity_to_idx[e]
                for e in filter_entities
                if e in self.entity_to_idx
            }
            # Set filtered scores to very low value
            for idx in filter_indices:
                scores[idx] = -float("inf")

        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]

        # Convert to entity labels with scores
        results = []
        for idx in top_indices:
            if idx in self.idx_to_entity and scores[idx] > -float("inf"):
                entity = self.idx_to_entity[idx]
                score = float(scores[idx])
                results.append((entity, score))

        return results

    def predict_head(
        self,
        relation: str,
        tail: str,
        k: int = 10,
        filter_entities: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Predict top-k head entities for a given (?, relation, tail).

        Args:
            relation: Relation label
            tail: Tail entity label
            k: Number of top entities to return
            filter_entities: Optional set of entities to filter from results

        Returns:
            List of (entity, score) tuples sorted by score (descending)
        """
        self._ensure_initialized()

        if tail not in self.entity_to_idx:
            raise KeyError(f"Entidade tail não encontrada: {tail}")
        if relation not in self.relation_to_idx:
            raise KeyError(f"Relação não encontrada: {relation}")

        rel_idx = self.relation_to_idx[relation]
        tail_idx = self.entity_to_idx[tail]

        # Ensure model is initialized
        if self.transe_manager is None or self.transe_manager.model is None:
            raise RuntimeError("TransE manager ou modelo não inicializado")

        # Score all entities as potential heads
        scores = []
        with torch.no_grad():
            # Batch processing
            batch_size = 1000
            num_entities = len(self.entity_to_idx)

            for start_idx in range(0, num_entities, batch_size):
                end_idx = min(start_idx + batch_size, num_entities)
                batch_indices = list(range(start_idx, end_idx))

                # Create batch tensors
                heads = torch.tensor(batch_indices, dtype=torch.long)
                relations = torch.full((len(batch_indices),), rel_idx, dtype=torch.long)
                tails = torch.full((len(batch_indices),), tail_idx, dtype=torch.long)

                # Move to device
                device = next(self.transe_manager.model.parameters()).device
                heads = heads.to(device)
                relations = relations.to(device)
                tails = tails.to(device)

                # Get scores
                batch_scores = self.transe_manager.model(heads, relations, tails)
                scores.extend(batch_scores.cpu().numpy())

        # Convert to numpy array
        scores = np.array(scores)

        # Filter entities if requested
        if filter_entities:
            filter_indices = {
                self.entity_to_idx[e]
                for e in filter_entities
                if e in self.entity_to_idx
            }
            for idx in filter_indices:
                scores[idx] = -float("inf")

        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]

        # Convert to entity labels with scores
        results = []
        for idx in top_indices:
            if idx in self.idx_to_entity and scores[idx] > -float("inf"):
                entity = self.idx_to_entity[idx]
                score = float(scores[idx])
                results.append((entity, score))

        return results

    def batch_score_triples(self, triples: list[tuple[str, str, str]]) -> np.ndarray:
        """
        Score multiple triples in batch.

        Args:
            triples: List of (head, relation, tail) tuples

        Returns:
            Array of scores for each triple
        """
        self._ensure_initialized()

        # Convert to indices
        head_indices = []
        rel_indices = []
        tail_indices = []
        valid_mask = []

        for head, rel, tail in triples:
            try:
                head_indices.append(self.entity_to_idx[head])
                rel_indices.append(self.relation_to_idx[rel])
                tail_indices.append(self.entity_to_idx[tail])
                valid_mask.append(True)
            except KeyError:
                # Use dummy indices for invalid triples
                head_indices.append(0)
                rel_indices.append(0)
                tail_indices.append(0)
                valid_mask.append(False)

        # Convert to tensors
        heads = torch.tensor(head_indices, dtype=torch.long)
        relations = torch.tensor(rel_indices, dtype=torch.long)
        tails = torch.tensor(tail_indices, dtype=torch.long)

        # Ensure model is initialized
        if self.transe_manager is None or self.transe_manager.model is None:
            raise RuntimeError("TransE manager ou modelo não inicializado")

        # Move to device
        device = next(self.transe_manager.model.parameters()).device
        heads = heads.to(device)
        relations = relations.to(device)
        tails = tails.to(device)

        # Get scores
        with torch.no_grad():
            scores = self.transe_manager.model(heads, relations, tails)
            scores = scores.cpu().numpy()

        # Set invalid scores to -inf
        scores[~np.array(valid_mask)] = -float("inf")

        return scores

    def get_entity_embedding(self, entity: str) -> np.ndarray | None:
        """
        Get embedding vector for an entity.

        Args:
            entity: Entity label

        Returns:
            Embedding vector or None if entity not found
        """
        self._ensure_initialized()

        if entity not in self.entity_to_idx:
            return None

        idx = self.entity_to_idx[entity]

        if self.transe_manager is None or self.transe_manager.model is None:
            raise RuntimeError("TransE manager ou modelo não inicializado")

        with torch.no_grad():
            embedding = self.transe_manager.model.entity_embeddings.weight[idx]
            return embedding.cpu().numpy()

    def get_relation_embedding(self, relation: str) -> np.ndarray | None:
        """
        Get embedding vector for a relation.

        Args:
            relation: Relation label

        Returns:
            Embedding vector or None if relation not found
        """
        self._ensure_initialized()

        if relation not in self.relation_to_idx:
            return None

        idx = self.relation_to_idx[relation]

        if self.transe_manager is None or self.transe_manager.model is None:
            raise RuntimeError("TransE manager ou modelo não inicializado")

        with torch.no_grad():
            embedding = self.transe_manager.model.relation_embeddings.weight[idx]
            return embedding.cpu().numpy()

    def find_similar_entities(
        self, entity: str, k: int = 10, metric: str = "cosine"
    ) -> list[tuple[str, float]]:
        """
        Find k most similar entities to the given entity.

        Args:
            entity: Entity label
            k: Number of similar entities to return
            metric: Distance metric ("cosine" or "euclidean")

        Returns:
            List of (entity, similarity) tuples
        """
        self._ensure_initialized()

        if entity not in self.entity_to_idx:
            raise KeyError(f"Entidade não encontrada: {entity}")

        # Get query embedding
        query_idx = self.entity_to_idx[entity]

        # Ensure model is initialized
        if self.transe_manager is None or self.transe_manager.model is None:
            raise RuntimeError("TransE manager ou modelo não inicializado")

        with torch.no_grad():
            query_emb = self.transe_manager.model.entity_embeddings.weight[query_idx]
            all_embs = self.transe_manager.model.entity_embeddings.weight

            if metric == "cosine":
                # Normalize embeddings
                query_emb = F.normalize(query_emb, p=2, dim=0)
                all_embs = F.normalize(all_embs, p=2, dim=1)
                # Compute cosine similarity
                similarities = torch.matmul(all_embs, query_emb)
                # Get top-k (excluding query entity itself)
                values, indices = torch.topk(similarities, k + 1)
            else:  # euclidean
                # Compute L2 distances
                distances = torch.norm(all_embs - query_emb.unsqueeze(0), p=2, dim=1)
                # Get top-k smallest distances
                neg_values, indices = torch.topk(-distances, k + 1)  # Negative for ascending order
                values = -neg_values  # Convert back to positive distances

        # Convert to entity labels
        results = []
        for i in range(len(indices)):
            idx = int(indices[i].item())
            if idx != query_idx and idx in self.idx_to_entity:
                entity_label = self.idx_to_entity[idx]
                score = float(values[i])
                results.append((entity_label, score))

        return results[:k]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the loaded model and data.

        Returns:
            Dictionary with model and data statistics
        """
        self._ensure_initialized()

        # Ensure model is initialized before accessing its attributes
        device = None
        model_parameters = None
        if (
            self.transe_manager is not None
            and self.transe_manager.model is not None
        ):
            device = str(next(self.transe_manager.model.parameters()).device)
            model_parameters = sum(
                p.numel() for p in self.transe_manager.model.parameters()
            )

        stats = {
            "num_entities": len(self.entity_to_idx),
            "num_relations": len(self.relation_to_idx),
            "model_config": {
                "embedding_dim": self.transe_config["model"]["embedding_dim"],
                "margin": self.transe_config["model"]["margin"],
                "norm": self.transe_config["model"]["norm"],
            },
            "device": device,
            "model_parameters": model_parameters,
        }

        # Add training metrics if available
        if self.transe_manager is not None and hasattr(self.transe_manager, "last_val_metrics"):
            stats["last_validation_metrics"] = self.transe_manager.last_val_metrics

        return stats
