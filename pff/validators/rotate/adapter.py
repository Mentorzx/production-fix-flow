"""RotatE Ensemble Adapter Module.

Provides adapter classes to integrate RotatE with the existing ensemble pipeline,
making it compatible with AdvancedEnsembleTrainer and other components.

Design Patterns Applied:
    - **Adapter Pattern:** Adapts RotatE interface to standard KGE interface.
    - **Facade Pattern:** Simplifies RotatE usage for ensemble integration.
    - **Dependency Injection:** Accepts model, mappings, and config as dependencies.

The adapter provides a standardized KGE interface, enabling seamless integration
with the ensemble pipeline and downstream ML models.

Example:
    >>> from pff.validators.rotate.adapter import RotatEEnsembleAdapter
    >>> adapter = RotatEEnsembleAdapter(
    ...     kg_config_path="config/models/kg.yaml",
    ...     rotate_config_path="config/models/rotate.yaml"
    ... )
    >>> embedding = adapter.get_entity_embedding("entity_name")
    >>> score = adapter.score_triple("head", "relation", "tail")

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pff.utils import FileManager, logger
from pff.validators.kg.config import KGConfig
from pff.validators.rotate.config import RotatEConfig
from pff.validators.rotate.core import RotatEModel
from pff.validators.rotate.rotate_service import RotatEScorerService


class RotatEEnsembleAdapter:
    """Adapter for RotatE to work with AdvancedEnsembleTrainer.
    
    Design Pattern: Adapter
        Provides a standardized KGE interface, enabling the ensemble
        pipeline to use RotatE seamlessly with downstream models.
    
    Interface methods:
        - get_entity_embedding(entity_id: str) -> np.ndarray
        - get_relation_embedding(relation_id: str) -> np.ndarray
        - score_triple(head: str, relation: str, tail: str) -> float
        - score_triples_batch(triples: list) -> np.ndarray
        - get_all_entity_embeddings() -> np.ndarray
        - get_all_relation_embeddings() -> np.ndarray
    
    Note: RotatE uses complex embeddings, so this adapter concatenates
    real and imaginary parts to provide real-valued embeddings compatible
    with downstream models like LightGBM.
    
    Attributes:
        scorer_service: RotatEScorerService for scoring operations.
        entity_to_idx: Mapping from entity names to indices.
        relation_to_idx: Mapping from relation names to indices.
        embedding_dim: Dimension of combined embeddings (2 * complex_dim).
    
    Example:
        >>> adapter = RotatEEnsembleAdapter(kg_config_path, rotate_config_path)
        >>> h_emb = adapter.get_entity_embedding("Entity1")
        >>> r_emb = adapter.get_relation_embedding("relation1")
        >>> score = adapter.score_triple("Entity1", "relation1", "Entity2")
    """
    
    def __init__(
        self,
        kg_config_path: str | Path,
        rotate_config_path: str | Path,
        load_best_model: bool = True,
    ) -> None:
        """Initialize RotatE ensemble adapter.
        
        Args:
            kg_config_path: Path to KG configuration YAML.
            rotate_config_path: Path to RotatE configuration YAML.
            load_best_model: Whether to load the best checkpoint.
        """
        self.kg_config_path = Path(kg_config_path)
        self.rotate_config_path = Path(rotate_config_path)
        
        # Initialize KG config
        self.kg_config = KGConfig(self.kg_config_path)
        
        # Initialize scorer service (handles model loading)
        self.scorer_service = RotatEScorerService(
            kg_config=self.kg_config,
            rotate_config_path=self.rotate_config_path,
            load_best_model=load_best_model,
        )
        
        # Cache mappings for quick lookup
        self.entity_to_idx = self.scorer_service.entity_to_idx
        self.idx_to_entity = self.scorer_service.idx_to_entity
        self.relation_to_idx = self.scorer_service.relation_to_idx
        self.idx_to_relation = self.scorer_service.idx_to_relation
        
        # Cache embeddings for performance
        self._entity_embeddings: np.ndarray | None = None
        self._relation_embeddings: np.ndarray | None = None
        
        # Get embedding dimension
        if self.scorer_service.model is not None:
            self.embedding_dim = self.scorer_service.model.embedding_dim * 2
        else:
            self.embedding_dim = 512  # Default fallback
        
        logger.info(
            f"RotatE adapter inicializado: {len(self.entity_to_idx):,} entidades, "
            f"{len(self.relation_to_idx)} relacoes, dim={self.embedding_dim}"
        )
    
    def get_entity_embedding(self, entity: str) -> np.ndarray:
        """Get combined embedding for an entity.
        
        Concatenates real and imaginary parts of the complex embedding.
        
        Args:
            entity: Entity name or ID.
            
        Returns:
            Combined embedding array [2 * complex_dim].
        """
        idx = self.entity_to_idx.get(entity)
        
        if idx is None:
            logger.debug(f"Unknown entity: {entity}, returning mean embedding")
            return self._get_mean_entity_embedding()
        
        embeddings = self._get_cached_entity_embeddings()
        return embeddings[idx]
    
    def get_relation_embedding(self, relation: str) -> np.ndarray:
        """Get embedding for a relation.
        
        Converts phase angles to cos/sin representation.
        
        Args:
            relation: Relation name or ID.
            
        Returns:
            Combined embedding array [2 * complex_dim].
        """
        idx = self.relation_to_idx.get(relation)
        
        if idx is None:
            logger.debug(f"Unknown relation: {relation}, returning mean embedding")
            return self._get_mean_relation_embedding()
        
        embeddings = self._get_cached_relation_embeddings()
        return embeddings[idx]
    
    def score_triple(self, head: str, relation: str, tail: str) -> float:
        """Score a single triple.
        
        Args:
            head: Head entity name.
            relation: Relation name.
            tail: Tail entity name.
            
        Returns:
            RotatE score (lower = more likely valid).
        """
        return self.scorer_service.score_triple(head, relation, tail)
    
    def score_triples_batch(self, triples: list[tuple[str, str, str]]) -> np.ndarray:
        """Score multiple triples efficiently.
        
        Args:
            triples: List of (head, relation, tail) tuples.
            
        Returns:
            Array of scores.
        """
        return self.scorer_service.score_triple_batch(triples)
    
    def score_to_probability(self, score: float) -> float:
        """Convert raw score to probability.
        
        Args:
            score: Raw RotatE score.
            
        Returns:
            Probability in [0, 1] (higher = more likely valid).
        """
        return self.scorer_service.score_to_probability(score)
    
    def get_all_entity_embeddings(self) -> np.ndarray:
        """Get all entity embeddings as a matrix.
        
        Returns:
            Array of shape [num_entities, 2 * complex_dim].
        """
        return self._get_cached_entity_embeddings()
    
    def get_all_relation_embeddings(self) -> np.ndarray:
        """Get all relation embeddings as a matrix.
        
        Returns:
            Array of shape [num_relations, 2 * complex_dim].
        """
        return self._get_cached_relation_embeddings()
    
    def _get_cached_entity_embeddings(self) -> np.ndarray:
        """Get entity embeddings with caching.
        
        Returns:
            Combined entity embeddings [num_entities, 2 * dim].
        """
        if self._entity_embeddings is None:
            self._entity_embeddings = self.scorer_service.get_combined_entity_embeddings()
        return self._entity_embeddings
    
    def _get_cached_relation_embeddings(self) -> np.ndarray:
        """Get relation embeddings with caching.
        
        Returns:
            Combined relation embeddings [num_relations, 2 * dim].
        """
        if self._relation_embeddings is None:
            self._relation_embeddings = self.scorer_service.get_combined_relation_embeddings()
        return self._relation_embeddings
    
    def _get_mean_entity_embedding(self) -> np.ndarray:
        """Get mean entity embedding for unknown entities.
        
        Returns:
            Mean embedding vector.
        """
        embeddings = self._get_cached_entity_embeddings()
        return np.mean(embeddings, axis=0)
    
    def _get_mean_relation_embedding(self) -> np.ndarray:
        """Get mean relation embedding for unknown relations.
        
        Returns:
            Mean embedding vector.
        """
        embeddings = self._get_cached_relation_embeddings()
        return np.mean(embeddings, axis=0)
    
    def clear_cache(self) -> None:
        """Clear embedding caches."""
        self._entity_embeddings = None
        self._relation_embeddings = None
        logger.debug("Embedding caches cleared")
    
    @property
    def model(self) -> RotatEModel | None:
        """Get underlying RotatE model.
        
        Returns:
            RotatEModel instance or None if not loaded.
        """
        return self.scorer_service.model
    
    @property
    def num_entities(self) -> int:
        """Get number of entities.
        
        Returns:
            Number of entities in the KG.
        """
        return len(self.entity_to_idx)
    
    @property
    def num_relations(self) -> int:
        """Get number of relations.
        
        Returns:
            Number of relations in the KG.
        """
        return len(self.relation_to_idx)


class RotatEEmbeddingAdapter:
    """Adapter that provides RotatE embeddings in standard format.
    
    This adapter is designed for the HPO pipeline and feature extraction,
    converting complex embeddings to real-valued format.
    
    Design Pattern: Adapter
        Converts RotatE's complex embeddings to real-valued format
        compatible with feature extraction pipelines.
    
    Attributes:
        model: RotatE model.
        entity_embeddings: Combined entity embeddings (real + imag).
        relation_embeddings: Combined relation embeddings (cos + sin).
    """
    
    def __init__(
        self,
        model: RotatEModel,
        entity_to_idx: dict[str, int],
        relation_to_idx: dict[str, int],
    ) -> None:
        """Initialize RotatE embedding adapter.
        
        Args:
            model: Trained RotatE model.
            entity_to_idx: Entity name to index mapping.
            relation_to_idx: Relation name to index mapping.
        """
        self.model = model
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        
        # Extract and combine embeddings
        self._extract_embeddings()
        
        logger.debug(
            f"RotatE embedding adapter: entidades={self.entity_embeddings.shape}, "
            f"relacoes={self.relation_embeddings.shape}"
        )
    
    def _extract_embeddings(self) -> None:
        """Extract and combine RotatE embeddings."""
        with torch.no_grad():
            # Entity embeddings: concatenate real and imaginary
            ent_real, ent_imag = self.model.get_entity_embeddings()
            self.entity_embeddings = np.concatenate([
                ent_real.cpu().numpy(),
                ent_imag.cpu().numpy()
            ], axis=1)
            
            # Relation embeddings: convert phases to cos/sin
            phases = self.model.get_relation_phases().cpu().numpy()
            self.relation_embeddings = np.concatenate([
                np.cos(phases),
                np.sin(phases)
            ], axis=1)
    
    def get_entity_embedding(self, entity_idx: int) -> np.ndarray:
        """Get entity embedding by index.
        
        Args:
            entity_idx: Entity index.
            
        Returns:
            Combined embedding vector.
        """
        if 0 <= entity_idx < len(self.entity_embeddings):
            return self.entity_embeddings[entity_idx]
        return np.zeros(self.entity_embeddings.shape[1])
    
    def get_relation_embedding(self, relation_idx: int) -> np.ndarray:
        """Get relation embedding by index.
        
        Args:
            relation_idx: Relation index.
            
        Returns:
            Combined embedding vector.
        """
        if 0 <= relation_idx < len(self.relation_embeddings):
            return self.relation_embeddings[relation_idx]
        return np.zeros(self.relation_embeddings.shape[1])
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (combined).
        
        Returns:
            Dimension of combined embeddings.
        """
        return self.entity_embeddings.shape[1]
