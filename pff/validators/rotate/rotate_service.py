"""RotatE Scorer Service Module.

Provides a service interface for scoring triples using trained RotatE models.
Compatible with the ensemble wrapper architecture.

Design Patterns Applied:
    - **Adapter Pattern:** Adapts RotatE model to ensemble-compatible interface.
    - **Service Locator:** Manages model loading and caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pff import settings
from pff.utils import FileManager, logger
from pff.validators.kg.calibration import ScoreCalibrator
from pff.validators.kg.config import KGConfig
from pff.validators.rotate.core import RotatEModel
from pff.validators.rotate.config import RotatEConfig


class RotatEScorerService:
    """Service for scoring triples using a trained RotatE model.

    This service provides methods for:
    - Loading pre-trained RotatE models
    - Scoring individual triples
    - Converting raw scores to probabilities via Platt scaling
    - Batch scoring for efficiency

    Attributes:
        model: The trained RotatE model.
        entity_to_idx: Mapping from entity names to indices.
        relation_to_idx: Mapping from relation names to indices.
        calibrator: Optional ScoreCalibrator for probability conversion.

    Example:
        >>> service = RotatEScorerService(kg_config, rotate_config_path)
        >>> score = service.score_triple("Entity1", "relation", "Entity2")
        >>> prob = service.score_to_probability(score)
    """

    def __init__(
        self,
        kg_config: KGConfig,
        rotate_config_path: Path,
        load_best_model: bool = True,
    ) -> None:
        """Initialize the RotatE scorer service.

        Args:
            kg_config: Knowledge graph configuration.
            rotate_config_path: Path to RotatE configuration YAML.
            load_best_model: Whether to load the best checkpoint (default: True).
        """
        self.kg_config = kg_config
        self.rotate_config_path = Path(rotate_config_path)
        self.file_manager = FileManager()

        # Load config
        self.config_data = self.file_manager.read(self.rotate_config_path)
        self.rotate_config = RotatEConfig(
            embedding_dim=self.config_data.get("model", {}).get("embedding_dim", 256),
            gamma=self.config_data.get("model", {}).get("gamma", 12.0),
            epsilon=self.config_data.get("model", {}).get("epsilon", 2.0),
        )

        # Initialize mappings
        self.entity_to_idx: dict[str, int] = {}
        self.relation_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.idx_to_relation: dict[int, str] = {}

        # Model and calibrator
        self.model: RotatEModel | None = None
        self.calibrator: ScoreCalibrator | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load components
        self._load_mappings()
        if load_best_model:
            self._load_model()
            self._load_calibrator()

        logger.info(
            f"RotatE scorer inicializado: {len(self.entity_to_idx):,} entidades, "
            f"{len(self.relation_to_idx)} relacoes"
        )

    def _load_mappings(self) -> None:
        """Load entity and relation mappings from output directory."""
        from pff.validators.rotate.mapping_utils import load_mappings

        try:
            outputs_config = self.config_data.get("outputs", {})
            model_dir = Path(outputs_config.get("dir", settings.OUTPUTS_DIR / "rotate"))
            maps_path = model_dir / "maps"
            
            entity_map_path = maps_path / "rotate_entity_map_raw.parquet"
            relation_map_path = maps_path / "rotate_relation_map_raw.parquet"
            
            # Fallback to transe maps if rotate maps don't exist
            if not entity_map_path.exists():
                entity_map_path = maps_path / "transe_entity_map_raw.parquet"
            if not relation_map_path.exists():
                relation_map_path = maps_path / "transe_relation_map_raw.parquet"
            
            if not entity_map_path.exists() or not relation_map_path.exists():
                logger.warning(f"Mappings not found in {maps_path}")
                return
            
            (
                self.entity_to_idx,
                self.idx_to_entity,
                self.relation_to_idx,
                self.idx_to_relation,
            ) = load_mappings(entity_map_path, relation_map_path)
        except Exception as e:
            logger.warning(f"Failed to load mappings: {e}", exc_info=True)

    def _load_model(self) -> None:
        """Load the best RotatE model checkpoint."""
        outputs_config = self.config_data.get("outputs", {})
        model_dir = Path(outputs_config.get("dir", settings.OUTPUTS_DIR / "rotate"))
        checkpoint_dir = model_dir / "checkpoints"
        best_model_path = checkpoint_dir / "best_model.pt"

        if not best_model_path.exists():
            logger.warning(f"Checkpoint not found: {best_model_path}")
            return

        try:
            checkpoint = torch.load(best_model_path, map_location=self._device, weights_only=True)

            # Get model dimensions from checkpoint
            num_entities = checkpoint.get(
                "num_entities", len(self.entity_to_idx)
            )
            num_relations = checkpoint.get(
                "num_relations", len(self.relation_to_idx)
            )

            # Create model
            self.model = RotatEModel(
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=self.rotate_config.embedding_dim,
                gamma=self.rotate_config.gamma,
                epsilon=self.rotate_config.epsilon,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self._device)
            self.model.eval()

            logger.info(f"Modelo RotatE carregado de: {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to load RotatE model: {e}", exc_info=True)

    def _load_calibrator(self) -> None:
        """Load score calibrator if available."""
        outputs_config = self.config_data.get("outputs", {})
        model_dir = Path(outputs_config.get("dir", settings.OUTPUTS_DIR / "rotate"))
        calibrator_path = model_dir / "score_calibrator.pkl"

        if not calibrator_path.exists():
            logger.debug(f"Calibrator not found: {calibrator_path}")
            return

        try:
            self.calibrator = ScoreCalibrator.load(calibrator_path)
            logger.debug("Calibrador RotatE carregado")
        except Exception as e:
            logger.warning(f"Failed to load calibrator: {e}", exc_info=True)

    def score_triple(self, head: str, relation: str, tail: str) -> float:
        """Score a single triple.

        Args:
            head: Head entity name.
            relation: Relation name.
            tail: Tail entity name.

        Returns:
            Raw score from RotatE model (lower = more likely).

        Raises:
            ValueError: If model not loaded or entity/relation unknown.
        """
        if self.model is None:
            raise ValueError("Modelo RotatE nao carregado")

        head_idx = self.entity_to_idx.get(head)
        rel_idx = self.relation_to_idx.get(relation)
        tail_idx = self.entity_to_idx.get(tail)

        if head_idx is None or rel_idx is None or tail_idx is None:
            # Return neutral score for unknown entities
            return self.rotate_config.gamma

        with torch.no_grad():
            score = self.model.score_triple(head_idx, rel_idx, tail_idx)

        return float(score)

    def score_triple_batch(
        self, triples: list[tuple[str, str, str]]
    ) -> np.ndarray:
        """Score a batch of triples efficiently.

        Args:
            triples: List of (head, relation, tail) tuples.

        Returns:
            Array of scores for each triple.
        """
        if self.model is None:
            return np.full(len(triples), self.rotate_config.gamma)

        # Convert to indices
        indexed_triples = []
        valid_mask = []
        for head, relation, tail in triples:
            h_idx = self.entity_to_idx.get(head)
            r_idx = self.relation_to_idx.get(relation)
            t_idx = self.entity_to_idx.get(tail)

            if h_idx is not None and r_idx is not None and t_idx is not None:
                indexed_triples.append([h_idx, r_idx, t_idx])
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        # Score valid triples
        scores = np.full(len(triples), self.rotate_config.gamma)
        if indexed_triples:
            triple_tensor = torch.tensor(indexed_triples, dtype=torch.long)
            batch_scores = self.model.score_triples_batch(triple_tensor)
            valid_indices = [i for i, m in enumerate(valid_mask) if m]
            scores[valid_indices] = batch_scores

        return scores

    def score_to_probability(self, score: float) -> float:
        """Convert raw score to probability.

        Uses Platt scaling if calibrator is available, otherwise
        applies sigmoid transformation based on gamma.

        Args:
            score: Raw RotatE score.

        Returns:
            Probability in [0, 1] range (higher = more likely valid).
        """
        if self.calibrator is not None:
            return float(self.calibrator.transform(np.array([score]))[0])

        # Fallback: sigmoid based on gamma margin
        # Lower score = more likely valid, so we invert
        normalized = (self.rotate_config.gamma - score) / self.rotate_config.gamma
        return float(1 / (1 + np.exp(-5 * normalized)))

    def get_entity_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Get entity embeddings as real and imaginary parts.

        Returns:
            Tuple of (real_embeddings, imag_embeddings) arrays.
        """
        if self.model is None:
            raise ValueError("Modelo RotatE nao carregado")

        with torch.no_grad():
            real, imag = self.model.get_entity_embeddings()
            return real.cpu().numpy(), imag.cpu().numpy()

    def get_relation_embeddings(self) -> np.ndarray:
        """Get relation phase embeddings.

        Returns:
            Array of relation phase angles.
        """
        if self.model is None:
            raise ValueError("Modelo RotatE nao carregado")

        with torch.no_grad():
            phases = self.model.get_relation_phases()
            return phases.cpu().numpy()

    def get_combined_entity_embeddings(self) -> np.ndarray:
        """Get entity embeddings concatenated (real + imag).

        This format is compatible with LightGBM hybrid models.

        Returns:
            Array with shape (num_entities, 2 * complex_dim).
        """
        real, imag = self.get_entity_embeddings()
        return np.concatenate([real, imag], axis=1)

    def get_combined_relation_embeddings(self) -> np.ndarray:
        """Get relation embeddings as cos/sin components.

        This format is compatible with LightGBM hybrid models.

        Returns:
            Array with shape (num_relations, 2 * complex_dim).
        """
        phases = self.get_relation_embeddings()
        real = np.cos(phases)
        imag = np.sin(phases)
        return np.concatenate([real, imag], axis=1)
