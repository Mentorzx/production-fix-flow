"""RotatE Validator Module.

Knowledge Graph Embedding using Rotational transformations in Complex Space.
Based on Sun et al. 2019 "RotatE: Knowledge Graph Embedding by Relational
Rotation in Complex Space" (ICLR 2019).

Design Patterns Applied:
    - **Strategy Pattern:** RotatEStrategy implements KGEModelStrategy for
      interchangeable model selection.
    - **Factory Pattern:** Model creation via ModelFactory.create(ModelType.ROTATE).
    - **Template Method:** Training loop follows BaseTrainer structure.
    - **Observer Pattern:** Training metrics observed via TrainingObserver.
    - **Builder Pattern:** RotatEConfig built from YAML configuration.

Mathematical Foundation:
    RotatE models relations as rotations in complex space:
        h ∘ r = t
    Where:
        - h, t ∈ ℂ^d are entity embeddings (complex vectors)
        - r = e^(iθ) is a relation-specific rotation (phase angles)
        - ∘ denotes element-wise (Hadamard) product

    Scoring function:
        d_r(h, t) = ||h ∘ r - t||

    Key advantages over TransE:
        1. Captures anti-symmetric relations naturally (θ ≠ 0, π)
        2. Models composition via rotation composition
        3. Better for hierarchical (N-1) relation patterns
        4. Memory efficient (half of ComplEx for same expressiveness)

Exports:
    - RotatEModel: PyTorch model with complex embeddings
    - RotatEConfig: Configuration dataclass
    - RotatEManager: Training and evaluation orchestrator

Example:
    >>> from pff.validators.rotate import RotatEModel, RotatEConfig
    >>> config = RotatEConfig(embedding_dim=256, gamma=12.0)
    >>> model = RotatEModel(num_entities=5000, num_relations=50, config=config)
    >>> scores = model.score_triples_batch(triples)

Author: PFF Team
Date: 2025-11-25
"""

from pff.validators.rotate.core import RotatEModel, RotatEDataset
from pff.validators.rotate.config import RotatEConfig, RotatEConfigBuilder
from pff.validators.rotate.manager import RotatEManager

# Lazy imports for ensemble components to avoid circular imports
# Use: from pff.validators.rotate.rotate_service import RotatEScorerService
# Use: from pff.validators.rotate.wrappers import RotatEWrapper, RotatEHybridWrapper
# Use: from pff.validators.rotate.trainer import RotatETrainer, RotatETrainerConfig
# Use: from pff.validators.rotate.adapter import RotatEEnsembleAdapter, RotatETransEAdapter
#
# SRP Components (created 2025-11-26):
# Use: from pff.validators.rotate.checkpoint_manager import RotatECheckpointManager
# Use: from pff.validators.rotate.data_loader import RotatEDataLoader
# Use: from pff.validators.rotate.metrics_reporter import RotatEMetricsReporter
#
# SOTA Components:
# Use: from pff.validators.rotate.contrastive import ContrastiveLossFactory, LossType
# Use: from pff.validators.rotate.negative_sampling import NegativeSamplerFactory, NegativeSamplingStrategy
# Use: from pff.validators.rotate.sota_utils import LabelSmoothingLoss, ReciprocalRelationAugmenter

__all__ = [
    # Core
    "RotatEModel",
    "RotatEDataset",
    "RotatEConfig",
    "RotatEConfigBuilder",
    "RotatEManager",
]
