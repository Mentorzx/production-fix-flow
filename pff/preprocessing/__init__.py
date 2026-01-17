"""
Central KG Preprocessing Package for PFF Telecom.

This package provides a unified preprocessing pipeline for Knowledge Graph data,
ensuring consistency between the main training pipeline and HPO experiments.

Design Patterns:
- Strategy Pattern: Different preprocessing strategies (dedup, self-loops, etc.)
- Pipeline Pattern: Sequential preprocessing steps
- Observer Pattern: Progress reporting hooks

Key Features:
- Deduplication of triples
- Self-loop removal
- Inverse relation augmentation (with leakage prevention)
- Attribute/structural relation classification
- Degree-based features
- Consistent split handling

Advanced Features (SOTA):
- Hub downsampling (reduce dominance of high-degree nodes)
- Semantic inverse relations (worksIn → employs)
- Entity resolution (deduplicate similar entities)
- Relation cardinality classification (1:1, 1:N, N:1, N:N)
- Path counting (k-hop features)
- Textualization (BERT-ready text generation)

Usage:
    from pff.preprocessing import KGPreprocessingPipeline, PreprocessingConfig

    config = PreprocessingConfig.from_yaml("config/models/kg.yaml")
    pipeline = KGPreprocessingPipeline(config)

    # For pre-split data (single DataFrame)
    processed_df = pipeline.preprocess_all(raw_df)

    # For already-split data (separate train/valid/test)
    train, valid, test = pipeline.preprocess_splits(train_df, valid_df, test_df)
"""

from .config import PreprocessingConfig
from .pipeline import KGPreprocessingPipeline
from .strategies import (
    DeduplicationStrategy,
    SelfLoopRemovalStrategy,
    InverseRelationStrategy,
    AttributeRelationClassifier,
    DegreeFeatureExtractor,
    EntityDegreeFilter,
    RelationSupportFilter,
)
from .split import SafeSplitter, LeakageChecker

# Advanced SOTA strategies
from .advanced_strategies import (
    HubDownsamplingStrategy,
    SemanticInverseStrategy,
    EntityResolutionStrategy,
    RelationCardinalityClassifier,
    PathCountingStrategy,
    TextualizationStrategy,
)
from .utils import filter_attribute_relations

__all__ = [
    # Core
    "PreprocessingConfig",
    "KGPreprocessingPipeline",
    # Basic strategies
    "DeduplicationStrategy",
    "SelfLoopRemovalStrategy",
    "InverseRelationStrategy",
    "AttributeRelationClassifier",
    "DegreeFeatureExtractor",
    "EntityDegreeFilter",
    "RelationSupportFilter",
    # Split utilities
    "SafeSplitter",
    "LeakageChecker",
    # Advanced SOTA strategies
    "HubDownsamplingStrategy",
    "SemanticInverseStrategy",
    "EntityResolutionStrategy",
    "RelationCardinalityClassifier",
    "PathCountingStrategy",
    "TextualizationStrategy",
    # Helpers
    "filter_attribute_relations",
]
