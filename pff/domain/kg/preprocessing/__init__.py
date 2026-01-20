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
    from pff.domain.kg.preprocessing import KGPreprocessingPipeline, PreprocessingConfig

    config = PreprocessingConfig.from_yaml("config/models/kg.yaml")
    pipeline = KGPreprocessingPipeline(config)

    processed_df = pipeline.preprocess_all(raw_df)

    train, valid, test = pipeline.preprocess_splits(train_df, valid_df, test_df)
"""

from .advanced_strategies import (
    EntityResolutionStrategy,
    HubDownsamplingStrategy,
    PathCountingStrategy,
    RelationCardinalityClassifier,
    SemanticInverseStrategy,
    TextualizationStrategy,
)
from .config import PreprocessingConfig
from .pipeline import KGPreprocessingPipeline
from .split import LeakageChecker, SafeSplitter
from .strategies import (
    AttributeRelationClassifier,
    DeduplicationStrategy,
    DegreeFeatureExtractor,
    EntityDegreeFilter,
    InverseRelationStrategy,
    RelationSupportFilter,
    SelfLoopRemovalStrategy,
)
from .utils import filter_attribute_relations

__all__ = [
    "PreprocessingConfig",
    "KGPreprocessingPipeline",
    "DeduplicationStrategy",
    "SelfLoopRemovalStrategy",
    "InverseRelationStrategy",
    "AttributeRelationClassifier",
    "DegreeFeatureExtractor",
    "EntityDegreeFilter",
    "RelationSupportFilter",
    "SafeSplitter",
    "LeakageChecker",
    "HubDownsamplingStrategy",
    "SemanticInverseStrategy",
    "EntityResolutionStrategy",
    "RelationCardinalityClassifier",
    "PathCountingStrategy",
    "TextualizationStrategy",
    "filter_attribute_relations",
]
