"""
Preprocessing Configuration for KG Data.

Design Pattern: Builder + Configuration Object
- Immutable configuration after construction
- Fluent builder for programmatic configuration
- YAML/dict loading for file-based configuration

This module centralizes all preprocessing parameters to ensure
consistency between main pipeline and HPO experiments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pff.shared import FileManager, logger
from pff.shared.core.config import KG_PIPELINE_CONFIG_PATH

DEFAULT_ATTRIBUTE_RELATIONS = frozenset(
    {
        "id",
        "externalId",
        "providerProductId",
        "providerCustomerId",
        "providerContractId",
        "providerProductExternalId",
        "startDateTime",
        "endDateTime",
        "createdAt",
        "updatedAt",
        "modifiedAt",
        "effectiveDate",
        "expirationDate",
        "value",
        "name",
        "description",
        "label",
        "title",
        "timeZone",
        "locale",
        "currency",
        "status",
        "state",
        "type",
        "category",
    }
)


ATTRIBUTE_HANDLING_MARK = "mark"
ATTRIBUTE_HANDLING_REMOVE = "remove"
ATTRIBUTE_HANDLING_SEPARATE = "separate"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable configuration for KG preprocessing.

    Attributes:
        remove_duplicates: Whether to remove exact duplicate triples
        remove_self_loops: Whether to remove triples where head == tail
        add_inverse_relations: Whether to create inverse (t, r_inv, h) triples
        inverse_suffix: Suffix for inverse relation names
        apply_inverse_to_all_splits: If True, add inverses to all splits consistently
        min_entity_degree: Minimum degree to keep an entity (0 = keep all)
        min_relation_support: Minimum triples per relation (0 = keep all)
        attribute_relations: Relations to treat as attributes (exclude from LP target)
        exclude_attribute_from_prediction: Whether to exclude attribute relations from evaluation
        compute_degree_features: Whether to compute degree-based features
        split_before_inverse: Whether to do train/val/test split BEFORE adding inverses (CRITICAL)
        check_leakage: Whether to verify no leakage between splits after inverse addition
        chronological_split: Whether to use time-based splitting (requires timestamp column)
        timestamp_column: Column name for chronological splitting
        attribute_handling: How to handle attribute relations ("mark", "remove", "separate")
        allowed_reflexive_relations: Relations where self-loops are allowed
    """

    remove_duplicates: bool = True
    remove_self_loops: bool = True

    add_inverse_relations: bool = True
    inverse_suffix: str = "_inv"
    apply_inverse_to_all_splits: bool = True

    min_entity_degree: int = 0
    min_relation_support: int = 0
    relation_support_policy: str = "warn"

    attribute_relations: frozenset[str] = field(default_factory=lambda: DEFAULT_ATTRIBUTE_RELATIONS)
    attribute_patterns: tuple[str, ...] = tuple()
    exclude_attribute_from_prediction: bool = True
    attribute_handling: str = ATTRIBUTE_HANDLING_MARK
    allowed_reflexive_relations: frozenset[str] = field(default_factory=frozenset)

    compute_degree_features: bool = True
    output_dir: str = "outputs/preprocessing"

    split_before_inverse: bool = True
    check_leakage: bool = True

    chronological_split: bool = False
    timestamp_column: str = "timestamp"

    fix_leakage: bool = True
    resplit_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    ensure_transductive: bool = True
    stratified_by_relation: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PreprocessingConfig:
        """Create config from a dictionary/mapping.

        Args:
            data: Configuration dictionary

        Returns:
            PreprocessingConfig instance
        """
        attr_relations = data.get("attribute_relations", DEFAULT_ATTRIBUTE_RELATIONS)
        if isinstance(attr_relations, (list, set)):
            attr_relations = frozenset(attr_relations)
        elif isinstance(attr_relations, frozenset):
            pass
        else:
            attr_relations = DEFAULT_ATTRIBUTE_RELATIONS
        attribute_patterns_raw = data.get("attribute_patterns", ())
        if isinstance(attribute_patterns_raw, str):
            attribute_patterns: tuple[str, ...] = (attribute_patterns_raw,)
        elif isinstance(attribute_patterns_raw, (list, tuple)):
            attribute_patterns = tuple(str(p) for p in attribute_patterns_raw)
        else:
            attribute_patterns = tuple()
        attribute_handling = str(data.get("attribute_handling", ATTRIBUTE_HANDLING_MARK)).lower()
        allowed_reflexive_relations = data.get("allowed_reflexive_relations", frozenset())
        if isinstance(allowed_reflexive_relations, (list, set)):
            allowed_reflexive_relations = frozenset(allowed_reflexive_relations)
        elif not isinstance(allowed_reflexive_relations, frozenset):
            allowed_reflexive_relations = frozenset()

        raw_ratios = data.get("resplit_ratios", (0.8, 0.1, 0.1))
        if isinstance(raw_ratios, (list, tuple)) and len(raw_ratios) == 3:
            resplit_ratios: tuple[float, float, float] = (
                float(raw_ratios[0]),
                float(raw_ratios[1]),
                float(raw_ratios[2]),
            )
        else:
            resplit_ratios = (0.8, 0.1, 0.1)

        return cls(
            remove_duplicates=bool(data.get("remove_duplicates", True)),
            remove_self_loops=bool(data.get("remove_self_loops", True)),
            add_inverse_relations=bool(data.get("add_inverse_relations", True)),
            inverse_suffix=str(data.get("inverse_suffix", "_inv")),
            apply_inverse_to_all_splits=bool(data.get("apply_inverse_to_all_splits", True)),
            min_entity_degree=int(data.get("min_entity_degree", 0)),
            min_relation_support=int(data.get("min_relation_support", 0)),
            relation_support_policy=str(data.get("relation_support_policy", "warn")).lower(),
            attribute_relations=attr_relations,
            attribute_patterns=attribute_patterns,
            exclude_attribute_from_prediction=bool(
                data.get("exclude_attribute_from_prediction", True)
            ),
            attribute_handling=attribute_handling,
            allowed_reflexive_relations=allowed_reflexive_relations,
            compute_degree_features=bool(data.get("compute_degree_features", True)),
            output_dir=str(data.get("output_dir", "outputs/preprocessing")),
            split_before_inverse=bool(data.get("split_before_inverse", True)),
            check_leakage=bool(data.get("check_leakage", True)),
            chronological_split=bool(data.get("chronological_split", False)),
            timestamp_column=str(data.get("timestamp_column", "timestamp")),
            fix_leakage=bool(data.get("fix_leakage", True)),
            resplit_ratios=resplit_ratios,
            ensure_transductive=bool(data.get("ensure_transductive", True)),
            stratified_by_relation=bool(data.get("stratified_by_relation", True)),
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str | None = None) -> PreprocessingConfig:
        """Load config from YAML file.

        Args:
            config_path: Path to YAML config. If None, uses KG_PIPELINE_CONFIG_PATH.

        Returns:
            PreprocessingConfig instance
        """
        fm = FileManager()
        if config_path is None:
            default_path = Path("config/preprocessing.yaml")
            path = default_path if fm.exists(default_path) else KG_PIPELINE_CONFIG_PATH
        else:
            path = Path(config_path)
        try:
            raw = fm.read(path, return_native=True)
            if raw is None:
                raw = {}
            if not isinstance(raw, dict):
                logger.warning(
                    f"Config loaded from {path} is not a dict (got {type(raw)}). "
                    "Using empty config."
                )
                raw = {}
            preprocessing_config = raw.get("preprocessing", raw.get("data_optimizer", {}))
            if not isinstance(preprocessing_config, dict):
                preprocessing_config = {}
            return cls.from_mapping(preprocessing_config)
        except Exception as exc:
            logger.warning(f"Failed to load processing config from {path}: {exc}")
            return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "remove_duplicates": self.remove_duplicates,
            "remove_self_loops": self.remove_self_loops,
            "add_inverse_relations": self.add_inverse_relations,
            "inverse_suffix": self.inverse_suffix,
            "apply_inverse_to_all_splits": self.apply_inverse_to_all_splits,
            "min_entity_degree": self.min_entity_degree,
            "min_relation_support": self.min_relation_support,
            "relation_support_policy": self.relation_support_policy,
            "attribute_relations": list(self.attribute_relations),
            "attribute_patterns": list(self.attribute_patterns),
            "exclude_attribute_from_prediction": self.exclude_attribute_from_prediction,
            "attribute_handling": self.attribute_handling,
            "allowed_reflexive_relations": list(self.allowed_reflexive_relations),
            "compute_degree_features": self.compute_degree_features,
            "output_dir": self.output_dir,
            "split_before_inverse": self.split_before_inverse,
            "check_leakage": self.check_leakage,
            "chronological_split": self.chronological_split,
            "timestamp_column": self.timestamp_column,
            "fix_leakage": self.fix_leakage,
            "resplit_ratios": list(self.resplit_ratios),
            "ensure_transductive": self.ensure_transductive,
            "stratified_by_relation": self.stratified_by_relation,
        }


class PreprocessingConfigBuilder:
    """Fluent builder for PreprocessingConfig.

    Usage:
        config = (PreprocessingConfigBuilder()
            .with_deduplication(True)
            .with_inverse_relations(True, suffix="_reverse")
            .with_min_degree(2)
            .build())
    """

    def __init__(self) -> None:
        """Execute init."""

        self._config: dict[str, Any] = {}

    def with_deduplication(self, enabled: bool = True) -> PreprocessingConfigBuilder:
        """Enable/disable duplicate removal."""
        self._config["remove_duplicates"] = enabled
        return self

    def with_self_loop_removal(self, enabled: bool = True) -> PreprocessingConfigBuilder:
        """Enable/disable self-loop removal."""
        self._config["remove_self_loops"] = enabled
        return self

    def with_inverse_relations(
        self, enabled: bool = True, suffix: str = "_inv", all_splits: bool = True
    ) -> PreprocessingConfigBuilder:
        """Configure inverse relation augmentation."""
        self._config["add_inverse_relations"] = enabled
        self._config["inverse_suffix"] = suffix
        self._config["apply_inverse_to_all_splits"] = all_splits
        return self

    def with_min_degree(self, min_degree: int) -> PreprocessingConfigBuilder:
        """Set minimum entity degree filter."""
        self._config["min_entity_degree"] = min_degree
        return self

    def with_min_relation_support(self, min_support: int) -> PreprocessingConfigBuilder:
        """Set minimum relation support filter."""
        self._config["min_relation_support"] = min_support
        return self

    def with_attribute_relations(self, relations: set[str]) -> PreprocessingConfigBuilder:
        """Set attribute relation names."""
        self._config["attribute_relations"] = relations
        return self

    def with_attribute_patterns(self, patterns: tuple[str, ...]) -> PreprocessingConfigBuilder:
        """Set regex/substring patterns for attribute relations."""
        self._config["attribute_patterns"] = patterns
        return self

    def with_degree_features(self, enabled: bool = True) -> PreprocessingConfigBuilder:
        """Enable/disable degree feature computation."""
        self._config["compute_degree_features"] = enabled
        return self

    def with_leakage_check(self, enabled: bool = True) -> PreprocessingConfigBuilder:
        """Enable/disable leakage verification."""
        self._config["check_leakage"] = enabled
        return self

    def with_chronological_split(
        self, enabled: bool = True, timestamp_col: str = "timestamp"
    ) -> PreprocessingConfigBuilder:
        """Enable chronological splitting."""
        self._config["chronological_split"] = enabled
        self._config["timestamp_column"] = timestamp_col
        return self

    def with_leakage_fix(
        self,
        enabled: bool = True,
        resplit_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        ensure_transductive: bool = True,
        stratified_by_relation: bool = True,
    ) -> PreprocessingConfigBuilder:
        """Enable automatic leakage fix via re-splitting.

        When leakage is detected between splits, this option will:
        1. Unify all splits into a single pool
        2. Remove cross-split duplicates
        3. Re-split using SOTA stratified random split
        4. Ensure transductive coverage (all valid/test entities in train)
        5. Add inverses independently to each split

        Args:
            enabled: Whether to fix leakage automatically
            resplit_ratios: Train/valid/test ratios for re-split
            ensure_transductive: Move triples to ensure entity coverage
            stratified_by_relation: Stratify split by relation
        """
        self._config["fix_leakage"] = enabled
        self._config["resplit_ratios"] = resplit_ratios
        self._config["ensure_transductive"] = ensure_transductive
        self._config["stratified_by_relation"] = stratified_by_relation
        return self

    def build(self) -> PreprocessingConfig:
        """Build the final configuration."""
        return PreprocessingConfig.from_mapping(self._config)
