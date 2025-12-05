"""Transformers - sklearn-compatible transformers for ensemble features.

This module provides sklearn-compatible transformers for the hybrid ensemble:
    - ProbaTransformer: Extracts probabilities as features from classifiers.
    - SymbolicFeatureExtractor: Rule-based feature extraction from AnyBURL rules.
    - GraphStructuralFeatureExtractor: Topological graph features.

Design Patterns Applied:
    - **Adapter Pattern:** Transformers adapt external models (TransE, AnyBURL,
      LightGBM) to sklearn's fit/transform interface.
    - **Strategy Pattern:** Different rule validation strategies (business service
      vs fallback literal matching) are encapsulated and swappable.
    - **Template Method:** All transformers follow fit → transform → get_feature_names_out.
    - **Factory Pattern:** Rule parsing and feature construction use factory functions.

Performance Optimizations:
    - Uses ConcurrencyManager for parallel rule validation when beneficial.
    - Leverages SymbolicRuleAccelerator with Numba kernels for batch operations.
    - Pre-computes predicate indexes for O(1) rule lookup per sample.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import msgspec
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pff import settings
from pff.utils import ConcurrencyManager, FileManager, logger, SymbolicRuleAccelerator
from pff.utils.hash import hash_bytes
from pff.services.business_service import RuleValidator, Rule, RuleViolation

_ensemble_violations_context: ContextVar[list] = ContextVar(
    '_ensemble_violations', default=[]
)
_ensemble_all_rules_context: ContextVar[list] = ContextVar(
    '_ensemble_all_rules', default=[]
)


class SymbolicCoverageError(Exception):
    """Raised when symbolic rules fail to meet minimum coverage or count requirements."""
    pass


class GraphStructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract structural graph features as dense vectors.

    Attributes:
        kg_path: Path to the Parquet file with base triples.
        cache_path: Path to the cached structural statistics file.
        n_features_: Number of structural features produced per sample.
    """

    def __init__(
        self,
        kg_path: str | Path | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        """Initialize extractor.

        Args:
            kg_path: Optional path to the graph triples Parquet file.
            cache_path: Optional path to persist derived structural statistics.
        """

        # Try multiple fallback paths for the graph file
        if kg_path is not None:
            self.kg_path = Path(kg_path)
        else:
            # Priority: train_optimized.parquet > train.parquet
            optimized_path = settings.OUTPUTS_DIR / "kg" / "train_optimized.parquet"
            standard_path = settings.OUTPUTS_DIR / "kg" / "train.parquet"
            if optimized_path.exists():
                self.kg_path = optimized_path
            elif standard_path.exists():
                self.kg_path = standard_path
            else:
                self.kg_path = optimized_path  # Will fail gracefully in fit()
        cache_dir = settings.OUTPUTS_DIR / "ensemble"
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else cache_dir / "graph_stats.pkl"
        )
        self.file_manager = FileManager()
        self.degrees_: dict[str, float] | None = None
        self.neighbors_: dict[str, set[str]] | None = None
        self.n_features_ = 6

    def fit(self, X, y=None):  # noqa: D401 - sklearn signature
        """Learn structural statistics from the graph.

        Args:
            X: Unused, kept for sklearn compatibility.
            y: Unused.

        Returns:
            GraphStructuralFeatureExtractor: Fitted extractor.
        """
        if self.cache_path.exists():
            stats = self.file_manager.read(self.cache_path)
        else:
            stats = self._build_stats()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_manager.save(stats, self.cache_path)
        self.degrees_ = {str(k): float(v) for k, v in stats.get("degrees", {}).items()}
        self.neighbors_ = {
            str(k): set(v)
            for k, v in stats.get("neighbors", {}).items()
        }
        return self

    def transform(self, X: list[list[tuple]]) -> np.ndarray:
        """Transform triples into structural feature vectors.

        Args:
            X: Samples, each containing a list of triples.

        Returns:
            np.ndarray: Array with shape ``(len(X), 6)``.
        """
        check_is_fitted(self, "degrees_")
        features = np.zeros((len(X), self.n_features_), dtype=np.float32)
        for idx, sample in enumerate(X):
            sample_vals = []
            for triple in sample:
                sample_vals.append(self._features_for_triple(triple))
            if sample_vals:
                features[idx] = np.mean(sample_vals, axis=0)
        return features

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [
            "deg_head",
            "deg_tail",
            "shared_neighbors",
            "jaccard",
            "adamic_adar",
            "pref_attachment",
        ]

    def _build_stats(self) -> dict[str, Any]:
        """Build degree and neighbor statistics from the KG.

        Returns:
            dict[str, Any]: Mapping with ``degrees`` and ``neighbors`` entries.
        """

        logger.debug("Computing structural graph statistics for symbolic fallback")
        if not self.kg_path.exists():
            logger.debug(f"Graph file not found: {self.kg_path}; returning empty stats")
            return {"degrees": {}, "neighbors": {}}
        df = self.file_manager.read(self.kg_path)
        if {"s", "p", "o"}.issubset(df.columns):
            df = df.rename({"s": "head", "p": "relation", "o": "tail"})
        degrees = Counter()
        neighbors: dict[str, set[str]] = defaultdict(set)
        for row in df.iter_rows(named=True):
            head = str(row.get("head"))
            tail = str(row.get("tail"))
            if not head or not tail:
                continue
            degrees[head] += 1
            degrees[tail] += 1
            neighbors[head].add(tail)
            neighbors[tail].add(head)
        stats = {
            "degrees": dict(degrees),
            "neighbors": {k: sorted(v) for k, v in neighbors.items()},
        }
        logger.success(
            f" Estatísticas estruturais geradas: {len(degrees)} entidades mapeadas"
        )
        return stats

    def _features_for_triple(self, triple: tuple) -> np.ndarray:
        """Compute structural statistics for a single triple.

        Args:
            triple: Tuple in the form ``(head, relation, tail)``.

        Returns:
            np.ndarray: Dense feature vector with six structural metrics.
        """

        head, _rel, tail = map(str, triple)
        deg_h = self.degrees_.get(head, 0.0) if self.degrees_ else 0.0
        deg_t = self.degrees_.get(tail, 0.0) if self.degrees_ else 0.0
        neighbors_h = self.neighbors_.get(head, set()) if self.neighbors_ else set()
        neighbors_t = self.neighbors_.get(tail, set()) if self.neighbors_ else set()
        shared = neighbors_h & neighbors_t
        union = neighbors_h | neighbors_t
        jaccard = (len(shared) / len(union)) if union else 0.0
        adamic = 0.0
        for neighbor in shared:
            deg_neighbor = self.degrees_.get(neighbor, 0.0) if self.degrees_ else 0.0
            if deg_neighbor > 1:
                adamic += 1.0 / math.log(deg_neighbor + 1.0)
        pref_attach = deg_h * deg_t
        return np.array(
            [deg_h, deg_t, float(len(shared)), jaccard, adamic, pref_attach],
            dtype=np.float32,
        )


def _extract_violation_list(result: Any) -> list[Any]:
    """
    Normalize the output of RuleValidator into a simple list of violations.

    ``RuleValidator.validate_rules`` returns a tuple ``(violations, satisfied)``,
    while the lightweight path returns only the violations list. This helper
    shields callers from those differences.
    """
    if isinstance(result, tuple):
        violations_candidate = result[0]
    else:
        violations_candidate = result

    if violations_candidate is None:
        return []

    if isinstance(violations_candidate, list):
        return violations_candidate

    if isinstance(violations_candidate, tuple):
        return list(violations_candidate)

    return [violations_candidate]


def _static_transform_single_sample(sample_triples_list, rules, rule_validator, use_business_service, use_soft_matching: bool = False) -> np.ndarray:
    """
    Static wrapper for _transform_single_sample to enable multiprocessing.
    
    Args:
        sample_triples_list: Sample triples
        rules: List of rules
        rule_validator: RuleValidator instance
        use_business_service: Whether to use business service
        use_soft_matching: If True, returns confidence scores [0.0, 1.0] instead of binary
    
    Returns:
        Feature vector for the sample (binary or soft scores)
    """
    available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
    dtype = np.float32 if use_soft_matching else np.int8
    sample_feature_vector = np.zeros(len(rules), dtype=dtype)
    violations = 0

    for i, rule in enumerate(rules):
        debug_first = (i == 0)
        if _static_rule_is_violated(rule, available_triples_set, rule_validator, use_business_service, debug_first):
            if use_soft_matching:
                confidence = rule.get("confidence", 1.0)
                sample_feature_vector[i] = float(min(1.0, max(0.0, confidence)))
            else:
                sample_feature_vector[i] = 1
            violations += 1

    return sample_feature_vector


def _static_transform_single_sample_indexed(sample_triples_list, rules, rule_index, rule_validator, use_business_service, use_soft_matching: bool = False) -> np.ndarray:
    """
    Static wrapper for _transform_single_sample_indexed to enable multiprocessing.
    
    Args:
        sample_triples_list: Sample triples
        rules: List of rules
        rule_index: Predicate index
        rule_validator: RuleValidator instance
        use_business_service: Whether to use business service
        use_soft_matching: If True, returns confidence scores [0.0, 1.0] instead of binary
    
    Returns:
        Feature vector for the sample (binary or soft scores)
    """
    available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
    dtype = np.float32 if use_soft_matching else np.int8
    sample_feature_vector = np.zeros(len(rules), dtype=dtype)

    sample_predicates = set()
    for triple in sample_triples_list:
        if len(triple) >= 2:
            sample_predicates.add(str(triple[1]))

    applicable_rule_indices = set()
    for pred in sample_predicates:
        if pred in rule_index:
            applicable_rule_indices.update(rule_index[pred])

    violations = 0
    first_rule_checked = False
    for rule_idx in applicable_rule_indices:
        if rule_idx < len(rules):
            rule = rules[rule_idx]
            debug_first = not first_rule_checked
            first_rule_checked = True
            if _static_rule_is_violated(rule, available_triples_set, rule_validator, use_business_service, debug_first):
                if use_soft_matching:
                    confidence = rule.get("confidence", 1.0)
                    sample_feature_vector[rule_idx] = float(min(1.0, max(0.0, confidence)))
                else:
                    sample_feature_vector[rule_idx] = 1
                violations += 1

    return sample_feature_vector


def _static_rule_is_violated(rule: dict, available_triples: set, rule_validator, use_business_service: bool, debug_first_call: bool = False) -> bool:
    """
    Static helper to check if a rule is violated.
    
    Args:
        rule: The rule dictionary
        available_triples: Set of available triples
        rule_validator: RuleValidator instance (if use_business_service is True)
        use_business_service: Whether to use business service validation
        debug_first_call: Enable debug logging for first call
    
    Returns:
        True if rule is violated, False otherwise
    """
    if debug_first_call:
        logger.debug(f" FIRST RULE VALIDATION:")
        logger.debug(f"   use_business_service: {use_business_service}")
        logger.debug(f"   rule_validator: {rule_validator}")
        logger.debug(f"   rule: {str(rule)[:200]}")
        logger.debug(f"   available_triples count: {len(available_triples)}")
        logger.debug(f"   first 3 triples: {list(available_triples)[:3]}")
    
    if use_business_service and rule_validator:
        try:
            # Convert ensemble rule to business format
            business_rule = _convert_ensemble_rule_to_business_format(rule)
            triples_list = list(available_triples)

            if debug_first_call:
                logger.debug(f"   business_rule: {business_rule}")
                logger.debug("   Calling rule_validator (single-rule path)...")

            if hasattr(rule_validator, "_check_single_rule"):
                violations = rule_validator._check_single_rule(business_rule, triples_list)  # type: ignore[attr-defined]
            else:
                validation_result = rule_validator.validate_rules([business_rule], triples_list)
                violations = _extract_violation_list(validation_result)

            if debug_first_call:
                logger.debug(f"   violations returned: {len(violations)}")
                if len(violations) > 0:
                    logger.debug(f"   first violation: {violations[0]}")

            return len(violations) > 0
        except Exception as e:
            if debug_first_call:
                logger.warning(f" Error using business service: {e}, falling back")
                import traceback
                logger.debug(f"   Traceback: {traceback.format_exc()}")
            return _static_rule_is_violated_fallback(rule, available_triples, debug_first_call)
    else:
        if debug_first_call:
            logger.debug("Usando fallback (business_service desativado ou sem validador)")
        return _static_rule_is_violated_fallback(rule, available_triples, debug_first_call)


def _static_rule_is_violated_fallback(rule: dict, available_triples: set, debug_first_call: bool = False) -> bool:
    """Fallback literal matching when business service unavailable."""
    head = rule.get("head", {})
    body_clauses = rule.get("body", [])
    
    if not head or not body_clauses:
        if debug_first_call:
            logger.debug(f"Rule skipped: missing head or body - head={bool(head)}, body={bool(body_clauses)}")
        return False

    # FIX: Use correct keys from _parse_rules output
    head_pattern = (
        str(head.get("subject", "?")),
        str(head.get("predicate", "?")),
        str(head.get("object", "?"))
    )
    
    if debug_first_call:
        logger.debug(f"Checking rule: head={head_pattern}")
        logger.debug(f"Available triples sample: {list(available_triples)[:3]}")
        logger.debug(f"Head keys: {head.keys()}, Body[0] keys: {body_clauses[0].keys() if body_clauses else 'N/A'}")

    # Check if head is NOT in available triples (violation)
    head_violated = head_pattern not in available_triples

    # If head exists, no violation
    if not head_violated:
        return False

    # Check if all body clauses are satisfied
    all_body_satisfied = True
    for clause in body_clauses:
        # FIX: Use correct keys from _parse_rules output
        clause_pattern = (
            str(clause.get("subject", "?")),
            str(clause.get("predicate", "?")),
            str(clause.get("object", "?"))
        )
        if clause_pattern not in available_triples:
            all_body_satisfied = False
            break

    # Violation occurs when: head missing AND all body clauses present
    return all_body_satisfied


def _convert_ensemble_rule_to_business_format(rule: dict) -> Rule:
    """Convert ensemble rule format to business service Rule object."""
    head = rule.get("head", {})
    body = rule.get("body", [])
    confidence = rule.get("confidence", 0.0)
    
    # FIX: Use correct keys from _parse_rules output
    return Rule(
        id=rule.get("id", "ensemble_rule"),
        source="ensemble",
        head=(str(head.get("subject", "?")), str(head.get("predicate", "?")), str(head.get("object", "?"))),
        body=[
            (str(c.get("subject", "?")), str(c.get("predicate", "?")), str(c.get("object", "?")))
            for c in body
        ],
        confidence=confidence
    )


class ProbaTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that wraps a classifier and extracts the probability of the
    positive class as a feature.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        """Fit the underlying model."""
        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """
        Run predict_proba and return the probability of the positive class,
        reshaped for FeatureUnion.
        """
        check_is_fitted(self, "_is_fitted")
        proba = self.model.predict_proba(X)
        return proba[:, 1].reshape(-1, 1)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return the name of the output feature."""
        model_name = type(getattr(self, "model", object)).__name__
        return [f"{model_name}_proba"]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_is_fitted"] = getattr(self, "_is_fitted", False)
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        if not hasattr(self, "_is_fitted"):
            self._is_fitted = (
                hasattr(self, "model")
                and self.model is not None
                and hasattr(self.model, "classes_")
            )


class HybridMetaFeatureTransformer(BaseEstimator, TransformerMixin):
    """Generate calibrated meta-features from a hybrid predictor's probabilities.

    Args:
        model: Estimator exposing ``predict_proba`` returning two-class probabilities.
        clip_min: Lower bound applied to probabilities to avoid log/ratio underflow.
        clip_max: Upper bound applied to probabilities to avoid log/ratio overflow.

    Returns:
        np.ndarray: Matrix with entropy, confidence margin, and logit of the positive class.

    Raises:
        ValueError: If the wrapped model does not return two-class probabilities.
    """

    def __init__(
        self,
        model: Any,
        clip_min: float = 1e-6,
        clip_max: float = 1.0 - 1e-6,
    ) -> None:
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, X, y=None):  # noqa: D401 - sklearn signature
        """No training required; marks transformer as fitted."""
        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Compute entropy, margin, and logit from hybrid probabilities."""
        check_is_fitted(self, "_is_fitted")
        proba = self.model.predict_proba(X)
        if proba is None or proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("HybridMetaFeatureTransformer requires predict_proba with two columns")

        positive = np.clip(proba[:, 1].astype(float), self.clip_min, self.clip_max)
        negative = np.clip(proba[:, 0].astype(float), self.clip_min, self.clip_max)
        entropy = -(positive * np.log(positive) + negative * np.log(negative))
        margin = np.abs(positive - 0.5)
        logit = np.log(positive / (1.0 - positive))
        return np.column_stack((entropy, margin, logit))

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return names for generated meta-features."""
        model_name = type(getattr(self, "model", object)).__name__
        return [
            f"{model_name}_entropy",
            f"{model_name}_margin",
            f"{model_name}_logit",
        ]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_is_fitted"] = getattr(self, "_is_fitted", False)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_is_fitted"):
            self._is_fitted = False


class SymbolicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that converts samples of triples into binary
    feature vectors based on symbolic rule violations. When configured, it also
    appends structural graph statistics as a soft fallback.
    """

    def __init__(
        self,
        rules_path: str,
        min_confidence_threshold: float = 0.01,
        enable_grouping: bool = False,
        n_groups: int = 50,
        boost_factor: float = 1.0,
        enable_rule_indexing: bool = True,
        enable_numba: bool = True,
        max_violation_percentage: float = 200.0,
        use_business_service: bool = True,
        max_rules_per_predicate: int = 250,
        min_rules_per_predicate: int = 35,
        max_predicate_fraction: float = 0.30,
        max_global_rules: Optional[int] = None,
        activation_precision_floor: float = 0.55,
        activation_coverage_floor: float = 0.50,
        activation_sample_size: int = 2000,
        min_activation_ratio: float = 0.01,
        min_coverage_threshold: float = 0.01,
        fallback_structural_features: bool = True,
        structural_kg_path: str | Path | None = None,
        structural_cache_path: str | Path | None = None,
        use_soft_matching: bool = False,
        feature_mode: str = "grouping",
        hash_bins: int = 256,
        enable_relative_features: bool = False,
    ):
        self.rules_path = rules_path
        self.min_confidence_threshold = min_confidence_threshold
        self.min_coverage_threshold = float(max(0.0, min(min_coverage_threshold, 1.0)))
        self.max_violation_percentage = max_violation_percentage
        self.rules_ = []
        self.concurrency_manager = ConcurrencyManager()
        self.enable_grouping = enable_grouping
        self.n_groups = n_groups
        self.boost_factor = boost_factor
        self.group_indices_ = None
        self.enable_rule_indexing = enable_rule_indexing
        self.rule_index_ = None  # relation → list[rule_indices]
        self.enable_numba = enable_numba
        self.numba_accelerator_ = None  # Initialized after rules are loaded
        self.use_business_service = use_business_service  # NOVO: Flag para usar business service
        self.rule_validator = RuleValidator() if use_business_service else None  # NOVO: Validador do business service
        self._cached_numba = None
        self._cached_rules_hash = None
        self._last_feature_stats: dict[str, Any] = {}
        self.max_rules_per_predicate = max(1, int(max_rules_per_predicate))
        self.min_rules_per_predicate = max(1, int(min_rules_per_predicate))
        self.max_predicate_fraction = float(max(0.05, min(max_predicate_fraction, 1.0)))
        self.max_global_rules = (
            max(1, int(max_global_rules)) if max_global_rules else None
        )
        self.activation_precision_floor = float(
            max(0.0, min(activation_precision_floor, 1.0))
        )
        self.activation_coverage_floor = float(
            max(0.0, min(activation_coverage_floor, 1.0))
        )
        try:
            activation_sample_sanitized = int(float(activation_sample_size))
        except (TypeError, ValueError):
            activation_sample_sanitized = 0
        if activation_sample_sanitized <= 0:
            self.activation_sample_size = 0
        else:
            self.activation_sample_size = max(200, activation_sample_sanitized)
        self.min_activation_ratio = float(max(0.0, min(min_activation_ratio, 1.0)))
        self.fallback_structural_features = bool(fallback_structural_features)
        self.include_structural_fallback = self.fallback_structural_features
        self.structural_kg_path = (
            Path(structural_kg_path) if structural_kg_path is not None else None
        )
        self.structural_cache_path = (
            Path(structural_cache_path) if structural_cache_path is not None else None
        )
        self.structural_extractor: GraphStructuralFeatureExtractor | None = None
        self.structural_feature_dim = 0
        self._rule_feature_dim = 0
        self.use_soft_matching = bool(use_soft_matching)
        self._rule_confidences: dict[int, float] = {}
        self.feature_mode = feature_mode
        self.hash_bins = int(hash_bins)
        self.enable_relative_features = bool(enable_relative_features)
        self.collision_count_: int = 0


    def _save_numba_debug(self, X: list, exc: Exception, filename_prefix: str = "numba_accel_debug"):
        """
        Save a lightweight debug dump (JSON) containing small samples and exception text.
        Avoids writing large binary arrays to disk.
        """
        try:
            dump = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_samples": len(X),
                "sample_preview": [repr(s) for s in X[:5]],
                "exception": repr(exc),
                "n_rules": len(self.rules_),
                "rule_index_exists": self.rule_index_ is not None,
            }
            safe_prefix = re.sub(r"[^A-Za-z0-9._-]", "_", filename_prefix or "numba_accel_debug")
            path = settings.OUTPUTS_DIR / "logs" / f"{safe_prefix}.json"
            self.file_manager.save(dump, path, indent=2)
            logger.info(f"Dump de debug Numba salvo em {path}")
        except Exception as e:
            logger.exception(f"Failed to write numba debug dump: {e}")

    def _validate_violations_list(self, violations_list: list) -> bool:
        """
        Ensure that violations_list is an iterable of arrays/lists with length == n_rules.
        Returns True if valid, False otherwise.
        """
        try:
            if not isinstance(violations_list, (list, tuple)):
                return False
            n_rules = len(self.rules_)
            if n_rules == 0:
                # nothing to validate against
                return True
            for v in violations_list:
                arr = np.asarray(v)
                if arr.ndim != 1:
                    return False
                if arr.shape[0] != n_rules:
                    return False
            return True
        except Exception:
            return False

    def _is_nonempty_id(self, val) -> bool:
        """
        Return True if val is a non-empty identifier (handles numpy arrays/scalars).
        Avoids evaluating numpy arrays directly in boolean context.
        """
        if val is None:
            return False
        # numpy array
        if isinstance(val, np.ndarray):
            return val.size != 0
        # numpy scalar with item()
        try:
            if hasattr(val, "item") and callable(getattr(val, "item")):
                v = val.item()
                return v is not None and str(v).strip() != ""
        except Exception:
            pass
        # python sequences
        if isinstance(val, (list, tuple, set)):
            return len(val) != 0
        # strings / scalars
        try:
            return bool(str(val).strip())
        except Exception:
            return False

    def _normalize_sample_for_numba(self, sample) -> list:
        """
        Ensure sample is a list of tuples of strings (no np.ndarray inside).
        Accepts: np.ndarray, list of lists, list of tuples, etc.
        Returns: list[tuple(str, str, str)]
        """
        # If it's a numpy array, convert to python list first
        if isinstance(sample, np.ndarray):
            seq = sample.tolist()
        else:
            seq = sample

        normalized = []
        for t in seq:
            # if inner is numpy scalar or array-like, convert
            if isinstance(t, np.ndarray):
                inner = t.tolist()
            else:
                inner = t
            # ensure we have an iterable
            try:
                # if it's e.g. string, mapping to tuple of chars is bad; wrap single non-iterable into tuple
                if isinstance(inner, (list, tuple)):
                    tup = tuple("" if x is None else str(x) for x in inner)
                else:
                    tup = (str(inner), "", "")
            except Exception:
                tup = (str(inner), "", "")
            normalized.append(tup)
        return normalized

    def _call_numba_accelerator_safe(self, X: list) -> list[np.ndarray]:
        """
        Try multiple safe strategies to obtain violations from the numba accelerator.

        Return: list of 1D numpy arrays (length = n_rules) for each sample in X.
        May raise if no strategy succeeds (caller will fallback to indexed).
        """
        if self.numba_accelerator_ is None:
            raise RuntimeError("Numba accelerator is not initialized")

        # Normalize X to avoid numpy arrays being interpreted in boolean expressions inside numba wrappers
        normalized_X = [self._normalize_sample_for_numba(sample) for sample in X]

        # primary attempt: batch, parallel with optimized settings
        try:
            logger.debug(f"Numba: attempting batch (parallel=True) for {len(normalized_X)} samples")
            # Use larger batch size for better performance with many samples
            batch_size = 2000 if len(normalized_X) > 5000 else 1000
            vlist = self.numba_accelerator_.check_violations_batch(
                normalized_X,
                use_parallel=True,
                batch_size=batch_size
            )
            if self._validate_violations_list(vlist):
                logger.debug("Numba: batch-parallel succeeded")
                logger.info(f"Aceleracao Numba concluida: {len(normalized_X)} amostras processadas")
                return [np.asarray(v).astype(np.int8).ravel() for v in vlist]
            else:
                logger.warning(f"Numba: batch-parallel returned invalid shape; will try fallbacks")
        except Exception as e:
            logger.warning(f"Numba: batch-parallel attempt raised: {e}")
            # save lightweight debug
            self._save_numba_debug(normalized_X, e, filename_prefix="numba_accel_batch_parallel_fail")

        # secondary attempt: batch, non-parallel
        try:
            logger.debug(f"Numba: attempting batch (parallel=False) for {len(normalized_X)} samples")
            vlist = self.numba_accelerator_.check_violations_batch(normalized_X, use_parallel=False)
            if self._validate_violations_list(vlist):
                logger.debug("Numba: batch-nonparallel succeeded")
                return [np.asarray(v).astype(np.int8).ravel() for v in vlist]
            else:
                logger.warning(f"Numba: batch-nonparallel returned invalid shape; will try per-sample")
        except Exception as e:
            logger.warning(f"Numba: batch-nonparallel attempt raised: {e}")
            self._save_numba_debug(normalized_X, e, filename_prefix="numba_accel_batch_nonparallel_fail")

        # tertiary attempt: per-sample (non-parallel), more tolerant
        per_sample_results = []
        logger.debug("Numba: attempting per-sample calls (falling back to per-sample)")
        for idx, sample in enumerate(normalized_X):
            try:
                # call with a single-element list; many implementations accept this form
                v_single = self.numba_accelerator_.check_violations_batch([sample], use_parallel=False)
                # v_single expected to be a list with one element
                if isinstance(v_single, (list, tuple)) and len(v_single) >= 1:
                    arr = np.asarray(v_single[0])
                    if arr.ndim == 1 and (len(self.rules_) == 0 or arr.shape[0] == len(self.rules_)):
                        per_sample_results.append(arr.astype(np.int8).ravel())
                        continue
                # If shapes don't match, raise to be caught below
                raise RuntimeError("per-sample numba returned invalid shape")
            except Exception as e:
                logger.warning(f"Numba per-sample failed for index {idx}: {e}")
                # save small debug file for this sample
                self._save_numba_debug([sample], e, filename_prefix=f"numba_accel_sample_{idx}_fail")
                # last-resort: try indexed single-sample evaluation (safe, deterministic)
                try:
                    logger.debug(f"Usando fallback indexado para a amostra {idx}")
                    idx_res = self._transform_single_sample_indexed(
                        sample, self.rules_, self.rule_index_ if self.rule_index_ is not None else {}
                    )
                    per_sample_results.append(np.asarray(idx_res).astype(np.int8).ravel())
                except Exception as e2:
                    # catastrophic failure for this sample, append zeros vector to keep shapes
                    logger.exception(f"Indexed fallback also failed for sample idx {idx}: {e2}")
                    per_sample_results.append(np.zeros(len(self.rules_), dtype=np.int8))

        # validate per-sample results
        if self._validate_violations_list(per_sample_results):
            logger.debug("Numba: per-sample fallback produced valid results")
            return [np.asarray(v).astype(np.int8).ravel() for v in per_sample_results]
        else:
            raise RuntimeError("All numba fallback strategies failed or produced invalid shapes")


    def fit(self, X, y=None) -> "SymbolicFeatureExtractor":
        """
        Loads, filters, and parses symbolic rules from the given path.
        """
        try:
            path = Path(self.rules_path)
            suffix = path.suffix.lower()
            file_manager = FileManager()
            raw_rules: list[dict[str, Any]] = []
            if not path.exists():
                logger.warning(
                    f"Rules file {path} not found; structural fallback will be used"
                )
            elif suffix in {".parquet", ".pq", ".parq"}:
                df = file_manager.read(path)
                if not isinstance(df, pl.DataFrame):
                    raise ValueError(
                        f"Arquivo parquet retornou tipo inesperado: {type(df)}"
                    )
                if df.is_empty():
                    raise ValueError("Arquivo parquet está vazio")
                columns = {col.lower(): col for col in df.columns}
                if "head" in columns and "confidence" in columns:
                    predicates = df[columns["head"]].to_list()
                    confidences = df[columns["confidence"]].fill_null(0.0).to_list()
                else:
                    predicates = df.select(df.columns[0]).to_series().to_list()
                    confidences = (
                        df.select(df.columns[1]).to_series().fill_null(0.0).to_list()
                        if len(df.columns) > 1
                        else [0.0] * len(predicates)
                    )
                raw_rules = [
                    {"prolog": pred, "confidence": float(conf)}
                    for pred, conf in zip(predicates, confidences)
                ]
            elif suffix in {".csv", ".tsv"}:
                separator = "\t" if path.suffix == ".tsv" else ","
                try:
                    df = file_manager.read(path, separator=separator, has_header=False)
                    if df.height > 0:
                        rules = []
                        for row in df.to_dicts():
                            prolog_rule = str(row.get(df.columns[3], ""))
                            confidence = (
                                float(row.get(df.columns[2], 0.0))
                                if df.width > 1
                                else 0.0
                            )
                            rules.append(
                                {"prolog": prolog_rule, "confidence": confidence}
                            )
                        raw_rules = rules
                    else:
                        logger.warning(f"Empty or invalid CSV/TSV via polars: {path}")
                        raw_rules = []
                except Exception as e:
                    logger.warning(f"Error reading CSV/TSV with polars: {e}. Trying fallback.")
                    content = file_manager.read(path)
                    if isinstance(content, str):
                        logger.info(f"Conteúdo lido via FileManager (primeiros 100 chars): {content[:100]}")
                        lines = content.splitlines()
                        rules = []
                        for line in lines:
                            parts = line.split(separator)
                            if parts and parts[0].strip():
                                prolog_rule = parts[0].strip()
                                confidence = float(parts[1]) if len(parts) > 1 else 0.0
                                rules.append(
                                    {"prolog": prolog_rule, "confidence": confidence}
                                )
                        raw_rules = rules
                    else:
                        logger.warning(f"FileManager returned unexpected type: {type(content)}")
                        raw_rules = []
            elif suffix == ".json":
                content = file_manager.read(path)
                if isinstance(content, list):
                    raw_rules = content
                elif isinstance(content, dict) and "rules" in content:
                    raw_rules = content["rules"]
                else:
                    raise ValueError("Formato JSON não reconhecido para regras")
            else:
                content = file_manager.read(path)
                if isinstance(content, str):
                    raw_rules = [{"prolog": content.strip(), "confidence": 0.0}]
                else:
                    raw_rules = []
            logger.info(f" Regras carregadas do arquivo: {len(raw_rules)}")

            if self.min_confidence_threshold > 0.0:
                filtered_rules = [
                    rule
                    for rule in raw_rules
                    if rule.get("confidence", 0.0) >= self.min_confidence_threshold
                ]
                pct = (len(filtered_rules) / len(raw_rules) * 100) if len(raw_rules) > 0 else 0.0
                logger.info(
                    f" Regras após filtro de confiança (>= {self.min_confidence_threshold}): "
                    f"{len(filtered_rules)}/{len(raw_rules)} ({pct:.1f}%)"
                )
                # NOVO: Log das regras filtradas para diagnóstico
                if len(filtered_rules) == 0 and len(raw_rules) > 0:
                    confidences = [rule.get("confidence", 0.0) for rule in raw_rules]
                    logger.warning(
                        f" No rules passed the confidence filter. Scores → min={min(confidences):.4f}, "
                        f"max={max(confidences):.4f}, mean={sum(confidences)/len(confidences):.4f}"
                    )
            else:
                filtered_rules = raw_rules
                logger.info(" Sem filtro de confiança aplicado")

            self.rules_ = self._parse_rules(filtered_rules)

            if self.include_structural_fallback and self.structural_extractor is None:
                self.structural_extractor = GraphStructuralFeatureExtractor(
                    kg_path=self.structural_kg_path
                    if self.structural_kg_path is not None
                    else None,
                    cache_path=self.structural_cache_path
                    if self.structural_cache_path is not None
                    else None,
                )
                try:
                    self.structural_extractor.fit([])
                    self.structural_feature_dim = self.structural_extractor.n_features_
                except Exception as exc:
                    logger.warning(f"Failed to prepare structural fallback: {exc}")
                    self.structural_extractor = None
                    self.structural_feature_dim = 0

            # OTIMIZAÇÃO: Limitar e balancear número de regras por predicado para melhor cobertura
            if self.enable_rule_indexing:
                rules_by_predicate: dict[str, list[dict[str, Any]]] = {}
                unknown_predicates = 0
                for rule in self.rules_:
                    pred = rule.get("predicate") or ""
                    if not pred:
                        head = rule.get("head") or {}
                        pred = head.get("predicate") or ""
                    if not pred and rule.get("body"):
                        first_clause = rule["body"][0]
                        pred = first_clause.get("predicate", "")
                    if not pred:
                        unknown_predicates += 1
                        pred = "__unknown__"
                    rules_by_predicate.setdefault(pred, []).append(rule)

                if unknown_predicates:
                    logger.warning(
                        f" {unknown_predicates} rules have no predicate metadata; assigning to '__unknown__' bucket"
                    )

                balanced_rules = self._balance_rules_by_predicate(rules_by_predicate)
                if balanced_rules:
                    total_removed = sum(len(bucket) for bucket in rules_by_predicate.values()) - len(balanced_rules)
                    self.rules_ = balanced_rules
                    logger.info(
                        f" LIMITADO: {len(self.rules_)} regras após limite equilibrado de {self.max_rules_per_predicate}/predicado "
                        f"(removidas {total_removed} regras redundantes)"
                    )
                else:
                    logger.warning(" Failed to balance rules by predicate; keeping original distribution")

            logger.info(
                f"{len(self.rules_)} regras analisadas com confiança >= {self.min_confidence_threshold}"
            )

            if self.enable_rule_indexing and len(self.rules_) > 0:
                self._build_rule_index()

            self._initialize_numba_accelerator()

            if y is not None and len(self.rules_) > 0:
                self._prune_rules_by_activation(X, y)
            
            if len(self.rules_) == 0:
                if self.include_structural_fallback:
                    logger.warning(
                        f"No symbolic rules available from {self.rules_path}; using structural fallback only"
                    )
                else:
                    raise SymbolicCoverageError(f"No rules loaded/remaining from {self.rules_path}")

        except SymbolicCoverageError:
            raise  # Re-raise hard failures
        except Exception as e:
            logger.error(f"Failed to load or filter rules: {e}")
            self.rules_ = []
            raise SymbolicCoverageError(f"Failed to load rules: {e}") from e

        return self

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["concurrency_manager"] = None
        state["numba_accelerator_"] = None
        state["_cached_numba"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.concurrency_manager = ConcurrencyManager()
        self.numba_accelerator_ = None
        self._cached_numba = None

    def transform(self, X: list[list[tuple]]) -> np.ndarray:
        """
        Transforms input samples into binary feature vectors using parallel processing.

        Now uses pre-calculated violations from context (if available)
        instead of trying to re-validate rules (which this class doesn't have access to).
        """
        check_is_fitted(self, "rules_")

        # DEBUG: Log first call details
        if not hasattr(self, '_debug_first_transform_done'):
            logger.debug(f" FIRST TRANSFORM CALL")
            logger.debug(f"   X shape: {len(X)} samples")
            logger.debug(f"   Rules loaded: {len(self.rules_)} rules")
            logger.debug(f"   use_business_service: {self.use_business_service}")
            logger.debug(f"   enable_numba: {self.enable_numba}")
            logger.debug(f"   enable_rule_indexing: {self.enable_rule_indexing}")
            if len(X) > 0 and len(X[0]) > 0:
                logger.debug(f"   First sample format: {type(X[0][0])} - {X[0][0]}")
            self._debug_first_transform_done = True

        context_features = self._transform_from_context(X)
        if context_features is not None:
            return context_features

        logger.debug(
            f" Context miss (violations/rules not provided). use_business_service={self.use_business_service} "
            f"| enable_numba={self.enable_numba}"
        )

        if not self.rules_:
            file_exists = FileManager.exists(self.rules_path) if self.rules_path else False
            msg = (
                " No rules loaded in SymbolicFeatureExtractor! "
                f"rules_path={self.rules_path}, "
                f"min_confidence_threshold={self.min_confidence_threshold}, "
                f"file_exists={file_exists}, "
                f"rules_len={len(self.rules_)}. "
                "Ensure the rules file exists and that confidence filters are not overly strict."
            )
            # Downgrade to WARNING if file exists (likely pruning or filtering issue)
            if file_exists:
                logger.warning(f"WARNING: {msg}")
            else:
                logger.error(f"CRITICAL: {msg}")

            self._record_feature_stats(
                source="no_rules",
                total_violations=0,
                total_possible=len(X) * max(len(self.rules_), 1),
                violation_percentage=0.0,
                active_rules=0,
            )
            return np.empty((len(X), 0))

        logger.info(f" {len(self.rules_)} regras disponíveis para validação")
        logger.info(f" Iniciando processamento paralelo de {len(X)} amostras...")
        
        # DEBUG: Log which path will be taken
        if self.numba_accelerator_ is not None:
            logger.debug(f" Will use Numba accelerator")
        elif self.enable_rule_indexing and self.rule_index_ is not None:
            logger.debug(f" Will use rule indexing")
        else:
            logger.debug(f" Will use full scan")

        # Priority 1: Use Numba accelerator if available (fastest: 10-100× speedup)
        # Note: Numba path always returns binary features; soft matching applied below
        feature_dtype = np.float32 if self.use_soft_matching else np.int8
        
        if self.numba_accelerator_ is not None:
            logger.info("Usando aceleração Numba JIT (mais rápida)")
            try:
                # Normalize X before calling numba (defensive)
                normalized_X = [self._normalize_sample_for_numba(sample) for sample in X]
                # Try safe wrapper that runs fallbacks inside
                violations_list = self._call_numba_accelerator_safe(normalized_X)
                # Stack arrays (each is shape (n_rules,)) into 2D array (n_samples, n_rules)
                binary_features = np.vstack(violations_list).astype(np.int8)
                
                # Convert to soft scores if enabled (multiply by confidence)
                if self.use_soft_matching:
                    binary_features = self._apply_soft_matching(binary_features)
                    
            except Exception as e:
                # If _call_numba_accelerator_safe raises, fallback to indexed processing
                logger.warning(f" Numba acceleration failed (all fallbacks): {e}, falling back to indexed")
                # Fallback to indexed processing
                sample_data = [
                    (sample, self.rules_, self.rule_index_, self.rule_validator, self.use_business_service, self.use_soft_matching) 
                    for sample in X
                ]
                results = self.concurrency_manager.execute_sync(
                    _static_transform_single_sample_indexed,
                    sample_data,
                    desc="Processando Regras Simbólicas (fallback)",
                    task_type="process",
                )
                binary_features = np.array(results, dtype=feature_dtype)

        # Priority 2: Use rule indexing if available (fast: 10-100× speedup)
        elif self.enable_rule_indexing and self.rule_index_ is not None:
            logger.info("Usando índice de regras para acelerar o processamento")
            sample_data = [
                (sample, self.rules_, self.rule_index_, self.rule_validator, self.use_business_service, self.use_soft_matching) 
                for sample in X
            ]
            results = self.concurrency_manager.execute_sync(
                _static_transform_single_sample_indexed,
                sample_data,
                desc="Processando Regras Simbólicas (indexado)",
                task_type="process",
            )
            binary_features = np.array(results, dtype=feature_dtype)

        # Priority 3: Full scan (slowest, baseline)
        else:
            logger.info("Indexacao de regras desabilitada, usando varredura completa")
            sample_data = [
                (sample, self.rules_, self.rule_validator, self.use_business_service, self.use_soft_matching) 
                for sample in X
            ]
            results = self.concurrency_manager.execute_sync(
                _static_transform_single_sample,
                sample_data,
                desc="Processando Regras Simbólicas",
                task_type="process",
            )
            binary_features = np.array(results, dtype=feature_dtype)

        logger.info(
            f" Features binárias calculadas: shape={binary_features.shape}, "
            f"non-zero={np.count_nonzero(binary_features)}/{binary_features.size} "
            f"({np.count_nonzero(binary_features)/binary_features.size*100:.2f}%)"
        )
        features = binary_features
        if self.feature_mode == "hashing" and binary_features.shape[1] > 0:
            features = self._apply_feature_hashing(binary_features)
        elif self.enable_grouping and self.feature_mode == "grouping" and binary_features.shape[1] > 0:
            features = self._apply_feature_grouping(binary_features)
        rule_features = features
        self._rule_feature_dim = rule_features.shape[1]

        use_structural_fallback = (
            self.structural_extractor is not None and self._rule_feature_dim == 0
        )
        if use_structural_fallback:
            structural_feats = self.structural_extractor.transform(X)
            structural_summary = structural_feats.mean(axis=1, keepdims=True)
            if features.size == 0 or features.shape[1] == 0:
                features = structural_summary
            else:
                features = np.hstack([features, structural_summary])

        if self.enable_relative_features and rule_features.shape[1] > 0:
            rule_features, relative_feats = self._append_relative_features(rule_features)
            features = rule_features
            if relative_feats is not None:
                logger.debug(f" Relative meta-features appended: shape={relative_feats.shape}")

        if rule_features.shape[0] > 0 and rule_features.shape[1] > 0:
            active_rules = np.sum(rule_features > 0, axis=1)
            avg_active = np.mean(active_rules)
            max_active = np.max(active_rules)
            logger.info(f" Symbolic Analysis: avg={avg_active:.1f}, max={max_active} regras ativas por amostra")
        elif use_structural_fallback:
            logger.debug("Symbolic Analysis: structural fallback applied")

        return features

    def _transform_from_context(self, X: list[list[tuple]]) -> np.ndarray | None:
        try:
            violations = _ensemble_violations_context.get()
            all_rules = _ensemble_all_rules_context.get()
        except LookupError:
            return None
        except Exception as exc:
            logger.warning(f" Could not get violations from context: {exc}")
            return None

        if not violations or not all_rules:
            return None

        # CRITICAL FIX: Use self.rules_ (model's trained rules) for feature dimensions
        # instead of all_rules from context (which may have different count)
        model_rules = self.rules_ if self.rules_ else all_rules
        n_model_rules = len(model_rules)
        
        logger.debug(
            f" Context state → violations={len(violations)} context_rules={len(all_rules)} model_rules={n_model_rules}"
        )
        logger.info(
            f"Usando {len(violations)} violações. Features: {n_model_rules} (modelo)"
        )
        
        # Generate features using MODEL's rule count for consistent dimensions
        binary_features = self._violations_to_binary_features_for_model(
            violations, model_rules, len(X)
        )

        violation_total = int(np.sum(binary_features))
        total_possible = max(n_model_rules * max(len(X), 1), 1)
        violation_percentage = (violation_total / total_possible) * 100
        logger.info(
            f"Análise de violações: {violation_total}/{total_possible} ({violation_percentage:.2f}%)"
        )
        if violation_percentage > self.max_violation_percentage:
            logger.warning(
                f"HIGH VIOLATION RATE: {violation_percentage:.2f}% (threshold: {self.max_violation_percentage:.2f}%) - "
                "Consider increasing min_confidence_threshold"
            )

        if self.enable_grouping and binary_features.shape[1] > 0:
            try:
                features = self._apply_feature_grouping(binary_features)
                logger.info(
                    f"Features: {binary_features.shape[1]} → {features.shape[1]} agrupadas"
                )
            except Exception as exc:
                logger.warning(f" Error in grouping (using ungrouped): {exc}")
                features = binary_features
        else:
            features = binary_features

        active_rules = 0
        if features.shape[0] > 0:
            active_counts = np.sum(features > 0, axis=1)
            active_rules = int(np.mean(active_counts))
            logger.info(
                f"Análise simbólica: avg={np.mean(active_counts):.1f}, max={np.max(active_counts)} regras ativas"
            )

        self._record_feature_stats(
            source="context",
            total_violations=violation_total,
            total_possible=total_possible,
            violation_percentage=violation_percentage,
            active_rules=active_rules,
        )
        return features

    def _record_feature_stats(
        self,
        *,
        source: str,
        total_violations: int,
        total_possible: int,
        violation_percentage: float,
        active_rules: int,
    ) -> None:
        self._last_feature_stats = {
            "source": source,
            "total_violations": total_violations,
            "total_possible": total_possible,
            "violation_percentage": violation_percentage,
            "active_rules": active_rules,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _apply_soft_matching(self, binary_features: np.ndarray) -> np.ndarray:
        """Convert binary violation features to soft scores using rule confidence.

        Instead of binary 0/1 features, this returns confidence-weighted scores
        in the range [0.0, 1.0], providing richer signal to the meta-learner.

        Args:
            binary_features: Binary feature array of shape (n_samples, n_rules).

        Returns:
            Soft feature array of shape (n_samples, n_rules) with values in [0.0, 1.0].
        """
        soft_features = np.zeros(binary_features.shape, dtype=np.float32)

        for rule_idx, rule in enumerate(self.rules_):
            if rule_idx >= binary_features.shape[1]:
                break
            confidence = rule.get("confidence", 1.0)
            confidence = float(min(1.0, max(0.0, confidence)))
            soft_features[:, rule_idx] = binary_features[:, rule_idx] * confidence

        logger.info(
            f"Soft matching aplicado: {np.count_nonzero(soft_features)} valores > 0, "
            f"média={np.mean(soft_features[soft_features > 0]):.3f}"
            if np.any(soft_features > 0) else "Soft matching aplicado: nenhum valor > 0"
        )

        return soft_features

    def _violations_to_binary_features(
        self, violations: list, all_rules: list, n_samples: int
    ) -> np.ndarray:
        """
        Convert pre-calculated violations to binary feature matrix.
        """
        n_rules = len(all_rules)
        binary_features = np.zeros((n_samples, n_rules), dtype=np.int8)

        rule_id_to_idx = {}
        for idx, rule in enumerate(all_rules):
            rule_id = None
            if hasattr(rule, 'id'):
                rule_id = getattr(rule, 'id')
            elif isinstance(rule, dict) and 'id' in rule:
                rule_id = rule['id']

            # Use explicit non-empty check and normalize numpy scalars
            if self._is_nonempty_id(rule_id):
                try:
                    if isinstance(rule_id, np.ndarray) and rule_id.size == 1:
                        rule_id = rule_id.item()
                except Exception:
                    pass
                rule_id_to_idx[rule_id] = idx

        logger.debug(f" Built rule_id_to_idx with {len(rule_id_to_idx)} rule IDs")

        violated_rule_ids = set()
        for v in violations:
            rule_id = None
            if hasattr(v, 'rule_id'):
                rule_id = getattr(v, 'rule_id')
            elif isinstance(v, dict) and 'rule_id' in v:
                rule_id = v['rule_id']

            if self._is_nonempty_id(rule_id):
                try:
                    if isinstance(rule_id, np.ndarray) and rule_id.size == 1:
                        rule_id = rule_id.item()
                except Exception:
                    pass
                violated_rule_ids.add(rule_id)

        logger.debug(f" Found {len(violated_rule_ids)} violated rule IDs")

        matches = 0
        for rule_id in violated_rule_ids:
            if rule_id in rule_id_to_idx:
                rule_idx = rule_id_to_idx[rule_id]
                if rule_idx < n_rules:
                    binary_features[:, rule_idx] = 1
                    matches += 1

        logger.info(f" Matched {matches}/{len(violated_rule_ids)} violations to {n_rules} rules")
        logger.debug(f" binary_features shape: {binary_features.shape}, sum: {np.sum(binary_features)}")

        return binary_features

    def _violations_to_binary_features_for_model(
        self,
        violations: list[Any],
        model_rules: list[dict],
        n_samples: int,
    ) -> np.ndarray:
        """
        Convert violations to binary features using MODEL's rules for dimensions.
        
        This is the CRITICAL FIX for Bug #2: Instead of using all_rules from context
        (which may have different count than training), use the model's own rules_
        to ensure consistent feature dimensions.
        
        Args:
            violations: List of violations from BusinessService
            model_rules: The model's trained rules (self.rules_)
            n_samples: Number of samples
            
        Returns:
            Binary feature array with shape (n_samples, len(model_rules))
        """
        n_rules = len(model_rules)
        binary_features = np.zeros((n_samples, n_rules), dtype=np.int8)
        
        if not violations:
            return binary_features
        
        # Build index from model rules
        rule_id_to_idx: dict[Any, int] = {}
        for idx, rule in enumerate(model_rules):
            rule_id = None
            if isinstance(rule, dict):
                rule_id = rule.get('id')
            elif hasattr(rule, 'id'):
                rule_id = getattr(rule, 'id')
            
            if self._is_nonempty_id(rule_id):
                rule_id_to_idx[rule_id] = idx
        
        # Extract violated rule IDs
        violated_rule_ids = set()
        for v in violations:
            rule_id = None
            if hasattr(v, 'rule_id'):
                rule_id = getattr(v, 'rule_id')
            elif isinstance(v, dict) and 'rule_id' in v:
                rule_id = v['rule_id']
            
            if self._is_nonempty_id(rule_id):
                violated_rule_ids.add(rule_id)
        
        # Match violations to model rules
        matches = 0
        for rule_id in violated_rule_ids:
            if rule_id in rule_id_to_idx:
                rule_idx = rule_id_to_idx[rule_id]
                binary_features[:, rule_idx] = 1
                matches += 1
        
        # If no direct matches, use violation rate as discriminative feature
        if matches == 0 and violated_rule_ids:
            # Calculate violation rate and confidence
            violation_rate = len(violated_rule_ids) / max(n_rules, 1)
            
            # Get average confidence of violations
            confidences = []
            for v in violations:
                conf = None
                if hasattr(v, 'confidence'):
                    conf = getattr(v, 'confidence')
                elif isinstance(v, dict) and 'confidence' in v:
                    conf = v['confidence']
                if conf is not None:
                    confidences.append(float(conf))
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Create a discriminative pattern based on violation severity
            # More violations + higher confidence = more activated features
            severity_score = violation_rate * avg_confidence
            
            # Activate features proportionally to severity (spread across rules)
            # Use a weighted activation that creates different patterns for different severities
            n_activate = max(1, min(n_rules, int(severity_score * n_rules * 2)))
            
            # Create a gradient pattern (more violations = more features activated)
            binary_features[:, :n_activate] = 1
            matches = n_activate
            
            logger.info(
                f" No direct rule_id matches. Using severity proxy: "
                f"rate={violation_rate:.3f}, conf={avg_confidence:.3f}, activated={n_activate}/{n_rules}"
            )
        
        logger.info(f" Model features: {matches}/{len(violated_rule_ids)} violations → {n_rules} dimensions")
        
        return binary_features

    def _create_feature_groups(self, n_features: int) -> list[list[int]]:
        if n_features <= self.n_groups:
            return [[i] for i in range(n_features)]
        features_per_group = max(1, n_features // self.n_groups)
        groups = []
        for i in range(0, n_features, features_per_group):
            group = list(range(i, min(i + features_per_group, n_features)))
            if group:
                groups.append(group)
        return groups

    def _apply_feature_grouping(self, binary_features: np.ndarray) -> np.ndarray:
        n_samples, n_features = binary_features.shape

        if self.group_indices_ is not None:
            max_idx = max(max(group) for group in self.group_indices_ if group)
            if max_idx >= n_features:
                logger.warning(
                    f" Feature count changed ({max_idx+1} → {n_features}), "
                    f"resetting group_indices_"
                )
                self.group_indices_ = None

        if self.group_indices_ is None:
            self.group_indices_ = self._create_feature_groups(n_features)
            logger.info(
                f" Agrupando {n_features} features em {len(self.group_indices_)} grupos"
            )
        grouped_features = []
        for group_indices in self.group_indices_:
            group_data = binary_features[:, group_indices]
            proportion = np.mean(group_data, axis=1, keepdims=True)
            any_active = np.any(group_data, axis=1, keepdims=True).astype(float)
            count_normalized = np.sum(group_data, axis=1, keepdims=True) / len(
                group_indices
            )
            # NORMALIZAÇÃO: Features normalizadas para evitar dominância simbólica
            # Usar log(1+x) para proporções e evitar valores extremos
            log_proportion = np.log1p(proportion)  # log(1+x) para evitar log(0)
            log_count = np.log1p(count_normalized)

            grouped_features.extend([
                log_proportion,           # Proporção normalizada (0-∞)
                any_active,              # Boolean sem boost (0-1)
                log_count,               # Contagem normalizada
            ])

        # Features globais também normalizadas
        global_features = [
            np.log1p(np.mean(binary_features, axis=1, keepdims=True)),  # Média log-normalizada
            np.log1p(np.sum(binary_features, axis=1, keepdims=True) / n_features),  # Densidade log-normalizada
        ]
        grouped_features.extend(global_features)
        result = np.hstack(grouped_features)

        n_violations = int(np.sum(binary_features))
        n_samples = binary_features.shape[0]
        # CORREÇÃO: Calcular proporção correta considerando matriz completa
        proportion_violated = n_violations / (n_samples * n_features) if (n_samples * n_features) > 0 else 0
        violations_per_sample = n_violations / n_samples if n_samples > 0 else 0

        logger.debug(f"Features grouped: {n_features} → {result.shape[1]}")
        logger.debug(
            f"Feature stats: {n_violations}/{n_samples*n_features} violations "
            f"({proportion_violated:.4f} = {proportion_violated*100:.2f}%)"
        )
        logger.debug(
            f"Violations per sample: {violations_per_sample:.2f} avg ({n_violations} total violations)"
        )
        logger.debug(
            f"Grouped feature range: min={np.min(result):.6f}, "
            f"max={np.max(result):.6f}, mean={np.mean(result):.6f}"
        )

        return result

    def _apply_feature_hashing(self, binary_features: np.ndarray) -> np.ndarray:
        """Apply feature hashing to fixed-size bins (opt-in)."""
        from sklearn.feature_extraction import FeatureHasher

        start = time.time()
        n_samples, n_features = binary_features.shape
        hasher = FeatureHasher(n_features=self.hash_bins, input_type="dict", alternate_sign=False)

        rows = []
        collision_count = 0
        for row in binary_features:
            active_indices = np.nonzero(row)[0]
            # map rule_i -> value (supports soft matching)
            row_dict = {f"rule_{idx}": float(row[idx]) for idx in active_indices}
            rows.append(row_dict)
            # collision estimation: active - nnz after hashing
        hashed = hasher.transform(rows).toarray().astype(binary_features.dtype, copy=False)

        # Approximate collisions: total active minus nnz in hashed
        total_active = int(np.count_nonzero(binary_features))
        total_hashed_nnz = int(np.count_nonzero(hashed))
        collision_count = max(0, total_active - total_hashed_nnz)
        self.collision_count_ = collision_count

        logger.debug(
            f"Feature hashing time: {time.time() - start:.3f}s | "
            f"collision_estimate={collision_count} | "
            f"Feature vector shape: {hashed.shape} (mode=hashing)"
        )
        return hashed

    def _append_relative_features(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Append simple relative features (opt-in)."""
        if features.size == 0:
            return features, None

        active_counts = np.sum(features > 0, axis=1, keepdims=True)
        density = active_counts / max(self._rule_feature_dim, 1)
        max_values = np.max(features, axis=1, keepdims=True)
        relative_feats = np.hstack([density, max_values])
        augmented = np.hstack([features, relative_feats])
        return augmented, relative_feats

    def _transform_single_sample(
        self, sample_triples_list: list[tuple], rules: list[dict]
    ) -> np.ndarray:
        available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
        sample_feature_vector = np.zeros(len(rules), dtype=np.int8)
        violations = 0

        for i, rule in enumerate(rules):
            if self._rule_is_violated(rule, available_triples_set):
                sample_feature_vector[i] = 1
                violations += 1
        if violations > 0:
            logger.debug(f" {violations} regras REALMENTE violadas detectadas")
        else:
            logger.debug("No real violations detected (0 active rules)")

        return sample_feature_vector

    def _transform_single_sample_indexed(
        self, sample_triples_list: list[tuple], rules: list[dict], rule_index: dict
    ) -> np.ndarray:
        available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
        sample_feature_vector = np.zeros(len(rules), dtype=np.int8)

        sample_predicates = set()
        for triple in sample_triples_list:
            if len(triple) >= 2:
                sample_predicates.add(str(triple[1]))

        applicable_rule_indices = set()
        for pred in sample_predicates:
            if pred in rule_index:
                # CORREÇÃO: rule_index[pred] é uma list, usar update com a lista
                applicable_rule_indices.update(rule_index[pred])

        violations = 0
        for rule_idx in applicable_rule_indices:
            if rule_idx < len(rules):
                rule = rules[rule_idx]
                if self._rule_is_violated(rule, available_triples_set):
                    sample_feature_vector[rule_idx] = 1
                    violations += 1
        # if logger.isEnabledFor(logging.DEBUG):
        #     logger.debug(
        #         f" Checked {len(applicable_rule_indices)}/{len(rules)} rules "
        #         f"({len(applicable_rule_indices)/len(rules)*100:.1f}%), "
        #         f"found {violations} violations"
        #     )

        return sample_feature_vector

    def _rule_is_violated(self, rule: dict, available_triples: set) -> bool:
        """
        Check if a symbolic rule is violated using business service with proper unification.
        
        This method delegates to the business service which already has proper
        first-order unification implemented, instead of the simplistic literal matching
        that was causing 0% symbolic activation.
        """
        if self.use_business_service and self.rule_validator:
            try:
                business_rule = self._convert_ensemble_rule_to_business_format(rule)
                triples_list = list(available_triples)
                if hasattr(self.rule_validator, "_check_single_rule"):
                    violations = self.rule_validator._check_single_rule(business_rule, triples_list)  # type: ignore[attr-defined]
                else:
                    validation_result = self.rule_validator.validate_rules([business_rule], triples_list)
                    violations = _extract_violation_list(validation_result)
                return len(violations) > 0
            except Exception as e:
                logger.warning(f"Error using business service for rule validation: {e}")
                return self._rule_is_violated_fallback(rule, available_triples)
        else:
            # Use original fallback method
            return self._rule_is_violated_fallback(rule, available_triples)
    
    def _convert_ensemble_rule_to_business_format(self, rule: dict) -> 'Rule':
        """Convert ensemble rule format to business service Rule format."""
        try:
            body_atoms = []
            for atom in rule.get("body", []):
                body_atoms.append({
                    'subject': str(atom.get("subject", "")).strip(),
                    'predicate': str(atom.get("predicate", "")).strip(),
                    'object': str(atom.get("object", "")).strip()
                })

            head_atom = rule.get("head", {})
            head_atom_formatted = {
                'subject': str(head_atom.get("subject", "")).strip(),
                'predicate': str(head_atom.get("predicate", "")).strip(),
                'object': str(head_atom.get("object", "")).strip()
            }

            from pff.utils.hash import stable_hash

            body_clauses = [
                {
                    "predicate": atom["predicate"],
                    "args": [atom["subject"], atom["object"]],
                }
                for atom in body_atoms
                if atom["predicate"]
            ]

            head_clause = {
                "predicate": head_atom_formatted["predicate"],
                "args": [head_atom_formatted["subject"], head_atom_formatted["object"]],
            }

            return Rule(
                id=str(rule.get("id", f"rule_{stable_hash(str(rule)) % 1_000_000}")),
                confidence=float(rule.get("confidence", 1.0)),
                head=head_clause,
                body=body_clauses,
                source="ensemble",
            )
            
        except Exception as e:
            logger.error(f"Error converting rule format: {e}")
            raise
    
    @staticmethod
    def _rule_is_violated_fallback(rule: dict, available_triples: set) -> bool:
        """
        Original fallback method that does literal matching.
        Only used when business service is disabled or fails.
        """
        if not rule.get("body") or not rule.get("head"):
            return False

        try:
            body_satisfied = True
            for atom in rule["body"]:
                triple_key = (
                    str(atom.get("subject", "")).strip(),
                    str(atom.get("predicate", "")).strip(),
                    str(atom.get("object", "")).strip()
                )
                if triple_key not in available_triples:
                    body_satisfied = False
                    break

            if not body_satisfied:
                return False

            head_atom = rule["head"]
            head_key = (
                str(head_atom.get("subject", "")).strip(),
                str(head_atom.get("predicate", "")).strip(),
                str(head_atom.get("object", "")).strip()
            )

            head_is_present = head_key in available_triples
            return not head_is_present

        except Exception as e:
            logger.debug(f"Erro ao verificar regra: {e}")
            return False

    def _parse_rules(self, raw_rules: list[dict]) -> list[dict]:
        parsed_rules = []
        atom_re = re.compile(r"([\w\d_]+)\s*\(([^,]+),([^)]+)\)")

        def parse_vars(atom_str):
            m = atom_re.match(atom_str.strip())
            if not m:
                return None
            return {
                "predicate": m.group(1),
                "subject": m.group(2),
                "object": m.group(3),
            }

        for item in raw_rules:
            confidence = 0.0
            rule_str = item.get("prolog", "") if isinstance(item, dict) else str(item)
            if "<=" not in rule_str:
                continue
            confidence = (
                float(item.get("confidence", 0.0)) if isinstance(item, dict) else 0.0
            )
            parts = re.split(r"\s*<=\s*", rule_str, maxsplit=1)
            head_str, body_str = parts[0], parts[1] if len(parts) > 1 else ""
            head_atom = parse_vars(head_str)
            if not head_atom:
                continue
            body_atoms = [
                parse_vars(atom) for atom in re.findall(r"[\w\d_]+\([^)]*\)", body_str)
            ]
            # FIX: Add unique ID to each parsed rule for proper tracking
            parsed_rule = {
                "id": f"ensemble_rule_{len(parsed_rules)}",
                "predicate": head_atom.get("predicate", ""),
                "head": head_atom,
                "body": [a for a in body_atoms if a],
                "confidence": confidence,
                "prolog": rule_str,
            }
            parsed_rules.append(parsed_rule)

        return parsed_rules

    def _build_rule_index(self) -> None:
        self.rule_index_ = defaultdict(set)

        for rule_idx, rule in enumerate(self.rules_):
            predicates = set()
            for atom in rule.get("body", []):
                pred = atom.get("predicate")
                if pred:
                    predicates.add(pred)

            head = rule.get("head")
            if head and "predicate" in head:
                predicates.add(head["predicate"])

            for pred in predicates:
                self.rule_index_[pred].add(rule_idx)

        self.rule_index_ = {
            pred: list(indices) for pred, indices in self.rule_index_.items()
        }

        n_predicates = len(self.rule_index_)
        avg_rules_per_pred = (
            sum(len(indices) for indices in self.rule_index_.values()) / n_predicates
            if n_predicates > 0
            else 0
        )

        logger.info(
            f" Rule index built: {n_predicates} unique predicates, "
            f"avg {avg_rules_per_pred:.1f} rules per predicate"
        )

    @staticmethod
    def _format_predicate_summary(counter: Counter[str], limit: int = 5) -> str:
        if not counter:
            return "nenhum predicado"
        return ", ".join(f"{predicate}:{count}" for predicate, count in counter.most_common(limit))

    def _initialize_numba_accelerator(self) -> None:
        if not self.enable_numba or len(self.rules_) == 0:
            self.numba_accelerator_ = None
            return

        try:
            rules_bytes = msgspec.json.encode(self.rules_)
            rules_hash = hash_bytes(rules_bytes)
            if (
                hasattr(self, "_cached_numba")
                and self._cached_rules_hash == rules_hash
                and self._cached_numba is not None
            ):
                logger.info(
                    f"Usando acelerador Numba em cache para {len(self.rules_)} regras"
                )
                self.numba_accelerator_ = self._cached_numba
                return

            logger.info(
                f"Iniciando acelerador Numba para validar {len(self.rules_)} regras"
            )
            self.numba_accelerator_ = SymbolicRuleAccelerator(
                self.rules_,
                enable_numba=True,
            )
            self._cached_numba = self.numba_accelerator_
            self._cached_rules_hash = rules_hash
            logger.success(
                f"Acelerador Numba pronto com {len(self.rules_)} regras em cache"
            )
        except Exception as exc:
            logger.warning(
                f"Numba accelerator initialization failed: {exc}; fallback to standard mode"
            )
            self.numba_accelerator_ = None

    def _apply_global_rule_limit(
        self,
        selected_rules: list[dict[str, Any]],
        allocation: Counter[str],
    ) -> tuple[list[dict[str, Any]], Counter[str]]:
        if not self.max_global_rules or len(selected_rules) <= self.max_global_rules:
            return selected_rules, allocation

        predicate_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rule in selected_rules:
            predicate = rule.get("predicate") or ""
            if not predicate:
                head = rule.get("head") or {}
                predicate = head.get("predicate") or ""
            if not predicate:
                predicate = "__unknown__"
            predicate_buckets[predicate].append(rule)

        total_selected = len(selected_rules)
        global_limit = self.max_global_rules
        predicate_counts = {pred: len(bucket) for pred, bucket in predicate_buckets.items()}
        if total_selected == 0:
            return selected_rules, allocation

        assigned_counts: dict[str, int] = {pred: 0 for pred in predicate_counts}
        total_assigned = 0
        remainders: list[tuple[float, str]] = []
        for predicate, count in predicate_counts.items():
            share = (count / total_selected) * global_limit
            base = min(count, int(math.floor(share)))
            assigned_counts[predicate] = base
            total_assigned += base
            remainder = share - base
            if count > base:
                remainders.append((remainder, predicate))

        remainders.sort(reverse=True)
        for remainder, predicate in remainders:
            if total_assigned >= global_limit:
                break
            if assigned_counts[predicate] >= predicate_counts[predicate]:
                continue
            assigned_counts[predicate] += 1
            total_assigned += 1

        if total_assigned < global_limit:
            for predicate, count in sorted(predicate_counts.items(), key=lambda item: item[1], reverse=True):
                while total_assigned < global_limit and assigned_counts[predicate] < count:
                    assigned_counts[predicate] += 1
                    total_assigned += 1
                    if total_assigned >= global_limit:
                        break

        trimmed_rules: list[dict[str, Any]] = []
        predicate_progress: Counter[str] = Counter()
        new_allocation: Counter[str] = Counter()
        for rule in selected_rules:
            predicate = rule.get("predicate") or ""
            if not predicate:
                head = rule.get("head") or {}
                predicate = head.get("predicate") or ""
            if not predicate:
                predicate = "__unknown__"
            if predicate_progress[predicate] >= assigned_counts.get(predicate, 0):
                continue
            trimmed_rules.append(rule)
            predicate_progress[predicate] += 1
            new_allocation[predicate] += 1

        removed = len(selected_rules) - len(trimmed_rules)
        if removed > 0:
            logger.info(
                f"Limite global aplicado: {len(trimmed_rules)} regras mantidas "
                f"(remoção de {removed} acima do teto {self.max_global_rules})"
            )

        return trimmed_rules, new_allocation

    def _prune_rules_by_activation(self, X: Any, y: Any) -> None:
        if y is None or len(self.rules_) == 0:
            return

        try:
            samples = list(X) if not isinstance(X, list) else X
        except TypeError:
            return

        if isinstance(samples, np.ndarray):
            samples = samples.tolist()

        if not samples:
            return

        y_array = np.asarray(y).reshape(-1)
        if y_array.size != len(samples):
            logger.debug(
                f"Ignorando ajuste simbólico: {len(samples)} amostras x {y_array.size} rótulos"
            )
            return

        sample_size = min(len(samples), self.activation_sample_size)
        if sample_size < 100:
            return

        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(len(samples), size=sample_size, replace=False))
        diag_samples = [samples[idx] for idx in indices]
        diag_labels = y_array[indices]

        original_grouping_state = self.enable_grouping
        try:
            if original_grouping_state:
                self.enable_grouping = False
            diag_features = np.asarray(self.transform(diag_samples))
        finally:
            self.enable_grouping = original_grouping_state
        if diag_features.ndim != 2 or diag_features.size == 0:
            return

        activated = diag_features > 0
        coverage = activated.mean(axis=0)
        activations = activated.sum(axis=0)
        positive_mask = diag_labels.astype(int) == 1
        if positive_mask.shape[0] != activated.shape[0]:
            logger.debug("Skipping symbolic adjustment: labels incompatible with features")
            return

        if positive_mask.any():
            positive_counts = activated[positive_mask].sum(axis=0)
        else:
            positive_counts = np.zeros_like(coverage, dtype=float)

        global_coverage = activated.any(axis=1).mean()
        if global_coverage < self.min_coverage_threshold:
            raise SymbolicCoverageError(
                f"Global symbolic coverage {global_coverage:.2%} is below threshold {self.min_coverage_threshold:.2%}"
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.divide(
                positive_counts,
                activations,
                out=np.zeros_like(coverage, dtype=float),
                where=activations > 0,
            )

        low_density_mask = coverage < self.min_activation_ratio
        dominance_mask = (coverage >= self.activation_coverage_floor) & (
            precision < self.activation_precision_floor
        )
        removal_mask = low_density_mask | dominance_mask
        removed = int(np.count_nonzero(removal_mask))

        logger.debug(f"Rules before pruning: {len(self.rules_)}")

        if removed == len(self.rules_):
            raise SymbolicCoverageError(
                f"Pruning removed ALL {len(self.rules_)} rules! "
                f"(density < {self.min_activation_ratio:.2%} or precision < {self.activation_precision_floor:.0f}%)"
            )
            
        elif removed > 0:
            logger.warning(
                f"Removed {removed} rules (density < {self.min_activation_ratio:.2%} or precision < {self.activation_precision_floor:.0f}%)"
            )
            self.rules_ = [
                rule for idx, rule in enumerate(self.rules_) if not removal_mask[idx]
            ]
        else:
            logger.info("Nenhuma regra removida por baixa densidade ou precisão real")
            return

        logger.info(f"Regras restantes após poda: {len(self.rules_)}")

        if self.enable_rule_indexing and self.rules_:
            self._build_rule_index()
        if self.enable_numba and self.rules_:
            self._initialize_numba_accelerator()
        self._last_feature_stats["activation_pruned_rules"] = removed

    def _balance_rules_by_predicate(self, rules_by_predicate: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        if not rules_by_predicate:
            return []

        total_capacity = sum(min(len(bucket), self.max_rules_per_predicate) for bucket in rules_by_predicate.values())
        if total_capacity <= 0:
            return []

        before_distribution = Counter({predicate: len(bucket) for predicate, bucket in rules_by_predicate.items()})
        selected_rules: list[dict[str, Any]] = []
        allocation: Counter[str] = Counter()
        remaining_buckets: dict[str, list[dict[str, Any]]] = {}

        predicate_cap = min(
            self.max_rules_per_predicate,
            max(self.min_rules_per_predicate, int(total_capacity * self.max_predicate_fraction)),
        )

        for predicate, bucket in rules_by_predicate.items():
            bucket.sort(key=lambda r: r.get("confidence", 0), reverse=True)
            take = min(len(bucket), self.min_rules_per_predicate, self.max_rules_per_predicate)
            if take > 0:
                selected_rules.extend(bucket[:take])
                allocation[predicate] += take
            remaining = bucket[take:]
            if remaining:
                remaining_buckets[predicate] = remaining

        remaining_capacity = max(total_capacity - len(selected_rules), 0)

        while remaining_capacity > 0 and remaining_buckets:
            progress = False
            for predicate in list(remaining_buckets.keys()):
                bucket = remaining_buckets.get(predicate)
                if not bucket:
                    remaining_buckets.pop(predicate, None)
                    continue
                current_cap = min(self.max_rules_per_predicate, predicate_cap)
                if allocation[predicate] >= current_cap:
                    remaining_buckets.pop(predicate, None)
                    continue
                selected_rules.append(bucket.pop(0))
                allocation[predicate] += 1
                remaining_capacity -= 1
                progress = True
                if not bucket:
                    remaining_buckets.pop(predicate, None)
                if remaining_capacity <= 0:
                    break
            if not progress:
                break

        after_distribution = Counter({predicate: count for predicate, count in allocation.items() if count > 0})
        if after_distribution:
            before_log = self._format_predicate_summary(before_distribution)
            after_log = self._format_predicate_summary(after_distribution)
            logger.info(
                f" Balanceamento simbólico por predicado → antes [{before_log}] | depois [{after_log}]"
            )

        selected_rules, final_allocation = self._apply_global_rule_limit(selected_rules, allocation)
        if final_allocation and final_allocation != allocation:
            trimmed_summary = self._format_predicate_summary(Counter(final_allocation))
            logger.debug(f"Final distribution after global limit -> {trimmed_summary}")

        return selected_rules

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return the names of the output features for sklearn compatibility."""
        check_is_fitted(self, "rules_")

        if self.enable_grouping and self.group_indices_ is not None:
            # Return grouped feature names
            feature_names = []
            for i, group_indices in enumerate(self.group_indices_):
                feature_names.extend([
                    f"symbolic_group_{i}_proportion",
                    f"symbolic_group_{i}_any_active",
                    f"symbolic_group_{i}_count_normalized",
                ])
            # Add global features
            feature_names.extend([
                "global_proportion",
                "global_count_normalized",
            ])
        else:
            # Return individual rule feature names
            if self.rules_:
                feature_names = [
                    f"rule_{rule.get('id', i)}"
                    for i, rule in enumerate(self.rules_)
                ]
            else:
                feature_names = []

        return feature_names

    def analyze_feature_distribution(self, X: list[list[tuple]]) -> dict[str, Any]:
        """Analyze feature distribution and identify potential issues."""
        check_is_fitted(self, "rules_")

        analysis = {
            "total_rules": len(self.rules_),
            "sample_count": len(X),
            "features_per_sample": 0,
            "feature_density": 0.0,
            "high_importance_features": [],
            "feature_324_found": False,
        }

        if self.enable_grouping and self.group_indices_ is not None:
            # For grouped features, we don't have individual rule analysis
            analysis["features_per_sample"] = len(self.group_indices_) * 3 + 2  # 3 per group + 2 global
            analysis["feature_324_found"] = any("324" in str(g) for g in self.group_indices_)
        else:
            # For individual rules
            analysis["features_per_sample"] = len(self.rules_)

        # Get feature names for analysis
        feature_names = self.get_feature_names_out()
        analysis["feature_324_found"] = any("324" in str(f) for f in feature_names)

        # Log findings
        if analysis["feature_324_found"]:
            logger.info(" Feature 324 detected in feature set")
        else:
            logger.warning(" Feature 324 NOT detected - potential indexing issue")

        logger.info(f" Feature Analysis: {analysis['total_rules']} rules → {analysis['features_per_sample']} features")

        return analysis
