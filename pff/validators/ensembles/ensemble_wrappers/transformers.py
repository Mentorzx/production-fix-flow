"""
Transformers - sklearn-compatible transformers

This module contains:
- ProbaTransformer (extracts probabilities as features)
- SymbolicFeatureExtractor (rule-based feature extraction)

Part of Sprint 4 refactoring (ensemble_wrappers.py split into 3 files).
These transformers are used in sklearn pipelines and feature unions.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pff.utils import ConcurrencyManager, FileManager, logger

# ══════════════════════════════════════════════════════════════════════════
# CONTEXT VARIABLES (Sprint 16 - Bug Fix)
# ══════════════════════════════════════════════════════════════════════════
# These context variables allow Business Service to pass violations to
# SymbolicFeatureExtractor without breaking sklearn Pipeline API.
#
# Problem: SymbolicFeatureExtractor.transform() doesn't have access to rules
# Solution: Business Service sets violations in context before calling Ensemble
#
# Thread-safe: ContextVar is thread-local and async-safe
# ══════════════════════════════════════════════════════════════════════════

_ensemble_violations_context: ContextVar[list] = ContextVar(
    '_ensemble_violations', default=[]
)
_ensemble_all_rules_context: ContextVar[list] = ContextVar(
    '_ensemble_all_rules', default=[]
)


# ══════════════════════════════════════════════════════════════════════════
# PROBA TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════


class ProbaTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that wraps a classifier and extracts the probability of the
    positive class as a feature.

    This corrected version properly implements the scikit-learn fitted state
    protocol to avoid warnings about the Pipeline not being trained.
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


# ══════════════════════════════════════════════════════════════════════════
# SYMBOLIC FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════


class SymbolicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that converts samples of triples into binary
    feature vectors based on symbolic rule violations.

    Each feature in the output vector corresponds to a rule from the ruleset.
    The feature is 1 if the rule is violated by the sample, and 0 otherwise.
    A rule H :- B is considered violated if its body B is true given the
    sample's triples, but its head H is not.
    """

    def __init__(
        self,
        rules_path: str,
        min_confidence_threshold: float = 0.0,
        enable_grouping: bool = False,
        n_groups: int = 50,
        boost_factor: float = 10.0,
    ):
        self.rules_path = rules_path
        self.min_confidence_threshold = min_confidence_threshold
        self.rules_ = []
        self.concurrency_manager = ConcurrencyManager()
        self.enable_grouping = enable_grouping
        self.n_groups = n_groups
        self.boost_factor = boost_factor
        self.group_indices_ = None

    def fit(self, X, y=None) -> "SymbolicFeatureExtractor":
        """
        Loads, filters, and parses symbolic rules from the given path.
        """
        try:
            path = Path(self.rules_path)
            suffix = path.suffix.lower()
            file_manager = FileManager()
            if suffix in {".parquet", ".pq", ".parq"}:
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
                    df = pl.read_csv(path, separator=separator, has_header=False)
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
                        raw_rules = []
                except Exception:
                    content = file_manager.read(path)
                    if isinstance(content, str):
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
            if self.min_confidence_threshold > 0.0:
                filtered_rules = [
                    rule
                    for rule in raw_rules
                    if rule.get("confidence", 0.0) >= self.min_confidence_threshold
                ]
            else:
                filtered_rules = raw_rules
            self.rules_ = self._parse_rules(filtered_rules)
            logger.info(
                f"{len(self.rules_)} regras analisadas com confiança >= {self.min_confidence_threshold}"
            )
        except Exception as e:
            logger.error(f"Falha ao carregar ou filtrar regras: {e}")
            self.rules_ = []

        return self

    def transform(self, X: list[list[tuple]]) -> np.ndarray:
        """
        Transforms input samples into binary feature vectors using parallel processing.

        Sprint 16 Fix: Now uses pre-calculated violations from context (if available)
        instead of trying to re-validate rules (which this class doesn't have access to).
        """
        check_is_fitted(self, "rules_")

        # Sprint 16 Fix: Try to get violations from context first
        try:
            violations = _ensemble_violations_context.get()
            all_rules = _ensemble_all_rules_context.get()

            if violations and all_rules:
                # Use pre-calculated violations from Business Service
                logger.info(
                    f"🔍 [Sprint 16] Using {len(violations)} pre-calculated violations "
                    f"from {len(all_rules)} rules"
                )
                binary_features = self._violations_to_binary_features(
                    violations, all_rules, len(X)
                )

                logger.info(
                    f"🔍 [Sprint 16 Debug] binary_features shape: {binary_features.shape}, "
                    f"violations in matrix: {np.sum(binary_features)}"
                )

                # Apply grouping if enabled
                if self.enable_grouping and binary_features.shape[1] > 0:
                    try:
                        features = self._apply_feature_grouping(binary_features)
                        logger.info(f"✅ Features: {binary_features.shape[1]} → {features.shape[1]} agrupadas")
                    except Exception as e:
                        logger.warning(f"⚠️ Error in grouping (using ungrouped): {e}")
                        features = binary_features
                else:
                    features = binary_features

                # Log active rules
                if features.shape[0] > 0:
                    active_rules = np.sum(features > 0, axis=1)
                    logger.info(f"🔍 Symbolic Analysis: {active_rules[0]} regras ativas")

                return features

        except Exception as e:
            logger.warning(f"⚠️ Could not get violations from context: {e}")

        # Fallback to old behavior (will likely return empty features)
        if not self.rules_:
            logger.warning(
                "Nenhuma regra carregada no SymbolicFeatureExtractor. Retornando features vazias."
            )
            return np.empty((len(X), 0))

        logger.info(f"Iniciando processamento paralelo de {len(X)} amostras...")
        sample_data = [(sample, self.rules_) for sample in X]
        results = self.concurrency_manager.execute_sync(
            SymbolicFeatureExtractor._transform_single_sample,
            sample_data,
            desc="Processando Regras Simbólicas",
            task_type="process",
        )
        binary_features = np.array(results, dtype=np.int8)
        if self.enable_grouping and binary_features.shape[1] > 0:
            features = self._apply_feature_grouping(binary_features)
        else:
            features = binary_features
        if features.shape[0] > 0:
            active_rules = np.sum(features > 0, axis=1)
            logger.info(f"🔍 Symbolic Analysis: {active_rules[0]} regras ativas")

        return features

    def _violations_to_binary_features(
        self, violations: list, all_rules: list, n_samples: int
    ) -> np.ndarray:
        """
        Convert pre-calculated violations to binary feature matrix.

        Sprint 16 Fix: This method allows SymbolicFeatureExtractor to use
        violations calculated by Business Service instead of trying to
        re-validate rules (which it doesn't have access to).

        Args:
            violations: List of RuleViolation objects from Business Service
            all_rules: List of all Rule objects (for dimensionality)
            n_samples: Number of samples (usually 1 for single prediction)

        Returns:
            np.ndarray: Binary matrix where 1 = rule violated, 0 = rule satisfied
                        Shape: (n_samples, n_rules)
        """
        n_rules = len(all_rules)

        # Create binary feature matrix (all zeros initially)
        binary_features = np.zeros((n_samples, n_rules), dtype=np.int8)

        # Build map of rule IDs to indices
        # RuleViolation.rule_id corresponds to Rule.id
        rule_id_to_idx = {}
        for idx, rule in enumerate(all_rules):
            # Extract rule ID from rule
            rule_id = None
            if hasattr(rule, 'id'):
                rule_id = rule.id
            elif isinstance(rule, dict) and 'id' in rule:
                rule_id = rule['id']

            if rule_id:
                rule_id_to_idx[rule_id] = idx

        logger.debug(f"🔍 [Sprint 16] Built rule_id_to_idx with {len(rule_id_to_idx)} rule IDs")

        # Get violated rule IDs
        violated_rule_ids = set()
        for v in violations:
            # Extract rule_id from violation
            rule_id = None
            if hasattr(v, 'rule_id'):
                rule_id = v.rule_id
            elif isinstance(v, dict) and 'rule_id' in v:
                rule_id = v['rule_id']

            if rule_id:
                violated_rule_ids.add(rule_id)

        logger.debug(f"🔍 [Sprint 16] Found {len(violated_rule_ids)} violated rule IDs")

        # Mark violated rules as 1
        matches = 0
        for rule_id in violated_rule_ids:
            if rule_id in rule_id_to_idx:
                rule_idx = rule_id_to_idx[rule_id]
                if rule_idx < n_rules:  # Safety check
                    binary_features[:, rule_idx] = 1
                    matches += 1

        logger.info(f"🔍 [Sprint 16] Matched {matches}/{len(violated_rule_ids)} violations to {n_rules} rules")
        logger.debug(f"🔍 [Sprint 16] binary_features shape: {binary_features.shape}, sum: {np.sum(binary_features)}")

        return binary_features

    def _create_feature_groups(self, n_features: int) -> list[list[int]]:
        """
        Divides the feature indices into groups for ensemble processing.

        Args:
            n_features (int): The total number of features to be grouped.

        Returns:
            list[list[int]]: A list of groups, where each group is a list of feature indices.

        Notes:
            - If the number of features is less than or equal to the number of groups (`self.n_groups`), each feature is placed in its own group.
            - Otherwise, features are divided as evenly as possible among the groups.
        """
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
        """
        Applies feature grouping and transformation to a binary feature matrix.
        This method groups binary features according to predefined or dynamically created groups,
        and computes aggregated statistics for each group, such as the proportion of active features,
        whether any feature in the group is active, and the normalized count of active features.
        These aggregated features are then boosted by a configurable factor. Additionally, global
        statistics across all features are appended.
        If all resulting features are zero and the boost factor is positive, the method injects
        small random noise and "phantom activations" to ensure that downstream models (e.g., XGBoost)
        do not ignore these features entirely.
        Args:
            binary_features (np.ndarray): A 2D numpy array of shape (n_samples, n_features)
                containing binary (0/1) feature values.
        Returns:
            np.ndarray: A 2D numpy array of shape (n_samples, n_transformed_features) containing
                the grouped and transformed features, possibly with added noise if all features
                were originally zero.
        """
        n_samples, n_features = binary_features.shape

        # Sprint 16 Fix: Reset group_indices_ if feature count changed
        if self.group_indices_ is not None:
            # Check if any group contains invalid indices
            max_idx = max(max(group) for group in self.group_indices_ if group)
            if max_idx >= n_features:
                logger.warning(
                    f"⚠️ [Sprint 16] Feature count changed ({max_idx+1} → {n_features}), "
                    f"resetting group_indices_"
                )
                self.group_indices_ = None

        if self.group_indices_ is None:
            self.group_indices_ = self._create_feature_groups(n_features)
            logger.info(
                f"📊 Agrupando {n_features} features em {len(self.group_indices_)} grupos"
            )
        grouped_features = []
        for group_indices in self.group_indices_:
            group_data = binary_features[:, group_indices]
            proportion = np.mean(group_data, axis=1, keepdims=True)
            any_active = np.any(group_data, axis=1, keepdims=True).astype(float)
            count_normalized = np.sum(group_data, axis=1, keepdims=True) / len(
                group_indices
            )
            grouped_features.extend(
                [
                    proportion * self.boost_factor,
                    any_active * self.boost_factor,
                    count_normalized * self.boost_factor,
                ]
            )
        global_features = [
            np.mean(binary_features, axis=1, keepdims=True) * self.boost_factor,
            np.sum(binary_features, axis=1, keepdims=True)
            / n_features
            * self.boost_factor,
        ]
        grouped_features.extend(global_features)
        result = np.hstack(grouped_features)

        # Sprint 16 Debug: Show feature statistics
        n_violations = int(np.sum(binary_features))
        proportion_violated = n_violations / n_features if n_features > 0 else 0
        logger.info(f"✅ Features: {n_features} → {result.shape[1]} agrupadas")
        logger.info(
            f"🔍 [Sprint 16] Feature stats: {n_violations}/{n_features} violations "
            f"({proportion_violated:.4f} = {proportion_violated*100:.2f}%)"
        )
        logger.info(
            f"🔍 [Sprint 16] Grouped feature range: min={np.min(result):.6f}, "
            f"max={np.max(result):.6f}, mean={np.mean(result):.6f}"
        )

        return result

    @staticmethod
    def _transform_single_sample(
        sample_triples_list: list[tuple], rules: list[dict]
    ) -> np.ndarray:
        """
        Process a single sample against all rules. Designed for parallel execution.
        """
        available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
        sample_feature_vector = np.zeros(len(rules), dtype=np.int8)
        violations = 0

        for i, rule in enumerate(rules):
            if SymbolicFeatureExtractor._rule_is_violated(rule, available_triples_set):
                sample_feature_vector[i] = 1
                violations += 1
        if violations > 0:
            logger.debug(f"✅ {violations} regras REALMENTE violadas detectadas")
        else:
            logger.debug("✅ Nenhuma violação real detectada (0 regras ativas)")

        return sample_feature_vector

    @staticmethod
    def _rule_is_violated(rule: dict, available_triples: set) -> bool:
        """
        Determines whether a given logical rule is violated based on available triples.
        A rule is considered violated if:
        - All atoms in the rule's body are present in `available_triples`.
        - The atom in the rule's head is NOT present in `available_triples`.
        Args:
            rule (dict): A dictionary representing the rule, with keys "body" (list of atoms)
                and "head" (single atom). Each atom is a dict with "subject", "predicate", and "object".
            available_triples (set): A set of tuples representing available triples,
                where each tuple is (subject, predicate, object).
        Returns:
            bool: True if the rule is violated, False otherwise.
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
        """
        Parses a list of raw rule representations into a structured format.
        Each rule in `raw_rules` can be either a dictionary with a "prolog" key (and optional "confidence"),
        or a string containing a rule in Prolog-like syntax (with "<=" separating head and body).
        The function extracts the head and body atoms, their predicates, subjects, and objects, as well as the confidence score.
        Args:
            raw_rules (list[dict]): A list of rule representations, where each item is either a dict with a "prolog" key
                (and optional "confidence" key), or a string containing a rule.
        Returns:
            list[dict]: A list of parsed rules, where each rule is a dictionary with the following keys:
                - "head": dict with keys "predicate", "subject", "object" representing the head atom.
                - "body": list of dicts, each with keys "predicate", "subject", "object" for body atoms.
                - "confidence": float, the confidence score associated with the rule.
                - "prolog": str, the original rule string.
        """
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
            parts = re.split(r"\s*<=\s*", rule_str, 1)
            head_str, body_str = parts[0], parts[1] if len(parts) > 1 else ""
            head_atom = parse_vars(head_str)
            if not head_atom:
                continue
            body_atoms = [
                parse_vars(atom) for atom in re.findall(r"[\w\d_]+\([^)]*\)", body_str)
            ]
            parsed_rule = {
                "head": head_atom,
                "body": [a for a in body_atoms if a],
                "confidence": confidence,
                "prolog": rule_str,
            }
            parsed_rules.append(parsed_rule)

        return parsed_rules
