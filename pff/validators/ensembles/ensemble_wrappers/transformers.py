"""
Transformers - sklearn-compatible transformers

This module contains:
- ProbaTransformer (extracts probabilities as features)
- SymbolicFeatureExtractor (rule-based feature extraction)
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from datetime import datetime

import msgspec
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pff.utils import ConcurrencyManager, FileManager, logger, SymbolicRuleAccelerator
from pff.services.business_service import RuleValidator, Rule, RuleViolation

_ensemble_violations_context: ContextVar[list] = ContextVar(
    '_ensemble_violations', default=[]
)
_ensemble_all_rules_context: ContextVar[list] = ContextVar(
    '_ensemble_all_rules', default=[]
)


# Global helper functions for multiprocessing (must be picklable)
def _static_transform_single_sample(sample_triples_list, rules, rule_validator, use_business_service) -> np.ndarray:
    """
    Static wrapper for _transform_single_sample to enable multiprocessing.
    
    Args:
        sample_triples_list: Sample triples
        rules: List of rules
        rule_validator: RuleValidator instance
        use_business_service: Whether to use business service
    
    Returns:
        Binary feature vector for the sample
    """
    available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
    sample_feature_vector = np.zeros(len(rules), dtype=np.int8)
    violations = 0

    for i, rule in enumerate(rules):
        debug_first = (i == 0)
        if _static_rule_is_violated(rule, available_triples_set, rule_validator, use_business_service, debug_first):
            sample_feature_vector[i] = 1
            violations += 1

    return sample_feature_vector


def _static_transform_single_sample_indexed(sample_triples_list, rules, rule_index, rule_validator, use_business_service) -> np.ndarray:
    """
    Static wrapper for _transform_single_sample_indexed to enable multiprocessing.
    
    Args:
        sample_triples_list: Sample triples
        rules: List of rules
        rule_index: Predicate index
        rule_validator: RuleValidator instance
        use_business_service: Whether to use business service
    
    Returns:
        Binary feature vector for the sample
    """
    available_triples_set = {tuple(map(str, t)) for t in sample_triples_list}
    sample_feature_vector = np.zeros(len(rules), dtype=np.int8)

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
        logger.debug(f"🔍 FIRST RULE VALIDATION:")
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
                logger.debug(f"   Calling rule_validator.validate_rules()...")
            
            violations = rule_validator.validate_rules([business_rule], triples_list)
            
            if debug_first_call:
                logger.debug(f"   violations returned: {len(violations)}")
                if len(violations) > 0:
                    logger.debug(f"   first violation: {violations[0]}")
            
            return len(violations) > 0
        except Exception as e:
            if debug_first_call:
                logger.warning(f"⚠️ Error using business service: {e}, falling back")
                import traceback
                logger.debug(f"   Traceback: {traceback.format_exc()}")
            return _static_rule_is_violated_fallback(rule, available_triples, debug_first_call)
    else:
        if debug_first_call:
            logger.debug(f"   Using fallback (business_service disabled or no validator)")
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


class SymbolicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that converts samples of triples into binary
    feature vectors based on symbolic rule violations.
    """

    def __init__(
        self,
        rules_path: str,
        min_confidence_threshold: float = 0.10,  # INCREASED from 0.05 to reduce overfitting and sparsity
        enable_grouping: bool = False,
        n_groups: int = 50,
        boost_factor: float = 1.0,  # Reduzido de 10.0 para evitar dominância simbólica
        enable_rule_indexing: bool = True,
        enable_numba: bool = True,
        max_violation_percentage: float = 200.0,  # Novo parâmetro para validação
        use_business_service: bool = True,  # NOVO: Usar business service com unificação
    ):
        self.rules_path = rules_path
        self.min_confidence_threshold = min_confidence_threshold
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


    def _save_numba_debug(self, X: list, exc: Exception, filename_prefix: str = "numba_accel_debug"):
        """
        Save a lightweight debug dump (JSON) containing small samples and exception text.
        Avoids writing large binary arrays to disk.
        """
        try:
            dump = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "n_samples": len(X),
                "sample_preview": [repr(s) for s in X[:5]],
                "exception": repr(exc),
                "n_rules": len(self.rules_),
                "rule_index_exists": self.rule_index_ is not None,
            }
            path = Path(f"{filename_prefix}.json")
            path.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
            logger.info(f"Numba debug dump saved to {path}")
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
                logger.info(f"✅ Numba acceleration successful: processed {len(normalized_X)} samples")
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
                    logger.debug(f"Using indexed fallback for sample idx {idx}")
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
            logger.info(f"📋 Regras carregadas do arquivo: {len(raw_rules)}")

            if self.min_confidence_threshold > 0.0:
                filtered_rules = [
                    rule
                    for rule in raw_rules
                    if rule.get("confidence", 0.0) >= self.min_confidence_threshold
                ]
                logger.info(
                    f"🔍 Regras após filtro de confiança (>= {self.min_confidence_threshold}): "
                    f"{len(filtered_rules)}/{len(raw_rules)} ({len(filtered_rules)/len(raw_rules)*100:.1f}%)"
                )
            else:
                filtered_rules = raw_rules
                logger.info("🔍 Sem filtro de confiança aplicado")

            self.rules_ = self._parse_rules(filtered_rules)

            # OTIMIZAÇÃO: Limitar número de regras por predicado para melhor performance
            if self.enable_rule_indexing:
                rules_by_predicate = {}
                for rule in self.rules_:
                    pred = rule.get("predicate")
                    if pred not in rules_by_predicate:
                        rules_by_predicate[pred] = []
                    rules_by_predicate[pred].append(rule)

                # Limitar a 100 regras por predicado (top confidence)
                # OPTIMIZED (Sprint 23): Reduced from 1000 to 100 for better performance
                # With 32 predicates: 32 × 100 = ~3,200 rules (was 32 × 1000 = ~32,000)
                max_rules_per_predicate = 100
                filtered_by_predicate = []
                total_removed = 0

                for pred, pred_rules in rules_by_predicate.items():
                    # Ordenar por confiança e pegar as N melhores
                    pred_rules.sort(key=lambda r: r.get("confidence", 0), reverse=True)
                    top_rules = pred_rules[:max_rules_per_predicate]
                    filtered_by_predicate.extend(top_rules)

                    removed_count = len(pred_rules) - len(top_rules)
                    total_removed += removed_count

                self.rules_ = filtered_by_predicate
                logger.info(
                    f"🔧 LIMITADO: {len(self.rules_)} regras após limite de {max_rules_per_predicate}/predicado "
                    f"(removidas {total_removed} regras redundantes)"
                )

            logger.info(
                f"{len(self.rules_)} regras analisadas com confiança >= {self.min_confidence_threshold}"
            )

            # Build rule index for faster filtering
            if self.enable_rule_indexing and len(self.rules_) > 0:
                self._build_rule_index()

            # Build Numba accelerator for maximum performance
            if self.enable_numba and len(self.rules_) > 0:
                try:
                    # OTIMIZAÇÃO: Verificar se já existe cache para este conjunto de regras
                    # Use msgspec (faster than json.dumps, follows project utils guidelines)
                    rules_bytes = msgspec.json.encode(self.rules_)
                    rules_hash = hash(rules_bytes)

                    if hasattr(self, '_cached_numba') and self._cached_rules_hash == rules_hash:
                        logger.info("⚡ Using cached Numba accelerator (reusing compiled kernels)")
                        self.numba_accelerator_ = self._cached_numba
                    else:
                        logger.info("⚡ Initializing Numba accelerator for rule validation...")
                        self.numba_accelerator_ = SymbolicRuleAccelerator(
                            self.rules_,
                            enable_numba=True
                        )
                        # Cache para reutilização
                        self._cached_numba = self.numba_accelerator_
                        self._cached_rules_hash = rules_hash
                        logger.success(f"✅ Numba accelerator ready with {len(self.rules_)} rules (cached)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Numba accelerator: {e}, using standard mode")
                    self.numba_accelerator_ = None

        except Exception as e:
            logger.error(f"Falha ao carregar ou filtrar regras: {e}")
            self.rules_ = []

        return self

    def transform(self, X: list[list[tuple]]) -> np.ndarray:
        """
        Transforms input samples into binary feature vectors using parallel processing.

        Now uses pre-calculated violations from context (if available)
        instead of trying to re-validate rules (which this class doesn't have access to).
        """
        check_is_fitted(self, "rules_")

        # DEBUG: Log first call details
        if not hasattr(self, '_debug_first_transform_done'):
            logger.debug(f"🔍 FIRST TRANSFORM CALL")
            logger.debug(f"   X shape: {len(X)} samples")
            logger.debug(f"   Rules loaded: {len(self.rules_)} rules")
            logger.debug(f"   use_business_service: {self.use_business_service}")
            logger.debug(f"   enable_numba: {self.enable_numba}")
            logger.debug(f"   enable_rule_indexing: {self.enable_rule_indexing}")
            if len(X) > 0 and len(X[0]) > 0:
                logger.debug(f"   First sample format: {type(X[0][0])} - {X[0][0]}")
            self._debug_first_transform_done = True

        try:
            violations = _ensemble_violations_context.get()
            all_rules = _ensemble_all_rules_context.get()

            # DEBUG: Log context state
            logger.debug(f"🔍 Context state:")
            logger.debug(f"   violations: {type(violations)}, len={len(violations) if violations else 0}")
            logger.debug(f"   all_rules: {type(all_rules)}, len={len(all_rules) if all_rules else 0}")

            if (violations is not None and all_rules is not None and
                len(violations) > 0 and len(all_rules) > 0):
                # Use pre-calculated violations from Business Service
                logger.info(
                    f"🔍 Using {len(violations)} pre-calculated violations "
                    f"from {len(all_rules)} rules"
                )
                binary_features = self._violations_to_binary_features(
                    violations, all_rules, len(X)
                )

                logger.info(
                    f"🔍 binary_features shape: {binary_features.shape}, "
                    f"violations in matrix: {np.sum(binary_features)}"
                )

                # VALIDAÇÃO: Monitorar percentual de violações para detectar overfitting
                total_violations = np.sum(binary_features)
                total_possible = len(all_rules) * len(X)
                violation_percentage = (total_violations / total_possible) * 100

                logger.info(
                    f"📊 Violation Analysis: {total_violations}/{total_possible} "
                    f"({violation_percentage:.2f}%)"
                )

                # Alerta se percentual de violações for muito alta (indicativo de overfitting)
                if violation_percentage > self.max_violation_percentage:
                    logger.warning(
                        f"⚠️ HIGH VIOLATION RATE: {violation_percentage:.2f}% "
                        f"(threshold: {self.max_violation_percentage}%) - "
                        f"Consider increasing min_confidence_threshold"
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
            logger.info(f"🔄 Using fallback: calculating violations manually for {len(X)} samples")
            logger.debug(f"🔍 FALLBACK MODE:")
            logger.debug(f"   Exception: {e}")
            logger.debug(f"   Will use business_service: {self.use_business_service}")

        if not self.rules_:
            logger.error(
                f"❌ CRITICAL: Nenhuma regra carregada no SymbolicFeatureExtractor! "
                f"rules_path={self.rules_path}, "
                f"min_confidence_threshold={self.min_confidence_threshold}, "
                f"file_exists={Path(self.rules_path).exists() if self.rules_path else 'N/A'}. "
                f"Verifique se o arquivo de regras existe e se min_confidence_threshold não é muito alto."
            )
            return np.empty((len(X), 0))

        logger.info(f"✅ {len(self.rules_)} regras disponíveis para validação")
        logger.info(f"🚀 Iniciando processamento paralelo de {len(X)} amostras...")
        
        # DEBUG: Log which path will be taken
        if self.numba_accelerator_ is not None:
            logger.debug(f"🔍 Will use Numba accelerator")
        elif self.enable_rule_indexing and self.rule_index_ is not None:
            logger.debug(f"🔍 Will use rule indexing")
        else:
            logger.debug(f"🔍 Will use full scan")

        # Priority 1: Use Numba accelerator if available (fastest: 10-100× speedup)
        if self.numba_accelerator_ is not None:
            logger.info("⚡ Using Numba JIT acceleration (FASTEST)")
            try:
                # Normalize X before calling numba (defensive)
                normalized_X = [self._normalize_sample_for_numba(sample) for sample in X]
                # Try safe wrapper that runs fallbacks inside
                violations_list = self._call_numba_accelerator_safe(normalized_X)
                # Stack arrays (each is shape (n_rules,)) into 2D array (n_samples, n_rules)
                binary_features = np.vstack(violations_list).astype(np.int8)
            except Exception as e:
                # If _call_numba_accelerator_safe raises, fallback to indexed processing
                logger.warning(f"⚠️ Numba acceleration failed (all fallbacks): {e}, falling back to indexed")
                # Fallback to indexed processing
                sample_data = [
                    (sample, self.rules_, self.rule_index_, self.rule_validator, self.use_business_service) 
                    for sample in X
                ]
                results = self.concurrency_manager.execute_sync(
                    _static_transform_single_sample_indexed,
                    sample_data,
                    desc="Processando Regras Simbólicas (fallback)",
                    task_type="process",
                )
                binary_features = np.array(results, dtype=np.int8)

        # Priority 2: Use rule indexing if available (fast: 10-100× speedup)
        elif self.enable_rule_indexing and self.rule_index_ is not None:
            logger.info("🗂️ Using rule index for faster processing")
            sample_data = [
                (sample, self.rules_, self.rule_index_, self.rule_validator, self.use_business_service) 
                for sample in X
            ]
            results = self.concurrency_manager.execute_sync(
                _static_transform_single_sample_indexed,
                sample_data,
                desc="Processando Regras Simbólicas (indexado)",
                task_type="process",
            )
            binary_features = np.array(results, dtype=np.int8)

        # Priority 3: Full scan (slowest, baseline)
        else:
            logger.info("⚠️ Rule indexing disabled, using full scan")
            sample_data = [
                (sample, self.rules_, self.rule_validator, self.use_business_service) 
                for sample in X
            ]
            results = self.concurrency_manager.execute_sync(
                _static_transform_single_sample,
                sample_data,
                desc="Processando Regras Simbólicas",
                task_type="process",
            )
            binary_features = np.array(results, dtype=np.int8)

        logger.info(
            f"📊 Features binárias calculadas: shape={binary_features.shape}, "
            f"non-zero={np.count_nonzero(binary_features)}/{binary_features.size} "
            f"({np.count_nonzero(binary_features)/binary_features.size*100:.2f}%)"
        )
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

        logger.debug(f"🔍 Built rule_id_to_idx with {len(rule_id_to_idx)} rule IDs")

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

        logger.debug(f"🔍 Found {len(violated_rule_ids)} violated rule IDs")

        matches = 0
        for rule_id in violated_rule_ids:
            if rule_id in rule_id_to_idx:
                rule_idx = rule_id_to_idx[rule_id]
                if rule_idx < n_rules:
                    binary_features[:, rule_idx] = 1
                    matches += 1

        logger.info(f"🔍 Matched {matches}/{len(violated_rule_ids)} violations to {n_rules} rules")
        logger.debug(f"🔍 binary_features shape: {binary_features.shape}, sum: {np.sum(binary_features)}")

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
                    f"⚠️ Feature count changed ({max_idx+1} → {n_features}), "
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

        logger.info(f"✅ Features: {n_features} → {result.shape[1]} agrupadas")
        logger.info(
            f"🔍 Feature stats: {n_violations}/{n_samples*n_features} violations "
            f"({proportion_violated:.4f} = {proportion_violated*100:.2f}%)"
        )
        logger.info(
            f"📊 Violations per sample: {violations_per_sample:.2f} avg ({n_violations} total violations)"
        )
        logger.info(
            f"🔍 Grouped feature range: min={np.min(result):.6f}, "
            f"max={np.max(result):.6f}, mean={np.mean(result):.6f}"
        )

        return result

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
            logger.debug(f"✅ {violations} regras REALMENTE violadas detectadas")
        else:
            logger.debug("✅ Nenhuma violação real detectada (0 regras ativas)")

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
        #         f"✅ Checked {len(applicable_rule_indices)}/{len(rules)} rules "
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
                # Convert ensemble rule format to business service format
                business_rule = self._convert_ensemble_rule_to_business_format(rule)
                
                # Convert available triples to business service format
                triples_list = list(available_triples)
                
                # Use business service with proper unification
                violations = self.rule_validator.validate_rules([business_rule], triples_list)
                
                # Rule is violated if we found any violations
                return len(violations) > 0
                
            except Exception as e:
                logger.warning(f"Error using business service for rule validation: {e}")
                # Fallback to original method
                return self._rule_is_violated_fallback(rule, available_triples)
        else:
            # Use original fallback method
            return self._rule_is_violated_fallback(rule, available_triples)
    
    def _convert_ensemble_rule_to_business_format(self, rule: dict) -> 'Rule':
        """Convert ensemble rule format to business service Rule format."""
        try:
            # Extract body atoms
            body_atoms = []
            for atom in rule.get("body", []):
                body_atoms.append({
                    'subject': str(atom.get("subject", "")).strip(),
                    'predicate': str(atom.get("predicate", "")).strip(),
                    'object': str(atom.get("object", "")).strip()
                })
            
            # Extract head atom
            head_atom = rule.get("head", {})
            head_atom_formatted = {
                'subject': str(head_atom.get("subject", "")).strip(),
                'predicate': str(head_atom.get("predicate", "")).strip(),
                'object': str(head_atom.get("object", "")).strip()
            }
            
            # Create business service Rule
            # FIX: Rule expects tuples, not dicts
            business_rule = Rule(
                id=rule.get("id", f"rule_{hash(str(rule)) % 1000000}"),  # Use existing ID if available
                confidence=rule.get("confidence", 1.0),
                head=(
                    head_atom_formatted['subject'],
                    head_atom_formatted['predicate'],
                    head_atom_formatted['object']
                ),
                body=[
                    (atom['subject'], atom['predicate'], atom['object'])
                    for atom in body_atoms
                ],
                source="ensemble"
            )
            
            return business_rule
            
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
            parts = re.split(r"\s*<=\s*", rule_str, 1)
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
                "head": head_atom,
                "body": [a for a in body_atoms if a],
                "confidence": confidence,
                "prolog": rule_str,
            }
            parsed_rules.append(parsed_rule)

        return parsed_rules

    def _build_rule_index(self) -> None:
        from collections import defaultdict

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
            f"🗂️ Rule index built: {n_predicates} unique predicates, "
            f"avg {avg_rules_per_pred:.1f} rules per predicate"
        )

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
            logger.info("✅ Feature 324 detected in feature set")
        else:
            logger.warning("⚠️ Feature 324 NOT detected - potential indexing issue")

        logger.info(f"📊 Feature Analysis: {analysis['total_rules']} rules → {analysis['features_per_sample']} features")

        return analysis
