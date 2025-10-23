from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from pff import settings
from pff.utils import ConcurrencyManager, FileManager, logger, DiskCache
from pff.utils.research import _TripleIndexStrategy

from pff.utils.numba_kernels import (
    VocabularyEncoder,
    find_matching_triples_accelerated,
    NUMBA_AVAILABLE,
)


@dataclass
class Rule:
    """
    Represents a validation rule with metadata.

    Attributes:
        id: Unique identifier for the rule
        confidence: Confidence score for the rule (0-1)
        head: Head predicate of the rule
        body: List of body predicates
        source: Source of the rule (manual, anyburl)
        total_predictions: Total predictions for AnyBURL rules
        correct_predictions: Correct predictions for AnyBURL rules
        occurrences: Number of times this exact rule pattern appears (v10.8.0)
        aggregated_confidence: Sum of confidences from all occurrences (v10.8.0)
    """

    id: str
    confidence: float
    head: dict[str, Any]
    body: list[dict[str, Any]]
    source: str
    total_predictions: int = 0
    correct_predictions: int = 0
    occurrences: int = 1  # v10.8.0: Track rule frequency
    aggregated_confidence: float = 0.0  # v10.8.0: Sum of confidence from duplicates


@dataclass
class RuleViolation:
    """
    Represents a rule violation found during validation.

    Attributes:
        rule_id: ID of the violated rule
        confidence: Confidence of the violated rule
        description: Human-readable description of the violation
        bindings: Variable bindings when violation was detected
    """

    rule_id: str
    confidence: float
    description: str
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert violation to dictionary format."""
        return {
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "description": self.description,
            "bindings": self.bindings,
        }


class RuleEngine:
    """
    Dynamic rule engine for loading and managing validation rules.

    This engine loads rules from both manual JSON files and AnyBURL TSV files,
    parsing them into a unified internal representation.
    """

    def __init__(self):
        self.rule_index: dict[str, Rule] = {}
        self.manual_rules: list[Rule] = []
        self.anyburl_rules: list[Rule] = []
        self.file_manager = FileManager()

    def _parse_pattern(self, pattern_str: str) -> tuple[dict, list[dict]]:
        """
        Parse a Datalog-like pattern string into head and body structures.

        Args:
            pattern_str: Pattern string like "head(A,B) <= body1(A,C), body2(C,B)"

        Returns:
            Tuple of (head dict, body list)

        Raises:
            ValueError: If pattern format is invalid
        """
        if "<=" not in pattern_str:
            raise ValueError(f"Padrão de regra inválido, falta '<=': {pattern_str}")

        head_str, body_str = pattern_str.split("<=", 1)

        def parse_single_clause(clause_str: str) -> dict:
            """Parse a single clause like 'predicate(arg1,arg2)' into a dict."""
            clause_str = clause_str.strip()
            match = re.match(r"(\w+)\((.*?)\)", clause_str)
            if not match:
                raise ValueError(f"Cláusula malformada: {clause_str}")

            predicate = match.group(1)
            args_str = match.group(2)
            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

            return {"predicate": predicate, "args": args}

        head = parse_single_clause(head_str)

        body_clauses_parts = [
            c.strip() for c in body_str.strip().split("),") if c.strip()
        ]
        body = []
        for i, clause_part in enumerate(body_clauses_parts):
            if i < len(body_clauses_parts) - 1:
                clause_full = clause_part + ")"
            else:
                clause_full = clause_part
            body.append(parse_single_clause(clause_full))

        return head, body

    def load_manual_rules(self, filepath: Path | None = None) -> None:
        """
        Load manual rules from a JSON file with robust parsing and validation.

        Args:
            filepath: Path to manual rules JSON file
        """
        if filepath is None:
            filepath = settings.PATTERNS_DIR / "manual_rules.json"

        try:
            rules_data = self.file_manager.read(filepath)
            if not isinstance(rules_data, dict):
                logger.error(
                    f"❌ Erro de formato em '{filepath}'. Esperado um dicionário com listas de regras, mas foi recebido {type(rules_data).__name__}."
                )
                return

            for rule_category, rules_list in rules_data.items():
                if not isinstance(rules_list, list):
                    logger.warning(
                        f"Ignorando chave '{rule_category}' em '{filepath}', pois não contém uma lista."
                    )
                    continue

                for i, rule_data in enumerate(rules_list):
                    try:
                        required_keys = {"id", "confidence", "pattern"}
                        if not required_keys.issubset(rule_data.keys()):
                            logger.warning(
                                f"Regra em '{rule_category}' #{i+1} com chaves ausentes ignorada: {rule_data.get('id', 'ID_DESCONHECIDO')}"
                            )
                            continue

                        head, body = self._parse_pattern(rule_data["pattern"])
                        rule = Rule(
                            id=rule_data["id"],
                            confidence=float(rule_data["confidence"]),
                            head=head,
                            body=body,
                            source="manual",
                        )
                        self.manual_rules.append(rule)
                        self.rule_index[rule.id] = rule

                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Erro ao processar regra em '{rule_category}' #{i+1} (ID: {rule_data.get('id', 'N/A')}). Erro: {e}. Regra ignorada."
                        )

            logger.success(
                f"✅ {len(self.manual_rules)} regras manuais carregadas de {filepath}"
            )

        except FileNotFoundError:
            logger.warning(f"⚠️ Arquivo de regras manuais não encontrado: {filepath}")
        except Exception:
            logger.exception(
                f"❌ Erro inesperado ao carregar ou processar as regras manuais de {filepath}"
            )
            raise

    def _load_and_aggregate_rules(self, filepath: Path) -> list[Rule]:
        """
        Internal method to load and aggregate rules from TSV.
        This method will be cached by DiskCache.

        Args:
            filepath: Path to AnyBURL TSV file

        Returns:
            List of aggregated rules
        """
        rules = []
        try:
            df = pl.read_csv(filepath, separator="\t", has_header=False)
            df = df.rename(
                {
                    "column_1": "total_predictions",
                    "column_2": "correct_predictions",
                    "column_3": "confidence",
                    "column_4": "rule_text",
                }
            )

            for idx, row in enumerate(df.to_dicts()):
                try:
                    rule_text = row["rule_text"]
                    head, body = self._parse_pattern(rule_text)

                    rule = Rule(
                        id=f"anyburl_{idx}",
                        confidence=float(row["confidence"]),
                        head=head,
                        body=body,
                        source="anyburl",
                        total_predictions=int(row["total_predictions"]),
                        correct_predictions=int(row["correct_predictions"]),
                    )
                    rules.append(rule)

                except Exception as e:
                    logger.debug(f"Erro ao processar regra AnyBURL #{idx}: {e}")
                    continue

            logger.success(f"✅ {len(rules)} regras AnyBURL carregadas de {filepath}")
            logger.info(f"📊 Agregando {len(rules)} regras...")
            aggregated_rules = _aggregate_duplicate_rules(rules)
            logger.success(f"✅ {len(aggregated_rules)} regras únicas após agregação")

            return aggregated_rules

        except FileNotFoundError:
            logger.warning(f"⚠️ Arquivo de regras AnyBURL não encontrado: {filepath}")
            return []
        except Exception as e:
            logger.error(f"❌ Erro ao carregar regras AnyBURL: {e}")
            raise

    def load_anyburl_rules(self, filepath: Path | None = None) -> None:
        """
        Load AnyBURL rules from TSV file with DiskCache caching.

        **v10.8.0 - DiskCache Integration:**
        - Uses project's DiskCache for consistent caching
        - Auto-invalidation when TSV file changes
        - Expected speedup: ~3-4s per run

        Args:
            filepath: Path to AnyBURL TSV file
        """
        if filepath is None:
            filepath = settings.PYCLAUSE_DIR / "rules_anyburl.tsv"

        cache = DiskCache(root=settings.OUTPUTS_DIR / ".cache" / "aggregated_rules")
        file_stat = filepath.stat()
        cache_key = f"{filepath.name}_{file_stat.st_mtime}_{file_stat.st_size}"
        
        @cache(ttl=7 * 24 * 3600)
        def cached_load(key: str, path: Path) -> list[Rule]:
            return self._load_and_aggregate_rules(path)

        self.anyburl_rules = cached_load(cache_key, filepath)
        for rule in self.anyburl_rules:
            self.rule_index[rule.id] = rule

    def get_all_rules(self) -> list[Rule]:
        """
        Get all loaded rules from all sources.

        Returns:
            Combined list of all rules
        """
        return self.manual_rules + self.anyburl_rules


def _aggregate_duplicate_rules(rules: list[Rule]) -> list[Rule]:
    """
    Aggregate duplicate rules by body pattern, preserving frequency as confidence signal.

    This function groups rules with identical body predicates, counting occurrences
    and summing confidences. This allows validating each unique pattern ONCE instead
    of validating the same pattern thousands of times.

    Args:
        rules: List of rules (may contain many duplicates)

    Returns:
        List of unique rules with aggregation metadata:
        - occurrences: Count of how many times this pattern appeared
        - aggregated_confidence: Sum of all confidence scores for this pattern

    Example:
        Input: [Rule1(conf=0.8), Rule1_dup(conf=0.9), Rule2(conf=0.7)]
        Output: [Rule1(conf=0.8, occurrences=2, agg_conf=1.7), Rule2(conf=0.7, occurrences=1, agg_conf=0.7)]

    """
    if not rules:
        return []

    from collections import defaultdict

    groups: dict[str, list[Rule]] = defaultdict(list)
    fm = FileManager()

    for rule in rules:
        body_str = fm.json_dumps(rule.body, sort_keys=True)
        groups[body_str].append(rule)

    aggregated = []
    for body_key, group in groups.items():
        representative = group[0]
        occurrences = len(group)
        aggregated_confidence = sum(r.confidence for r in group)
        aggregated_rule = Rule(
            id=representative.id,
            confidence=representative.confidence,
            head=representative.head,
            body=representative.body,
            source=representative.source,
            total_predictions=representative.total_predictions,
            correct_predictions=representative.correct_predictions,
            occurrences=occurrences,
            aggregated_confidence=aggregated_confidence,
        )
        aggregated.append(aggregated_rule)

    unique_count = len(aggregated)
    total_count = len(rules)
    duplicate_ratio = (1 - unique_count / total_count) * 100 if total_count > 0 else 0

    logger.info(
        f"📊 Rule aggregation: {total_count:,} rules → {unique_count:,} unique patterns "
        f"({duplicate_ratio:.1f}% duplicates)"
    )

    return aggregated


class RuleValidator:
    """
    Validates data against rules using unification and pattern matching.

    This class implements the core validation logic, checking if rules
    are satisfied by the given triples through variable unification.
    """

    def __init__(self):
        self.triple_index = _TripleIndexStrategy()

    def validate_rules(
        self, rules: list[Rule], triples: list[tuple[Any, str, Any]]
    ) -> tuple[list[RuleViolation], list[Rule]]:
        """
        Validate all rules against the given triples in parallel.

        Args:
            rules: List of rules to validate (128K+ supported)
            triples: List of (subject, predicate, object) triples (1-10K typical)

        Returns:
            Tuple of (violations list, satisfied rules list)
        """
        if not rules:
            return [], []

        already_aggregated = any(r.occurrences > 1 or r.aggregated_confidence > 0 for r in rules[:10])

        if not already_aggregated:
            import time
            t_agg_start = time.time()
            rules = _aggregate_duplicate_rules(rules)
            t_agg_end = time.time()
            logger.info(f"⚡ Rule aggregation completed in {t_agg_end - t_agg_start:.2f}s")
        else:
            logger.info("✅ Rules already aggregated (loaded from cache), skipping aggregation")

        from pff.utils.resource_manager import get_resource_manager
        import sys

        resource_manager = get_resource_manager()
        estimated_task_size = sys.getsizeof(rules[0]) if rules else 5000  # ~5 KB
        shared_data_size = sum(sys.getsizeof(t) for t in triples[:10]) * len(triples) // 10
        limits = resource_manager.calculate_limits(
            task_count=len(rules),
            estimated_task_size=estimated_task_size,
            shared_data_size=shared_data_size,
        )

        logger.info(
            f"🚀 Adaptive allocation: {limits.optimal_workers} workers, "
            f"{limits.max_pending_futures} max pending, "
            f"{limits.safe_memory_limit / 1024**3:.1f} GB safe limit"
        )

        import time
        t0 = time.time()
        triple_index = TripleIndex(triples)
        index_build_time = time.time() - t0
        logger.info(
            f"🚀 Triple index built: {len(triples)} triples in {index_build_time:.2f}s "
            f"(enables 5-10x speedup)"
        )

        import functools
        shared_data = (triples, triple_index)
        fn_with_index = functools.partial(_run_rule_check_indexed, shared_data)
        args_list = [(rule,) for rule in rules]
        cm = ConcurrencyManager()
        results: list[list[RuleViolation]] = cm.execute_sync(
            fn=fn_with_index,
            args_list=args_list,
            task_type="process",
            max_workers=limits.optimal_workers,  # 🚀 Adaptive workers
            desc=f"🚀 Validating {len(rules):,} rules (indexed)",
        )
        violations = []
        satisfied_rules = []
        for i, rule_violations in enumerate(results):
            if rule_violations:
                violations.extend(rule_violations)
            else:
                satisfied_rules.append(rules[i])

        return violations, satisfied_rules

    def _check_single_rule(
        self, rule: Rule, triples: list[tuple[Any, str, Any]]
    ) -> list[RuleViolation]:
        """
        Check if a single rule is satisfied by the triples.

        Args:
            rule: Rule to check
            triples: List of triples

        Returns:
            List of violations (empty if rule is satisfied)
        """
        violations: list[RuleViolation] = []
        self._find_rule_violations(rule.body, triples, 0, {}, violations, rule)

        return violations

    def _find_rule_violations(
        self,
        body_predicates: list[dict],
        triples: list[tuple],
        pred_idx: int,
        bindings: dict[str, Any],
        violations: list[RuleViolation],
        rule: Rule,
    ) -> None:
        """
        Recursively find violations by checking if body predicates are satisfied.
        """
        if pred_idx >= len(body_predicates):
            if not self._check_head_satisfied(rule.head, triples, bindings):
                substituted_head = self._substitute_vars(rule.head["args"], bindings)
                head_str = f"{rule.head['predicate']}({', '.join(map(str, substituted_head))})"
                bindings_str = ", ".join(f"{k}='{v}'" for k, v in bindings.items())
                description = (
                    f"Conclusão esperada '{head_str}' não encontrada. "
                    f"A violação ocorreu porque as condições da regra foram satisfeitas com as variáveis: [{bindings_str}]"
                )
                violation = RuleViolation(
                    rule_id=rule.id,
                    confidence=rule.confidence,
                    description=description,
                    bindings=bindings.copy(),
                )
                violations.append(violation)
            return

        pattern = body_predicates[pred_idx]

        for triple in triples:
            new_bindings = self._try_unify(pattern, triple, bindings)
            if new_bindings is not None:
                self._find_rule_violations(
                    body_predicates,
                    triples,
                    pred_idx + 1,
                    new_bindings,
                    violations,
                    rule,
                )

    def _try_unify(
        self,
        pattern: dict[str, Any],
        triple: tuple[Any, str, Any],
        bindings: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Try to unify a pattern with a triple given current bindings.

        Args:
            pattern: Pattern to match
            triple: Triple to match against
            bindings: Current variable bindings

        Returns:
            Updated bindings if unification succeeds, None otherwise
        """
        subject, predicate, obj = triple

        if pattern["predicate"] != predicate and pattern["predicate"] != "*":
            return None
        args = pattern.get("args", [])
        if len(args) < 2:
            return None
        new_bindings = bindings.copy()
        if not self._bind_or_check(args[0], subject, new_bindings):
            return None
        if not self._bind_or_check(args[1], obj, new_bindings):
            return None

        return new_bindings

    def _bind_or_check(self, var: str, value: Any, bindings: dict[str, Any]) -> bool:
        """
        Bind variable to value or check consistency with existing binding.

        Args:
            var: Variable name (uppercase) or literal
            value: Value to bind/check
            bindings: Current bindings

        Returns:
            True if successful, False otherwise
        """
        if var.isupper():
            if var in bindings:
                return str(bindings[var]) == str(value)
            else:
                bindings[var] = value
                return True
        else:
            return var == str(value)

    def _check_head_satisfied(
        self,
        head_pattern: dict[str, Any],
        triples: list[tuple],
        bindings: dict[str, Any],
    ) -> bool:
        """
        Check if head pattern exists in triples with current bindings.

        Args:
            head_pattern: Head predicate pattern
            triples: List of triples
            bindings: Current variable bindings

        Returns:
            True if head is satisfied
        """
        substituted_args = self._substitute_vars(head_pattern["args"], bindings)
        for triple in triples:
            subject, predicate, obj = triple
            if predicate == head_pattern["predicate"]:
                if len(substituted_args) >= 2:
                    if (
                        str(subject) == substituted_args[0]
                        and str(obj) == substituted_args[1]
                    ):
                        return True

        return False

    def _substitute_vars(self, args: list[str], bindings: dict[str, Any]) -> list[str]:
        """
        Substitute variables with their bound values.

        Args:
            args: List of variables/literals
            bindings: Variable bindings

        Returns:
            List with variables replaced by values
        """
        result = []
        for arg in args:
            if arg.isupper() and arg in bindings:
                result.append(str(bindings[arg]))
            else:
                result.append(arg)
        return result


class ModelIntegration:
    """
    Integrates ML models for hybrid scoring.

    Manages ensemble model (TransE + LightGBM + Symbolic Rules + XGBoost meta-learner).
    Falls back to individual TransE and LightGBM models if ensemble is not available.
    """

    def __init__(self):
        self.ensemble_model = None
        self.transe_model = None
        self.lightgbm_model = None
        self.models_loaded = False
        self.lgbm_feature_names: list[str] = []

    def load_models(self, models_dir: Path) -> bool:
        """
        Load ensemble model (preferred) or individual TransE and LightGBM models (fallback).

        Args:
            models_dir: Directory containing model files

        Returns:
            True if models loaded successfully
        """
        try:
            ensemble_path = models_dir / "ensemble" / "stacking_model_advanced.joblib"
            if ensemble_path.exists():
                self.ensemble_model = joblib.load(ensemble_path)
                logger.info("✅ Modelo Ensemble carregado (TransE + LightGBM + Symbolic + XGBoost)")
                self.models_loaded = True
                return True
            logger.warning("⚠️ Ensemble não encontrado, carregando modelos individuais...")
            transe_path = models_dir / "transe" / "transe_model.pkl"
            if transe_path.exists():
                self.transe_model = joblib.load(transe_path)
                logger.info("✅ Modelo TransE carregado")
            lgb_path = models_dir / "transe" / "lightgbm_model.bin"
            if lgb_path.exists():
                import lightgbm as lgb
                self.lightgbm_model = lgb.Booster(model_file=str(lgb_path))
                self.lgbm_feature_names = self.lightgbm_model.feature_name()
                logger.info("✅ Modelo LightGBM carregado")
            
            self.models_loaded = bool(self.transe_model or self.lightgbm_model)  
            return self.models_loaded

        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
            return False

    def predict_hybrid_score(
        self,
        triples: list[tuple[Any, str, Any]],
        violations: list[Any] | None = None,
        all_rules: list[Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Generate hybrid score from models with XAI explanations.

        Prefers ensemble model (TransE + LightGBM + Symbolic + XGBoost).
        Falls back to individual models if ensemble is not available.

        Args:
            triples: List of triples for prediction
            violations: List of rule violations (NEW - FIX Bug #1)
            all_rules: List of all rules for feature extraction (NEW - FIX Bug #1)

        Returns:
            Tuple of (score, xai_report) where xai_report contains:
                - individual_scores: Dict with each model's score
                - ensemble_decision: Final ensemble score
                - violation_analysis: Violation-based features
                - decision_explanation: Human-readable explanation
        """
        xai_report = {
            "individual_scores": {},
            "ensemble_decision": 0.5,
            "violation_analysis": {},
            "decision_explanation": "",
        }

        if not self.models_loaded:
            xai_report["decision_explanation"] = "⚠️ Modelos não carregados, retornando score neutro"
            return 0.5, xai_report
        if violations is not None and all_rules is not None:
            violation_features = self._extract_violation_features(violations, all_rules)
            xai_report["violation_analysis"] = violation_features
            logger.debug(
                f"🔍 [XAI] Violation Features: {len(violations)} violations, "
                f"rate={violation_features['violation_rate']:.3f}, "
                f"avg_conf={violation_features['avg_confidence']:.3f}"
            )
        if self.ensemble_model:
            try:
                from pff.validators.ensembles.ensemble_wrappers.transformers import (
                    _ensemble_violations_context,
                    _ensemble_all_rules_context,
                )
                token_violations = _ensemble_violations_context.set(
                    violations if violations is not None else []
                )
                token_rules = _ensemble_all_rules_context.set(
                    all_rules if all_rules is not None else []
                )

                try:
                    proba = self.ensemble_model.predict_proba([triples])
                    base_ensemble_score = float(proba[0, 1])
                    xai_report["individual_scores"]["ensemble_base"] = base_ensemble_score
                    logger.info(f"📊 [XAI] Base Ensemble Score: {base_ensemble_score:.4f}")
                finally:
                    # Always reset context (cleanup)
                    _ensemble_violations_context.reset(token_violations)
                    _ensemble_all_rules_context.reset(token_rules)

                if self.lightgbm_model:
                    try:
                        features = self._extract_features(triples)
                        lgb_score = float(self.lightgbm_model.predict(features)[0])  # type: ignore[index,arg-type]
                        xai_report["individual_scores"]["lightgbm"] = lgb_score
                        logger.info(f"   └─ LightGBM: {lgb_score:.4f}")
                    except Exception as e:
                        logger.warning(f"   └─ LightGBM: erro ({e})")

                if self.transe_model:
                    try:
                        transe_scores = []
                        for triple in triples[:5]:  # Sample first 5 for XAI
                            head, relation, tail = map(str, triple)
                            raw_score = self.transe_model.score_triple(head, relation, tail)
                            normalized_score = 1 / (1 + np.exp(-raw_score))
                            transe_scores.append(normalized_score)
                        avg_transe = float(np.mean(transe_scores)) if transe_scores else 0.5
                        xai_report["individual_scores"]["transe"] = avg_transe
                        logger.info(f"   └─ TransE: {avg_transe:.4f} (sampled {len(transe_scores)} triples)")
                    except Exception as e:
                        logger.warning(f"   └─ TransE: erro ({e})")

                final_score = base_ensemble_score
                violation_penalty = 0.0
                if violations and len(violations) > 0 and all_rules:
                    total_rules = len(all_rules)
                    violation_rate = len(violations) / total_rules
                    violation_penalty = min(violation_rate * 10, 0.3)
                    final_score = max(0.0, base_ensemble_score - violation_penalty)
                    xai_report["individual_scores"]["violation_penalty"] = -violation_penalty
                    logger.info(
                        f"   └─ Violation Analysis: {len(violations)} violations "
                        f"({violation_rate:.4%} of {total_rules:,} rules)"
                    )
                    logger.info(
                        f"   └─ Violation Penalty: -{violation_penalty:.4f} "
                        f"(rate-based: {violation_rate:.4%} * 10)"
                    )
                    logger.info(
                        f"📉 [XAI] Adjusted: {base_ensemble_score:.4f} → {final_score:.4f}"
                    )
                xai_report["ensemble_decision"] = final_score
                explanation_parts = []
                explanation_parts.append(f"Ensemble base score: {base_ensemble_score:.4f}")

                if "lightgbm" in xai_report["individual_scores"]:
                    lgb = xai_report["individual_scores"]["lightgbm"]
                    explanation_parts.append(f"LightGBM contribution: {lgb:.4f}")
                if "transe" in xai_report["individual_scores"]:
                    transe = xai_report["individual_scores"]["transe"]
                    explanation_parts.append(f"TransE contribution: {transe:.4f}")
                if violation_penalty > 0 and violations:
                    explanation_parts.append(
                        f"Violation penalty: -{violation_penalty:.4f} "
                        f"({len(violations)} rule violations detected)"
                    )

                explanation_parts.append(f"Final decision: {final_score:.4f}")

                if final_score < 0.3:
                    explanation_parts.append("⛔ Recommendation: REJECT (high violation rate)")
                elif final_score < 0.5:
                    explanation_parts.append("⚠️ Recommendation: REVIEW (moderate violations)")
                else:
                    explanation_parts.append("✅ Recommendation: ACCEPT (low violations)")

                xai_report["decision_explanation"] = " | ".join(explanation_parts)
                logger.info(f"🎯 [XAI] {xai_report['decision_explanation']}")

                return final_score, xai_report

            except Exception as e:
                logger.warning(f"⚠️ Erro na predição do Ensemble: {e}")
                logger.warning("   Caindo para modelos individuais...")
                xai_report["decision_explanation"] = f"Erro no Ensemble: {e}, usando fallback"

        scores = []
        logger.info("📊 [XAI] Using fallback individual models")

        if self.lightgbm_model:
            try:
                features = self._extract_features(triples)
                lgb_score = float(self.lightgbm_model.predict(features)[0])  # type: ignore[index,arg-type]
                scores.append(lgb_score)
                xai_report["individual_scores"]["lightgbm"] = lgb_score
                logger.info(f"   └─ LightGBM: {lgb_score:.4f}")
            except Exception as e:
                logger.warning(f"   └─ LightGBM: erro ({e})")

        if self.transe_model:
            try:
                transe_scores = []
                for triple in triples:
                    head, relation, tail = map(str, triple)
                    raw_score = self.transe_model.score_triple(head, relation, tail)
                    normalized_score = 1 / (1 + np.exp(-raw_score))
                    scaled_score = 0.8 * normalized_score + 0.1
                    transe_scores.append(scaled_score)

                avg_transe_score = float(np.mean(transe_scores))
                scores.append(avg_transe_score)
                xai_report["individual_scores"]["transe"] = avg_transe_score
                logger.info(f"   └─ TransE: {avg_transe_score:.4f}")
            except Exception as e:
                logger.warning(f"   └─ TransE: erro ({e})")
                scores.append(0.5)

        final_score = sum(scores) / len(scores) if scores else 0.5
        xai_report["ensemble_decision"] = final_score
        xai_report["decision_explanation"] = (
            f"Fallback models average: {final_score:.4f} "
            f"(LightGBM: {xai_report['individual_scores'].get('lightgbm', 'N/A')}, "
            f"TransE: {xai_report['individual_scores'].get('transe', 'N/A')})"
        )

        logger.info(f"🎯 [XAI] {xai_report['decision_explanation']}")
        return final_score, xai_report

    def _extract_features(self, triples: list[tuple[Any, str, Any]]) -> np.ndarray:
        """
        Extract features from triples for LightGBM.

        Args:
            triples: List of triples

        Returns:
            Feature array
        """
        if not self.lgbm_feature_names:
            raise ValueError("Nomes das features do LightGBM não foram carregados.")
        predicate_counts: dict[str, int] = defaultdict(int)
        for _, predicate, _ in triples:
            predicate_counts[predicate] += 1
        features = [
            predicate_counts.get(feature_name, 0)
            for feature_name in self.lgbm_feature_names
        ]

        return np.array([features])

    def _extract_violation_features(
        self, violations: list[Any], all_rules: list[Any]
    ) -> dict[str, Any]:
        """
        Extract features from rule violations for Ensemble.

        This is the CRITICAL FIX for Bug #1: Instead of passing only triples
        to the Ensemble (which forces it to re-validate), we extract violation
        features HERE in the Business Service and pass them to the Ensemble.

        Args:
            violations: List of RuleViolation objects
            all_rules: List of all rules (for context)

        Returns:
            Dictionary with violation features:
                - num_violations: Total number of violations
                - violation_rate: Violations / total_rules ratio
                - avg_confidence: Average confidence of violations
                - violated_rule_ids: Set of violated rule IDs
        """
        num_violations = len(violations)
        total_rules = len(all_rules)

        features = {
            "num_violations": num_violations,
            "violation_rate": num_violations / max(total_rules, 1),
            "avg_confidence": 0.0,
            "violated_rule_ids": set(),
        }

        if violations:
            confidences = [v.confidence for v in violations if hasattr(v, "confidence")]
            features["avg_confidence"] = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            features["violated_rule_ids"] = {
                v.rule_id for v in violations if hasattr(v, "rule_id")
            }

        return features


class BusinessService:
    """
    Main business validation service with dynamic rule loading and XAI.

    This service validates JSON data against dynamically loaded rules
    from both manual definitions and AnyBURL inferences, providing
    detailed validation reports with confidence scores.
    """

    def __init__(self):
        logger.info("🚀 Inicializando Business Service com XAI...")
        self.file_manager = FileManager()
        self.triple_strategy = _TripleIndexStrategy()
        self.rule_engine = RuleEngine()
        self.rule_validator = RuleValidator()
        self.model_integration = ModelIntegration()
        self.triples_cache = DiskCache(root=".cache/triples_cache")
        self._load_rules()
        self._load_models()

    def _load_rules(self) -> None:
        """
        Load all validation rules from configured sources.
        """
        manual_path = settings.PATTERNS_DIR / "manual_rules.json"
        if manual_path.exists():
            self.rule_engine.load_manual_rules(manual_path)

        anyburl_path = settings.PYCLAUSE_DIR / "rules_anyburl.tsv"
        if anyburl_path.exists():
            self.rule_engine.load_anyburl_rules(anyburl_path)

        total_rules = len(self.rule_engine.get_all_rules())
        logger.info(f"📊 Total de {total_rules} regras carregadas")

        if total_rules == 0:
            logger.warning("⚠️ Nenhuma regra foi carregada!")

    def _load_models(self) -> None:
        """
        Load ML models for hybrid scoring.
        """
        success = self.model_integration.load_models(settings.OUTPUTS_DIR)
        if not success:
            logger.warning("⚠️ Operando sem modelos ML - apenas validação por regras")

    def validate(self, input_data: dict | str) -> dict[str, Any]:
        """
        Validate input JSON against all loaded rules.

        Args:
            input_data: JSON data or path to validate

        Returns:
            Validation report dictionary containing:
                - is_valid: Overall validation status
                - confidence_score: Average confidence of satisfied rules
                - hybrid_score: Combined ML model score
                - total_violations: Number of rule violations
                - top_10_violations: List of top 10 violations
        """
        try:
            if isinstance(input_data, str):
                file_path = Path(input_data)
                if not file_path.is_absolute():
                    file_path = settings.DATA_DIR / file_path.name
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Arquivo de dados da tarefa não encontrado em: {file_path}"
                    )
                input_data = self.file_manager.read(file_path)
            cache_key = self.triple_strategy._generate_cache_key(input_data)
            triples = self.triples_cache._load_from_cache(cache_key, ttl=None)
            if triples is not None:
                logger.success(f"⚡️ Cache HIT para triplas. Chave: {cache_key[:10]}... Carregando {len(triples)} triplas do cache.")
            else:
                # logger.info(f"🐢 Cache MISS. Gerando triplas para a chave: {cache_key[:10]}...")
                triples = self.triple_strategy._normalize_to_triples_optimized(input_data)
                self.triples_cache._save_to_cache(cache_key, triples)
                # logger.debug(f"💾 {len(triples)} triplas salvas no cache. Chave: {cache_key[:10]}...")
            logger.debug(f"📊 {len(triples)} triplas extraídas do JSON")
            all_rules = self.rule_engine.get_all_rules()
            violations, satisfied_rules = self.rule_validator.validate_rules(
                all_rules, triples
            )
            confidence_score = self._calculate_confidence_score(satisfied_rules)
            hybrid_score, xai_report = self.model_integration.predict_hybrid_score(
                triples, violations=violations, all_rules=all_rules
            )
            top_10_violations = []
            if violations:
                violations.sort(key=lambda v: v.confidence, reverse=True)
                for v in violations[:10]:
                    top_10_violations.append({
                        "rule_id": v.rule_id,
                        "description": v.description,
                        "confidence": v.confidence,
                    })
            is_valid = len(violations) == 0 and hybrid_score > 0.5
            logger.info(
                f"✅ Validação concluída: {'VÁLIDO' if is_valid else 'INVÁLIDO'}"
            )
            logger.info(f"   - Violações: {len(violations)}")
            logger.info(f"   - Confiança: {confidence_score:.4f}")
            logger.info(f"   - Score híbrido: {hybrid_score:.4f}")
            result = {
                "is_valid": is_valid,
                "confidence_score": confidence_score,
                "hybrid_score": hybrid_score,
                "total_violations": len(violations),
                "num_violations": len(violations),  # Compatibility with tests
                "top_10_violations": top_10_violations,
                "confidence": confidence_score,
                "dominant_expert": "N/A",
                "diagnostic": top_10_violations[0]['description'] if top_10_violations else "Nenhuma violação encontrada",
                "xai_report": xai_report,
                "xai_summary": {
                    "decision": xai_report["decision_explanation"],
                    "models": xai_report["individual_scores"],
                    "violations": xai_report["violation_analysis"],
                }
            }

            logger.info("═" * 80)
            logger.info("🔬 [XAI] RELATÓRIO DE EXPLICABILIDADE")
            logger.info("═" * 80)

            if "ensemble_base" in xai_report["individual_scores"]:
                logger.info(f"📊 Score Base Ensemble: {xai_report['individual_scores']['ensemble_base']:.4f}")
            if "lightgbm" in xai_report["individual_scores"]:
                logger.info(f"   └─ LightGBM: {xai_report['individual_scores']['lightgbm']:.4f}")
            if "transe" in xai_report["individual_scores"]:
                logger.info(f"   └─ TransE: {xai_report['individual_scores']['transe']:.4f}")
            if "violation_penalty" in xai_report["individual_scores"]:
                penalty = xai_report["individual_scores"]["violation_penalty"]
                logger.info(f"   └─ Penalty (violations): {penalty:.4f}")

            logger.info(f"🎯 Decisão Final: {xai_report['ensemble_decision']:.4f}")
            logger.info(f"💡 Explicação: {xai_report['decision_explanation']}")
            logger.info("═" * 80)

            return result
        except Exception as e:
            logger.exception(f"❌ Erro durante validação: {e}")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "hybrid_score": 0.0,
                "total_violations": -1,
                "top_10_violations": [],
                "confidence": 0.0,
                "dominant_expert": "N/A",
                "diagnostic": f"Erro de validação: {str(e)}"
            }

    def _calculate_confidence_score(self, satisfied_rules: list[Rule]) -> float:
        """
        Calculate weighted average confidence of satisfied rules.

        Args:
            satisfied_rules: List of rules that were satisfied

        Returns:
            Weighted average confidence score
        """
        if not satisfied_rules:
            return 0.0
        total_weight = sum(rule.confidence for rule in satisfied_rules)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(rule.confidence**2 for rule in satisfied_rules)
        return weighted_sum / total_weight


def _run_rule_check(rule: Rule, triples: list[tuple]) -> list[RuleViolation]:
    """
    Executes a rule check on a list of triples using the provided Rule.

    **DEPRECATED:** Use _run_rule_check_shared instead to avoid memory explosion.

    Args:
        rule (Rule): The rule to be checked.
        triples (list[tuple]): A list of triples to validate against the rule.
    Returns:
        list[RuleViolation]: A list of rule violations found during validation.
    """
    temp_validator = RuleValidator()
    return temp_validator._check_single_rule(rule, triples)


class TripleIndex:
    """
    High-performance triple index using hash-based lookups.

    Provides O(1) average-case lookup instead of O(n) linear search.
    Expected speedup: 5-10x for rule validation.

    Structure:
        spo: dict[subject][predicate] = set(objects)
        pos: dict[predicate][object] = set(subjects)
        osp: dict[object][subject] = set(predicates)

    Example:
        >>> index = TripleIndex(triples)
        >>> index.exists("Alice", "knows", "Bob")  # O(1)
        True
        >>> index.get_objects("Alice", "knows")  # O(1)
        {"Bob", "Charlie"}
    """

    def __init__(self, triples: list[tuple[Any, str, Any]]):
        """
        Build triple index from list of (subject, predicate, object) tuples.

        Time complexity: O(n) where n = number of triples
        Space complexity: O(n) - stores each triple 3 times for fast lookups

        Args:
            triples: List of (subject, predicate, object) tuples
        """
        self.spo: dict[Any, dict[str, set[Any]]] = {}
        self.pos: dict[str, dict[Any, set[Any]]] = {}
        self.osp: dict[Any, dict[Any, set[str]]] = {}

        for s, p, o in triples:
            # spo: subject → predicate → objects
            if s not in self.spo:
                self.spo[s] = {}
            if p not in self.spo[s]:
                self.spo[s][p] = set()
            self.spo[s][p].add(o)

            # pos: predicate → object → subjects
            if p not in self.pos:
                self.pos[p] = {}
            if o not in self.pos[p]:
                self.pos[p][o] = set()
            self.pos[p][o].add(s)

            # osp: object → subject → predicates
            if o not in self.osp:
                self.osp[o] = {}
            if s not in self.osp[o]:
                self.osp[o][s] = set()
            self.osp[o][s].add(p)

    def exists(self, subject: Any, predicate: str, obj: Any) -> bool:
        """
        Check if triple (s, p, o) exists. O(1) average case.

        Args:
            subject: Triple subject
            predicate: Triple predicate
            obj: Triple object

        Returns:
            True if triple exists, False otherwise
        """
        return obj in self.spo.get(subject, {}).get(predicate, set())

    def get_objects(self, subject: Any, predicate: str) -> set[Any]:
        """Get all objects for (subject, predicate). O(1) average case."""
        return self.spo.get(subject, {}).get(predicate, set())

    def get_subjects(self, predicate: str, obj: Any) -> set[Any]:
        """Get all subjects for (predicate, object). O(1) average case."""
        return self.pos.get(predicate, {}).get(obj, set())

    def get_predicates(self, subject: Any, obj: Any) -> set[str]:
        """Get all predicates for (subject, object). O(1) average case."""
        return self.osp.get(obj, {}).get(subject, set())


def _bind_or_check_standalone(var: Any, value: Any, bindings: dict[str, Any]) -> bool:
    """Standalone version of _bind_or_check without instance dependencies."""
    if isinstance(var, str) and var.isupper():  # Variable
        if var in bindings:
            return bindings[var] == value
        else:
            bindings[var] = value
            return True
    else:
        return var == value


def _substitute_vars_standalone(args: list[Any], bindings: dict[str, Any]) -> list[Any]:
    """Standalone version of _substitute_vars without instance dependencies."""
    return [bindings.get(arg, arg) if isinstance(arg, str) and arg.isupper() else arg for arg in args]


def _try_unify_standalone(
    pattern: dict[str, Any],
    triple: tuple[Any, str, Any],
    bindings: dict[str, Any],
) -> dict[str, Any] | None:
    """Standalone version of _try_unify without instance dependencies."""
    subject, predicate, obj = triple

    if pattern["predicate"] != predicate and pattern["predicate"] != "*":
        return None
    args = pattern.get("args", [])
    if len(args) < 2:
        return None

    new_bindings = bindings.copy()

    if not _bind_or_check_standalone(args[0], subject, new_bindings):
        return None
    if not _bind_or_check_standalone(args[1], obj, new_bindings):
        return None

    return new_bindings


def _check_head_satisfied_standalone(head: dict, triples: list[tuple], bindings: dict[str, Any]) -> bool:
    """
    Standalone version of _check_head_satisfied without instance dependencies.

    **DEPRECATED:** Use _check_head_satisfied_indexed for O(1) lookup.
    This version has O(n) complexity due to linear search.
    """
    substituted_args = _substitute_vars_standalone(head["args"], bindings)
    predicate = head["predicate"]

    if len(substituted_args) < 2:
        return False

    subject, obj = substituted_args[0], substituted_args[1]

    for triple in triples:
        if triple[0] == subject and triple[1] == predicate and triple[2] == obj:
            return True

    return False


def _check_head_satisfied_indexed(
    head: dict, triple_index: TripleIndex, bindings: dict[str, Any]
) -> bool:
    """
    🚀 OPTIMIZED: Check if head is satisfied using indexed lookup.

    Complexity: O(1) average case (vs O(n) for linear search)
    Expected speedup: 5-10x

    Args:
        head: Head predicate to check
        triple_index: Pre-built triple index for fast lookups
        bindings: Variable bindings

    Returns:
        True if head is satisfied, False otherwise
    """
    substituted_args = _substitute_vars_standalone(head["args"], bindings)
    predicate = head["predicate"]

    if len(substituted_args) < 2:
        return False

    subject, obj = substituted_args[0], substituted_args[1]

    # O(1) hash lookup instead of O(n) linear search
    return triple_index.exists(subject, predicate, obj)


def _find_rule_violations_standalone(
    body_predicates: list[dict],
    triples: list[tuple],
    pred_idx: int,
    bindings: dict[str, Any],
    violations: list[RuleViolation],
    rule: Rule,
    encoder: VocabularyEncoder | None = None,
) -> None:
    """
    Standalone version of _find_rule_violations without instance dependencies.
    
    **DEPRECATED:** Use _find_rule_violations_indexed for O(1) triple lookups.

    Args:
        encoder: Optional VocabularyEncoder for Numba acceleration (Sprint 17)
    """
    if pred_idx >= len(body_predicates):
        if not _check_head_satisfied_standalone(rule.head, triples, bindings):
            substituted_head = _substitute_vars_standalone(rule.head["args"], bindings)
            head_str = f"{rule.head['predicate']}({', '.join(map(str, substituted_head))})"
            bindings_str = ", ".join(f"{k}='{v}'" for k, v in bindings.items())
            description = (
                f"Conclusão esperada '{head_str}' não encontrada. "
                f"A violação ocorreu porque as condições da regra foram satisfeitas com as variáveis: [{bindings_str}]"
            )
            violation = RuleViolation(
                rule_id=rule.id,
                confidence=rule.confidence,
                description=description,
                bindings=bindings.copy(),
            )
            violations.append(violation)
        return

    pattern = body_predicates[pred_idx]

    if encoder is not None and NUMBA_AVAILABLE and len(triples) > 10:
        matching_indices = find_matching_triples_accelerated(pattern, triples, encoder)
        for idx in matching_indices:
            triple = triples[idx]
            new_bindings = _try_unify_standalone(pattern, triple, bindings)
            if new_bindings is not None:
                _find_rule_violations_standalone(
                    body_predicates,
                    triples,
                    pred_idx + 1,
                    new_bindings,
                    violations,
                    rule,
                    encoder,  # Pass encoder through recursion
                )
    else:
        for triple in triples:
            new_bindings = _try_unify_standalone(pattern, triple, bindings)
            if new_bindings is not None:
                _find_rule_violations_standalone(
                    body_predicates,
                    triples,
                    pred_idx + 1,
                    new_bindings,
                    violations,
                    rule,
                    encoder,  # Pass encoder through recursion even in fallback
                )


def _find_rule_violations_indexed(
    body_predicates: list[dict],
    triples: list[tuple],
    triple_index: TripleIndex,
    pred_idx: int,
    bindings: dict[str, Any],
    violations: list[RuleViolation],
    rule: Rule,
) -> None:
    """
    Same logic as _find_rule_violations_standalone but uses TripleIndex
    for O(1) head satisfaction checks.
    """
    if pred_idx >= len(body_predicates):
        if not _check_head_satisfied_indexed(rule.head, triple_index, bindings):
            substituted_head = _substitute_vars_standalone(rule.head["args"], bindings)
            head_str = f"{rule.head['predicate']}({', '.join(map(str, substituted_head))})"
            bindings_str = ", ".join(f"{k}='{v}'" for k, v in bindings.items())
            description = (
                f"Conclusão esperada '{head_str}' não encontrada. "
                f"A violação ocorreu porque as condições da regra foram satisfeitas com as variáveis: [{bindings_str}]"
            )
            violation = RuleViolation(
                rule_id=rule.id,
                confidence=rule.confidence,
                description=description,
                bindings=bindings.copy(),
            )
            violations.append(violation)
        return

    pattern = body_predicates[pred_idx]

    for triple in triples:
        new_bindings = _try_unify_standalone(pattern, triple, bindings)
        if new_bindings is not None:
            _find_rule_violations_indexed(
                body_predicates,
                triples,
                triple_index,
                pred_idx + 1,
                new_bindings,
                violations,
                rule,
            )


def _run_rule_check_indexed(
    shared_data: tuple[list[tuple], TripleIndex], rule: Rule
) -> list[RuleViolation]:
    """
    Uses pre-built TripleIndex for O(1) head satisfaction checks instead of O(n) linear search.

    Args:
        shared_data: Tuple of (triples_list, triple_index) shared across workers
        rule: Rule to validate

    Returns:
        List of rule violations found
    """
    shared_triples, triple_index = shared_data
    violations: list[RuleViolation] = []
    _find_rule_violations_indexed(
        rule.body, shared_triples, triple_index, 0, {}, violations, rule
    )
    return violations
