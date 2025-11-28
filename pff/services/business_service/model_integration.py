"""
Model Integration - ML Model Integration for Hybrid Scoring.

This module integrates ML models (RotatE, LightGBM, Ensemble) for hybrid
scoring with XAI (Explainable AI) capabilities.

Design Patterns Applied:
    - **Dependency Injection:** Uses ViolationPenaltyCalculator for penalty logic.
    - **Strategy Pattern:** Delegates penalty computation to injected calculator.
    - **Facade Pattern:** Provides a unified interface to multiple ML models.

Performance:
    - Ensemble model is preferred (RotatE + LightGBM + Symbolic + XGBoost)
    - Falls back to individual models if ensemble is not available
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pff import settings
from pff.config import VALIDATOR_CONFIG_PATH
from pff.utils import FileManager, logger
from pff.services.violation_penalty import PenaltyConfig, ViolationPenaltyCalculator


# Load validator config
_file_manager = FileManager()
_validator_config = _file_manager.read(VALIDATOR_CONFIG_PATH)


class ModelIntegration:
    """
    Integrates ML models for hybrid scoring.

    Manages ensemble model (RotatE + LightGBM + Symbolic Rules + XGBoost meta-learner).
    Falls back to individual RotatE and LightGBM models if ensemble is not available.

    Design Patterns Applied:
        - **Dependency Injection:** Uses ViolationPenaltyCalculator for penalty logic.
        - **Strategy Pattern:** Delegates penalty computation to injected calculator.
        - **Facade Pattern:** Unified interface to multiple ML models.
    """

    def __init__(self, penalty_calculator: ViolationPenaltyCalculator | None = None):
        """
        Initialize model integration.

        Args:
            penalty_calculator: Optional injected penalty calculator (DI pattern)
        """
        self.ensemble_model = None
        self.rotate_model = None
        self.lightgbm_model = None
        self.models_loaded = False
        self.lgbm_feature_names: list[str] = []

        # Injected penalty calculator (SRP extraction)
        violation_cfg = _validator_config.get("violation_scoring", {})
        self._penalty_calculator = penalty_calculator or ViolationPenaltyCalculator(
            PenaltyConfig.from_config(violation_cfg)
        )

        # XAI parameters
        xai_cfg = _validator_config.get("xai", {})
        self._rotate_sample_size = xai_cfg.get("rotate_sample_size", 5)

        # Scoring parameters from config (avoid magic numbers)
        scoring_cfg = _validator_config.get("scoring", {})
        self._rotate_scale = scoring_cfg.get("rotate_scale", 0.8)
        self._rotate_offset = scoring_cfg.get("rotate_offset", 0.1)

    def load_models(self, models_dir: Path) -> bool:
        """
        Load ensemble model (preferred) or individual RotatE and LightGBM models (fallback).

        Args:
            models_dir: Directory containing model files

        Returns:
            True if models loaded successfully
        """
        try:
            ensemble_path = models_dir / "ensemble" / "stacking_model_advanced.joblib"
            if ensemble_path.exists():
                self.ensemble_model = joblib.load(ensemble_path)
                logger.info(" Modelo Ensemble carregado (RotatE + LightGBM + Symbolic + XGBoost)")
                self.models_loaded = True
                return True
            logger.warning("Ensemble model not found, falling back to individual models")
            rotate_path = models_dir / "rotate" / "rotate_model.pkl"
            if rotate_path.exists():
                self.rotate_model = joblib.load(rotate_path)
                logger.info(" Modelo RotatE carregado")
            lgb_path = models_dir / "rotate" / "lightgbm_model.bin"
            if lgb_path.exists():
                import lightgbm as lgb

                self.lightgbm_model = lgb.Booster(model_file=str(lgb_path))
                self.lgbm_feature_names = self.lightgbm_model.feature_name()
                logger.info(" Modelo LightGBM carregado")

            self.models_loaded = bool(self.rotate_model or self.lightgbm_model)
            return self.models_loaded

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def predict_hybrid_score(
        self,
        triples: list[tuple[Any, str, Any]],
        violation_payload: dict[str, Any] | None = None,
        *,
        violations: list[Any] | None = None,
        all_rules: list[Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Generate hybrid score from models with XAI explanations.

        Prefers ensemble model (RotatE + LightGBM + Symbolic + XGBoost).
        Falls back to individual models if ensemble is not available.

        Args:
            triples: List of triples for prediction
            violation_payload: Optional dict with violations and rules
            violations: List of rule violations
            all_rules: List of all rules for feature extraction

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

        payload = self._build_violation_payload(
            violation_payload, violations=violations, all_rules=all_rules
        )

        if not self.models_loaded:
            xai_report["decision_explanation"] = " Modelos não carregados, retornando score neutro"
            return 0.5, xai_report

        # Always compute violation features for bonus/penalty calculation
        violation_features: dict[str, Any] = self._extract_violation_features(
            payload.get("violations") or [], payload.get("rules") or []
        )
        xai_report["violation_analysis"] = violation_features

        if violation_features.get("num_violations", 0) > 0:
            logger.debug(
                " [XAI] Violation Features: count=%s rate=%.3f avg_conf=%.3f",
                violation_features["num_violations"],
                violation_features["violation_rate"],
                violation_features["avg_confidence"],
            )

        if self.ensemble_model:
            try:
                with self._symbolic_context(payload):
                    proba = self.ensemble_model.predict_proba([triples])
                    base_ensemble_score = float(proba[0, 1])
                    xai_report["individual_scores"]["ensemble_base"] = base_ensemble_score
                    logger.debug(f"[XAI] Base Ensemble Score: {base_ensemble_score:.4f}")

                violation_penalty = 0.0
                penalty_context: dict[str, Any] = {}
                if violation_features:
                    (
                        violation_penalty,
                        penalty_context,
                    ) = self._penalty_calculator.compute(violation_features)
                    if penalty_context:
                        xai_report["violation_analysis"].update(penalty_context)
                else:
                    xai_report["violation_analysis"] = {}

                if self.lightgbm_model:
                    try:
                        features = self._extract_features(triples)
                        lgb_score = float(self.lightgbm_model.predict(features)[0])
                        xai_report["individual_scores"]["lightgbm"] = lgb_score
                        logger.debug(f"[XAI] LightGBM score: {lgb_score:.4f}")
                    except Exception as e:
                        logger.warning(f"LightGBM scoring error: {e}")

                if self.rotate_model:
                    try:
                        rotate_scores = []
                        sample_size = self._rotate_sample_size
                        for triple in triples[:sample_size]:
                            head, relation, tail = map(str, triple)
                            raw_score = self.rotate_model.score_triple(head, relation, tail)
                            normalized_score = 1 / (1 + np.exp(-raw_score))
                            rotate_scores.append(normalized_score)
                        avg_rotate = float(np.mean(rotate_scores)) if rotate_scores else 0.5
                        xai_report["individual_scores"]["rotate"] = avg_rotate
                        logger.debug(f"[XAI] RotatE score: {avg_rotate:.4f} (sampled {len(rotate_scores)} triples)")
                    except Exception as e:
                        logger.warning(f"RotatE scoring error: {e}")

                final_score = min(1.0, max(0.0, base_ensemble_score - violation_penalty))

                # Log penalty or bonus (debug level)
                if violation_penalty > 0:
                    xai_report["individual_scores"]["violation_penalty"] = -violation_penalty
                    logger.debug(
                        f"[XAI] Violation penalty: -{violation_penalty:.4f} "
                        f"(reason={penalty_context.get('penalty_reason', 'rate')})"
                    )
                elif violation_penalty < 0:
                    bonus = -violation_penalty
                    xai_report["individual_scores"]["no_violations_bonus"] = bonus
                    logger.debug(
                        f"[XAI] No-violations bonus: +{bonus:.4f} "
                        f"(reason={penalty_context.get('penalty_reason', 'clean')})"
                    )

                xai_report["ensemble_decision"] = final_score
                explanation_parts = []
                explanation_parts.append(f"Ensemble base score: {base_ensemble_score:.4f}")

                if "lightgbm" in xai_report["individual_scores"]:
                    lgb = xai_report["individual_scores"]["lightgbm"]
                    explanation_parts.append(f"LightGBM contribution: {lgb:.4f}")
                if "rotate" in xai_report["individual_scores"]:
                    rotate = xai_report["individual_scores"]["rotate"]
                    explanation_parts.append(f"RotatE contribution: {rotate:.4f}")
                if violation_penalty > 0 and payload["violations"]:
                    explanation_parts.append(
                        f"Violation penalty: -{violation_penalty:.4f} "
                        f"({len(payload['violations'])} rule violations detected)"
                    )

                explanation_parts.append(f"Final decision: {final_score:.4f}")

                if final_score < 0.3:
                    explanation_parts.append(" Recommendation: REJECT (high violation rate)")
                elif final_score < 0.5:
                    explanation_parts.append(" Recommendation: REVIEW (moderate violations)")
                else:
                    explanation_parts.append(" Recommendation: ACCEPT (low violations)")

                xai_report["decision_explanation"] = " | ".join(explanation_parts)
                logger.info(f" [XAI] Decisao final: {final_score:.4f}")
                logger.debug(f"[XAI] Full explanation: {xai_report['decision_explanation']}")

                return final_score, xai_report

            except Exception as e:
                logger.warning(f"Ensemble prediction error: {e}")
                logger.warning("Falling back to individual models")
                xai_report["decision_explanation"] = f"Erro no Ensemble: {e}, usando fallback"

        # Fallback to individual models
        scores = []
        logger.info(" [XAI] Utilizando modelos individuais (modo fallback)")

        if self.lightgbm_model:
            try:
                features = self._extract_features(triples)
                lgb_score = float(self.lightgbm_model.predict(features)[0])
                scores.append(lgb_score)
                xai_report["individual_scores"]["lightgbm"] = lgb_score
                logger.debug(f"[XAI] LightGBM score: {lgb_score:.4f}")
            except Exception as e:
                logger.warning(f"LightGBM prediction error: {e}")

        if self.rotate_model:
            try:
                rotate_scores = []
                for triple in triples:
                    head, relation, tail = map(str, triple)
                    raw_score = self.rotate_model.score_triple(head, relation, tail)
                    normalized_score = 1 / (1 + np.exp(-raw_score))
                    # Use config values instead of magic numbers
                    scaled_score = self._rotate_scale * normalized_score + self._rotate_offset
                    rotate_scores.append(scaled_score)

                avg_rotate_score = float(np.mean(rotate_scores))
                scores.append(avg_rotate_score)
                xai_report["individual_scores"]["rotate"] = avg_rotate_score
                logger.debug(f"[XAI] RotatE score: {avg_rotate_score:.4f}")
            except Exception as e:
                logger.warning(f"RotatE prediction error: {e}")
                scores.append(0.5)

        final_score = sum(scores) / len(scores) if scores else 0.5
        xai_report["ensemble_decision"] = final_score
        xai_report["decision_explanation"] = (
            f"Fallback models average: {final_score:.4f} "
            f"(LightGBM: {xai_report['individual_scores'].get('lightgbm', 'N/A')}, "
            f"RotatE: {xai_report['individual_scores'].get('rotate', 'N/A')})"
        )

        logger.info(f" [XAI] Score final (fallback): {final_score:.4f}")
        logger.debug(f"[XAI] Full explanation: {xai_report['decision_explanation']}")
        return final_score, xai_report

    def _extract_features(self, triples: list[tuple[Any, str, Any]]) -> np.ndarray:
        """
        Extract features from triples for LightGBM.

        Uses vectorized NumPy operations for better performance.

        Args:
            triples: List of triples

        Returns:
            Feature array with shape (1, num_features)
        """
        if not self.lgbm_feature_names:
            raise ValueError("LightGBM feature names not loaded.")

        # Vectorized: extract predicates and count using NumPy
        if not triples:
            return np.zeros((1, len(self.lgbm_feature_names)), dtype=np.float64)

        predicates = np.array([t[1] for t in triples], dtype=object)
        unique_preds, counts = np.unique(predicates, return_counts=True)
        predicate_counts = dict(zip(unique_preds, counts))

        # Build feature vector in one pass
        features = np.array(
            [predicate_counts.get(fn, 0) for fn in self.lgbm_feature_names],
            dtype=np.float64
        )

        return features.reshape(1, -1)

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
            Dictionary with violation features
        """
        num_violations = len(violations)
        total_rules = len(all_rules)

        violation_rate = num_violations / max(total_rules, 1)
        features = {
            "num_violations": num_violations,
            "violation_rate": violation_rate,
            "avg_confidence": 0.0,
            "violated_rule_ids": set(),
            "total_rules": total_rules,
            "violations_per_k_rules": violation_rate * 1000,
        }

        if violations:
            confidences = [v.confidence for v in violations if hasattr(v, "confidence")]
            features["avg_confidence"] = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            features["violated_rule_ids"] = {
                v.rule_id for v in violations if hasattr(v, "rule_id")
            }

        severity = features["avg_confidence"] * features["violation_rate"]
        features["severity_score"] = severity

        return features

    def _build_violation_payload(
        self,
        violation_payload: dict[str, Any] | None,
        *,
        violations: list[Any] | None,
        all_rules: list[Any] | None,
    ) -> dict[str, Any]:
        """Build violation payload from various input sources."""
        payload = {
            "violations": violations or [],
            "rules": all_rules or [],
            "metadata": {},
        }
        if violation_payload:
            payload["violations"] = violation_payload.get("violations", payload["violations"]) or []
            payload["rules"] = violation_payload.get("rules", payload["rules"]) or []
            payload["metadata"] = violation_payload.get("metadata", {})
        return payload

    @contextmanager
    def _symbolic_context(self, payload: dict[str, Any]):
        """Context manager for setting symbolic features during prediction."""
        from pff.validators.ensembles.ensemble_wrappers.transformers import (
            _ensemble_all_rules_context,
            _ensemble_violations_context,
        )

        token_rules = None
        token_violations = None
        try:
            token_violations = _ensemble_violations_context.set(payload.get("violations", []) or [])
            token_rules = _ensemble_all_rules_context.set(payload.get("rules", []) or [])
            yield
        finally:
            if token_violations is not None:
                _ensemble_violations_context.reset(token_violations)
            if token_rules is not None:
                _ensemble_all_rules_context.reset(token_rules)
