"""
Business Service Core - Main Validation Service.

This module provides the main BusinessService class that orchestrates
rule validation, ML scoring, and caching.

Design Patterns Applied:
    - **Facade Pattern:** BusinessService provides a unified interface to all
      validation components.
    - **Template Method:** `validate()` defines the validation skeleton with
      customizable steps.
    - **Dependency Injection:** Services receive FileManager, RuleEngine, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff import settings
from pff.config import VALIDATOR_CONFIG_PATH
from pff.utils import DiskCache, FileManager, logger
from pff.utils.dev.research import _TripleIndexStrategy

from .model_integration import ModelIntegration
from .models import Rule
from .rule_engine import RuleEngine
from .rule_validator import RuleValidator


# Load validator config
_file_manager = FileManager()
_validator_config = _file_manager.read(VALIDATOR_CONFIG_PATH)


class BusinessService:
    """
    Main business validation service with dynamic rule loading and XAI.

    This service validates JSON data against dynamically loaded rules
    from both manual definitions and AnyBURL inferences, providing
    detailed validation reports with confidence scores.

    Design Patterns:
        - **Facade Pattern:** Unified interface to validation components.
        - **Template Method:** `validate()` defines the validation skeleton.
    """

    def __init__(self):
        """Initialize the business service."""
        logger.info(" Inicializando Business Service com XAI...")
        self.file_manager = FileManager()
        self.triple_strategy = _TripleIndexStrategy()
        self.rule_engine = RuleEngine()
        self.rule_validator = RuleValidator()
        self.model_integration = ModelIntegration()
        
        # Cache config from validator.yaml
        cache_cfg = _validator_config.get("cache", {})
        triples_subdir = cache_cfg.get("triples_cache_subdir", "triples_cache")
        self.triples_cache = DiskCache(root=settings.CACHE_DIR / triples_subdir)
        
        self._load_rules()
        self._load_models()

    def _load_rules(self) -> None:
        """Load all validation rules from configured sources."""
        manual_path = settings.PATTERNS_DIR / "manual_rules.json"
        if manual_path.exists():
            self.rule_engine.load_manual_rules(manual_path)

        anyburl_path = settings.PYCLAUSE_DIR / "rules_anyburl.tsv"
        if anyburl_path.exists():
            self.rule_engine.load_anyburl_rules(anyburl_path)

        total_rules = len(self.rule_engine.get_all_rules())
        logger.info(f" Total de {total_rules} regras carregadas")

        if total_rules == 0:
            logger.warning("No rules were loaded!")

    def _load_models(self) -> None:
        """Load ML models for hybrid scoring."""
        success = self.model_integration.load_models(settings.OUTPUTS_DIR)
        if not success:
            logger.warning("Operating without ML models - rule validation only")

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
                logger.success(
                    f" Cache HIT para triplas. Chave: {cache_key[:10]}... "
                    f"Carregando {len(triples)} triplas do cache."
                )
            else:
                triples = self.triple_strategy._normalize_to_triples_optimized(input_data)
                self.triples_cache._save_to_cache(cache_key, triples)

            logger.debug(f"{len(triples)} triples extracted from JSON")

            all_rules = self.rule_engine.get_all_rules()
            violations, satisfied_rules = self.rule_validator.validate_rules(
                all_rules, triples
            )

            confidence_score = self._calculate_confidence_score(satisfied_rules)

            violation_payload = {
                "violations": violations,
                "rules": all_rules,
                "metadata": {
                    "cache_key": cache_key,
                    "triple_count": len(triples),
                },
            }

            hybrid_score, xai_report = self.model_integration.predict_hybrid_score(
                triples,
                violation_payload=violation_payload,
                violations=violations,
                all_rules=all_rules,
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
                f" Validação concluída: {'VÁLIDO' if is_valid else 'INVÁLIDO'}"
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
                "diagnostic": top_10_violations[0]["description"] if top_10_violations else "Nenhuma violação encontrada",
                "xai_report": xai_report,
                "xai_summary": {
                    "decision": xai_report["decision_explanation"],
                    "models": xai_report["individual_scores"],
                    "violations": xai_report["violation_analysis"],
                },
            }

            logger.info("═" * 80)
            logger.info(" [XAI] RELATORIO DE EXPLICABILIDADE")
            logger.info("═" * 80)

            if "ensemble_base" in xai_report["individual_scores"]:
                logger.info(f" Score Base Ensemble: {xai_report['individual_scores']['ensemble_base']:.4f}")
            if "lightgbm" in xai_report["individual_scores"]:
                logger.debug(f"[XAI] LightGBM: {xai_report['individual_scores']['lightgbm']:.4f}")
            if "rotate" in xai_report["individual_scores"]:
                logger.debug(f"[XAI] RotatE: {xai_report['individual_scores']['rotate']:.4f}")
            if "violation_penalty" in xai_report["individual_scores"]:
                penalty = xai_report["individual_scores"]["violation_penalty"]
                logger.info(f" Penalidade por violacoes: {penalty:.4f}")

            logger.info(f" Decisao Final: {xai_report['ensemble_decision']:.4f}")
            logger.debug(f"[XAI] Full explanation: {xai_report['decision_explanation']}")
            logger.info("═" * 80)

            return result

        except Exception as e:
            logger.exception(f"Validation error: {e}")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "hybrid_score": 0.0,
                "total_violations": -1,
                "top_10_violations": [],
                "confidence": 0.0,
                "dominant_expert": "N/A",
                "diagnostic": f"Erro de validação: {str(e)}",
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
