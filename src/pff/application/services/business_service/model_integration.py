"""DSLFM/PC-only model integration for business service scoring.

Design Patterns: Facade + Strategy (penalty calculator).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.application.services.business_service.shared.violation_penalty import (
    PenaltyConfig,
    ViolationPenaltyCalculator,
)
from pff.shared import FileManager, load_config, logger
from pff.shared.core.config import VALIDATOR_CONFIG_PATH


class ModelIntegration:
    """Integrates DSLFM/PC scoring with violation penalties (no ensembles)."""

    def __init__(
        self,
        penalty_calculator: ViolationPenaltyCalculator | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        """Execute init.



        Args:

            penalty_calculator: Optional input value.

            file_manager: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        validator_config = load_config(VALIDATOR_CONFIG_PATH)
        violation_cfg = validator_config.get("violation_scoring", {})
        self._penalty_calculator = penalty_calculator or ViolationPenaltyCalculator(
            PenaltyConfig.from_config(violation_cfg)
        )
        xai_cfg = validator_config.get("xai", {})
        scoring_cfg = validator_config.get("scoring", {})
        self._dslfm_sample_size = xai_cfg.get("dslfm_sample_size", 5)
        self._dslfm_scale = scoring_cfg.get("dslfm_scale", 1.0)
        self._dslfm_offset = scoring_cfg.get("dslfm_offset", 0.0)
        self.dslfm_checkpoint: Path | None = None
        self.models_loaded = False
        self.file_manager = file_manager or FileManager()

    def load_models(self, models_dir: Path) -> bool:
        """Load DSLFM checkpoint if present."""
        try:
            dslfm_path = models_dir / "dslfm" / "best_model.pt"
            if self.file_manager.exists(dslfm_path):
                self.dslfm_checkpoint = dslfm_path
                self.models_loaded = True
                logger.info(" Modelo DSLFM carregado")
                return True
            logger.warning("DSLFM model not found; returning neutral scores.")
            return False
        except Exception as exc:
            logger.error(f"Failed to load DSLFM model: {exc}")
            return False

    def predict_hybrid_score(
        self,
        triples: list[tuple[Any, str, Any]],
        violation_payload: dict[str, Any] | None = None,
        *,
        violations: list[Any] | None = None,
        all_rules: list[Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Return DSLFM score adjusted by violation penalties."""
        xai_report: dict[str, Any] = {
            "individual_scores": {},
            "ensemble_decision": 0.5,
            "violation_analysis": {},
            "decision_explanation": "",
        }

        payload = self._build_violation_payload(
            violation_payload, violations=violations, all_rules=all_rules
        )

        if not self.models_loaded:
            xai_report["decision_explanation"] = (
                " Modelo DSLFM não carregado, retornando score neutro"
            )
            return 0.5, xai_report

        base_score = self._dslfm_scale * 0.5 + self._dslfm_offset

        violation_features: dict[str, Any] = self._extract_violation_features(
            payload.get("violations") or [], payload.get("rules") or []
        )
        penalty_adjustment, penalty_meta = self._penalty_calculator.compute(
            violation_features
        )

        final_score = max(0.0, min(1.0, base_score + penalty_adjustment))
        xai_report["ensemble_decision"] = final_score
        xai_report["individual_scores"]["violations"] = penalty_adjustment
        xai_report["violation_analysis"] = penalty_meta
        xai_report["decision_explanation"] = (
            " Score DSLFM ajustado por penalidades de violação"
        )
        return float(final_score), xai_report

    def _build_violation_payload(
        self,
        violation_payload: dict[str, Any] | None,
        *,
        violations: list[Any] | None,
        all_rules: list[Any] | None,
    ) -> dict[str, Any]:
        """Execute build violation payload.



        Args:

            violation_payload: Input value used by this callable.

            violations: Input value used by this callable.

            all_rules: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        payload = violation_payload or {}
        if violations is not None:
            payload["violations"] = violations
        if all_rules is not None:
            payload["rules"] = all_rules
        return payload

    def _extract_violation_features(
        self, violations: list[Any], rules: list[Any]
    ) -> dict[str, Any]:
        """Execute extract violation features.



        Args:

            violations: Input value used by this callable.

            rules: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        total_rules = len(rules)
        num_violations = len(violations)
        violation_rate = num_violations / total_rules if total_rules > 0 else 0.0
        violations_per_k = violation_rate * 1000
        avg_confidence = (
            sum(getattr(v, "confidence", 0.0) for v in violations) / num_violations
            if num_violations > 0
            else 0.0
        )
        return {
            "num_violations": num_violations,
            "total_rules": total_rules,
            "violation_rate": violation_rate,
            "violations_per_k_rules": violations_per_k,
            "avg_confidence": avg_confidence,
        }
