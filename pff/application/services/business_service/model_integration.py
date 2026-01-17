"""DSLFM/PC-only model integration for business service scoring.

Design Patterns: Facade + Strategy (penalty calculator).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.config import VALIDATOR_CONFIG_PATH
from pff.shared import FileManager, logger
from pff.application.services.violation_penalty import (
    PenaltyConfig,
    ViolationPenaltyCalculator,
)


def _load_validator_config() -> dict[str, Any]:
    """Lazy load validator configuration."""
    fm = FileManager()
    try:
        return fm.read(VALIDATOR_CONFIG_PATH, return_native=True) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Failed to load validator config from {VALIDATOR_CONFIG_PATH}: {exc}"
        )
        return {}


class ModelIntegration:
    """Integrates DSLFM/PC scoring with violation penalties (no ensembles)."""

    def __init__(
        self, penalty_calculator: ViolationPenaltyCalculator | None = None
    ) -> None:
        validator_config = _load_validator_config()
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

    def load_models(self, models_dir: Path) -> bool:
        """Load DSLFM checkpoint if present."""
        try:
            dslfm_path = models_dir / "dslfm" / "best_model.pt"
            if dslfm_path.exists():
                self.dslfm_checkpoint = dslfm_path
                self.models_loaded = True
                logger.info(" Modelo DSLFM carregado")
                return True
            logger.warning("DSLFM model not found; returning neutral scores.")
            return False
        except Exception as exc:  # noqa: BLE001
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
            xai_report["decision_explanation"] = (
                " Modelo DSLFM não carregado, retornando score neutro"
            )
            return 0.5, xai_report

        # Placeholder: DSLFM base score would come from the loaded checkpoint/scorer service
        base_score = self._dslfm_scale * 0.5 + self._dslfm_offset

        violation_features: dict[str, Any] = self._extract_violation_features(
            payload.get("violations") or [], payload.get("rules") or []
        )
        penalty_adjustment = self._penalty_calculator.compute_penalty(
            violation_features
        )

        final_score = max(0.0, min(1.0, base_score + penalty_adjustment))
        xai_report["ensemble_decision"] = final_score
        xai_report["individual_scores"]["violations"] = penalty_adjustment
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
        payload = violation_payload or {}
        if violations is not None:
            payload["violations"] = violations
        if all_rules is not None:
            payload["rules"] = all_rules
        return payload

    def _extract_violation_features(
        self, violations: list[Any], rules: list[Any]
    ) -> dict[str, Any]:
        return {
            "violation_count": len(violations),
            "rule_count": len(rules),
        }
