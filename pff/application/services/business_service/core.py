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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pff.domain.audit import (
    AuditReportBuilder,
    AuditReportSchemaValidator,
    build_audit_run_ids,
    canonicalize_json_document,
    records_to_triples,
)
from pff.domain.audit.anomaly_scoring import AnomalyScoringConfig
from pff.domain.audit.findings import (
    drift_to_findings,
    graph_validation_report_to_findings,
    neuro_symbolic_scores_to_findings,
    schema_report_to_findings,
)
from pff.domain.audit.graph_constraints import GraphConstraintsValidator
from pff.domain.audit.input_validation import AuditInputSchemaValidator
from pff.domain.audit.json_patch import suggest_repairs_from_schema_report
from pff.domain.audit.profile import AuditProfileConfig, build_profile, compute_drift
from pff.domain.ports.persistence.audit_ports import (
    AuditAnalysisPort,
    AuditReportsPort,
    AuditStoragePort,
)
from pff.shared import DiskCache, FileManager, logger
from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.config import VALIDATOR_CONFIG_PATH, settings
from pff.shared.research import _TripleIndexStrategy

from .model_integration import ModelIntegration
from .models import Rule
from .rule_engine import RuleEngine
from .rule_validator import RuleValidator

HYBRID_SCORE_VALIDITY_THRESHOLD = 0.5


def _load_validator_config() -> dict[str, Any]:
    """Lazy load validator configuration."""
    fm = FileManager()
    try:
        return fm.read(VALIDATOR_CONFIG_PATH, return_native=True) or {}
    except Exception as exc:
        logger.warning(f"Failed to load validator config from {VALIDATOR_CONFIG_PATH}: {exc}")
        return {}


@dataclass(frozen=True)
class AuditExecutionResult:
    """Result payload for a Postgres-first audit execution."""

    report: dict[str, Any]
    run_id: str


class BusinessService:
    """
    Main business validation service with dynamic rule loading and XAI.

    This service validates JSON data against dynamically loaded rules
    from manual definitions,
    providing detailed validation reports with confidence scores.

    Design Patterns:
        - **Facade Pattern:** Unified interface to validation components.
        - **Template Method:** `validate()` defines the validation skeleton.
    """

    def __init__(
        self,
        *,
        file_manager: FileManager | None = None,
        rule_engine: RuleEngine | None = None,
        rule_validator: RuleValidator | None = None,
        model_integration: ModelIntegration | None = None,
        triple_strategy: _TripleIndexStrategy | None = None,
        audit_storage: AuditStoragePort | None = None,
        audit_analysis_repo: AuditAnalysisPort | None = None,
        audit_reports_repo: AuditReportsPort | None = None,
        audit_report_builder: AuditReportBuilder | None = None,
    ):
        """Initialize the business service (DI-friendly)."""
        logger.info("inicializando_business_service")
        self.file_manager = file_manager or FileManager()
        self.triple_strategy = triple_strategy or _TripleIndexStrategy()
        self.rule_engine = rule_engine or RuleEngine()
        self.rule_validator = rule_validator or RuleValidator()
        self.model_integration = model_integration or ModelIntegration()

        validator_config = _load_validator_config()
        cache_cfg = validator_config.get("cache", {})
        triples_subdir = cache_cfg.get("triples_cache_subdir", "triples_cache")
        self.triples_cache = DiskCache(root=settings.CACHE_DIR / triples_subdir)

        self._audit_storage = audit_storage
        self._audit_analysis_repo = audit_analysis_repo
        self._audit_reports_repo = audit_reports_repo
        self._audit_report_builder = audit_report_builder or AuditReportBuilder(
            outputs_dir=settings.OUTPUTS_DIR,
            schema_validator=AuditReportSchemaValidator(file_manager=self.file_manager),
            file_manager=self.file_manager,
        )

        self._load_rules()
        self._load_models()

    def __enter__(self) -> BusinessService:
        """Context manager entry for dependency-injected lifecycles."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit (best-effort cleanup)."""
        self.close()
        return False

    def close(self) -> None:
        """Release background resources that are safe to stop."""
        janitor = getattr(self.triples_cache, "_janitor", None)
        if janitor is not None:
            try:
                janitor.stop()
            except Exception as exc:
                logger.warning(f"Failed to stop disk cache janitor: {exc}")

    def _load_rules(self) -> None:
        """Load all validation rules from configured sources."""
        manual_path = settings.OUTPUTS_DIR / "ensemble" / "rules" / "manual_rules.json"
        if FileManager.exists(manual_path):
            self.rule_engine.load_manual_rules(manual_path)

        total_rules = len(self.rule_engine.get_all_rules())
        logger.info(f"regras_carregadas total={total_rules}")

        if total_rules == 0:
            logger.warning("No rules were loaded!")

    def _load_models(self) -> None:
        """Load ML models for hybrid scoring."""
        success = self.model_integration.load_models(settings.OUTPUTS_DIR)
        if not success:
            logger.warning("Operating without ML models - rule validation only")

    def _ensure_models_loaded(self) -> None:
        """Ensure ML models are loaded for scoring paths."""
        if not getattr(self.model_integration, "models_loaded", False):
            self._load_models()

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
                if not self.file_manager.exists(file_path):
                    raise FileNotFoundError(
                        f"Arquivo de dados da tarefa não encontrado em: {file_path}"
                    )
                input_data = self.file_manager.read(file_path, return_native=True)

            cache_key = self.triple_strategy._generate_cache_key(input_data)
            triples = self.triples_cache._load_from_cache(cache_key, ttl=None)

            if triples is not None:
                logger.success(
                    f"cache_triplas_acerto chave_prefixo={cache_key[:10]} triplas={len(triples):,}"
                )
            else:
                triples = self.triple_strategy._normalize_to_triples_optimized(input_data)
                self.triples_cache._save_to_cache(cache_key, triples)

            logger.debug(f"{len(triples)} triples extracted from JSON")

            validation_cfg = _load_validator_config().get("validation", {})
            prefer_manual_rules = bool(
                validation_cfg.get("manual_rules_only_for_small_payloads", True)
            )
            manual_payload_max = int(validation_cfg.get("manual_rules_payload_max", 200))
            if (
                prefer_manual_rules
                and len(triples) <= manual_payload_max
                and self.rule_engine.manual_rules
            ):
                all_rules = self.rule_engine.manual_rules
                logger.debug(f"Using only manual rules for small payload ({len(triples)} triples)")
            else:
                all_rules = self.rule_engine.get_all_rules()
            violations, satisfied_rules = self.rule_validator.validate_rules(all_rules, triples)

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
                    top_10_violations.append(
                        {
                            "rule_id": v.rule_id,
                            "description": v.description,
                            "confidence": v.confidence,
                        }
                    )

            is_valid = len(violations) == 0 and hybrid_score > HYBRID_SCORE_VALIDITY_THRESHOLD

            logger.info(
                "validacao_concluida "
                f"status={'valido' if is_valid else 'invalido'} "
                f"violacoes={len(violations):,} "
                f"confianca={confidence_score:.6f} "
                f"score_hibrido={hybrid_score:.6f}"
            )

            result = {
                "is_valid": is_valid,
                "confidence_score": confidence_score,
                "hybrid_score": hybrid_score,
                "total_violations": len(violations),
                "num_violations": len(violations),
                "top_10_violations": top_10_violations,
                "confidence": confidence_score,
                "dominant_expert": "N/A",
                "diagnostic": (
                    top_10_violations[0]["description"]
                    if top_10_violations
                    else "Nenhuma violação encontrada"
                ),
                "xai_report": xai_report,
                "xai_summary": {
                    "decision": xai_report["decision_explanation"],
                    "models": xai_report["individual_scores"],
                    "violations": xai_report["violation_analysis"],
                },
            }

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

    def audit_document(
        self,
        document: Any,
        *,
        baseline_key: Any,
        schema_version: str | int,
        input_schema: dict[str, Any] | None = None,
        schema_id: str | None = None,
        constraints: dict[str, Any] | None = None,
        scored_items: list[dict[str, Any]] | None = None,
        export_outputs: bool = False,
        meta_overrides: dict[str, Any] | None = None,
    ) -> AuditExecutionResult:
        """Run a Postgres-first JSON→Graph→JSON audit pipeline and persist the report.

        Args:
            document: Input JSON-like payload.
            baseline_key: Stable key for baseline artifacts (profile/calibration/EVT).
            schema_version: Input document schema version.
            input_schema: Optional JSON Schema for mechanical validation.
            schema_id: Optional schema identifier for report metadata.
            constraints: Optional SHACL-like graph constraints configuration.
            scored_items: Optional neuro-symbolic scored items aligned to
                json_pointer.
            export_outputs: When True, also writes
                `outputs/audit/<run_id>/report/audit_report.json`.
            meta_overrides: Optional metadata overrides merged into
                `report.meta`.

        Returns:
            AuditExecutionResult with report payload and run_id.
        """

        payload = run_coroutine_sync(
            self._audit_document_async(
                document=document,
                baseline_key=baseline_key,
                schema_version=schema_version,
                input_schema=input_schema,
                schema_id=schema_id,
                constraints=constraints,
                scored_items=scored_items,
                export_outputs=export_outputs,
                meta_overrides=meta_overrides,
            )
        )
        return AuditExecutionResult(report=payload["report"], run_id=str(payload["run_id"]))

    async def _audit_document_async(
        self,
        *,
        document: Any,
        baseline_key: Any,
        schema_version: str | int,
        input_schema: dict[str, Any] | None,
        schema_id: str | None,
        constraints: dict[str, Any] | None,
        scored_items: list[dict[str, Any]] | None,
        export_outputs: bool,
        meta_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._audit_storage is None:
            raise RuntimeError(
                "Audit storage not initialized. Inject AuditStoragePort to use audit features."
            )
        if self._audit_analysis_repo is None:
            raise RuntimeError("Audit analysis repo not initialized. Inject AuditAnalysisPort.")
        if self._audit_reports_repo is None:
            raise RuntimeError("Audit reports repo not initialized. Inject AuditReportsPort.")

        run_ids = build_audit_run_ids(
            document=document,
            baseline_key=baseline_key,
            schema_version=schema_version,
        )

        records = canonicalize_json_document(document, document_id=run_ids.document_id)
        triples = records_to_triples(records, run_id=run_ids.run_id)

        await self._audit_storage.persist_canonicalization(
            run_id=run_ids.run_id,
            document_id=run_ids.document_id,
            baseline_id=run_ids.baseline_id,
            records=records,
            triples=triples,
        )

        schema_report: list[dict[str, Any]] = []
        if input_schema is not None:
            schema_report = AuditInputSchemaValidator(schema=input_schema).validate(document)
            await self._audit_analysis_repo.save_schema_report(
                run_id=run_ids.run_id,
                schema_report=schema_report,
                schema_id=schema_id,
                schema_version=schema_version,
            )

        profile_cfg = AuditProfileConfig.load(file_manager=self.file_manager)
        baseline_profile = await self._audit_analysis_repo.load_baseline_profile(
            baseline_id=run_ids.baseline_id
        )
        baseline_bootstrapped = False
        if baseline_profile is None:
            baseline_profile = build_profile(records, config=profile_cfg)
            digest = {
                "baseline_source": "bootstrapped",
                "baseline_run_id": run_ids.run_id,
                "baseline_profile_hash": baseline_profile.get("profile_hash"),
            }
            await self._audit_analysis_repo.save_baseline_profile(
                baseline_id=run_ids.baseline_id,
                profile=baseline_profile,
                digest=digest,
            )
            baseline_bootstrapped = True

        edges_map: dict[str, list[float]] = {}
        fields = baseline_profile.get("fields", {}) if isinstance(baseline_profile, dict) else {}
        if isinstance(fields, dict):
            for field_path, entry in fields.items():
                if not isinstance(entry, dict):
                    continue
                hist = entry.get("numeric_hist")
                if isinstance(hist, dict) and isinstance(hist.get("edges"), list):
                    edges_map[str(field_path)] = [float(x) for x in hist["edges"]]

        current_profile = build_profile(
            records, config=profile_cfg, numeric_bin_edges_by_field=edges_map
        )
        drift_report = compute_drift(
            baseline_profile=baseline_profile,
            current_profile=current_profile,
            config=profile_cfg,
        )
        await self._audit_analysis_repo.save_run_profile(
            run_id=run_ids.run_id,
            profile_current=current_profile,
            drift=drift_report,
        )

        findings: list[dict[str, Any]] = []
        if schema_report:
            findings.extend(schema_report_to_findings(schema_report))
            if input_schema is not None:
                repairs = suggest_repairs_from_schema_report(
                    document=document,
                    schema=input_schema,
                    schema_report=schema_report,
                )
                if repairs:
                    findings.append(
                        {
                            "severity": "info",
                            "layer": "schema",
                            "message": (
                                f"Suggested repairs derived from JSON Schema "
                                f"violations: count={len(repairs)}"
                            ),
                            "suggested_repairs": repairs,
                            "broken_invariants": [{"name": "json_schema_repairs"}],
                        }
                    )

        drift_thresholds = profile_cfg.drift_thresholds or {}
        findings.extend(drift_to_findings(drift_report, thresholds=drift_thresholds))

        if baseline_bootstrapped:
            findings.append(
                {
                    "severity": "info",
                    "layer": "profile",
                    "message": (
                        f"Baseline profile bootstrapped for baseline_id={run_ids.baseline_id}"
                    ),
                    "evidence": {"baseline_id": run_ids.baseline_id},
                    "broken_invariants": [{"name": "baseline_profile_bootstrapped"}],
                }
            )

        if constraints is not None:
            validator = GraphConstraintsValidator(constraints=constraints)
            triple_dicts = [
                {
                    "s": t.s,
                    "p": t.p,
                    "o": t.o,
                    "json_pointer": t.json_pointer,
                    "record_hash": t.record_hash,
                    "triple_hash": t.triple_hash,
                }
                for t in triples
            ]
            graph_report = validator.validate(triple_dicts)
            if graph_report:
                findings.extend(graph_validation_report_to_findings(graph_report))

        if scored_items is not None:
            anomaly_cfg = AnomalyScoringConfig.load(file_manager=self.file_manager)
            findings.extend(
                neuro_symbolic_scores_to_findings(
                    scored_items,
                    p_value_warning=anomaly_cfg.p_value_warning,
                    p_value_error=anomaly_cfg.p_value_error,
                    max_findings=anomaly_cfg.max_findings,
                )
            )

        merged_meta: dict[str, Any] = {}
        if meta_overrides:
            merged_meta.update(dict(meta_overrides))
        if schema_id is not None:
            merged_meta.setdefault("schema_id", schema_id)

        report, built_ids, paths = self._audit_report_builder.build_report(
            document=document,
            baseline_key=baseline_key,
            schema_version=schema_version,
            findings=findings,
            meta_overrides=merged_meta,
        )

        if built_ids.run_id != run_ids.run_id:
            raise RuntimeError(
                "AuditRunIds mismatch between precomputed ids and report builder: "
                f"precomputed={run_ids.run_id} built={built_ids.run_id}"
            )

        await self._audit_reports_repo.save_report(run_id=built_ids.run_id, report=report)

        if export_outputs:
            self._audit_report_builder.write_report(report, paths=paths)

        logger.info(
            "auditoria_concluida "
            f"run_id={built_ids.run_id} findings={len(findings):,} export_outputs={export_outputs}"
        )
        return {"report": report, "run_id": built_ids.run_id}
