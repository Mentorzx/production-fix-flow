"""
Rule filtering utilities for AnyBURL outputs.

This module centralizes all rule-threshold logic so that optimization
pipelines do not reimplement filtering heuristics. It can be reused by
training scripts, validators, or batch jobs that need to post-process
AnyBURL rule dumps.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pff.utils import FileManager, logger
from pff.validators.kg.anyburl import RuleParser


@dataclass
class RuleFilterConfig:
    min_confidence: float = 0.5
    min_support: float = 5.0
    confidence_quantile: float = 0.75
    support_quantile: float = 0.6
    target_ratio: float = 0.35
    min_rules: int = 400
    max_rules_limit: int = 20_000
    relax_attempts: int = 4
    relax_decay: float = 0.85
    min_conf_floor: float = 0.15
    min_support_floor: float = 3.0
    activation_precision_floor: float = 0.55
    activation_min_predictions: int = 50
    activation_lift_floor: float = 1.15


@dataclass
class RuleFilterResult:
    filtered_rules_path: Path
    filtered_metadata: list[dict[str, Any]]
    original_metadata: list[dict[str, Any]]
    metrics: dict[str, float]
    metadata_lookup: Dict[str, dict[str, Any]]


class AnyBURLRuleFilter:
    """Filters AnyBURL rules applying configurable confidence/support heuristics."""

    def __init__(self, config: RuleFilterConfig, parser: RuleParser | None = None):
        self.config = config
        self.parser = parser or RuleParser()
        self.file_manager = FileManager()
        self._last_activation_filter_stats: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config_path: Path) -> "AnyBURLRuleFilter":
        """Create a filter instance from a YAML config file."""
        fm = FileManager()
        if not config_path.exists():
            logger.warning(
                f"Rule filter configuration not found ({config_path}). "
                "Using default settings."
            )
            return cls(RuleFilterConfig())

        # FileManager.read() handles YAML parsing automatically (AGENTS.md §5)
        config_payload = fm.read(config_path)
        if not isinstance(config_payload, dict):
            config_payload = {}

        # Accept nested rule_filter section (preferred) or top-level defaults (legacy)
        if "rule_filter" in config_payload and isinstance(config_payload["rule_filter"], dict):
            config_payload = config_payload["rule_filter"]
        else:
            logger.warning("rule_filter section missing in config; using top-level defaults and built-in fallbacks")

        # Validate required nested structures
        hpo_ranges = config_payload.get("hpo_ranges", {})
        required_keys = {
            "confidence_quantile",
            "support_quantile",
            "target_ratio",
            "max_length_cyclic",
            "max_length_acyclic",
        }
        missing_keys = [k for k in required_keys if k not in hpo_ranges]
        if missing_keys:
            logger.warning(
                f"rule_filter.hpo_ranges missing keys {missing_keys}; using defaults where absent"
            )

        defaults = config_payload.get("defaults", config_payload)
        config_kwargs: dict[str, Any] = {}
        for field in fields(RuleFilterConfig):
            if field.name in defaults:
                config_kwargs[field.name] = defaults[field.name]

        config = RuleFilterConfig(**config_kwargs)
        return cls(config)

    def filter_rules(
        self,
        *,
        rules_path: Path,
        output_dir: Path,
        rule_confidence: float,
        rule_support: float,
        target_entity_ratio: float,
        max_rules: int | None = None,
    ) -> RuleFilterResult:
        """Filter the AnyBURL rule file and persist a trimmed TSV."""
        if not rules_path.exists():
            raise FileNotFoundError(f"Arquivo de regras do AnyBURL não encontrado: {rules_path}")

        rules, metadata = self.parser.parse_rules_file(rules_path)
        if not rules:
            logger.warning("No AnyBURL rules were loaded; returning original file without filters.")
            return RuleFilterResult(
                filtered_rules_path=rules_path,
                filtered_metadata=[],
                original_metadata=metadata,
                metrics=self._build_metrics([], [], rule_confidence, rule_support),
                metadata_lookup={},
            )

        confidences = np.array([float(item.get("confidence", 0.0)) for item in metadata])
        supports = np.array([float(item.get("support", 0.0)) for item in metadata])

        applied_conf = max(
            float(rule_confidence),
            float(target_entity_ratio),
            self.config.min_confidence,
        )
        applied_support = max(float(rule_support), self.config.min_support)

        if confidences.size > 0:
            dynamic_conf = float(np.quantile(confidences, self.config.confidence_quantile))
            applied_conf = max(applied_conf, dynamic_conf)
        if supports.size > 0:
            dynamic_support = float(np.quantile(supports, self.config.support_quantile))
            applied_support = max(applied_support, dynamic_support)

        filtered_pairs = self._filter_pairs(rules, metadata, applied_conf, applied_support)
        filtered_pairs = self._apply_activation_filters(filtered_pairs)
        filtered_pairs = filtered_pairs[: self.config.max_rules_limit]

        target_rules = min(
            max(self.config.min_rules, int(len(rules) * self.config.target_ratio)),
            len(rules),
        )
        filtered_pairs = self._maybe_relax_thresholds(
            rules,
            metadata,
            filtered_pairs,
            target_rules,
            applied_conf,
            applied_support,
        )
        filtered_pairs = self._apply_activation_filters(filtered_pairs)

        if len(filtered_pairs) < target_rules:
            fallback_pairs = list(zip(rules, metadata))
            fallback_pairs.sort(key=self._sort_key, reverse=True)
            filtered_pairs = fallback_pairs[:target_rules]
            filtered_pairs = self._apply_activation_filters(filtered_pairs)
            logger.info(
                f"O limite mínimo de regras não foi atingido ({len(filtered_pairs)}/{len(rules)}). "
                "Retornando às regras mais bem classificadas sem filtros adicionais."
            )

        if max_rules and len(filtered_pairs) > max_rules:
            filtered_pairs.sort(key=self._sort_key, reverse=True)
            filtered_pairs = filtered_pairs[:max_rules]
            logger.info(
                f"Limite global aplicado: {len(filtered_pairs)}/{len(rules)} regras mantidas (teto={max_rules})"
            )

        filtered_rules = [pair[0] for pair in filtered_pairs]
        filtered_metadata = [pair[1] for pair in filtered_pairs]
        metadata_lookup = {
            meta.get("rule", ""): meta for meta in filtered_metadata if meta.get("rule")
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        filtered_rules_path = output_dir / "rules_filtered.tsv"
        
        df_rules = pl.DataFrame({
            "num_predictions": [m.get('num_predictions', 0) for m in filtered_metadata],
            "support": [m.get('support', 0) for m in filtered_metadata],
            "confidence": [m.get('confidence', 0.0) for m in filtered_metadata],
            "rule": [m.get('rule', '') for m in filtered_metadata]
        })
        
        self.file_manager.save(df_rules, filtered_rules_path, include_header=False, separator="\t")

        logger.info(
            f"Filtro AnyBURL → {len(filtered_metadata)}/{len(metadata)} regras "
            f"(conf ≥ {applied_conf:.3f}, suporte ≥ {applied_support:.1f})"
        )
        if self._last_activation_filter_stats:
            stats = self._last_activation_filter_stats
            logger.info(
                "Estatísticas pós-ativação → "
                f"mantidas {stats.get('kept_rules', 0)}/{stats.get('initial_rules', 0)} | "
                f"precisão média {stats.get('mean_activation_precision', 0.0):.3f} | "
                f"lift médio {stats.get('mean_activation_lift', 0.0):.3f} | "
                f"descartes (precisão={stats.get('removed_low_precision', 0)}, "
                f"lift={stats.get('removed_low_lift', 0)}, "
                f"ativação={stats.get('removed_low_predictions', 0)})"
            )

        metrics = self._build_metrics(filtered_metadata, metadata, applied_conf, applied_support)

        return RuleFilterResult(
            filtered_rules_path=filtered_rules_path,
            filtered_metadata=filtered_metadata,
            original_metadata=metadata,
            metrics=metrics,
            metadata_lookup=metadata_lookup,
        )

    def _filter_pairs(
        self,
        rules: list[str],
        metadata: list[dict[str, Any]],
        conf_threshold: float,
        support_threshold: float,
    ) -> list[tuple[str, dict[str, Any]]]:
        return [
            (rule, meta)
            for rule, meta in zip(rules, metadata)
            if float(meta.get("confidence", 0.0)) >= conf_threshold
            and float(meta.get("support", 0.0)) >= support_threshold
        ]

    def _maybe_relax_thresholds(
        self,
        rules: list[str],
        metadata: list[dict[str, Any]],
        current_pairs: list[tuple[str, dict[str, Any]]],
        target_rules: int,
        initial_conf: float,
        initial_support: float,
    ) -> list[tuple[str, dict[str, Any]]]:
        relaxed_pairs = current_pairs
        conf_threshold = initial_conf
        support_threshold = initial_support

        attempts = 0
        while len(relaxed_pairs) < target_rules and attempts < self.config.relax_attempts and rules:
            attempts += 1
            conf_threshold = max(self.config.min_conf_floor, conf_threshold * self.config.relax_decay)
            support_threshold = max(self.config.min_support_floor, support_threshold * self.config.relax_decay)
            logger.warning(
                f"Filter below target ({len(relaxed_pairs)}/{len(rules)}). "
                f"Relaxing limits → conf {initial_conf:.3f}→{conf_threshold:.3f}, "
                f"support {initial_support:.1f}→{support_threshold:.1f}."
            )
            relaxed_pairs = self._filter_pairs(rules, metadata, conf_threshold, support_threshold)
            relaxed_pairs.sort(key=self._sort_key, reverse=True)
            relaxed_pairs = relaxed_pairs[: self.config.max_rules_limit]

        return relaxed_pairs

    @staticmethod
    def _calculate_activation_lift(meta: dict[str, Any]) -> float:
        precision = float(meta.get("confidence", 0.0))
        activations = float(meta.get("num_predictions", meta.get("support", 0.0)))
        total_predictions = max(activations, 0.0)
        positive_hits = max(0.0, precision * total_predictions)
        negative_hits = max(0.0, total_predictions - positive_hits)
        return (positive_hits + 1.0) / (negative_hits + 1.0)

    def _apply_activation_filters(
        self, pairs: list[tuple[str, dict[str, Any]]]
    ) -> list[tuple[str, dict[str, Any]]]:
        if not pairs:
            self._last_activation_filter_stats = {}
            return pairs

        activation_filtered: list[tuple[str, dict[str, Any]]] = []
        removed_low_precision = 0
        removed_low_predictions = 0
        removed_low_lift = 0
        kept_precisions: list[float] = []
        kept_lifts: list[float] = []
        for rule, meta in pairs:
            precision = float(meta.get("confidence", 0.0))
            activations = float(meta.get("num_predictions", meta.get("support", 0.0)))
            if precision < self.config.activation_precision_floor:
                removed_low_precision += 1
                continue
            if activations < self.config.activation_min_predictions:
                removed_low_predictions += 1
                continue
            lift = self._calculate_activation_lift(meta)
            if lift < self.config.activation_lift_floor:
                removed_low_lift += 1
                continue
            activation_filtered.append((rule, meta))
            kept_precisions.append(precision)
            kept_lifts.append(lift)

        if not activation_filtered and pairs:
            logger.warning(
                "Activation filter removed all rules; keeping previous set to avoid empty output"
            )
            self._last_activation_filter_stats = {}
            return pairs

        self._last_activation_filter_stats = {
            "initial_rules": len(pairs),
            "kept_rules": len(activation_filtered),
            "removed_low_precision": removed_low_precision,
            "removed_low_predictions": removed_low_predictions,
            "removed_low_lift": removed_low_lift,
            "mean_activation_precision": float(np.mean(kept_precisions)) if kept_precisions else 0.0,
            "mean_activation_lift": float(np.mean(kept_lifts)) if kept_lifts else 0.0,
        }
        total_removed = removed_low_precision + removed_low_predictions + removed_low_lift
        if total_removed > 0:
            logger.info(
                f"Regras removidas por baixa ativação: {total_removed} "
                f"(precisão < {self.config.activation_precision_floor:.2f}: {removed_low_precision}, "
                f"lift < {self.config.activation_lift_floor:.2f}: {removed_low_lift}, "
                f"ativação < {self.config.activation_min_predictions}: {removed_low_predictions})"
            )
        return activation_filtered

    def _build_metrics(
        self,
        filtered_metadata: list[dict[str, Any]],
        original_metadata: list[dict[str, Any]],
        applied_conf: float,
        applied_support: float,
    ) -> dict[str, float]:
        confidences_filtered = [float(item.get("confidence", 0.0)) for item in filtered_metadata]
        supports_filtered = [float(item.get("support", 0.0)) for item in filtered_metadata]
        lifts_filtered = [self._calculate_activation_lift(item) for item in filtered_metadata]

        metrics: dict[str, float] = {
            "rule_count": float(len(filtered_metadata)),
            "avg_confidence": float(np.mean(confidences_filtered)) if confidences_filtered else 0.0,
            "avg_support": float(np.mean(supports_filtered)) if supports_filtered else 0.0,
            "mean_activation_precision": float(np.mean(confidences_filtered)) if confidences_filtered else 0.0,
            "mean_activation_lift": float(np.mean(lifts_filtered)) if lifts_filtered else 0.0,
            "high_confidence_ratio": float(
                sum(1 for c in confidences_filtered if c >= applied_conf) / len(confidences_filtered)
            )
            if confidences_filtered
            else 0.0,
            "applied_conf_threshold": float(applied_conf),
            "applied_support_threshold": float(applied_support),
            "original_rule_count": float(len(original_metadata)),
        }
        return metrics

    @staticmethod
    def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
        meta = item[1]
        return (
            float(meta.get("confidence", 0.0)),
            float(meta.get("support", 0.0)),
        )
