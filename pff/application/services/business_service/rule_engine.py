"""
Rule Engine - Dynamic Rule Loading and Management.

This module provides the rule engine for loading and managing validation rules
from manual JSON sources.

Design Patterns Applied:
    - **Factory Pattern:** Creates Rule instances from various sources.
    - **Strategy Pattern:** Different loading strategies for different file formats.

Performance:
    - Apenas regras manuais são carregadas
    - Rule aggregation para deduplicar padrões
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from collections import defaultdict

from pff import settings
from pff.config import VALIDATOR_CONFIG_PATH
from pff.shared import FileManager, logger

from .models import Rule


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


class RuleEngine:
    """
    Dynamic rule engine for loading and managing validation rules.

    This engine loads rules from manual JSON files, parsing them into a unified
    internal representation.

    Design Patterns:
        - **Factory Pattern:** Creates Rule instances from various sources.
    """

    def __init__(self):
        """Initialize the rule engine."""
        self.rule_index: dict[str, Rule] = {}
        self.manual_rules: list[Rule] = []
        self.file_manager = FileManager()
        self.validator_config = _load_validator_config()

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
            raise ValueError(
                f"Invalid rule pattern, missing '<=' separator: {pattern_str}"
            )

        head_str, body_str = pattern_str.split("<=", 1)

        def parse_single_clause(clause_str: str) -> dict:
            """Parse a single clause like 'predicate(arg1,arg2)' into a dict."""
            clause_str = clause_str.strip()
            match = re.match(r"(\w+)\((.*?)\)", clause_str)
            if not match:
                raise ValueError(f"Malformed clause: {clause_str}")

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
            primary = settings.OUTPUTS_DIR / "ensemble" / "rules" / "manual_rules.json"
            fallback = settings.PATTERNS_DIR / "manual_rules.json"
            filepath = primary if primary.exists() else fallback

        try:
            rules_data = self.file_manager.read(filepath, return_native=True)
            if not isinstance(rules_data, dict):
                logger.error(
                    f"Erro de formato em '{filepath}'. Esperado um dicionário com "
                    f"listas de regras, mas foi recebido "
                    f"{type(rules_data).__name__}."
                )
                return

            for rule_category, rules_list in rules_data.items():
                if not isinstance(rules_list, list):
                    logger.warning(
                        f"Ignorando chave '{rule_category}' em '{filepath}', "
                        f"pois não contém uma lista."
                    )
                    continue

                for i, rule_data in enumerate(rules_list):
                    try:
                        required_keys = {"id", "confidence", "pattern"}
                        if not required_keys.issubset(rule_data.keys()):
                            logger.warning(
                                f"Regra em '{rule_category}' #{i + 1} com "
                                f"chaves ausentes ignorada: "
                                f"{rule_data.get('id', 'ID_DESCONHECIDO')}"
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
                            f"Erro ao processar regra em '{rule_category}' #{i + 1} "
                            f"(ID: {rule_data.get('id', 'N/A')}). "
                            f"Erro: {e}. Regra ignorada."
                        )

            logger.success(
                f" {len(self.manual_rules)} regras manuais carregadas de {filepath}"
            )

        except FileNotFoundError:
            logger.warning(f"Manual rules file not found: {filepath}")
        except Exception:
            logger.exception(
                f" Erro inesperado ao carregar ou processar as regras manuais de {filepath}"
            )
            raise

    def get_all_rules(self) -> list[Rule]:
        """
        Get all loaded rules from all sources.

        Returns:
            Combined list of all rules
        """
        return list(self.manual_rules)


def aggregate_duplicate_rules(rules: list[Rule]) -> list[Rule]:
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
        Output: [Rule1(conf=0.8, occurrences=2, agg_conf=1.7),
                 Rule2(conf=0.7, occurrences=1, agg_conf=0.7)]
    """
    if not rules:
        return []

    # Filter out None values (defensive programming for mock tests)
    rules = [r for r in rules if r is not None]
    if not rules:
        return []

    groups: dict[str, list[Rule]] = defaultdict(list)
    fm = FileManager()

    for rule in rules:
        rule_key = fm.json_dumps({"body": rule.body, "head": rule.head}, sort_keys=True)
        groups[rule_key].append(rule)

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
        f" Rule aggregation: {total_count:,} rules → {unique_count:,} unique patterns "
        f"({duplicate_ratio:.1f}% duplicates)"
    )

    return aggregated


# Backward compatibility alias
_aggregate_duplicate_rules = aggregate_duplicate_rules
