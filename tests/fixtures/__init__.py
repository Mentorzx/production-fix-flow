"""
Test fixtures for PFF unit tests.

This module provides sample data for fast, deterministic tests that don't
depend on production assets under data/models/.

Contents:
- sample_rules.tsv: Sample legacy rule records in TSV format
- sample_metrics.json: Sample ensemble metrics for validation tests
- valid_entity.json: Valid entity triples (no violations expected)
- invalid_entity.json: Invalid entity triples (multiple violations expected)

NOTE: Entity fixtures use pre-flattened triple format (list of [s, p, o])
to ensure predicates match manual_rules.json expectations.
"""

from pathlib import Path
import json

from pff.shared.core.file_manager import FileManager

FIXTURES_DIR = Path(__file__).parent

SAMPLE_RULES_PATH = FIXTURES_DIR / "sample_rules.tsv"
SAMPLE_METRICS_PATH = FIXTURES_DIR / "sample_metrics.json"
VALID_ENTITY_PATH = FIXTURES_DIR / "valid_entity.json"
INVALID_ENTITY_PATH = FIXTURES_DIR / "invalid_entity.json"


def get_sample_rules() -> list[dict]:
    """Load sample rules from fixtures.

    Returns:
        List of rule dictionaries with head_coverage, body_coverage,
        confidence, and rule_string fields.
    """
    rules = []
    content = FileManager.read_text(SAMPLE_RULES_PATH)
    for line in content.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            rules.append(
                {
                    "head_coverage": int(parts[0]),
                    "body_coverage": int(parts[1]),
                    "confidence": float(parts[2]),
                    "rule_string": parts[3],
                }
            )
    return rules


def get_sample_metrics() -> dict:
    """Load sample metrics from fixtures.

    Returns:
        Dictionary with Feature_Balance, Ensemble_Final, etc.
    """
    with open(SAMPLE_METRICS_PATH) as f:
        return json.load(f)


def get_valid_entity_triples() -> list[tuple[str, str, str]]:
    """Load valid entity triples from fixtures.

    Returns triples that satisfy all manual_rules.json rules:
    - status='active' with relatedParty, product, account, paymentMethod
    - relatedParty with id, name, role
    - productCharacteristic present

    Returns:
        List of (subject, predicate, object) triples.
    """
    with open(VALID_ENTITY_PATH) as f:
        data = json.load(f)
    return [tuple(t) for t in data]


def get_invalid_entity_triples() -> list[tuple[str, str, str]]:
    """Load invalid entity triples from fixtures.

    Returns triples that violate manual_rules.json:
    - status='active' without relatedParty (violates man_006)
    - status='active' without productCharacteristic (violates man_001-005)
    - paymentMethod.status='BARRED' without entity status='suspended' (violates man_013)

    Returns:
        List of (subject, predicate, object) triples.
    """
    with open(INVALID_ENTITY_PATH) as f:
        data = json.load(f)
    return [tuple(t) for t in data]
