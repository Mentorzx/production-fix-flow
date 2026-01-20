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

# import json
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
    """Carrega métricas de exemplo dos fixtures.

    Returns:
        Dicionário com Feature_Balance, Ensemble_Final, etc.
    """
    fm = FileManager()
    return fm.read(SAMPLE_METRICS_PATH, return_native=True)


def get_valid_entity_triples() -> list[tuple[str, str, str]]:
    """Carrega triplas de entidade válidas dos fixtures.

    Retorna triplas que satisfazem todas as regras de manual_rules.json:
    - status='active' com relatedParty, product, account, paymentMethod
    - relatedParty com id, name, role
    - productCharacteristic presente

    Returns:
        Lista de triplas (sujeito, predicado, objeto).
    """
    fm = FileManager()
    data = fm.read(VALID_ENTITY_PATH, return_native=True)
    return [tuple(t) for t in data]


def get_invalid_entity_triples() -> list[tuple[str, str, str]]:
    """Carrega triplas de entidade inválidas dos fixtures.

    Retorna triplas que violam manual_rules.json:
    - status='active' sem relatedParty (viola man_006)
    - status='active' sem productCharacteristic (viola man_001-005)
    - paymentMethod.status='BARRED' sem entity status='suspended' (viola man_013)

    Returns:
        Lista de triplas (sujeito, predicado, objeto).
    """
    fm = FileManager()
    data = fm.read(INVALID_ENTITY_PATH, return_native=True)
    return [tuple(t) for t in data]
