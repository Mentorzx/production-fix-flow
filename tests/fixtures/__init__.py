"""
Test fixtures for PFF unit tests.

This module provides sample data for fast, deterministic tests that don't
depend on production assets under data/models/.

Contents:
- sample_rules.tsv: Sample AnyBURL rules in TSV format
- sample_metrics.json: Sample ensemble metrics for validation tests
"""
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

SAMPLE_RULES_PATH = FIXTURES_DIR / "sample_rules.tsv"
SAMPLE_METRICS_PATH = FIXTURES_DIR / "sample_metrics.json"


def get_sample_rules() -> list[dict]:
    """Load sample rules from fixtures.
    
    Returns:
        List of rule dictionaries with head_coverage, body_coverage,
        confidence, and rule_string fields.
    """
    rules = []
    with open(SAMPLE_RULES_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                rules.append({
                    "head_coverage": int(parts[0]),
                    "body_coverage": int(parts[1]),
                    "confidence": float(parts[2]),
                    "rule_string": parts[3]
                })
    return rules


def get_sample_metrics() -> dict:
    """Load sample metrics from fixtures.
    
    Returns:
        Dictionary with Feature_Balance, Ensemble_Final, etc.
    """
    import json
    with open(SAMPLE_METRICS_PATH) as f:
        return json.load(f)
