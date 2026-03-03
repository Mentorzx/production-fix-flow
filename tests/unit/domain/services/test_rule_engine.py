"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/services/test_rule_engine.py

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.application.services.business_service.models import Rule
from pff.application.services.business_service.rule_engine import (
    RuleEngine,
    aggregate_duplicate_rules,
)


def test_aggregate_duplicate_rules_groups_by_head_and_body() -> None:
    """Execute test aggregate duplicate rules groups by head and body.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    rules = [
        Rule(
            id="r1",
            confidence=0.8,
            head={"predicate": "h", "args": ["A", "B"]},
            body=[{"predicate": "p", "args": ["A", "C"]}],
            source="manual",
        ),
        Rule(
            id="r1_dup",
            confidence=0.6,
            head={"predicate": "h", "args": ["A", "B"]},
            body=[{"predicate": "p", "args": ["A", "C"]}],
            source="manual",
        ),
        Rule(
            id="r2",
            confidence=0.4,
            head={"predicate": "h2", "args": ["X", "Y"]},
            body=[{"predicate": "q", "args": ["X", "Z"]}],
            source="manual",
        ),
    ]

    aggregated = aggregate_duplicate_rules(rules)

    assert len(aggregated) == 2
    by_head = {r.head["predicate"]: r for r in aggregated}

    first = by_head["h"]
    assert first.occurrences == 2
    assert first.aggregated_confidence == 1.4
    assert first.confidence == 0.8

    second = by_head["h2"]
    assert second.occurrences == 1
    assert second.aggregated_confidence == 0.4


class _FakeFileManager:
    def __init__(self) -> None:
        self.read_calls: list[Path | str] = []

    def read(self, path: Path | str, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        self.read_calls.append(path)
        return {
            "manual": [
                {
                    "id": "r_manual_1",
                    "confidence": 0.9,
                    "pattern": "h(A,B) <= p(A,B)",
                }
            ]
        }

    def exists(self, path: Path | str) -> bool:  # noqa: ARG002
        return False


class _FakeSettings:
    DATA_DIR = Path("/tmp/pff_data")
    OUTPUTS_DIR = Path("/tmp/pff_outputs")
    CACHE_DIR = Path("/tmp/pff_cache")
    PATTERNS_DIR = Path("/tmp/pff_patterns")


def test_rule_engine_uses_injected_file_manager_to_load_rules() -> None:
    """RuleEngine must load manual rules through injected file manager."""
    fake_manager = _FakeFileManager()
    engine = RuleEngine(file_manager=fake_manager)
    manual_path = Path("/tmp/manual_rules.json")

    engine.load_manual_rules(manual_path)

    assert fake_manager.read_calls == [manual_path]
    assert len(engine.manual_rules) == 1
    assert engine.manual_rules[0].id == "r_manual_1"


def test_rule_engine_uses_injected_config_loader() -> None:
    """RuleEngine must read validator config through injected loader."""

    calls: list[Path] = []

    def fake_loader(path: Path) -> dict[str, Any]:
        calls.append(path)
        return {"validation": {"max_recursion_depth": 33}}

    engine = RuleEngine(file_manager=_FakeFileManager(), config_loader=fake_loader)

    assert calls
    assert engine.validator_config["validation"]["max_recursion_depth"] == 33


def test_rule_engine_uses_injected_settings_for_default_manual_rules_path() -> None:
    """RuleEngine must resolve default manual rules path using injected settings."""

    fake_manager = _FakeFileManager()
    engine = RuleEngine(file_manager=fake_manager, settings_obj=_FakeSettings())

    engine.load_manual_rules()

    assert fake_manager.read_calls
    assert fake_manager.read_calls[0] == Path("/tmp/pff_patterns/manual_rules.json")
