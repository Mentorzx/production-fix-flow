from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.application.services.business_service.models import Rule
from pff.application.services.business_service.rule_validator import run_rule_check_indexed
from pff.application.services.business_service.triple_index import TripleIndex


def test_run_rule_check_indexed_uses_injected_config_loader() -> None:
    """run_rule_check_indexed should resolve validation config via injected loader."""

    calls: list[Path] = []

    def fake_loader(path: Path) -> dict[str, Any]:
        calls.append(path)
        return {"validation": {"max_recursion_depth": 7}}

    rule = Rule(
        id="r_empty_body",
        confidence=0.9,
        head={"predicate": "h", "args": ["A", "B"]},
        body=[],
        source="manual",
    )

    result = run_rule_check_indexed(([], TripleIndex([])), rule, config_loader=fake_loader)

    assert calls
    assert isinstance(result, list)
    assert len(result) == 1
