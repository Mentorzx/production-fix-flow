from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.application.services.business_service.shared.rule_builder import RuleSourceFactory
from pff.application.services.polars_extensions import DataFrameCache


class FakeFileManager:
    def __init__(self) -> None:
        self.ensure_dir_calls: list[Path] = []
        self.read_calls: list[tuple[Path, bool]] = []

    def ensure_dir(self, path: Path) -> None:
        self.ensure_dir_calls.append(path)

    def read(self, path: Path, return_native: bool = False, **_: Any) -> dict[str, Any]:
        self.read_calls.append((path, return_native))
        return {
            "manual": [
                {
                    "id": "rule_1",
                    "confidence": 0.9,
                    "pattern": "head(A,B) <= body(A,B)",
                }
            ]
        }


class FakeSettings:
    DATA_DIR = Path("/tmp/pff_data")
    OUTPUTS_DIR = Path("/tmp/pff_outputs")
    CACHE_DIR = Path("/tmp/pff_cache_root")
    PATTERNS_DIR = Path("/tmp/pff_patterns")


def test_dataframe_cache_accepts_injected_file_manager() -> None:
    fake = FakeFileManager()

    DataFrameCache(cache_dir=Path("/tmp/pff_cache_test"), file_manager=fake)

    assert fake.ensure_dir_calls, "DataFrameCache should use injected file manager"


def test_dataframe_cache_uses_injected_settings_when_cache_dir_not_provided() -> None:
    fake = FakeFileManager()

    cache = DataFrameCache(file_manager=fake, settings_obj=FakeSettings())

    assert cache.cache_dir == Path("/tmp/pff_cache_root/dataframes")


def test_rule_source_factory_propagates_injected_file_manager() -> None:
    fake = FakeFileManager()
    rules = RuleSourceFactory.load_rules(
        Path("manual_rules.json"),
        source_type="manual",
        file_manager=fake,
    )

    assert len(rules) == 1
    assert fake.read_calls, "ManualRuleSource should read via injected file manager"
    assert fake.read_calls[0][1] is True
