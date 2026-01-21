from __future__ import annotations

from pathlib import Path

from pff.domain.kg.config import KGConfig


def test_kg_config_resolves_relative_to_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = KGConfig("config/models/kg.yaml")
    assert cfg.configuration_path.name == "kg.yaml"
