from __future__ import annotations

from pathlib import Path

from pff.domain.kg.config import KGConfig
from pff.shared.core.config import settings


def test_kg_config_resolves_relative_to_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = KGConfig("config/models/kg.yaml")
    assert cfg.configuration_path.name == "kg.yaml"


def test_kg_config_missing_required_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")

    cfg = KGConfig("config/models/kg.yaml")
    missing = cfg.missing_required_files()

    assert set(missing) == {cfg.train_path, cfg.valid_path, cfg.test_path}
    assert cfg.validate(log_missing=False) is False
