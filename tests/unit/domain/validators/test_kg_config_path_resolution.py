"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_kg_config_path_resolution.py

"""

from __future__ import annotations

from pathlib import Path

from pff.domain.kg.config import KGConfig
from pff.shared.core.config import settings


def test_kg_config_resolves_relative_to_repo_root(monkeypatch, tmp_path: Path) -> None:
    """Execute test kg config resolves relative to repo root.



    Args:

        monkeypatch: Input value used by this callable.

        tmp_path: Input value used by this callable.

    """

    monkeypatch.chdir(tmp_path)
    cfg = KGConfig("config/models/kg.yaml")
    assert cfg.configuration_path.name == "kg.yaml"


def test_kg_config_missing_required_files(tmp_path: Path, monkeypatch) -> None:
    """Execute test kg config missing required files.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    monkeypatch.setattr(settings, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")

    cfg = KGConfig("config/models/kg.yaml")
    missing = cfg.missing_required_files()

    assert set(missing) == {cfg.train_path, cfg.valid_path, cfg.test_path}
    assert cfg.validate(log_missing=False) is False
