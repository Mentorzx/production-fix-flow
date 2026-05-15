from __future__ import annotations

import tomllib
from pathlib import Path


def test_static_tool_python_versions_match_project_runtime() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["tool"]["mypy"]["python_version"] == "3.12"
    assert pyproject["tool"]["pylint"]["main"]["py-version"] == "3.12"
    assert pyproject["tool"]["pyright"]["pythonVersion"] == "3.12"


def test_main_dependency_lock_defaults_to_cpu_torch() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["tool"]["poetry"]["dependencies"]
    lockfile = Path("poetry.lock").read_text()

    assert dependencies["torch"] == {"version": "2.7.0", "source": "pytorch-cpu"}
    assert "triton" not in dependencies
    assert 'name = "triton"' not in lockfile
    assert 'name = "nvidia-cuda-runtime-cu12"' not in lockfile
    assert 'url = "https://download.pytorch.org/whl/cpu"' in lockfile
