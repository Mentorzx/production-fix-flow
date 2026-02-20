"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_runtime_init.py

"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pff.shared.system import runtime


def _fake_settings(tmp_path: Path) -> SimpleNamespace:
    """Execute fake settings.



    Args:

        tmp_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    root = tmp_path / "root"
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    logs = tmp_path / "logs"
    cache = tmp_path / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        ROOT_DIR=root,
        DATA_DIR=data,
        OUTPUTS_DIR=outputs,
        LOGS_DIR=logs,
        CACHE_DIR=cache,
    )


def test_initialize_runtime_sets_cache_and_env(monkeypatch, tmp_path: Path) -> None:
    """Execute test initialize runtime sets cache and env.



    Args:

        monkeypatch: Input value used by this callable.

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    settings = _fake_settings(tmp_path)
    env_path = settings.ROOT_DIR / ".env"
    env_path.write_text("RUNTIME_TEST_KEY=runtime_value\n# comment\nINVALID\n", encoding="utf-8")

    monkeypatch.setattr(runtime, "settings", settings)
    monkeypatch.setattr(runtime, "_is_main_process", lambda: True)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.delenv("PFF_CLEAN_MODE", raising=False)
    monkeypatch.delenv("RUNTIME_TEST_KEY", raising=False)
    monkeypatch.delenv("JOBLIB_TEMP_FOLDER", raising=False)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)

    called = {"count": 0}

    def _fake_apply() -> None:
        called["count"] += 1

    monkeypatch.setattr("pff.shared.core.config.apply_permanent_configurations", _fake_apply)

    runtime.initialize_runtime(version="test")

    assert settings.DATA_DIR.exists()
    assert settings.OUTPUTS_DIR.exists()
    assert settings.LOGS_DIR.exists()
    assert settings.CACHE_DIR.joinpath("joblib").exists()
    assert settings.CACHE_DIR.joinpath("pycache").exists()
    assert called["count"] == 1
    assert runtime.os.environ["RUNTIME_TEST_KEY"] == "runtime_value"
    assert runtime.os.environ["JOBLIB_TEMP_FOLDER"].endswith("/cache/joblib")
    assert runtime.os.environ["PYTHONPYCACHEPREFIX"].endswith("/cache/pycache")


def test_initialize_runtime_clean_mode_skips_outputs_and_logs(monkeypatch, tmp_path: Path) -> None:
    """Execute test initialize runtime clean mode skips outputs and logs.



    Args:

        monkeypatch: Input value used by this callable.

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    settings = _fake_settings(tmp_path)
    monkeypatch.setattr(runtime, "settings", settings)
    monkeypatch.setattr(runtime, "_is_main_process", lambda: True)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setenv("PFF_CLEAN_MODE", "1")
    monkeypatch.delenv("JOBLIB_TEMP_FOLDER", raising=False)

    monkeypatch.setattr("pff.shared.core.config.apply_permanent_configurations", lambda: None)

    runtime.initialize_runtime(version="test")

    assert settings.DATA_DIR.exists()
    assert not settings.OUTPUTS_DIR.exists()
    assert not settings.LOGS_DIR.exists()
    assert "JOBLIB_TEMP_FOLDER" not in runtime.os.environ
