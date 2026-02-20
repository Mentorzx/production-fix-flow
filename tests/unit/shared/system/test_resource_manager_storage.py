"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/system/test_resource_manager_storage.py

"""

import os

from pff.shared.system.resource_manager import _detect_storage_type


def test_detect_storage_type_prefers_wsl() -> None:
    """Execute test detect storage type prefers wsl."""

    assert _detect_storage_type(is_wsl=True) == "wsl"


def test_detect_storage_type_env_override(monkeypatch) -> None:
    """Execute test detect storage type env override.



    Args:

        monkeypatch: Input value used by this callable.

    """

    monkeypatch.setenv("PFF_STORAGE_TYPE", "nvme")
    assert _detect_storage_type(is_wsl=False) == "nvme"
    monkeypatch.delenv("PFF_STORAGE_TYPE", raising=False)
    os.environ.pop("PFF_STORAGE_TYPE", None)
