"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/services/test_line_service_config.py

"""

from __future__ import annotations

from pathlib import Path

from pff.application.services.line_service.config import load_line_service_config


class _FakeFileManager:
    def __init__(self, payload: dict | None = None, *, exists: bool = True) -> None:
        """Execute init.



        Args:

            payload: Optional input value.

            exists: Optional input value.

        """

        self._payload = payload
        self._exists = exists

    def exists(self, _path: Path) -> bool:
        """Execute exists.



        Args:

            _path: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return self._exists

    def read(self, _path: Path, **_kwargs):
        """Execute read.



        Args:

            _path: Input value used by this callable.

            **_kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        return self._payload


def test_load_line_service_config_uses_injected_file_manager() -> None:
    """Execute test load line service config uses injected file manager.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    payload = {
        "coalescing_delay_s": 7,
        "circuit_breaker": {
            "read": {"fail_max": 9, "timeout_duration_s": 90},
            "write": {"fail_max": 4, "timeout_duration_s": 40},
        },
    }
    fm = _FakeFileManager(payload=payload, exists=True)

    config = load_line_service_config(
        path=Path("config/line_service.yaml"),
        file_manager=fm,
    )

    assert config.coalescing_delay_s == 7
    assert config.read_breaker.fail_max == 9
    assert config.read_breaker.timeout_duration_s == 90.0
    assert config.write_breaker.fail_max == 4
    assert config.write_breaker.timeout_duration_s == 40.0


def test_load_line_service_config_falls_back_to_defaults_when_missing() -> None:
    """Execute test load line service config falls back to defaults when missing.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    fm = _FakeFileManager(payload=None, exists=False)
    config = load_line_service_config(
        path=Path("config/line_service.yaml"),
        file_manager=fm,
    )

    assert config.coalescing_delay_s == 10
    assert config.read_breaker.fail_max == 5
    assert config.read_breaker.timeout_duration_s == 60.0
    assert config.write_breaker.fail_max == 3
    assert config.write_breaker.timeout_duration_s == 30.0
