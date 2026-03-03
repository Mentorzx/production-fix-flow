"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/services/test_model_integration.py

"""

from __future__ import annotations

from pathlib import Path

from pff.application.services.business_service.model_integration import ModelIntegration


class _FakeFileManager:
    def __init__(self, exists_result: bool) -> None:
        """Execute init.



        Args:

            exists_result: Input value used by this callable.

        """

        self.exists_result = exists_result

    def exists(self, _path: Path | str) -> bool:
        """Execute exists.



        Args:

            _path: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return self.exists_result


def test_model_integration_uses_injected_file_manager_for_model_lookup() -> None:
    """Execute test model integration uses injected file manager for model lookup.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    model_integration = ModelIntegration(file_manager=_FakeFileManager(exists_result=True))

    loaded = model_integration.load_models(Path("/tmp/outputs"))

    assert loaded is True
    assert model_integration.models_loaded is True
    assert model_integration.dslfm_checkpoint == Path("/tmp/outputs/dslfm/best_model.pt")


def test_model_integration_returns_false_when_checkpoint_missing() -> None:
    """Execute test model integration returns false when checkpoint missing.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    model_integration = ModelIntegration(file_manager=_FakeFileManager(exists_result=False))

    loaded = model_integration.load_models(Path("/tmp/outputs"))

    assert loaded is False
    assert model_integration.models_loaded is False
    assert model_integration.dslfm_checkpoint is None


def test_model_integration_uses_injected_config_loader() -> None:
    """ModelIntegration must read tuning values through injected loader."""

    calls: list[Path] = []

    def fake_loader(path: Path) -> dict[str, dict[str, float]]:
        calls.append(path)
        return {
            "violation_scoring": {
                "rate_floor": 9.0,
                "penalty_multiplier": 0.07,
            },
            "xai": {"dslfm_sample_size": 11},
            "scoring": {"dslfm_scale": 1.2, "dslfm_offset": 0.05},
        }

    model_integration = ModelIntegration(
        file_manager=_FakeFileManager(exists_result=False),
        config_loader=fake_loader,
    )

    assert calls
    assert model_integration._dslfm_sample_size == 11
    assert model_integration._dslfm_scale == 1.2
    assert model_integration._dslfm_offset == 0.05
    assert model_integration._penalty_calculator.config.rate_floor == 9.0
