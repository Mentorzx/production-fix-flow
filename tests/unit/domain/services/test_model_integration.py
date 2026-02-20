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

    def exists(self, _path: Path) -> bool:
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
