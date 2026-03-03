"""Regression tests for IntelligentPreprocessor file-manager port injection."""

from __future__ import annotations

from pff.application.services.intelligent_preprocessor import IntelligentPreprocessor


class _FakeFileManager:
    def __init__(self) -> None:
        self.saved_payload = None
        self.saved_path = None

    def save(self, data, path, **_kwargs):  # noqa: ANN001
        self.saved_payload = data
        self.saved_path = path


def test_intelligent_preprocessor_uses_injected_file_manager_port() -> None:
    """Manifest generation must call save on the injected file-manager port."""
    fake_manager = _FakeFileManager()
    preprocessor = IntelligentPreprocessor(file_manager=fake_manager)

    tasks = [{"msisdn": "5511999999999", "sequence": "baseline"}]
    preprocessor.generate_manifest_file(tasks, "test_manifest.yaml", exec_id="exec-1")

    assert fake_manager.saved_payload is not None
    assert fake_manager.saved_payload["execution_id"] == "exec-1"
    assert fake_manager.saved_payload["tasks"] == tasks
    assert fake_manager.saved_path is not None
