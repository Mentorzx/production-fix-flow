"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/infrastructure/cleanup/test_training_artifacts_clean_command.py

"""

from pathlib import Path

from pff.infrastructure.cleanup.commands.filesystem import TrainingArtifactsCleanCommand
from pff.shared.core.config import settings


def test_training_artifacts_clean_command_removes_known_temp_files(
    tmp_path: Path, monkeypatch
) -> None:
    """Execute test training artifacts clean command removes known temp files.



    Args:

        tmp_path: Input value used by this callable.

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    outputs_dir = tmp_path / "outputs"
    root_dir = tmp_path / "root"
    dslfm_dir = outputs_dir / "dslfm"
    dslfm_dir.mkdir(parents=True)
    root_dir.mkdir(parents=True)

    target_files = [
        dslfm_dir / "temp_trial_01",
        dslfm_dir / "model_temp.yaml",
        outputs_dir / "temp_config_trial_1.yaml",
        root_dir / "temp_config_trial_2.yaml",
    ]
    for file_path in target_files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("tmp", encoding="utf-8")

    monkeypatch.setattr(settings, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(settings, "ROOT_DIR", root_dir)

    command = TrainingArtifactsCleanCommand()
    command.execute()

    for file_path in target_files:
        assert not file_path.exists()
