from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import optuna
import pytest

from pff.infrastructure.hpo.config_loader import load_storage_settings
from pff.infrastructure.hpo.runner import _write_checkpoint
from pff.infrastructure.hpo.storage import create_optuna_storage
from pff.infrastructure.hpo.trials.study import create_study_and_run
from pff.shared.core.file_manager import FileManager
from pff.shared.ops.global_interrupt_manager import get_interrupt_manager


def test_checkpoint_write_bypasses_interrupt_short_circuit(tmp_path: Path) -> None:
    manager = get_interrupt_manager()
    manager.reset()
    manager.force_stop("test-interrupt")

    checkpoint_path = tmp_path / "checkpoint.json"
    payload = {"status": "running", "completed_trials": 0}

    mock_store = MagicMock()
    mock_store.upsert_checkpoint = AsyncMock(return_value=None)

    _write_checkpoint(checkpoint_path, payload, checkpoint_key="k", store=mock_store)

    mock_store.upsert_checkpoint.assert_called_once()
    manager.reset()


def test_create_study_and_run_resumes_from_existing_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = get_interrupt_manager()
    manager.reset()

    storage_path = tmp_path / "optuna_study.db"
    checkpoint_path = tmp_path / "checkpoint.json"
    output_dir = tmp_path / "run"
    fm = FileManager()

    def objective(trial: optuna.trial.Trial) -> float:
        return float(trial.suggest_float("x", 0.0, 1.0))

    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.study.load_optuna_settings",
        lambda _: {
            "tpe": {
                "n_startup_trials": 1,
                "multivariate": False,
                "group": False,
                "constant_liar": False,
                "seed": 42,
            },
            "hyperband": {"min_resource": 1, "max_resource": 2, "reduction_factor": 2},
        },
    )
    monkeypatch.setattr(
        "pff.infrastructure.hpo.trials.study.load_live_plot_settings",
        lambda _: {"enabled": False},
    )

    storage_backend = str(load_storage_settings(fm).get("backend", "sqlite")).lower()
    study_name = "resume_test"
    suffix = hashlib.sha1(str(tmp_path).encode("utf-8")).hexdigest()[:8]
    study_name = f"{study_name}_{suffix}"
    if storage_backend in {"postgres", "postgresql", "rdb", "rdbstorage"}:
        storage, storage_url = create_optuna_storage(storage_path=storage_path, file_manager=fm)
        study_not_found = getattr(optuna.exceptions, "StudyNotFound", KeyError)
        try:
            optuna.delete_study(
                study_name=study_name,
                storage=storage if storage is not None else storage_url,
            )
        except (KeyError, study_not_found):
            pass

    first = create_study_and_run(
        study_name=study_name,
        storage_path=storage_path,
        checkpoint_path=checkpoint_path,
        checkpoint_key="k",
        checkpoint_store=MagicMock(),
        output_dir=output_dir,
        work_dir=output_dir,
        n_trials=2,
        expected_trials=2,
        resume_mode=False,
        checkpoint_data=None,
        hpo_memory_config={},
        trial_memory=SimpleNamespace(),
        warmstart_callback=None,
        objective_fn=objective,
        artifact_manager=MagicMock(store=MagicMock(), study_name=study_name),
        enable_mlflow=False,
        file_manager=fm,
    )

    expects_file = storage_backend not in {
        "postgres",
        "postgresql",
        "rdb",
        "rdbstorage",
        "grpc",
        "grpc_proxy",
    }
    if expects_file:
        assert fm.exists(storage_path)
    assert first["n_trials"] == 2

    # Manually write checkpoint file since _write_checkpoint now only uses Postgres
    fm.save({"status": "running", "completed_trials": 2}, checkpoint_path)

    second = create_study_and_run(
        study_name=study_name,
        storage_path=storage_path,
        checkpoint_path=checkpoint_path,
        checkpoint_key="k",
        checkpoint_store=MagicMock(),
        output_dir=output_dir,
        work_dir=output_dir,
        n_trials=2,
        expected_trials=4,
        resume_mode=True,
        checkpoint_data=fm.read(checkpoint_path, return_native=True),
        hpo_memory_config={},
        trial_memory=SimpleNamespace(),
        warmstart_callback=None,
        objective_fn=objective,
        artifact_manager=MagicMock(store=MagicMock(), study_name=study_name),
        enable_mlflow=False,
        file_manager=fm,
    )

    assert second["n_trials"] == 4
    if expects_file:
        assert fm.exists(storage_path)
    manager.reset()
