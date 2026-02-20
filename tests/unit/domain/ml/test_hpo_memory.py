"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/ml/test_hpo_memory.py

"""

import optuna

from pff.infrastructure.hpo.runner import HPOMemoryConfig, PersistentBestTrialMemory
from pff.shared.core.file_manager import FileManager


def test_persistent_memory_records_and_warmstarts(tmp_path):
    """Execute test persistent memory records and warmstarts.



    Args:

        tmp_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    output_dir = tmp_path / "outputs" / "kg_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_config = HPOMemoryConfig(
        enabled=True,
        top_k_trials=2,
        warmstart_trials=2,
        storage_subdir="memory",
        min_score_delta=0.0,
    )
    file_manager = FileManager()
    memory = PersistentBestTrialMemory(output_dir, memory_config, file_manager=file_manager)

    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        """Execute objective.



        Args:

            trial: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=2)

    for frozen_trial in study.trials:
        memory.record_trial(
            study,
            frozen_trial,
            {
                "ensemble_metrics": {"weighted_score": frozen_trial.value},
                "model_metrics": {},
            },
        )

    # Expect DataFrame when reading parquet with return_native=True
    saved_payload = file_manager.read(memory.memory_path, return_native=True)
    assert saved_payload is not None
    import polars as pl

    assert isinstance(saved_payload, pl.DataFrame)
    assert len(saved_payload) == 2

    new_study = optuna.create_study(direction="maximize")
    added = memory.warmstart_study(new_study)

    assert added > 0
    # Enqueued trials have value=None, but params should match the best from previous study
    # Since objective(x) returns x, params['x'] == best_value

    # Run the new study to process enqueued trials
    new_study.optimize(objective, n_trials=added)

    assert any(trial.value == study.best_value for trial in new_study.trials)
