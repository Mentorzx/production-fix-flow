"""Tests for warm-start seed injection behavior."""

from pathlib import Path

import optuna

from pff.infrastructure.hpo.runner import HPOMemoryConfig, PersistentBestTrialMemory


def _build_memory(tmp_path: Path) -> PersistentBestTrialMemory:
    """Execute build memory.



    Args:

        tmp_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    config = HPOMemoryConfig(
        enabled=True,
        top_k_trials=3,
        warmstart_trials=3,
        storage_subdir="hpo_replay",
    )
    memory = PersistentBestTrialMemory(output_dir=tmp_path, config=config)
    distribution = optuna.distributions.FloatDistribution(0.0, 1.0)
    serialized = memory._serialize_distributions({"x": distribution})
    memory.entries = [
        {"value": 0.7, "params": {"x": 0.2}, "distributions": serialized},
        {"value": 0.6, "params": {"x": 0.4}, "distributions": serialized},
        {"value": 0.5, "params": {"x": 0.6}, "distributions": {}},
    ]
    memory.set_current_distributions({})
    return memory


def test_warmstart_injection_skips_attrs_for_enqueued(tmp_path: Path) -> None:
    """Ensure enqueued warmups are not flagged as warm-start trials."""
    study = optuna.create_study(direction="maximize")
    memory = _build_memory(tmp_path)

    injected = memory.warmstart_study(study)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert injected == 2
    assert len(completed) == 2
    assert all(t.user_attrs.get("warmstart") for t in completed)

    queued_trial = study.ask()
    assert not queued_trial.user_attrs.get("warmstart")
    # Use storage API instead of deprecated Trial.system_attrs (Optuna ≥3.1)
    sys_attrs = study._storage.get_trial_system_attrs(queued_trial._trial_id)
    assert not sys_attrs.get("warmstart_seed")
