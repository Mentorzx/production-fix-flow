"""Tests for shared HPO local-optima diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pff.infrastructure.hpo.callbacks_internal.observers import StagnationDetector
from pff.infrastructure.hpo.optimization_diagnostics import (
    analyze_multi_region_evidence,
    analyze_stagnation,
    build_local_optima_diagnostics,
)


def _trial(
    trial_id: int,
    value: float,
    params: dict[str, object],
    *,
    warmstart: bool = False,
    state: str = "COMPLETE",
) -> dict[str, object]:
    return {
        "id": trial_id,
        "value": value,
        "params": params,
        "warmstart": warmstart,
        "state": state,
    }


def test_analyze_stagnation_detects_plateau() -> None:
    """Plateaus after enough trials should mark the study as stagnant."""
    scores = [0.4, 0.5, 0.6, 0.61, 0.611, 0.611, 0.611, 0.611, 0.611, 0.611, 0.611, 0.611]

    result = analyze_stagnation(scores, window_size=7, min_trials=10, improvement_threshold=0.02)

    assert result["stagnant"] is True
    assert result["status"] == "stagnant"
    assert result["trials_since_improvement"] == 7
    assert result["recent_range"] == pytest.approx(0.0)


def test_analyze_stagnation_preserves_exploration_signal() -> None:
    """Fresh improvements should keep the study in exploring mode."""
    scores = [0.2, 0.25, 0.31, 0.38, 0.43, 0.49, 0.54, 0.6, 0.66, 0.72, 0.78, 0.83]

    result = analyze_stagnation(scores, window_size=7, min_trials=10, improvement_threshold=0.02)

    assert result["stagnant"] is False
    assert result["status"] == "exploring"
    assert result["trials_since_improvement"] == 0
    assert result["recent_range"] is not None


def test_analyze_stagnation_requires_min_trials() -> None:
    """Plateaus below the minimum evidence threshold must stay inconclusive."""
    scores = [0.4, 0.5, 0.6, 0.61, 0.611, 0.611, 0.611, 0.611, 0.611]

    result = analyze_stagnation(scores, window_size=7, min_trials=10, improvement_threshold=0.02)

    assert result["stagnant"] is False
    assert result["status"] == "insufficient_evidence"


def test_multi_region_evidence_detects_two_competitive_regions() -> None:
    """Competitive elite signatures should surface multiple-region evidence."""
    search_space = {
        "lr": {"name": "FloatDistribution", "attributes": {"low": 1e-4, "high": 1.0, "log": True}},
        "layers": {"choices": [2, 4, 8]},
    }
    trials = [
        _trial(1, 0.9500, {"lr": 0.001, "layers": 2}),
        _trial(2, 0.9490, {"lr": 0.0012, "layers": 2}),
        _trial(3, 0.9480, {"lr": 0.0011, "layers": 2}),
        _trial(4, 0.9470, {"lr": 0.11, "layers": 4}),
        _trial(5, 0.9465, {"lr": 0.12, "layers": 4}),
        _trial(6, 0.9460, {"lr": 0.10, "layers": 4}),
        _trial(7, 0.8100, {"lr": 0.4, "layers": 8}),
        _trial(8, 0.7900, {"lr": 0.5, "layers": 8}),
        _trial(9, 0.7700, {"lr": 0.6, "layers": 8}),
        _trial(10, 0.7500, {"lr": 0.7, "layers": 8}),
        _trial(11, 0.7300, {"lr": 0.8, "layers": 8}),
        _trial(12, 0.7100, {"lr": 0.9, "layers": 8}),
    ]

    result = analyze_multi_region_evidence(trials, search_space, direction="maximize")

    assert result["detected"] is True
    assert result["status"] == "multiple_regions"
    assert result["region_count"] == 2
    assert len(result["summary_labels"]) == 2


def test_multi_region_evidence_falls_back_when_all_trials_are_warmstart() -> None:
    """Warmstart-only studies should still analyze completed trials when nothing else exists."""
    search_space = {
        "lr": {"name": "FloatDistribution", "attributes": {"low": 1e-4, "high": 1.0, "log": True}},
        "layers": {"choices": [2, 4, 8]},
    }
    trials = [
        _trial(1, 0.9500, {"lr": 0.001, "layers": 2}, warmstart=True),
        _trial(2, 0.9490, {"lr": 0.0012, "layers": 2}, warmstart=True),
        _trial(3, 0.9480, {"lr": 0.0011, "layers": 2}, warmstart=True),
        _trial(4, 0.9470, {"lr": 0.11, "layers": 4}, warmstart=True),
        _trial(5, 0.9465, {"lr": 0.12, "layers": 4}, warmstart=True),
        _trial(6, 0.9460, {"lr": 0.10, "layers": 4}, warmstart=True),
        _trial(7, 0.8100, {"lr": 0.4, "layers": 8}, warmstart=True),
        _trial(8, 0.7900, {"lr": 0.5, "layers": 8}, warmstart=True),
        _trial(9, 0.7700, {"lr": 0.6, "layers": 8}, warmstart=True),
        _trial(10, 0.7500, {"lr": 0.7, "layers": 8}, warmstart=True),
        _trial(11, 0.7300, {"lr": 0.8, "layers": 8}, warmstart=True),
        _trial(12, 0.7100, {"lr": 0.9, "layers": 8}, warmstart=True),
    ]

    result = analyze_multi_region_evidence(trials, search_space, direction="maximize")

    assert result["eligible_trials"] == 12
    assert result["detected"] is True


def test_multi_region_evidence_rejects_single_dominant_region() -> None:
    """A single elite signature should not be promoted as multi-region evidence."""
    search_space = {
        "lr": {"name": "FloatDistribution", "attributes": {"low": 1e-4, "high": 1.0, "log": True}},
        "layers": {"choices": [2, 4, 8]},
    }
    trials = [
        _trial(1, 0.9500, {"lr": 0.001, "layers": 2}),
        _trial(2, 0.9495, {"lr": 0.0011, "layers": 2}),
        _trial(3, 0.9490, {"lr": 0.0012, "layers": 2}),
        _trial(4, 0.9485, {"lr": 0.0013, "layers": 2}),
        _trial(5, 0.9480, {"lr": 0.0014, "layers": 2}),
        _trial(6, 0.9475, {"lr": 0.0015, "layers": 2}),
        _trial(7, 0.8100, {"lr": 0.4, "layers": 8}),
        _trial(8, 0.7900, {"lr": 0.5, "layers": 8}),
        _trial(9, 0.7700, {"lr": 0.6, "layers": 8}),
        _trial(10, 0.7500, {"lr": 0.7, "layers": 8}),
        _trial(11, 0.7300, {"lr": 0.8, "layers": 8}),
        _trial(12, 0.7100, {"lr": 0.9, "layers": 8}),
    ]

    result = analyze_multi_region_evidence(trials, search_space, direction="maximize")

    assert result["detected"] is False
    assert result["status"] == "single_region"


def test_multi_region_evidence_requires_enough_trials() -> None:
    """The multi-region heuristic should stay neutral with sparse evidence."""
    search_space = {"lr": {"low": 1e-4, "high": 1.0}}
    trials = [_trial(idx + 1, 0.8 - idx * 0.01, {"lr": 0.1 + idx * 0.01}) for idx in range(11)]

    result = analyze_multi_region_evidence(trials, search_space, direction="maximize")

    assert result["detected"] is False
    assert result["status"] == "insufficient_evidence"


def test_build_local_optima_diagnostics_merges_stagnation_and_regions() -> None:
    """The dashboard payload should merge stagnation and region evidence coherently."""
    search_space = {
        "lr": {"name": "FloatDistribution", "attributes": {"low": 1e-4, "high": 1.0, "log": True}},
        "layers": {"choices": [2, 4, 8]},
    }
    plateau_scores = [0.95, 0.95, 0.95, 0.95, 0.9495, 0.9490, 0.9485, 0.9480, 0.9475, 0.9470, 0.9465, 0.9460]
    trials = [
        _trial(1, plateau_scores[0], {"lr": 0.001, "layers": 2}),
        _trial(2, plateau_scores[1], {"lr": 0.0012, "layers": 2}),
        _trial(3, plateau_scores[2], {"lr": 0.0011, "layers": 2}),
        _trial(4, plateau_scores[3], {"lr": 0.11, "layers": 4}),
        _trial(5, plateau_scores[4], {"lr": 0.12, "layers": 4}),
        _trial(6, plateau_scores[5], {"lr": 0.10, "layers": 4}),
        _trial(7, plateau_scores[6], {"lr": 0.4, "layers": 8}),
        _trial(8, plateau_scores[7], {"lr": 0.5, "layers": 8}),
        _trial(9, plateau_scores[8], {"lr": 0.6, "layers": 8}),
        _trial(10, plateau_scores[9], {"lr": 0.7, "layers": 8}),
        _trial(11, plateau_scores[10], {"lr": 0.8, "layers": 8}),
        _trial(12, plateau_scores[11], {"lr": 0.9, "layers": 8}),
    ]

    result = build_local_optima_diagnostics(
        trials,
        search_space,
        direction="maximize",
        current_sampler="TPESampler",
    )

    assert result["status"] == "stagnant"
    assert result["current_sampler"] == "TPESampler"
    assert result["best_trial_id"] == 1
    assert "sampler.type='cmaes'" in result["recommended_action"]


def test_stagnation_detector_keeps_warning_contract() -> None:
    """The observer must still emit the stagnation warning with the established wording."""
    detector = StagnationDetector(window_size=7, min_trials=10, improvement_threshold=0.02)
    scores = [0.4, 0.5, 0.6, 0.61, 0.611, 0.611, 0.611, 0.611, 0.611, 0.611, 0.611, 0.611]

    with patch("pff.infrastructure.hpo.callbacks_internal.observers.logger.warning") as warning:
        for number, score in enumerate(scores):
            trial = MagicMock()
            trial.number = number
            detector.on_trial_complete(trial, score)

    assert detector.is_stagnant() is True
    assert warning.call_count == 1
    message = warning.call_args[0][0]
    assert "HPO stagnation detected after 11 trials." in message
    assert "Recommend: restart with sampler.type='cmaes'" in message
