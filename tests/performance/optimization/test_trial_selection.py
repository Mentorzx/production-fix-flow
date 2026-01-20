from __future__ import annotations

from types import SimpleNamespace

from pff.domain.hpo.scoring import (
    ScoreWeights,
    TimeScaleConfig,
    _default_weights,
    compute_score,
    rename_metric_keys,
)
from pff.domain.hpo.selection import _quality_only_weights, _time_tradeoff, select_best_trials


class DummyTrial:
    def __init__(self, number: int, mrr: float, duration: float) -> None:
        self.number = number
        self.params = {"p": number}
        self.value = mrr
        self.state = "COMPLETE"
        self.user_attrs = {
            "mrr": mrr,
            "best_mrr": mrr,
            "hits1": mrr,
            "hits3": mrr,
            "hits10": mrr,
            "duration": duration,
        }


def _custom_weights() -> ScoreWeights:
    time_scale = TimeScaleConfig(
        t_best=1.0,
        t_target=10.0,
        t_worst=100.0,
        score_at_best=0.9,
        score_at_target=0.5,
        score_at_worst=0.1,
    )
    return ScoreWeights(
        rank_block=0.5,
        clf_block=0.0,
        time_block=0.5,
        rank_metrics={"mrr": 1.0},
        clf_metrics={},
        duration_weight=1.0,
        eps=0.02,
        time_scale=time_scale,
    )


def test_select_best_trials_prefers_quality_and_speed_separately() -> None:
    trials = [
        DummyTrial(number=0, mrr=0.9, duration=80.0),  # melhor qualidade, lento
        DummyTrial(number=1, mrr=0.78, duration=5.0),  # mais rapido, score tempo-aware maior
    ]
    study = SimpleNamespace(trials=trials)
    weights = _custom_weights()

    selection = select_best_trials(study, weights=weights)

    assert selection["best_time_aware"]["trial_number"] == 1
    assert selection["best_quality"]["trial_number"] == 0
    assert selection["best_tradeoff"]["trial_number"] in {0, 1}
    # trade-off deve escolher o campeao rapido pela relacao score/log(tempo)
    assert selection["best_tradeoff"]["trial_number"] == 1


def test_select_best_trials_handles_empty_study() -> None:
    empty_selection = select_best_trials(None)
    assert empty_selection["best_time_aware"] is None
    assert empty_selection["best_quality"] is None
    assert empty_selection["best_tradeoff"] is None


def test_select_best_trials_handles_missing_classification_metrics() -> None:
    """Seleção deve ser resiliente quando métricas de classificação estão ausentes."""
    trials = [
        DummyTrial(number=0, mrr=0.65, duration=15.0),
        DummyTrial(number=1, mrr=0.7, duration=25.0),
    ]
    # Remover métricas de classificação
    for trial in trials:
        trial.user_attrs.pop("auc", None)
        trial.user_attrs.pop("pr_auc", None)
        trial.user_attrs.pop("precision", None)
        trial.user_attrs.pop("recall", None)

    study = SimpleNamespace(trials=trials)
    selection = select_best_trials(study)

    assert selection["best_time_aware"]["trial_number"] in {0, 1}
    assert selection["best_quality"]["trial_number"] in {0, 1}
    assert selection["best_tradeoff"]["trial_number"] in {0, 1}


def test_select_best_trials_handles_zero_duration() -> None:
    """Duracao zero deve ser protegida por eps e nao explodir trade-off."""
    trials = [
        DummyTrial(number=0, mrr=0.4, duration=0.0),
        DummyTrial(number=1, mrr=0.35, duration=5.0),
    ]
    study = SimpleNamespace(trials=trials)

    selection = select_best_trials(study)

    assert selection["best_tradeoff"]["tradeoff_score"] > 0.0
    assert selection["best_tradeoff"]["trial_number"] == 0


def test_select_best_trials_handles_missing_duration_key() -> None:
    """Falta de duração não deve quebrar a seleção; trade-off deve usar eps."""
    trials = [
        DummyTrial(number=0, mrr=0.6, duration=1.0),
        DummyTrial(number=1, mrr=0.65, duration=2.0),
    ]
    # Remove duration from user_attrs
    for trial in trials:
        trial.user_attrs.pop("duration", None)

    study = SimpleNamespace(trials=trials)
    selection = select_best_trials(study)

    assert selection["best_tradeoff"]["tradeoff_score"] > 0.0
    assert selection["best_tradeoff"]["trial_number"] in {0, 1}


def test_select_best_trials_handles_nan_metrics() -> None:
    """Métricas NaN devem ser normalizadas com eps, sem quebrar seleção."""
    trials = [
        DummyTrial(number=0, mrr=float("nan"), duration=3.0),
        DummyTrial(number=1, mrr=0.55, duration=4.0),
    ]
    study = SimpleNamespace(trials=trials)

    selection = select_best_trials(study)

    assert selection["best_tradeoff"]["tradeoff_score"] > 0.0
    assert selection["best_tradeoff"]["trial_number"] == 1


def test_select_best_trials_uses_tradeoff_over_all_trials() -> None:
    """Best trade-off deve considerar todos os trials, não só campeões de tempo/qualidade."""
    trials = [
        DummyTrial(number=0, mrr=0.9, duration=200.0),  # melhor qualidade, lento
        DummyTrial(number=1, mrr=0.6, duration=2.0),  # trade-off ideal (rápido)
        DummyTrial(number=2, mrr=0.75, duration=30.0),  # intermediário
    ]
    study = SimpleNamespace(trials=trials)

    weights = _default_weights()
    selection = select_best_trials(study, weights=weights)

    quality_weights = _quality_only_weights(weights)
    metrics_history = [rename_metric_keys(t.user_attrs) for t in trials]
    tradeoff_scores: dict[int, float] = {}
    for idx, metrics in enumerate(metrics_history):
        history = metrics_history[:idx] + metrics_history[idx + 1 :]
        score_quality, _, _ = compute_score(metrics, history, weights=quality_weights)
        tradeoff_scores[idx] = _time_tradeoff(
            score_quality, metrics.get("duration", 0.0), weights.eps
        )

    expected_best = max(tradeoff_scores, key=tradeoff_scores.get)
    assert selection["best_tradeoff"]["trial_number"] == trials[expected_best].number
