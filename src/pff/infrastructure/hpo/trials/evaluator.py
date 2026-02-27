"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/hpo/trials/evaluator.py

"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from pff.domain.learning.dslfm.dslfm_kgc import load_dslfm_kgc_settings
from pff.domain.learning.ml.training_observer import TrainingEvent, TrainingObserver
from pff.infrastructure.hpo.callbacks_internal.visualizers import LiveTrainingObserver
from pff.infrastructure.persistence.model_persistence import FileSystemModelPersistence
from pff.domain.learning.ml.metrics import (
    BinaryMetricsBackend,
    BinaryMetricsInputs,
    compute_binary_metrics,
)
from pff.shared import logger
from pff.shared.core.file_manager import FileManager
from pff.shared.system.cuda import is_cuda_available


@dataclass(frozen=True)
class _BinaryMetricsBackend:
    accuracy_score: Callable[[np.ndarray, np.ndarray], float]
    auc: Callable[[np.ndarray, np.ndarray], float]
    average_precision_score: Callable[[np.ndarray, np.ndarray], float]
    matthews_corrcoef: Callable[[np.ndarray, np.ndarray], float]
    precision_recall_curve: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
    roc_auc_score: Callable[[np.ndarray, np.ndarray], float]


def _sk_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Execute sk average precision.



    Args:

        y_true: Input value used by this callable.

        y_score: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    from sklearn import metrics as sk_metrics

    return float(sk_metrics.average_precision_score(y_true, y_score))


def _sk_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Execute sk roc auc.



    Args:

        y_true: Input value used by this callable.

        y_score: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    from sklearn import metrics as sk_metrics

    return float(sk_metrics.roc_auc_score(y_true, y_score))


class BinaryMetricsObserver(TrainingObserver):
    """Observer that calculates binary classification metrics during training.

    This observer calls _compute_binary_metrics on evaluation epochs to provide
    metrics like MCC and AUC to the training progress logs and HPO pruners.
    """

    def __init__(
        self,
        manager: Any,
        valid_triples: np.ndarray,
        params: dict[str, Any],
        compute_every: int = 1,
    ) -> None:
        """Execute init.



        Args:

            manager: Input value used by this callable.

            valid_triples: Input value used by this callable.

            params: Input value used by this callable.

            compute_every: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.manager = manager
        self.valid_triples = valid_triples
        self.params = params
        self.compute_every = compute_every
        self._eval_count = 0

    def on_event(self, event: TrainingEvent) -> None:
        """Execute on event.



        Args:

            event: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if event.event_type == "epoch_end":
            if "mrr" in event.metrics:
                self._eval_count += 1
                if self._eval_count % self.compute_every == 0:
                    try:
                        current_params = self.params.copy()
                        if "binary_metrics_max_samples" not in current_params:
                            current_params["binary_metrics_max_samples"] = 2000

                        binary_metrics = _compute_binary_metrics(
                            self.manager,
                            self.valid_triples,
                            num_negatives=int(
                                current_params.get("binary_negatives") or 10
                            ),
                            seed=int(current_params.get("seed") or 1337) + event.epoch,
                            params=current_params,
                        )
                        event.metrics.update(binary_metrics)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to compute binary metrics at epoch {event.epoch}: {exc}"
                        )


def _find_scoring_model(model: Any) -> Any | None:
    """Execute find scoring model.



    Args:

        model: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if model is None:
        return None
    if hasattr(model, "score_triples_batch"):
        return model
    for attr in ("base_model", "_orig_mod", "model"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            found = _find_scoring_model(candidate)
            if found is not None:
                return found
    return None


def _resolve_binary_backend() -> BinaryMetricsBackend | None:
    """Execute resolve binary backend.



    Returns:

        Return value produced by the callable.

    """

    try:
        from sklearn import metrics as sk_metrics
        from pff_rust import (
            fast_average_precision_score,
            fast_matthews_corrcoef,
            fast_precision_recall_curve,
            fast_roc_auc_score,
        )

        return cast(
            BinaryMetricsBackend,
            _BinaryMetricsBackend(
                accuracy_score=sk_metrics.accuracy_score,
                auc=sk_metrics.auc,
                average_precision_score=fast_average_precision_score,
                matthews_corrcoef=fast_matthews_corrcoef,
                precision_recall_curve=fast_precision_recall_curve,
                roc_auc_score=fast_roc_auc_score,
            ),
        )
    except Exception:
        try:
            from sklearn import metrics as sk_metrics

            return cast(
                BinaryMetricsBackend,
                _BinaryMetricsBackend(
                    accuracy_score=sk_metrics.accuracy_score,
                    auc=sk_metrics.auc,
                    average_precision_score=_sk_average_precision,
                    matthews_corrcoef=sk_metrics.matthews_corrcoef,
                    precision_recall_curve=sk_metrics.precision_recall_curve,
                    roc_auc_score=_sk_roc_auc,
                ),
            )
        except Exception as exc:
            logger.warning(f"Binary metrics skipped: scikit-learn unavailable: {exc}")
            return None


def _load_binary_metric_runtime_config(
    *,
    params: dict[str, Any] | None,
    default_num_negatives: int,
) -> tuple[bool, int, Any, int, str, float]:
    """Execute load binary metric runtime config.



    Args:

        params: Input value used by this callable.

        default_num_negatives: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    from pff.infrastructure.hpo.config_loader import load_optimization_config

    settings = load_optimization_config(file_manager=FileManager())
    binary_cfg = settings.get("binary_metrics", {})
    if not isinstance(binary_cfg, dict):
        binary_cfg = {}
    params = params or {}
    enabled = bool(
        params.get("binary_metrics_enabled", binary_cfg.get("enabled", True))
    )
    num_negatives = int(
        params.get(
            "binary_metrics_num_negatives",
            binary_cfg.get("num_negatives", default_num_negatives),
        )
    )
    max_samples = params.get(
        "binary_metrics_max_samples", binary_cfg.get("max_samples", 5000)
    )
    batch_size = int(
        params.get("binary_metrics_batch_size", binary_cfg.get("batch_size", 4096))
    )
    device_pref = str(
        params.get("binary_metrics_device", binary_cfg.get("device", "auto"))
    ).lower()
    free_ratio_min = float(
        params.get(
            "binary_metrics_cuda_free_ratio_min",
            binary_cfg.get("cuda_free_ratio_min", 0.15),
        )
    )
    return enabled, num_negatives, max_samples, batch_size, device_pref, free_ratio_min


def _resolve_num_entities(scoring_model: Any) -> int:
    """Execute resolve num entities.



    Args:

        scoring_model: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    num_entities = getattr(scoring_model, "num_entities", 0)
    if num_entities > 0:
        return int(num_entities)
    model_config = getattr(scoring_model, "config", None)
    return int(getattr(model_config, "num_entities", 0))


def _prepare_binary_samples(
    *,
    val_triples: np.ndarray,
    num_entities: int,
    num_negatives: int,
    max_samples: Any,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute prepare binary samples.



    Args:

        val_triples: Input value used by this callable.

        num_entities: Input value used by this callable.

        num_negatives: Input value used by this callable.

        max_samples: Input value used by this callable.

        seed: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    rng = np.random.default_rng(seed)
    val_triples_arr = np.asarray(val_triples, dtype=np.int64)
    n_pos = int(val_triples_arr.shape[0])
    if (
        isinstance(max_samples, (int, np.integer))
        and max_samples > 0
        and n_pos > max_samples
    ):
        sampled_idx = rng.choice(n_pos, size=int(max_samples), replace=False)
        val_triples_arr = val_triples_arr[sampled_idx]
        n_pos = int(val_triples_arr.shape[0])
    n_total = n_pos * int(num_negatives)
    if n_total <= 0:
        return (
            val_triples_arr,
            np.empty((0, 3), dtype=np.int64),
            np.empty((0,), dtype=bool),
        )

    negatives_arr = np.repeat(val_triples_arr, int(num_negatives), axis=0)
    choice = rng.random(n_total) < 0.5
    rand_entities = rng.integers(0, num_entities, size=n_total, dtype=np.int64)
    negatives_arr[choice, 0] = rand_entities[choice]
    negatives_arr[~choice, 2] = rand_entities[~choice]
    return val_triples_arr, negatives_arr, choice


def _repair_negative_overlap(
    *,
    negatives_arr: np.ndarray,
    choice: np.ndarray,
    filter_arrays: Any,
    num_entities: int,
    seed: int,
) -> None:
    """Execute repair negative overlap.



    Args:

        negatives_arr: Input value used by this callable.

        choice: Input value used by this callable.

        filter_arrays: Input value used by this callable.

        num_entities: Input value used by this callable.

        seed: Input value used by this callable.

    """

    if not isinstance(filter_arrays, dict) or not filter_arrays:
        return
    rng = np.random.default_rng(seed + 1)
    max_attempts = 5
    for idx in range(negatives_arr.shape[0]):
        h, r, t = negatives_arr[idx]
        tails = filter_arrays.get((int(h), int(r)))
        if tails is None or not np.any(tails == t):
            continue
        for _ in range(max_attempts):
            replacement = rng.integers(0, num_entities, dtype=np.int64)
            if choice[idx]:
                h = replacement
            else:
                t = replacement
            tails = filter_arrays.get((int(h), int(r)))
            if tails is None or not np.any(tails == t):
                negatives_arr[idx, 0] = h
                negatives_arr[idx, 2] = t
                break


def _resolve_target_device(
    *,
    manager: Any,
    scoring_model: Any,
    device_pref: str,
    free_ratio_min: float,
) -> torch.device:
    """Execute resolve target device.



    Args:

        manager: Input value used by this callable.

        scoring_model: Input value used by this callable.

        device_pref: Input value used by this callable.

        free_ratio_min: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    device = getattr(getattr(manager, "model", None), "device", None)
    if device is None:
        try:
            device = next(scoring_model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        return torch.device("cuda")
    if is_cuda_available():
        from pff.shared.system.resource_manager import get_cuda_free_ratio

        free_ratio = get_cuda_free_ratio(default=0.0)
        if free_ratio is not None and free_ratio >= free_ratio_min:
            return torch.device("cuda")
    return torch.device("cpu")


def _score_tensor_batches(
    *,
    scoring_model: Any,
    tensor: torch.Tensor,
    batch_size: int,
    target_device: torch.device,
) -> torch.Tensor:
    """Execute score tensor batches.



    Args:

        scoring_model: Input value used by this callable.

        tensor: Input value used by this callable.

        batch_size: Input value used by this callable.

        target_device: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if tensor.numel() == 0:
        return torch.empty((0,), device=target_device)
    outputs: list[torch.Tensor] = []
    total = tensor.shape[0]
    step = max(int(batch_size), 1)
    for start in range(0, total, step):
        chunk = tensor[start : start + step]
        outputs.append(scoring_model.score_triples_batch(chunk))
    return torch.cat(outputs, dim=0)


def _run_binary_scoring(
    *,
    manager: Any,
    scoring_model: Any,
    val_triples_arr: np.ndarray,
    negatives_arr: np.ndarray,
    batch_size: int,
    device_pref: str,
    free_ratio_min: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Execute run binary scoring.



    Args:

        manager: Input value used by this callable.

        scoring_model: Input value used by this callable.

        val_triples_arr: Input value used by this callable.

        negatives_arr: Input value used by this callable.

        batch_size: Input value used by this callable.

        device_pref: Input value used by this callable.

        free_ratio_min: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    target_device = _resolve_target_device(
        manager=manager,
        scoring_model=scoring_model,
        device_pref=device_pref,
        free_ratio_min=free_ratio_min,
    )
    pos_tensor = torch.tensor(val_triples_arr, device=target_device, dtype=torch.long)
    neg_tensor = torch.tensor(negatives_arr, device=target_device, dtype=torch.long)

    original_device: torch.device | None = None
    try:
        original_device = next(scoring_model.parameters()).device
    except Exception:
        original_device = getattr(getattr(manager, "model", None), "device", None)
    moved_model = False
    if getattr(scoring_model, "device", None) != target_device:
        try:
            scoring_model.to(target_device)
            moved_model = True
        except Exception as exc:
            logger.warning(
                f"Failed to move model to {target_device} for binary metrics: {exc}"
            )
            if original_device is not None:
                target_device = original_device
                pos_tensor = pos_tensor.to(target_device)
                neg_tensor = neg_tensor.to(target_device)

    with torch.no_grad():
        inference_start = time.perf_counter()
        pos_scores = _score_tensor_batches(
            scoring_model=scoring_model,
            tensor=pos_tensor,
            batch_size=batch_size,
            target_device=target_device,
        )
        neg_scores = _score_tensor_batches(
            scoring_model=scoring_model,
            tensor=neg_tensor,
            batch_size=batch_size,
            target_device=target_device,
        )
        inference_elapsed = time.perf_counter() - inference_start

    total_triples_scored = len(pos_tensor) + len(neg_tensor)
    inference_latency_ms = (
        (inference_elapsed * 1000) / total_triples_scored
        if total_triples_scored > 0
        else 0.0
    )

    if moved_model and original_device is not None:
        try:
            scoring_model.to(original_device)
        except Exception as exc:
            logger.warning(f"Failed to restore model to {original_device}: {exc}")

    return pos_scores, neg_scores, float(inference_latency_ms)


def _compute_binary_loss(raw_scores: torch.Tensor, labels: np.ndarray) -> float:
    """Execute compute binary loss.



    Args:

        raw_scores: Input value used by this callable.

        labels: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    try:
        labels_t = torch.tensor(labels, dtype=torch.float32, device=raw_scores.device)
        return float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                raw_scores.float(),
                labels_t,
            ).item()
        )
    except Exception as exc:
        logger.debug(f"Failed to compute binary loss: {exc}")
        return 0.0


def _calibrate_probability_scores(
    raw_scores: torch.Tensor, labels: np.ndarray
) -> np.ndarray:
    """Execute calibrate probability scores.



    Args:

        raw_scores: Input value used by this callable.

        labels: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    try:
        scores_t = raw_scores.clone().detach().float()
        targets_t = torch.tensor(labels, dtype=torch.float32, device=scores_t.device)
        a = torch.zeros((), device=scores_t.device, requires_grad=True)
        bias_t = torch.zeros((), device=scores_t.device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [a, bias_t], max_iter=25, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            """Execute closure.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            optimizer.zero_grad()
            logits = a * scores_t + bias_t
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets_t
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        calibrated_logits = a.detach() * scores_t + bias_t.detach()
        return torch.sigmoid(calibrated_logits).cpu().numpy()
    except Exception as exc:
        logger.debug(f"Platt calibration failed: {exc}")
        return torch.sigmoid(raw_scores).cpu().numpy()


def _maybe_dump_binary_inputs(
    *,
    dump_dir: str | None,
    seed: int,
    num_negatives: int,
    labels: np.ndarray,
    prob_scores: np.ndarray,
    val_triples_arr: np.ndarray,
    negatives_arr: np.ndarray,
) -> None:
    """Execute maybe dump binary inputs.



    Args:

        dump_dir: Input value used by this callable.

        seed: Input value used by this callable.

        num_negatives: Input value used by this callable.

        labels: Input value used by this callable.

        prob_scores: Input value used by this callable.

        val_triples_arr: Input value used by this callable.

        negatives_arr: Input value used by this callable.

    """

    if not dump_dir:
        return
    try:
        from pff.shared.core.config import settings as app_settings

        dump_path = Path(dump_dir)
        if not dump_path.is_absolute():
            dump_path = app_settings.OUTPUTS_DIR / dump_path
        FileManager.ensure_dir(dump_path)
        stamp = int(time.time() * 1000)
        base = dump_path / f"binary_metrics_inputs_{stamp}"
        fm = FileManager()
        fm.save(np.asarray(labels), base.with_suffix(".labels.npy"))
        fm.save(np.asarray(prob_scores), base.with_suffix(".scores.npy"))
        fm.save(np.asarray(val_triples_arr), base.with_suffix(".pos_triples.npy"))
        fm.save(np.asarray(negatives_arr), base.with_suffix(".neg_triples.npy"))
        fm.save(
            {
                "seed": int(seed),
                "num_negatives": int(num_negatives),
                "n_labels": int(len(labels)),
            },
            base.with_suffix(".meta.json"),
        )
    except Exception as exc:
        logger.warning(f"Failed to dump binary metrics inputs: {exc}")


def _log_binary_metrics_summary(
    metrics: dict[str, float],
    binary_loss: float,
    inference_latency_ms: float,
    n_labels: int,
) -> None:
    """Execute log binary metrics summary.



    Args:

        metrics: Input value used by this callable.

        binary_loss: Input value used by this callable.

        inference_latency_ms: Input value used by this callable.

        n_labels: Input value used by this callable.

    """

    if metrics.get("auc", 0) > 0 or metrics.get("mcc", 0) > 0:
        logger.info(
            f"Metricas binarias (N={n_labels}): loss={binary_loss:.4f} "
            f"AUC={metrics.get('auc', 0):.4f} "
            f"PR_AUC={metrics.get('pr_auc', 0):.4f} MCC={metrics.get('mcc', 0):.4f} "
            f"AP={metrics.get('ap', 0):.4f} "
            f"Thresh={metrics.get('decision_threshold', 0):.3f} "
            f"InfLatency={inference_latency_ms:.4f}ms/triple"
        )
        return
    logger.warning("Classification metrics (MCC/AUC) are zero or failed.")


def _resolve_scoring_model(manager: Any) -> Any | None:
    """Execute resolve scoring model.



    Args:

        manager: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    model = getattr(manager, "model", None)
    if model is None:
        logger.warning("Binary metrics skipped: manager.model is None")
        return None
    scoring_model = _find_scoring_model(model)
    if scoring_model is None:
        logger.warning(
            f"Binary metrics skipped: model {type(model).__name__} does not expose 'score_triples_batch'"
        )
        return None
    return scoring_model


def _resolve_binary_runtime(
    *,
    params: dict[str, Any] | None,
    default_num_negatives: int,
) -> tuple[BinaryMetricsBackend, int, Any, int, str, float] | None:
    """Execute resolve binary runtime.



    Args:

        params: Input value used by this callable.

        default_num_negatives: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    backend = _resolve_binary_backend()
    if backend is None:
        return None
    enabled, num_negatives, max_samples, batch_size, device_pref, free_ratio_min = (
        _load_binary_metric_runtime_config(
            params=params,
            default_num_negatives=default_num_negatives,
        )
    )
    if not enabled:
        return None
    return backend, num_negatives, max_samples, batch_size, device_pref, free_ratio_min


def _prepare_binary_metric_inputs(
    *,
    manager: Any,
    scoring_model: Any,
    val_triples: np.ndarray,
    num_negatives: int,
    max_samples: Any,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Execute prepare binary metric inputs.



    Args:

        manager: Input value used by this callable.

        scoring_model: Input value used by this callable.

        val_triples: Input value used by this callable.

        num_negatives: Input value used by this callable.

        max_samples: Input value used by this callable.

        seed: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    num_entities = _resolve_num_entities(scoring_model)
    if num_entities <= 0:
        logger.warning("Binary metrics skipped: num_entities could not be determined")
        return None
    val_triples_arr, negatives_arr, choice = _prepare_binary_samples(
        val_triples=np.asarray(val_triples, dtype=np.int64),
        num_entities=num_entities,
        num_negatives=int(num_negatives),
        max_samples=max_samples,
        seed=seed,
    )
    if negatives_arr.size == 0:
        return None
    _repair_negative_overlap(
        negatives_arr=negatives_arr,
        choice=choice,
        filter_arrays=getattr(manager, "_filter_arrays", None),
        num_entities=num_entities,
        seed=seed,
    )
    return val_triples_arr, negatives_arr, choice


def _finalize_binary_metrics(
    *,
    backend: BinaryMetricsBackend,
    num_negatives: int,
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    inference_latency_ms: float,
    seed: int,
    val_triples_arr: np.ndarray,
    negatives_arr: np.ndarray,
) -> dict[str, float]:
    """Execute finalize binary metrics.



    Args:

        backend: Input value used by this callable.

        num_negatives: Input value used by this callable.

        pos_scores: Input value used by this callable.

        neg_scores: Input value used by this callable.

        inference_latency_ms: Input value used by this callable.

        seed: Input value used by this callable.

        val_triples_arr: Input value used by this callable.

        negatives_arr: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    raw_scores = torch.cat([pos_scores, neg_scores]).cpu()
    labels = np.concatenate(
        [
            np.ones(len(pos_scores), dtype=np.int64),
            np.zeros(len(neg_scores), dtype=np.int64),
        ]
    )
    binary_loss = _compute_binary_loss(raw_scores, labels)
    prob_scores = _calibrate_probability_scores(raw_scores, labels)

    metrics = compute_binary_metrics(
        BinaryMetricsInputs(labels=labels, prob_scores=prob_scores),
        backend=backend,
    )

    _maybe_dump_binary_inputs(
        dump_dir=os.getenv("PFF_BINARY_METRICS_DUMP_DIR"),
        seed=seed,
        num_negatives=int(num_negatives),
        labels=labels,
        prob_scores=prob_scores,
        val_triples_arr=val_triples_arr,
        negatives_arr=negatives_arr,
    )

    metrics["inference_latency"] = float(inference_latency_ms)
    metrics["binary_loss"] = float(binary_loss)
    _log_binary_metrics_summary(metrics, binary_loss, inference_latency_ms, len(labels))
    return metrics


def _compute_binary_metrics(
    manager: Any,
    val_triples: np.ndarray | None,
    *,
    num_negatives: int = 20,
    seed: int = 1337,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute binary metrics (ROC-AUC / PR-AUC / precision / recall / F1).

    This helper samples random corruptions from the full entity set to build a
    lightweight binary benchmark. It prefers a scoring-capable model
    (`score_triples_batch`) when the manager wraps multiple namespaces.

    If scikit-learn is unavailable, it returns an empty dict without failing
    the trial.

    Args:
        manager: Trained DSLFM manager instance with a `model` attribute.
        val_triples: Validation triples array with shape [N, 3].
        num_negatives: Negatives per positive triple.
        seed: RNG seed for deterministic negative sampling.

    Returns:
        Dictionary with computed metrics, or an empty dict when unavailable.
    """
    if val_triples is None or len(val_triples) == 0:
        return {}

    scoring_model = _resolve_scoring_model(manager)
    if scoring_model is None:
        return {}

    runtime = _resolve_binary_runtime(
        params=params,
        default_num_negatives=num_negatives,
    )
    if runtime is None:
        return {}

    backend, num_negatives, max_samples, batch_size, device_pref, free_ratio_min = (
        runtime
    )
    prepared = _prepare_binary_metric_inputs(
        manager=manager,
        scoring_model=scoring_model,
        val_triples=val_triples,
        num_negatives=int(num_negatives),
        max_samples=max_samples,
        seed=seed,
    )
    if prepared is None:
        return {}
    val_triples_arr, negatives_arr, _choice = prepared

    pos_scores, neg_scores, inference_latency_ms = _run_binary_scoring(
        manager=manager,
        scoring_model=scoring_model,
        val_triples_arr=val_triples_arr,
        negatives_arr=negatives_arr,
        batch_size=batch_size,
        device_pref=device_pref,
        free_ratio_min=free_ratio_min,
    )
    return _finalize_binary_metrics(
        backend=backend,
        num_negatives=int(num_negatives),
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        inference_latency_ms=inference_latency_ms,
        seed=seed,
        val_triples_arr=val_triples_arr,
        negatives_arr=negatives_arr,
    )


def _build_hpo_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """Translate HPO search-space parameter names into canonical config keys.

    HPO trials use human-friendly names (e.g. ``embedding_dim``,
    ``adversarial_temperature``, ``dslfm_epochs``) that don't map 1-to-1 to the
    underlying ``DSLFMKGCConfig`` / ``KGCTrainingConfig`` field names.  This
    helper normalizes them so that ``build_dslfm_configs`` can apply them as
    straight overrides.

    Args:
        params: Raw trial parameters dictionary.

    Returns:
        Cleaned overrides dict suitable for ``build_dslfm_configs``.
    """
    overrides: dict[str, Any] = dict(params)

    if "embedding_dim" in overrides:
        try:
            dim = int(overrides.pop("embedding_dim"))
            overrides["entity_dim"] = dim
            overrides["feature_dim"] = dim
        except Exception as exc:
            logger.warning(
                f"Invalid embedding_dim, expected int: value={params.get('embedding_dim')!r} error={exc}"
            )

    if "adversarial_temperature" in overrides:
        overrides.setdefault(
            "sampler_temperature", overrides.pop("adversarial_temperature")
        )

    if "dslfm_epochs" in overrides:
        overrides.setdefault("epochs", overrides.pop("dslfm_epochs"))

    if "kl_weight" in overrides:
        try:
            kl_value = float(overrides["kl_weight"])
        except Exception as exc:
            logger.warning(
                f"Invalid kl_weight, expected float: value={overrides.get('kl_weight')!r} error={exc}"
            )
        else:
            # Keep scheduler-consistent behavior: sampled kl_weight should drive the warmup ceiling.
            overrides.setdefault("max_kl_weight", kl_value)
            overrides.setdefault("min_kl_weight", 0.0)

    if "max_circuit_depth" in overrides:
        overrides.setdefault("max_circuit_depth", overrides["max_circuit_depth"])

    return overrides


def _train_dslfm_kgc_model(
    params: dict[str, Any],
    model_dir: Path,
    train_triples: np.ndarray,
    valid_triples: np.ndarray,
    num_entities: int,
    num_relations: int,
    relation_names: list[str] | None = None,
    *,
    use_bert: bool = True,
    trial: Any | None = None,
    trial_number_override: int | None = None,
    cv_fold_id: int | None = None,
) -> tuple[dict[str, Any], Path]:
    """Train DSLFM-KGC with HPO hyperparameters and return metrics + checkpoint.

    This function merges trial parameters with the project-level DSLFM-KGC YAML
    defaults via ``build_dslfm_configs`` (single source of truth).  For scoring
    stability, reranking is disabled (forced ``rerank_top_k=0``) because it has
    known MRR regressions in the current pipeline.

    Args:
        params: Trial hyperparameters dictionary.
        model_dir: Directory to store artifacts/checkpoints.
        train_triples: Training triples with shape [N, 3].
        valid_triples: Validation triples with shape [M, 3].
        num_entities: Number of entities.
        num_relations: Number of relations.
        relation_names: Optional relation names used by the BERT relation encoder.
        use_bert: Whether to enable the BERT relation encoder.
        trial: Optional Optuna trial for reporting/pruning.
        trial_number_override: Override for trial number (cross-validation).
        cv_fold_id: Optional cross-validation fold ID.

    Returns:
        Tuple with (training stats dict, checkpoint path).
    """
    from pff.domain.learning.dslfm.kgc_manager import (
        DSLFMKGCManager,
        build_dslfm_configs,
    )
    from pff.domain.learning.ml.training_observer import ConsoleObserver

    dslfm_settings = load_dslfm_kgc_settings(FileManager(), params.get("config_path"))

    overrides = _build_hpo_overrides(params)

    from pff.infrastructure.hpo.config_loader import load_optimization_config

    hpo_settings = load_optimization_config(file_manager=FileManager())
    time_budget = hpo_settings.get("time_budget_pruning", {})
    if not isinstance(time_budget, dict):
        time_budget = {}
    time_budget = params.get("time_budget_pruning", time_budget)
    if not isinstance(time_budget, dict):
        time_budget = {}
    overrides["time_budget"] = time_budget
    overrides["rerank_top_k"] = 0

    model_config, training_config = build_dslfm_configs(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=len(train_triples),
        raw_settings=dslfm_settings,
        overrides=overrides,
        checkpoint_dir=model_dir / "checkpoints",
        use_bert=use_bert,
        relation_names=relation_names,
    )

    logger.info(
        f"DSLFM Weights: community={model_config.community_weight} feature={model_config.feature_weight} "
        f"logic={model_config.lambda_logic} pc={model_config.lambda_pc}"
    )

    from pff.shared.core.config import settings

    hpo_plots_dir = settings.OUTPUTS_DIR / "optimization" / "plots"
    hpo_plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = model_dir / "checkpoints"
    persistence = FileSystemModelPersistence(checkpoint_dir)

    trial_number = getattr(trial, "number", 0) if trial else 0
    if trial_number_override is not None:
        trial_number = trial_number_override

    warmstart = False
    study_name: str | None = None
    if trial is not None:
        user_attrs = getattr(trial, "user_attrs", {}) or {}
        raw_study_name = getattr(getattr(trial, "study", None), "study_name", None)
        if isinstance(raw_study_name, str) and raw_study_name.strip():
            study_name = raw_study_name.strip()
        system_attrs: dict[str, Any] = {}
        storage = getattr(trial, "_storage", None)
        trial_id = getattr(trial, "_trial_id", None)
        if (
            storage is not None
            and trial_id is not None
            and hasattr(storage, "get_trial_system_attrs")
        ):
            try:
                loaded = storage.get_trial_system_attrs(trial_id)
                if isinstance(loaded, dict):
                    system_attrs = loaded
            except Exception:
                system_attrs = {}
        warmstart = bool(
            system_attrs.get("warmstart_seed")
            or user_attrs.get("warmstart")
            or user_attrs.get("warmstart_seed")
        )

    manager = DSLFMKGCManager(
        model_config,
        training_config,
        persistence_port=persistence,
        relation_names=(
            [str(r) for r in relation_names]
            if use_bert and relation_names is not None
            else None
        ),
        observers=[
            BinaryMetricsObserver(None, valid_triples, params),
            LiveTrainingObserver(
                hpo_plots_dir,
                trial_number,
                params,
                cv_fold_id=cv_fold_id,
                warmstart=warmstart,
                study_name=study_name,
            ),
            ConsoleObserver(verbose=False),
        ],
        seed=int(params.get("seed", 1337)),
    )

    for obs in manager.observers:
        if isinstance(obs, BinaryMetricsObserver):
            obs.manager = manager

    stats = manager.train(train_triples, valid_triples, trial=trial)
    binary_metrics = _compute_binary_metrics(
        manager,
        valid_triples,
        num_negatives=int(params.get("binary_negatives", 20)),
        seed=int(params.get("seed", 1337)),
        params=params,
    )
    if "final_metrics" in stats:
        stats["final_metrics"].update(binary_metrics)
    else:
        stats["final_metrics"] = binary_metrics
    if not binary_metrics:
        logger.warning(
            "Classification metrics missing for trial (AUC/PR/F1 not calculated)."
        )

    checkpoint_path = training_config.checkpoint_dir / "best_model.pt"

    return stats, checkpoint_path
