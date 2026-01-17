from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pff.shared import logger
from pff.shared.core.file_manager import FileManager
from pff.shared.system.cuda import is_cuda_available

from pff.domain.learning.dslfm.dslfm_kgc import load_dslfm_kgc_settings


from pff.domain.learning.ml.training_observer import TrainingEvent, TrainingObserver
from pff.infrastructure.hpo.callbacks_internal.visualizers import LiveTrainingObserver


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
        self.manager = manager
        self.valid_triples = valid_triples
        self.params = params
        self.compute_every = compute_every
        self._eval_count = 0

    def on_event(self, event: TrainingEvent) -> None:
        if event.event_type == "epoch_end":
            # Only compute if this is an evaluation epoch (has rank metrics)
            if "mrr" in event.metrics:
                self._eval_count += 1
                if self._eval_count % self.compute_every == 0:
                    try:
                        # Use a subset of negatives for speed during training if not already limited
                        current_params = self.params.copy()
                        if "binary_metrics_max_samples" not in current_params:
                            current_params["binary_metrics_max_samples"] = 2000

                        binary_metrics = _compute_binary_metrics(
                            self.manager,
                            self.valid_triples,
                            num_negatives=int(
                                current_params.get("binary_negatives", 10)
                            ),
                            seed=int(current_params.get("seed", 1337)) + event.epoch,
                            params=current_params,
                        )
                        event.metrics.update(binary_metrics)
                    except Exception as exc:
                        logger.warning(
                            f"Falha ao computar metricas binarias na epoca {event.epoch}: {exc}"
                        )


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

    model = getattr(manager, "model", None)
    if model is None:
        logger.warning("Binary metrics skipped: manager.model is None")
        return {}

    # Deep lookup for scoring method (handles torch.compile and wrappers)
    def _find_scoring_model(m: Any) -> Any | None:
        if hasattr(m, "score_triples_batch"):
            return m
        # Check wrappers like _CompiledModelWrapper or torch._dynamo
        for attr in ("base_model", "_orig_mod", "model"):
            candidate = getattr(m, attr, None)
            if candidate is not None:
                found = _find_scoring_model(candidate)
                if found is not None:
                    return found
        return None

    scoring_model = _find_scoring_model(model)
    if scoring_model is None:
        logger.warning(
            f"Binary metrics skipped: model {type(model).__name__} does not expose 'score_triples_batch'"
        )
        return {}

    try:
        from pff.shared.acceleration.numba_kernels import (
            fast_roc_auc_score as roc_auc_score,
            fast_matthews_corrcoef as matthews_corrcoef,
            fast_average_precision_score as average_precision_score,
            fast_precision_recall_curve as precision_recall_curve,
        )
        from sklearn.metrics import auc, accuracy_score
    except ImportError:
        try:
            from sklearn.metrics import (
                roc_auc_score,
                precision_recall_curve,
                auc,
                matthews_corrcoef,
                accuracy_score,
                average_precision_score,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Binary metrics skipped: scikit-learn unavailable: {exc}")
            return {}

    # ... (rest of configuration loading)
    from pff.infrastructure.hpo.config_loader import load_optimization_config

    settings = load_optimization_config(file_manager=FileManager())
    binary_cfg = (
        settings.get("binary_metrics", {}) if isinstance(settings, dict) else {}
    )
    if not isinstance(binary_cfg, dict):
        binary_cfg = {}

    params = params or {}
    enabled = bool(
        params.get("binary_metrics_enabled", binary_cfg.get("enabled", True))
    )
    if not enabled:
        return {}

    num_negatives = int(
        params.get(
            "binary_metrics_num_negatives",
            binary_cfg.get("num_negatives", num_negatives),
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

    rng = np.random.default_rng(seed)

    # Identify num_entities with fallback to config
    num_entities = getattr(scoring_model, "num_entities", 0)
    if num_entities <= 0:
        model_config = getattr(scoring_model, "config", None)
        num_entities = getattr(model_config, "num_entities", 0)

    if num_entities <= 0:
        logger.warning("Binary metrics skipped: num_entities could not be determined")
        return {}

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
        return {}

    negatives_arr = np.repeat(val_triples_arr, int(num_negatives), axis=0)
    choice = rng.random(n_total) < 0.5
    rand_entities = rng.integers(0, num_entities, size=n_total, dtype=np.int64)
    negatives_arr[choice, 0] = rand_entities[choice]
    negatives_arr[~choice, 2] = rand_entities[~choice]

    # Identify device
    scoring_model_any: Any = scoring_model
    device = getattr(getattr(manager, "model", None), "device", None)
    if device is None:
        try:
            device = next(scoring_model_any.parameters()).device
        except Exception:
            device = torch.device("cpu")

    def _select_device() -> torch.device:
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

    target_device = _select_device()
    pos_tensor = torch.tensor(val_triples_arr, device=target_device, dtype=torch.long)
    neg_tensor = torch.tensor(negatives_arr, device=target_device, dtype=torch.long)

    original_device = device
    moved_model = False

    if getattr(scoring_model_any, "device", None) != target_device:
        try:
            scoring_model_any.to(target_device)
            moved_model = True
        except Exception as exc:
            logger.warning(
                f"Falha ao mover modelo para {target_device} para metricas binarias: {exc}"
            )
            target_device = device
            pos_tensor = pos_tensor.to(target_device)
            neg_tensor = neg_tensor.to(target_device)

    def _score_in_batches(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return torch.empty((0,), device=target_device)
        outputs: list[torch.Tensor] = []
        total = tensor.shape[0]
        step = max(int(batch_size), 1)
        for start in range(0, total, step):
            chunk = tensor[start : start + step]
            outputs.append(scoring_model_any.score_triples_batch(chunk))
        return torch.cat(outputs, dim=0)

    with torch.no_grad():
        inference_start = time.perf_counter()
        pos_scores = _score_in_batches(pos_tensor)
        neg_scores = _score_in_batches(neg_tensor)
        inference_elapsed = time.perf_counter() - inference_start

    total_triples_scored = len(pos_tensor) + len(neg_tensor)
    inference_latency_ms = (
        (inference_elapsed * 1000) / total_triples_scored
        if total_triples_scored > 0
        else 0.0
    )

    if moved_model and original_device is not None:
        try:
            scoring_model_any.to(original_device)
        except Exception as exc:
            logger.warning(f"Falha ao restaurar modelo para {original_device}: {exc}")

    raw_scores = torch.cat([pos_scores, neg_scores]).cpu()
    labels = np.concatenate(
        [
            np.ones(len(pos_scores), dtype=np.int64),
            np.zeros(len(neg_scores), dtype=np.int64),
        ]
    )

    metrics: dict[str, float] = {}

    try:
        scores_t = raw_scores.clone().detach().float()
        targets_t = torch.tensor(labels, dtype=torch.float32, device=scores_t.device)
        a = torch.zeros((), device=scores_t.device, requires_grad=True)
        b = torch.zeros((), device=scores_t.device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [a, b], max_iter=25, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = a * scores_t + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets_t
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        calibrated_logits = a.detach() * scores_t + b.detach()
        prob_scores = torch.sigmoid(calibrated_logits).cpu().numpy()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Platt calibration failed: {exc}")
        prob_scores = torch.sigmoid(raw_scores).cpu().numpy()

    eps = 1e-12
    prob_scores = np.clip(prob_scores.astype(np.float64), eps, 1.0 - eps)

    # Identificação final e cálculo das métricas binárias
    try:
        metrics["brier"] = float(np.mean((prob_scores - labels) ** 2))
        metrics["nll"] = float(
            -np.mean(
                labels * np.log(prob_scores)
                + (1.0 - labels) * np.log(1.0 - prob_scores)
            )
        )
        n_bins = 15
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(prob_scores, edges[1:-1], right=True)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            acc = float(np.mean(labels[mask]))
            conf = float(np.mean(prob_scores[mask]))
            ece += float(np.sum(mask)) / float(len(labels)) * abs(acc - conf)
        metrics["ece"] = float(ece)
    except Exception as exc:
        logger.debug(f"Falha ao computar brier/nll/ece: {exc}")

    try:
        metrics["auc"] = float(roc_auc_score(labels, prob_scores))
    except Exception as exc:
        logger.warning(f"Falha ao computar ROC-AUC: {exc}")
        metrics["auc"] = 0.5

    try:
        precisions, recalls, thresholds = precision_recall_curve(labels, prob_scores)
        if len(precisions) > 1 and len(recalls) > 1:
            # Sort by recall (ascending) to ensure monotonicity for auc()
            sorted_indices = np.argsort(recalls)
            sorted_recalls = recalls[sorted_indices]
            sorted_precisions = precisions[sorted_indices]
            
            # Remove duplicates to avoid "neither increasing nor decreasing" error
            unique_mask = np.diff(sorted_recalls, prepend=-1) != 0
            unique_recalls = sorted_recalls[unique_mask]
            unique_precisions = sorted_precisions[unique_mask]
            
            if len(unique_recalls) >= 2:
                pr_auc = auc(unique_recalls, unique_precisions)
            else:
                pr_auc = 0.0
            metrics["pr_auc"] = float(pr_auc)

            f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
                precisions[:-1] + recalls[:-1] + 1e-12
            )
            best_idx = int(np.argmax(f1_scores))
            metrics["precision"] = float(precisions[best_idx])
            metrics["recall"] = float(recalls[best_idx])
            metrics["f1"] = float(f1_scores[best_idx])

            # Decisão de threshold para MCC: Best F1 ou 0.5 (fallback)
            decision_thresh = 0.5
            if len(thresholds) > best_idx:
                decision_thresh = float(thresholds[best_idx])

            metrics["decision_threshold"] = decision_thresh
            binary_preds = (prob_scores > decision_thresh).astype(np.int32)
            metrics["mcc"] = float(matthews_corrcoef(labels, binary_preds))
            metrics["accuracy"] = float(accuracy_score(labels, binary_preds))
            metrics["ap"] = float(average_precision_score(labels, prob_scores))
        else:
            metrics["mcc"] = 0.0
            metrics["pr_auc"] = 0.0
            metrics["accuracy"] = 0.0
            metrics["ap"] = 0.0
    except Exception as exc:
        logger.warning(f"Falha ao computar PR-AUC/MCC/Acc/AP: {exc}")
        metrics["mcc"] = 0.0
        metrics["accuracy"] = 0.0
        metrics["ap"] = 0.0

    metrics["inference_latency"] = float(inference_latency_ms)

    if metrics.get("auc", 0) > 0 or metrics.get("mcc", 0) > 0:
        logger.info(
            f"Metricas binarias (N={len(labels)}): AUC={metrics.get('auc', 0):.4f} "
            f"PR_AUC={metrics.get('pr_auc', 0):.4f} MCC={metrics.get('mcc', 0):.4f} "
            f"AP={metrics.get('ap', 0):.4f} "
            f"Thresh={metrics.get('decision_threshold', 0):.3f} "
            f"InfLatency={inference_latency_ms:.4f}ms/triple"
        )
    else:
        logger.warning("Metricas de classificacao (MCC/AUC) zeradas ou falharam.")

    return metrics


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
    defaults. For scoring stability, reranking is disabled (forced `rerank_top_k=0`)
    because it has known MRR regressions in the current pipeline.

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

    Returns:
        Tuple with (training stats dict, checkpoint path).
    """
    from pff.domain.learning.dslfm.core import DSLFMKGCConfig
    from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig
    from pff.domain.learning.ml.training_observer import ConsoleObserver

    settings = load_dslfm_kgc_settings(FileManager(), params.get("config_path"))
    kgc_settings = settings.get("kgc", {})
    model_defaults = kgc_settings.get("model", {})
    training_defaults = kgc_settings.get("training", {})
    compile_defaults = settings.get("compile", {})
    if not isinstance(compile_defaults, dict):
        compile_defaults = {}
    logic_defaults = settings.get("logic", {})
    pc_defaults = settings.get("pc", {})
    from pff.infrastructure.hpo.config_loader import load_optimization_config

    hpo_settings = load_optimization_config(file_manager=FileManager())
    time_budget = hpo_settings.get("time_budget_pruning", {})
    if not isinstance(time_budget, dict):
        time_budget = {}
    time_budget = params.get("time_budget_pruning", time_budget)
    if not isinstance(time_budget, dict):
        time_budget = {}

    def _get(section: dict[str, Any], key: str, fallback: Any) -> Any:
        return params.get(key, section.get(key, fallback))

    entity_dim = int(_get(model_defaults, "entity_dim", 256))
    feature_dim = int(_get(model_defaults, "feature_dim", 256))
    max_communities = int(_get(model_defaults, "max_communities", 128))
    hidden_dim = int(_get(model_defaults, "hidden_dim", 512))
    ibp_alpha = float(_get(model_defaults, "ibp_alpha", 1.0))
    kl_weight = float(_get(model_defaults, "kl_weight", 0.1))
    sparsity_weight = float(_get(model_defaults, "sparsity_weight", 0.01))
    temperature = float(_get(model_defaults, "temperature", 0.5))
    stochastic_latents = bool(_get(model_defaults, "stochastic_latents", False))
    encoder_dropout_p = float(_get(model_defaults, "encoder_dropout_p", 0.0))
    if "embedding_dim" in params:
        try:
            embedding_dim = int(params["embedding_dim"])
            entity_dim = embedding_dim
            feature_dim = embedding_dim
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Invalid embedding_dim, expected int: value={params.get('embedding_dim')!r} error={exc}"
            )

    sampler_type = str(_get(model_defaults, "sampler_type", "self_adversarial"))
    if sampler_type in {"self_adversarial", "uniform"} and "self_adversarial" in params:
        sampler_type = (
            "self_adversarial" if bool(params.get("self_adversarial")) else "uniform"
        )
    sampler_temperature = float(
        params.get(
            "adversarial_temperature", _get(model_defaults, "sampler_temperature", 1.0)
        )
    )
    learnable_temperature = bool(_get(model_defaults, "learnable_temperature", False))
    contrastive_temperature = float(
        params.get(
            "contrastive_temperature",
            _get(model_defaults, "contrastive_temperature", 0.5),
        )
    )
    negative_sample_size = int(
        params.get(
            "negative_sample_size", _get(model_defaults, "negative_sample_size", 0)
        )
    )
    num_global_negatives = int(
        params.get(
            "num_global_negatives", _get(model_defaults, "num_global_negatives", 0)
        )
    )
    lambda_logic = float(_get(logic_defaults, "lambda_logic", 0.0))
    t_norm = str(_get(logic_defaults, "t_norm", "product"))
    smoothing_epsilon = float(_get(logic_defaults, "smoothing_epsilon", 1e-6))
    lambda_pc = float(_get(pc_defaults, "lambda_pc", 0.0))
    pc_pruning_threshold = float(_get(pc_defaults, "pruning_threshold", 0.01))
    pc_grow_noise = float(_get(pc_defaults, "grow_noise", 0.01))
    pc_rebuild_every = int(_get(pc_defaults, "rebuild_every", 0))
    pc_max_depth = _get(pc_defaults, "max_circuit_depth", None)

    epochs = int(_get(training_defaults, "epochs", 100))
    epochs = int(params.get("dslfm_epochs", epochs))
    batch_size = int(_get(training_defaults, "batch_size", 256))
    effective_batch_size = int(_get(training_defaults, "effective_batch_size", 1024))
    learning_rate = float(_get(training_defaults, "learning_rate", 1e-4))
    validate_every = int(_get(training_defaults, "validate_every", 5))
    early_stopping_patience = int(
        _get(training_defaults, "early_stopping_patience", 10)
    )
    min_delta = float(_get(training_defaults, "min_delta", 0.0002))
    mixed_precision = bool(
        _get(training_defaults, "mixed_precision", is_cuda_available())
    )
    num_workers = int(_get(training_defaults, "num_workers", 0))
    pin_memory = bool(_get(training_defaults, "pin_memory", False))
    dataloader_prefetch_factor = int(
        _get(training_defaults, "dataloader_prefetch_factor", 4)
    )
    dataloader_persistent_workers = bool(
        _get(training_defaults, "dataloader_persistent_workers", True)
    )
    eval_batch_size = int(_get(training_defaults, "eval_batch_size", batch_size))
    regularization_warmup_epochs = int(
        _get(training_defaults, "regularization_warmup_epochs", 8)
    )
    regularization_start_scale = float(
        _get(training_defaults, "regularization_start_scale", 0.0)
    )

    model_config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=num_relations,
        entity_dim=entity_dim,
        feature_dim=feature_dim,
        max_communities=max_communities,
        hidden_dim=hidden_dim,
        ibp_alpha=ibp_alpha,
        kl_weight=kl_weight,
        sparsity_weight=sparsity_weight,
        temperature=temperature,
        stochastic_latents=stochastic_latents,
        encoder_dropout_p=encoder_dropout_p,
        use_bert_relations=use_bert and relation_names is not None,
        bert_model=params.get("bert_model", "bert-base-uncased"),
        use_checkpointing=bool(_get(model_defaults, "use_checkpointing", False)),
        sampler_type=sampler_type,
        sampler_temperature=sampler_temperature,
        learnable_temperature=learnable_temperature,
        contrastive_temperature=contrastive_temperature,
        negative_sample_size=negative_sample_size,
        num_global_negatives=num_global_negatives,
        lambda_logic=lambda_logic,
        t_norm=t_norm,
        smoothing_epsilon=smoothing_epsilon,
        lambda_pc=lambda_pc,
        pc_pruning_threshold=pc_pruning_threshold,
        pc_grow_noise=pc_grow_noise,
        pc_rebuild_every=pc_rebuild_every,
        pc_max_depth=pc_max_depth,
        logvar_clip_min=float(_get(model_defaults, "logvar_clip_min", -20.0)),
        logvar_clip_max=float(_get(model_defaults, "logvar_clip_max", 10.0)),
    )

    pin_memory = bool(params.get("pin_memory", pin_memory))
    num_workers = max(0, int(params.get("num_workers", num_workers)))
    dataloader_prefetch_factor = int(
        params.get("dataloader_prefetch_factor", dataloader_prefetch_factor)
    )
    dataloader_persistent_workers = bool(
        params.get("dataloader_persistent_workers", dataloader_persistent_workers)
    )

    training_config = KGCTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        validate_every=validate_every,
        early_stopping_patience=early_stopping_patience,
        min_delta=float(params.get("min_delta", min_delta)),
        checkpoint_dir=model_dir / "checkpoints",
        mixed_precision=mixed_precision,
        use_compile=bool(_get(training_defaults, "use_compile", False)),
        compile_mode=str(_get(compile_defaults, "mode", "reduce-overhead")),
        compile_dynamic=bool(_get(compile_defaults, "dynamic", True)),
        compile_fullgraph=bool(_get(compile_defaults, "fullgraph", False)),
        compile_backend=_get(compile_defaults, "backend", None),
        optimizer_fused=_get(training_defaults, "optimizer_fused", None),
        optimizer_foreach=_get(training_defaults, "optimizer_foreach", None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_persistent_workers=dataloader_persistent_workers,
        eval_batch_size=eval_batch_size,
        regularization_warmup_epochs=regularization_warmup_epochs,
        regularization_start_scale=regularization_start_scale,
        rerank_top_k=0,
        refresh_cache_on_val=True,
        max_grad_norm=_get(training_defaults, "max_grad_norm", None),
        time_budget=time_budget,
    )

    from pff import settings

    # Use the centralized optimization plots directory
    hpo_plots_dir = settings.OUTPUTS_DIR / "optimization" / "plots"
    hpo_plots_dir.mkdir(parents=True, exist_ok=True)

    trial_number = getattr(trial, "number", 0) if trial else 0
    if trial_number_override is not None:
        trial_number = trial_number_override

    manager = DSLFMKGCManager(
        model_config,
        training_config,
        relation_names=(
            [str(r) for r in relation_names]
            if use_bert and relation_names is not None
            else None
        ),
        observers=[
            BinaryMetricsObserver(None, valid_triples, params),
            LiveTrainingObserver(
                hpo_plots_dir, trial_number, params, cv_fold_id=cv_fold_id
            ),
            ConsoleObserver(verbose=False),
        ],
    )

    # Late bind manager to the observer
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
