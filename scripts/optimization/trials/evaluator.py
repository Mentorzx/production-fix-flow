from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pff.config import KG_PIPELINE_CONFIG_PATH, ROTATE_CONFIG_PATH
from pff.utils import ScoreCalibrator, logger
from pff.utils.core.file_manager import FileManager
from pff.utils.performance.training_observer import OptunaTrialObserver
from pff.validators.kg.config import KGConfig
from pff.validators.rotate.manager import RotatEManager

from .embedding_cache import (
    EmbeddingCache,
    create_cache_key_from_params,
    compute_data_hash,
)


def _train_rotate_model(
    params: dict[str, Any],
    model_dir: Path,
    checkpoint_dir: Path,
    config_dir: Path,
    file_manager: FileManager,
    *,
    trial: Any | None = None,
    enable_embedding_cache: bool = True,
    data_hash: str | None = None,
) -> tuple[RotatEManager, dict[str, float], Path]:
    """
    Train RotatE model with given hyperparameters and optional Optuna observer.
    
    Supports embedding caching to skip training when identical hyperparameters
    are detected from previous trials (SOTA optimization).
    
    Args:
        params: Trial hyperparameters dictionary
        model_dir: Directory to save model outputs
        checkpoint_dir: Directory for model checkpoints
        config_dir: Directory for trial configuration files
        file_manager: FileManager instance for I/O operations
        trial: Optional Optuna trial for progress reporting
        enable_embedding_cache: Whether to use embedding cache (default: True)
        data_hash: Pre-computed data hash for cache key (optional)
        
    Returns:
        Tuple of (RotatEManager, metrics dict, checkpoint path)
    """
    # Check embedding cache first (SOTA: avoid redundant training)
    cache_key = None
    if enable_embedding_cache and data_hash:
        cache = EmbeddingCache.get_instance()
        cache_key = create_cache_key_from_params(params, data_hash)
        
        cached = cache.get(cache_key)
        if cached is not None:
            # Create manager with cached embeddings
            rotate_config_data = file_manager.read(ROTATE_CONFIG_PATH) or {}
            model_cfg = rotate_config_data.setdefault("model", {})
            training_cfg = rotate_config_data.setdefault("training", {})
            
            model_cfg["embedding_dim"] = int(params.get("embedding_dim", 256))
            model_cfg["gamma"] = float(params.get("gamma", 12.0))
            model_cfg["epsilon"] = float(params.get("epsilon", 2.0))
            
            rotate_config_data["outputs"] = {
                "dir": str(model_dir),
                "save_model": True,
                "save_embeddings": True,
            }
            
            trial_rotate_config_path = config_dir / "rotate.yaml"
            file_manager.save(rotate_config_data, trial_rotate_config_path)
            
            # Initialize manager and restore cached embeddings
            rotate_manager = RotatEManager(
                rotate_config_path=trial_rotate_config_path,
                kg_config_path=KG_PIPELINE_CONFIG_PATH,
            )
            rotate_manager._setup_data()
            rotate_manager._setup_model()
            
            # Restore embeddings from cache
            if rotate_manager.model is not None:
                rotate_manager.model.entity_embedding.data = cached.entity_embeddings.to(
                    rotate_manager.model.entity_embedding.device
                )
                rotate_manager.model.relation_embedding.data = cached.relation_embeddings.to(
                    rotate_manager.model.relation_embedding.device
                )
            
            checkpoint_path = checkpoint_dir / "best_model.pt"
            return rotate_manager, cached.metrics, checkpoint_path

    # Standard training path
    rotate_config_data = file_manager.read(ROTATE_CONFIG_PATH) or {}

    model_cfg = rotate_config_data.setdefault("model", {})
    training_cfg = rotate_config_data.setdefault("training", {})

    model_cfg["embedding_dim"] = int(params.get("embedding_dim", model_cfg.get("embedding_dim", 128)))
    model_cfg["gamma"] = float(params.get("gamma", model_cfg.get("gamma", 12.0)))
    model_cfg["epsilon"] = float(params.get("epsilon", model_cfg.get("epsilon", 2.0)))
    model_cfg["regularization_weight"] = float(
        params.get("regularization_weight", model_cfg.get("regularization_weight", 1e-5))
    )

    training_cfg["epochs"] = int(params.get("rotate_epochs", training_cfg.get("epochs", 100)))
    training_cfg["batch_size"] = int(params.get("batch_size", training_cfg.get("batch_size", 512)))
    training_cfg["learning_rate"] = float(params.get("meta_learning_rate", training_cfg.get("learning_rate", 0.0001)))
    training_cfg["negative_sample_size"] = int(
        params.get("negative_sample_size", training_cfg.get("negative_sample_size", 256))
    )
    training_cfg["self_adversarial"] = bool(
        params.get("self_adversarial", training_cfg.get("self_adversarial", True))
    )
    training_cfg["adversarial_temperature"] = float(
        params.get("adversarial_temperature", training_cfg.get("adversarial_temperature", 1.0))
    )

    rotate_config_data.setdefault("checkpointing", {})
    rotate_config_data["checkpointing"]["save_dir"] = str(checkpoint_dir)
    rotate_config_data["outputs"] = {
        "dir": str(model_dir),
        "save_model": True,
        "save_embeddings": True,
        "save_checkpoints": False,
    }

    trial_rotate_config_path = config_dir / "rotate.yaml"
    file_manager.save(rotate_config_data, trial_rotate_config_path)

    logger.info("Treinando modelo RotatE (rotação em espaço complexo)...")
    rotate_manager = RotatEManager(
        rotate_config_path=trial_rotate_config_path,
        kg_config_path=KG_PIPELINE_CONFIG_PATH,
    )
    if trial is not None:
        try:
            rotate_manager.training_observer.add_observer(OptunaTrialObserver(trial))
        except Exception as observer_exc:  # noqa: BLE001
            logger.warning(f"Failed to attach Optuna observer: {observer_exc}")

    rotate_manager._setup_data()
    rotate_manager._setup_model()
    rotate_training_stats = rotate_manager.train(force_retrain=True)
    best_epoch = int(rotate_training_stats.get("best_epoch", 0))

    if rotate_manager.val_triples is not None and len(rotate_manager.val_triples) > 0:
        rotate_eval_raw = rotate_manager._validate(rotate_manager.val_triples)
    else:
        rotate_eval_raw = rotate_manager.last_val_metrics or {}

    rotate_metrics = {
        "mrr": float(rotate_eval_raw.get("mrr", 0.0)),
        "hits@1": float(rotate_eval_raw.get("hits@1", 0.0)),
        "hits@10": float(rotate_eval_raw.get("hits@10", 0.0)),
        "hits@3": float(rotate_eval_raw.get("hits@3", 0.0)),
        "mean_rank": float(rotate_eval_raw.get("mean_rank", 0.0)),
        "best_val_mrr": float(rotate_training_stats.get("best_val_mrr", 0.0)),
        "convergence_epoch": best_epoch,
        "best_epoch": best_epoch,
    }

    try:
        _train_rotate_score_calibrator(rotate_manager, model_dir)
    except Exception as calib_exc:  # noqa: BLE001
        logger.warning(f"Failed to train RotatE calibrator: {calib_exc}")

    # Cache embeddings for future trials with identical hyperparameters
    if enable_embedding_cache and cache_key is not None and rotate_manager.model is not None:
        try:
            cache = EmbeddingCache.get_instance()
            cache.put(
                cache_key,
                entity_embeddings=rotate_manager.model.entity_embedding.weight.detach().cpu().numpy(),
                relation_embeddings=rotate_manager.model.relation_embedding.weight.detach().cpu().numpy(),
                entity_to_idx=getattr(rotate_manager, "entity_to_idx", {}),
                relation_to_idx=getattr(rotate_manager, "relation_to_idx", {}),
                metrics=rotate_metrics,
                checkpoint_path=checkpoint_dir / "best_model.pt",
            )
        except Exception as cache_exc:  # noqa: BLE001
            logger.warning(f"Failed to cache embeddings: {cache_exc}")

    checkpoint_path = checkpoint_dir / "best_model.pt"
    return rotate_manager, rotate_metrics, checkpoint_path


def _train_rotate_score_calibrator(rotate_manager, output_dir: Path) -> None:
    """Fit Platt scaling using validation triples for RotatE model."""
    val_triples = getattr(rotate_manager, "val_triples", None)
    model = getattr(rotate_manager, "model", None)
    if val_triples is None or val_triples.size == 0 or model is None:
        logger.warning("No validation triples available; skipping RotatE calibration")
        return

    entity_count = len(getattr(rotate_manager, "entity_to_idx", {}))
    if entity_count == 0:
        logger.warning("Entity vocabulary is empty; skipping RotatE calibration")
        return

    rng = np.random.default_rng(42)
    scores: list[float] = []
    labels: list[int] = []
    for triple in val_triples:
        head, rel, tail = map(int, triple)
        pos_score = float(model.score_triple(head, rel, tail))
        scores.append(pos_score)
        labels.append(1)
        if rng.random() < 0.5:
            corrupted_head = int(rng.integers(0, entity_count))
            neg_score = float(model.score_triple(corrupted_head, rel, tail))
        else:
            corrupted_tail = int(rng.integers(0, entity_count))
            neg_score = float(model.score_triple(head, rel, corrupted_tail))
        scores.append(neg_score)
        labels.append(0)

    calibrator = ScoreCalibrator()
    try:
        calibrator.fit(np.array(scores), np.array(labels))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to fit RotatE score calibrator: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    calib_path = output_dir / "score_calibrator.pkl"
    FileManager().save(calibrator.to_dict(), calib_path)
    logger.info(f" Calibrador RotatE salvo em {calib_path}")
