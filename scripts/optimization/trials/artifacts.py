"""
Trial Artifact Manager Module

Manages storage and retrieval of trial results and model artifacts.

Design Patterns:
- Repository Pattern: Abstracts storage of trial results
- Single Responsibility Principle (SRP): Only handles artifact management
- Factory Pattern: Creates artifact paths and structures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.utils import logger
from pff.utils.core.file_manager import FileManager
from pff.utils.hash import stable_hash


class TrialArtifactManager:
    """
    SRP helper to store trial results and persist best artifacts.

    Repository Pattern: Manages trial results as a collection with
    hash-indexed lookup for O(1) retrieval by params.
    """

    def __init__(self, file_manager: FileManager | None = None) -> None:
        self.file_manager = file_manager or FileManager()
        self.trial_results: dict[int, dict[str, Any]] = {}
        # Hash-indexed lookup for O(1) retrieval by params
        self._params_index: dict[str, int] = {}

    def _hash_params(self, params: dict[str, Any]) -> str:
        """Create stable hash from params for indexing."""
        # Sort params for deterministic hash
        sorted_items = sorted(params.items())
        return stable_hash(str(sorted_items))

    def record_result(self, trial_number: int, trial_result: dict[str, Any]) -> None:
        """Store trial result for later persistence."""
        self.trial_results[trial_number] = trial_result
        # Index by params hash for O(1) lookup
        params = trial_result.get("params", {})
        if params:
            params_hash = self._hash_params(params)
            self._params_index[params_hash] = trial_number

    def _match_params(self, stored: dict[str, Any], trial_params: dict[str, Any]) -> bool:
        stored_params = stored.get("params", {})
        for name in set(stored_params.keys()) | set(trial_params.keys()):
            val1 = stored_params.get(name)
            val2 = trial_params.get(name)
            if isinstance(val1, float) and isinstance(val2, float):
                if abs(val1 - val2) > 1e-9:
                    return False
            elif val1 != val2:
                return False
        return True

    def get_trial_result(self, trial) -> dict[str, Any] | None:
        """
        Retrieve stored result matching a trial by number or params.

        Uses hash-indexed lookup for O(1) retrieval when possible.
        """
        # First try direct lookup by trial number
        result = self.trial_results.pop(getattr(trial, "number", -1), None)
        if result:
            # Clean up params index
            params = result.get("params", {})
            if params:
                params_hash = self._hash_params(params)
                self._params_index.pop(params_hash, None)
            return result

        # O(1) lookup by params hash
        trial_params = getattr(trial, "params", {})
        if trial_params:
            params_hash = self._hash_params(trial_params)
            trial_number = self._params_index.pop(params_hash, None)
            if trial_number is not None:
                return self.trial_results.pop(trial_number, None)

        # Fallback to O(n) search for edge cases
        for key, candidate in list(self.trial_results.items()):
            if self._match_params(candidate, trial_params):
                return self.trial_results.pop(key)
        return None

    def persist_best_models(self, best_models_dir: Path, trial_result: dict[str, Any]) -> None:
        """Persist best model artifacts into best_models_dir."""
        best_models_dir.mkdir(parents=True, exist_ok=True)
        for item in best_models_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                self.file_manager.delete_directory(item)

        model_paths = trial_result.get("model_paths", {})
        models_trained = trial_result.get("models_trained", {})

        if "rotate" in model_paths and model_paths["rotate"].exists():
            dest = best_models_dir / "best_rotate_model.pt"
            self.file_manager.copy_file(model_paths["rotate"], dest)
            logger.info(f"   Modelo RotatE salvo: {dest}")
        else:
            logger.warning(
                f"RotatE model NOT saved (models_trained={models_trained.get('rotate')}, path={'rotate' in model_paths})"
            )

        if "anyburl" in model_paths and model_paths["anyburl"].exists():
            dest_dir = best_models_dir / "anyburl"
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / "rules.tsv"
            self.file_manager.copy_file(model_paths["anyburl"], dest)
            logger.info(f"   Regras AnyBURL salvas: {dest}")
        else:
            logger.warning(
                f"AnyBURL model NOT saved (models_trained={models_trained.get('anyburl')}, path={'anyburl' in model_paths})"
            )

        if "lightgbm" in model_paths and model_paths["lightgbm"].exists():
            dest = best_models_dir / "best_lightgbm_model.bin"
            self.file_manager.copy_file(model_paths["lightgbm"], dest)
            logger.info(f"   Modelo LightGBM salvo: {dest}")
        else:
            logger.warning(
                f"LightGBM model NOT saved (models_trained={models_trained.get('lightgbm')}, path={'lightgbm' in model_paths})"
            )

        if "xgboost" in model_paths and model_paths["xgboost"].exists():
            dest = best_models_dir / "best_xgboost_model.joblib"
            self.file_manager.copy_file(model_paths["xgboost"], dest)
            logger.info(f"   Modelo XGBoost ensemble salvo: {dest}")
        else:
            logger.warning(
                "XGBoost model NOT saved "
                f"(models_trained={models_trained.get('xgboost')}, path={'xgboost' in model_paths})"
            )

    def persist_best_params(self, best_models_dir: Path, trial_result: dict[str, Any]) -> None:
        """Persist per-model best parameters using FileManager."""
        params = trial_result["params"]
        model_metrics = trial_result["model_metrics"]
        ensemble_metrics = trial_result.get("ensemble_metrics")

        if trial_result["models_trained"].get("rotate"):
            rotate_metrics_payload = {
                "mrr": model_metrics["rotate"].get("mrr", 0.0),
                "hits_at_1": model_metrics["rotate"].get("hits@1", 0.0),
                "hits_at_10": model_metrics["rotate"].get("hits@10", 0.0),
                "best_val_mrr": model_metrics["rotate"].get("best_val_mrr", 0.0),
            }
            rotate_params = {
                "model": "RotatE",
                "hyperparameters": {
                    "embedding_dim": params.get("embedding_dim"),
                    "gamma": params.get("gamma"),
                    "epsilon": params.get("epsilon"),
                    "learning_rate": params.get("meta_learning_rate"),
                    "epochs": params.get("rotate_epochs"),
                    "batch_size": params.get("batch_size"),
                    "negative_sample_size": params.get("negative_sample_size"),
                    "adversarial_temperature": params.get("adversarial_temperature"),
                    "self_adversarial": params.get("self_adversarial"),
                    "regularization_weight": params.get("regularization_weight"),
                },
                "metrics": rotate_metrics_payload,
                "weight_in_ensemble": params.get("neural_weight"),
            }
            rotate_file = best_models_dir / "best_params_rotate.json"
            self.file_manager.save(rotate_params, rotate_file)
            logger.info(f"   Parametros RotatE salvos: {rotate_file}")

        if trial_result["models_trained"].get("anyburl"):
            anyburl_metrics_payload = {
                "avg_confidence": model_metrics["anyburl"].get("avg_confidence", 0.0),
                "avg_support": model_metrics["anyburl"].get("avg_support", 0.0),
                "high_confidence_ratio": model_metrics["anyburl"].get("high_confidence_ratio", 0.0),
                "coverage": model_metrics["anyburl"].get("coverage", 0.0),
                "positive_rule_coverage": model_metrics["anyburl"].get("positive_rule_coverage", 0.0),
            }
            anyburl_params = {
                "model": "AnyBURL",
                "hyperparameters": {
                    "rule_confidence": params.get("rule_confidence"),
                    "rule_support": params.get("rule_support"),
                    "max_rule_length": params.get("max_rule_length"),
                },
                "metrics": anyburl_metrics_payload,
                "weight_in_ensemble": params.get("rules_weight"),
            }
            anyburl_file = best_models_dir / "best_params_anyburl.json"
            self.file_manager.save(anyburl_params, anyburl_file)
            logger.info(f"   Parametros AnyBURL salvos: {anyburl_file}")

        if trial_result["models_trained"].get("lightgbm"):
            lightgbm_payload = {
                "model": "LightGBM",
                "hyperparameters": {
                    "meta_learning_rate": params.get("meta_learning_rate"),
                    "meta_n_estimators": params.get("meta_n_estimators"),
                    "negative_ratio": params.get("negative_ratio"),
                },
                "metrics": {
                    "auc": model_metrics["lightgbm"].get("auc", 0.0),
                    "f1": model_metrics["lightgbm"].get("f1", 0.0),
                    "accuracy": model_metrics["lightgbm"].get("accuracy", 0.0),
                    "pr_auc": model_metrics["lightgbm"].get("pr_auc", 0.0),
                    "mcc": model_metrics["lightgbm"].get("mcc", 0.0),
                    "train_auc": model_metrics["lightgbm"].get("train_auc", 0.0),
                    "generalization_gap": model_metrics["lightgbm"].get("generalization_gap", 0.0),
                    "ece": model_metrics["lightgbm"].get("ece", 0.0),
                    "entropy": model_metrics["lightgbm"].get("entropy", 0.0),
                },
                "weight_in_ensemble": params.get("lightgbm_weight"),
            }
            lightgbm_file = best_models_dir / "best_params_lightgbm.json"
            self.file_manager.save(lightgbm_payload, lightgbm_file)
            logger.info(f"   Parametros LightGBM salvos: {lightgbm_file}")

        if ensemble_metrics:
            ensemble_file = best_models_dir / "best_params_ensemble.json"
            self.file_manager.save(ensemble_metrics, ensemble_file)
            logger.info(f"   Parametros ensemble salvos: {ensemble_file}")

    def cleanup_trial_dir(self, trial_dir: Path) -> None:
        """Remove temporary trial directory."""
        if trial_dir.exists():
            deleted = self.file_manager.delete_directory(trial_dir, ignore_errors=True)
            if deleted:
                logger.debug(f"  → Cleaned up trial directory: {trial_dir}")
