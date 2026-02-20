"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/shap_explainer.py

"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pff.shared.core.config import EXPLAINABILITY_CONFIG_PATH, settings
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger
from pff_rust import stable_hash

"""
SHAP explainability helpers.

Design Patterns:
- Factory Pattern: selects appropriate SHAP explainer automatically.
- Template Method: normalized pipeline (prepare → build → explain → persist).
- Adapter Pattern: accepts numpy/polars inputs and returns SHAP-native explanations.
"""


@dataclass
class ShapExplainerConfig:
    """Typed configuration for SHAP explainability."""

    enabled: bool = True
    max_background: int = 128
    max_samples: int = 512
    output_dir: Path = settings.OUTPUTS_DIR / "explainability"
    save_format: str = "parquet"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShapExplainerConfig:
        """Execute from dict.



        Args:

            data: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        shap_cfg = data.get("shap", data)
        output_dir_cfg = shap_cfg.get("output_dir", settings.OUTPUTS_DIR / "explainability")
        if output_dir_cfg is None or str(output_dir_cfg) == "":
            output_dir_cfg = settings.OUTPUTS_DIR / "explainability"
        return cls(
            enabled=bool(shap_cfg.get("enabled", True)),
            max_background=int(shap_cfg.get("max_background", 128)),
            max_samples=int(shap_cfg.get("max_samples", 512)),
            output_dir=Path(output_dir_cfg),
            save_format=str(shap_cfg.get("save_format", "parquet")),
        )


class ShapExplainerService:
    """
    SHAP explainability service.

    Pattern: Template Method (prepare → build → explain → persist).
    """

    def __init__(
        self,
        config_path: Path | None = None,
        config_data: dict[str, Any] | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        """Initialize SHAP explainer service."""
        self.file_manager = file_manager or FileManager()
        self.config = self._load_config(config_path, config_data)

    def _load_config(
        self,
        config_path: Path | None,
        config_data: dict[str, Any] | None,
    ) -> ShapExplainerConfig:
        """Execute load config.



        Args:

            config_path: Input value used by this callable.

            config_data: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if config_data is not None:
            return ShapExplainerConfig.from_dict(config_data)

        resolved_path = config_path or EXPLAINABILITY_CONFIG_PATH
        if resolved_path.exists():
            try:
                content = self.file_manager.read(resolved_path, return_native=True)
                return ShapExplainerConfig.from_dict(content or {})
            except Exception as exc:
                logger.warning(f"Failed to read SHAP config: {exc}")

        return ShapExplainerConfig()

    def explain(
        self,
        model: Any,
        data: Any,
        *,
        background_data: Any | None = None,
        feature_names: list[str] | None = None,
        save: bool = False,
        artifact_name: str = "shap_values",
    ):
        """
        Compute SHAP values for a model.

        Args:
            model: Fitted model with predict or predict_proba.
            data: Samples to explain (array-like).
            background_data: Optional background for the explainer.
            feature_names: Optional feature names for persistence.
            save: Whether to persist shap values to outputs.
            artifact_name: Base name for persisted artifact.

        Returns:
            shap.Explanation with values for the requested samples.
        """
        if not self.config.enabled:
            logger.info("SHAP desabilitado via configuração")
            return None

        X = self._to_numpy(data)
        if X.shape[0] == 0:
            logger.warning("No samples provided for SHAP computation")
            return None

        X_sampled = self._sample_rows(X, self.config.max_samples)
        background = self._prepare_background(background_data, X_sampled)

        explainer = self._build_explainer(model, background)
        if explainer is None:
            logger.warning("Unable to build SHAP explainer for the provided model")
            return None

        logger.debug(f"Computing SHAP values for {X_sampled.shape[0]} samples")
        explanation = explainer(X_sampled)

        if save:
            self._persist_explanation(explanation, feature_names or [], artifact_name)

        return explanation

    def _build_explainer(self, model: Any, background: np.ndarray):
        """Factory method to create a SHAP explainer."""
        try:
            import shap
        except Exception as exc:
            logger.error(f"Failed to import shap: {exc}")
            return None

        try:
            return shap.Explainer(model, background)
        except Exception as exc:
            logger.warning(
                f"SHAP explainer creation failed; falling back to KernelExplainer: {exc}"
            )
            try:
                return shap.KernelExplainer(model.predict, background)
            except Exception as inner_exc:
                logger.error(f"SHAP KernelExplainer failed: {inner_exc}")
                return None

    def _prepare_background(self, background_data: Any | None, X: np.ndarray) -> np.ndarray:
        """Execute prepare background.



        Args:

            background_data: Input value used by this callable.

            X: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if background_data is not None:
            background_np = self._to_numpy(background_data)
        else:
            background_np = X

        return self._sample_rows(background_np, self.config.max_background)

    def _sample_rows(self, data: np.ndarray, max_rows: int) -> np.ndarray:
        """Execute sample rows.



        Args:

            data: Input value used by this callable.

            max_rows: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if data.shape[0] <= max_rows:
            return data
        seed = stable_hash(
            (
                data.shape[0],
                data.shape[1] if data.ndim > 1 else 1,
                max_rows,
            )
        ) % (2**32)
        rng = np.random.default_rng(seed)
        indices = rng.choice(data.shape[0], size=max_rows, replace=False)
        return data[indices]

    def _persist_explanation(
        self,
        explanation: Any,
        feature_names: list[str],
        artifact_name: str,
    ) -> None:
        """Persist SHAP values using FileManager."""
        shap_values = explanation.values
        if shap_values is None:
            logger.warning("No SHAP values to persist")
            return

        values_array = np.asarray(shap_values)
        if values_array.ndim == 3:
            values_array = values_array[:, 0, :]

        feature_labels = feature_names or [f"f{i}" for i in range(values_array.shape[1])]
        df = pl.DataFrame(values_array, schema=feature_labels)

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = output_dir / f"{artifact_name}.{self.config.save_format}"
        logger.info(f"Salvando explicações SHAP em {artifact_path}")
        try:
            self.file_manager.save(df, artifact_path)
            logger.success(f"Valores SHAP salvos em {artifact_path}")
        except Exception as exc:
            logger.error(f"Failed to persist SHAP values: {exc}")

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        """Convert supported inputs to numpy array."""
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, pl.DataFrame):
            return np.asarray(data.to_numpy())
        if hasattr(data, "to_numpy"):
            return np.asarray(data.to_numpy())

        return np.asarray(data)

    def persist_async(self, coro: Any) -> None:
        """Helper to run async persistence without blocking."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            from pff.shared.acceleration.asyncio_runner import run_coroutine_sync

            run_coroutine_sync(coro)
        else:
            loop.create_task(coro)
