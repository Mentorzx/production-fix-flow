"""RotatE Ensemble Wrapper Module.

Provides scikit-learn compatible wrappers for RotatE models,
enabling their use in ensemble methods like StackingClassifier.

Design Patterns Applied:
    - **Adapter Pattern:** Adapts RotatE to sklearn interface.
    - **Template Method:** Inherits from BaseWrapper for common behavior.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from pff.utils import logger
from pff.utils.global_interrupt_manager import should_stop
from pff.validators.kg.config import KGConfig
from pff.validators.rotate.rotate_service import RotatEScorerService
from pff.validators.ensembles.ensemble_wrappers.base_wrapper import BaseWrapper


class RotatEWrapper(BaseWrapper):
    """Wrapper for RotatE model to behave like a scikit-learn classifier.

    This wrapper encapsulates the RotatE scoring service to make it
    compatible with the scikit-learn API, allowing it to be used as
    a base model in ensemble methods like StackingClassifier.

    RotatE uses complex embeddings with rotational transformations,
    which are particularly effective for sparse graphs and capturing
    symmetric/antisymmetric relation patterns.

    Attributes:
        kg_config_path: Path to KG configuration.
        rotate_config_path: Path to RotatE configuration.
        scorer_service_: Initialized scorer service.

    Example:
        >>> wrapper = RotatEWrapper(
        ...     kg_config_path="config/models/kg.yaml",
        ...     rotate_config_path="config/models/rotate.yaml"
        ... )
        >>> wrapper.fit(X_train, y_train)
        >>> predictions = wrapper.predict(X_test)
    """

    def __init__(self, kg_config_path: str, rotate_config_path: str):
        """Initialize RotatE wrapper.

        Args:
            kg_config_path: Path to KG configuration YAML.
            rotate_config_path: Path to RotatE configuration YAML.
        """
        super().__init__()
        self.kg_config_path = kg_config_path
        self.rotate_config_path = rotate_config_path
        self.scorer_service_: RotatEScorerService | None = None
        self.timeout = 30.0
        self._cache_key = (
            f"rotate_{Path(kg_config_path).stem}_{Path(rotate_config_path).stem}"
        )

    def __getstate__(self):
        """Custom serialization - remove non-picklable objects."""
        state = super().__getstate__()
        state["scorer_service_"] = None
        return state

    def __setstate__(self, state):
        """Custom deserialization - restore state."""
        super().__setstate__(state)

    def _ensure_scorer_service(self):
        """Initialize scorer service if not available, with caching."""
        if self.scorer_service_ is None:
            cached_config = self.cache_manager.get(self._cache_key)
            if cached_config is not None:
                logger.debug("RotatE scorer config carregado do cache")

            logger.info("Re-inicializando servico de scoring RotatE...")
            try:
                kg_config = KGConfig(self.kg_config_path)
                self.scorer_service_ = RotatEScorerService(
                    kg_config, Path(self.rotate_config_path), load_best_model=True
                )
                self.cache_manager.set(
                    self._cache_key,
                    {
                        "kg_config_path": self.kg_config_path,
                        "rotate_config_path": self.rotate_config_path,
                    },
                    ttl=3600,  # 1 hour cache
                )
            except Exception as e:
                logger.error(f"CRITICAL error initializing RotatE scorer: {str(e)}", exc_info=True)
                raise

    def fit(self, X, y=None):
        """Initialize RotatE scorer service with pre-trained model.

        Args:
            X: Input data (triples or feature vectors).
            y: Target labels (not used, kept for sklearn compatibility).

        Returns:
            self: Returns the fitted wrapper.
        """
        logger.info("Inicializando wrapper RotatE com modelo pre-treinado...")
        try:
            kg_config = KGConfig(self.kg_config_path)
            logger.debug(f"KG config carregado de: {self.kg_config_path}")
            self.scorer_service_ = RotatEScorerService(
                kg_config, Path(self.rotate_config_path), load_best_model=True
            )
            logger.success("Servico de scoring RotatE inicializado com sucesso")
        except Exception as e:
            logger.error(f"Failed to initialize RotatE: {str(e)}", exc_info=True)
            raise
        return self

    def predict(self, X: list[Any]) -> np.ndarray:
        """Predict classes based on probability scores.

        Args:
            X: List of samples, where each sample is a list of triples.

        Returns:
            Array of predicted classes (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: list[Any]) -> np.ndarray:
        """Predict probability of samples being valid using parallel processing.

        Args:
            X: List of samples, where each sample is a list of triples.

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class.
        """
        check_is_fitted(self, "scorer_service_")
        self._ensure_scorer_service()

        n_samples = len(X)
        probabilities = np.zeros((n_samples, 2))

        sample_data = [
            (idx, sample, self.scorer_service_) for idx, sample in enumerate(X)
        ]

        results = self.concurrency_manager.execute_sync(
            RotatEWrapper._score_sample_static,
            sample_data,
            max_workers=4,
            desc="Pontuando amostras RotatE",
            task_type="thread",
        )

        for idx, score in results:
            probabilities[idx, 1] = score
            probabilities[idx, 0] = 1 - score

        return probabilities

    @staticmethod
    def _score_sample_static(
        idx: int,
        sample_triples: list[Any],
        scorer_service: RotatEScorerService,
    ) -> tuple[int, float]:
        """Static method to score a single sample for multiprocessing.

        Args:
            idx: Sample index.
            sample_triples: List of triples for this sample.
            scorer_service: RotatE scorer service.

        Returns:
            Tuple of (index, probability).
        """
        if should_stop():
            return idx, 0.5

        scores = []
        for triple in sample_triples:
            try:
                head, relation, tail = map(str, triple)
                if scorer_service:
                    score = scorer_service.score_triple(head, relation, tail)
                    probability = scorer_service.score_to_probability(score)
                    scores.append(probability)
            except Exception:
                scores.append(0.5)

        return idx, float(np.mean(scores)) if scores else 0.5


class RotatEHybridWrapper(BaseWrapper):
    """Wrapper for RotatE + LightGBM hybrid model.

    This wrapper combines RotatE embeddings with LightGBM for enhanced
    triple classification. All dependencies must be injected via constructor.

    Attributes:
        lightgbm_model: Pre-trained LightGBM model.
        entity_to_idx: Mapping from entity names to indices.
        relation_to_idx: Mapping from relation names to indices.
        entity_embeddings: Combined entity embeddings (real + imag).
        relation_embeddings: Combined relation embeddings (cos + sin).
    """

    def __init__(
        self,
        lightgbm_model,
        entity_to_idx: dict[str, int],
        relation_to_idx: dict[str, int],
        entity_embeddings: np.ndarray,
        relation_embeddings: np.ndarray,
        lightgbm_model_path: str | Path | None = None,
    ):
        """Initialize RotatE hybrid wrapper.

        Args:
            lightgbm_model: Pre-trained LightGBM model.
            entity_to_idx: Mapping from entity names to indices.
            relation_to_idx: Mapping from relation names to indices.
            entity_embeddings: Combined entity embeddings (real + imag).
            relation_embeddings: Combined relation embeddings (cos + sin).
            lightgbm_model_path: Optional path for model serialization.
        """
        super().__init__()
        self.lightgbm_model_path = str(lightgbm_model_path) if lightgbm_model_path else None
        self.lightgbm_model = lightgbm_model
        self.model_ = lightgbm_model
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        # Compute mean embeddings for unknown entities
        self.mean_entity_embedding_ = (
            np.mean(self.entity_embeddings, axis=0)
            if self.entity_embeddings is not None
            else None
        )
        self.mean_relation_embedding_ = (
            np.mean(self.relation_embeddings, axis=0)
            if self.relation_embeddings is not None
            else None
        )

        self._embedding_dim = (
            self.entity_embeddings.shape[1]
            if self.entity_embeddings is not None
            else 0
        )

    def fit(self, X, y=None):
        """Fit method (no-op as model is pre-loaded).

        Args:
            X: Input data (ignored).
            y: Target labels (ignored).

        Returns:
            self: Returns the wrapper.
        """
        return self

    def predict(self, X: list[Any]) -> np.ndarray:
        """Predict classes based on probability scores.

        Args:
            X: List of samples.

        Returns:
            Array of predicted classes.
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: list[Any]) -> np.ndarray:
        """Predict probabilities using LightGBM with RotatE embeddings.

        Args:
            X: List of samples, each containing triples.

        Returns:
            Array of shape (n_samples, 2) with probabilities.
        """
        if self.model_ is None:
            logger.warning("LightGBM model not loaded")
            return np.full((len(X), 2), 0.5)

        # Extract features from triples
        features = self._extract_features(X)

        # Predict with LightGBM
        try:
            if hasattr(self.model_, "predict_proba"):
                return self.model_.predict_proba(features)
            else:
                probs = self.model_.predict(features)
                return np.column_stack([1 - probs, probs])
        except Exception as e:
            logger.error(f"RotatE hybrid prediction failed: {e}", exc_info=True)
            return np.full((len(X), 2), 0.5)

    def _extract_features(self, X: list[Any]) -> np.ndarray:
        """Extract embedding features from triple samples.

        Args:
            X: List of samples.

        Returns:
            Feature array for LightGBM.
        """
        feature_list = []
        for sample in X:
            sample_features = []
            for triple in sample:
                try:
                    head, relation, tail = map(str, triple)
                    head_idx = self.entity_to_idx.get(head)
                    rel_idx = self.relation_to_idx.get(relation)
                    tail_idx = self.entity_to_idx.get(tail)

                    # Get embeddings
                    h_emb = (
                        self.entity_embeddings[head_idx]
                        if head_idx is not None
                        else self.mean_entity_embedding_
                    )
                    r_emb = (
                        self.relation_embeddings[rel_idx]
                        if rel_idx is not None
                        else self.mean_relation_embedding_
                    )
                    t_emb = (
                        self.entity_embeddings[tail_idx]
                        if tail_idx is not None
                        else self.mean_entity_embedding_
                    )

                    # Concatenate embeddings
                    if h_emb is not None and r_emb is not None and t_emb is not None:
                        sample_features.append(np.concatenate([h_emb, r_emb, t_emb]))
                except Exception:
                    pass

            if sample_features:
                # Average features across triples in sample
                feature_list.append(np.mean(sample_features, axis=0))
            else:
                # Use zeros for samples without valid features
                feature_dim = self._embedding_dim * 3
                feature_list.append(np.zeros(feature_dim))

        return np.array(feature_list)
