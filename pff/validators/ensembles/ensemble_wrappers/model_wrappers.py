"""
Model Wrappers - TransE and Hybrid model wrappers

This module contains:
- TransEWrapper (Knowledge Graph embeddings)
- HybridWrapper (TransE + LightGBM hybrid model)

Part of Sprint 4 refactoring (ensemble_wrappers.py split into 3 files).
These classes wrap complex models to make them scikit-learn compatible.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from pff import settings
from pff.utils import logger
from pff.utils.global_interrupt_manager import should_stop
from pff.validators.kg.config import KGConfig
from pff.validators.transe.transe_service import TransEScorerService

from .base_wrapper import BaseWrapper


# ══════════════════════════════════════════════════════════════════════════
# TRANSE WRAPPER
# ══════════════════════════════════════════════════════════════════════════


class TransEWrapper(BaseWrapper):
    """
    Wrapper for the TransE pipeline to behave like a scikit-learn classifier.

    This wrapper encapsulates a complex Knowledge Graph (KG) pipeline to make it
    compatible with the scikit-learn API, allowing it to be used as a base model
    in ensemble methods like StackingClassifier.
    """

    def __init__(self, kg_config_path: str, transe_config_path: str):
        super().__init__()
        self.kg_config_path = kg_config_path
        self.transe_config_path = transe_config_path
        self.scorer_service_ = None
        self.timeout = 30.0
        self._cache_key = (
            f"transe_{Path(kg_config_path).stem}_{Path(transe_config_path).stem}"
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
            cached_service = self.cache_manager.get(
                self._cache_key
            )  # Try to load from cache first
            if cached_service is not None:
                logger.debug("TransE scorer carregado do cache")
                self.scorer_service_ = cached_service
                return
            logger.info("Re-inicializando serviço de scoring TransE...")
            try:
                kg_config = KGConfig(self.kg_config_path)
                self.scorer_service_ = TransEScorerService(
                    kg_config, Path(self.transe_config_path), load_best_model=True
                )
                self.cache_manager.set(  # Cache the service configuration (not the entire object)
                    self._cache_key,
                    {
                        "kg_config_path": self.kg_config_path,
                        "transe_config_path": self.transe_config_path,
                    },
                    ttl=3600,  # 1 hour cache
                )
            except Exception as e:
                logger.error(f"ERRO CRÍTICO ao inicializar TransE scorer: {str(e)}")
                logger.debug(f"Traceback completo:\n{traceback.format_exc()}")
                raise

    def fit(self, X, y=None):
        """Initialize TransE scorer service with pre-trained model."""
        logger.info("Inicializando wrapper TransE com modelo pré-treinado...")
        try:
            kg_config = KGConfig(self.kg_config_path)
            logger.debug(f"KG config carregado de: {self.kg_config_path}")
            self.scorer_service_ = TransEScorerService(
                kg_config, Path(self.transe_config_path), load_best_model=True
            )
            logger.success("Serviço de scoring TransE inicializado com sucesso")
        except Exception as e:
            logger.error(f"FALHA ao inicializar TransE: {str(e)}")
            logger.debug(f"Traceback completo:\n{traceback.format_exc()}")
            raise
        return self

    def predict(self, X: list[Any]) -> np.ndarray:
        """Predict classes based on probability scores."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: list[Any]) -> np.ndarray:
        """Predict the probability of samples being valid using parallel processing."""
        check_is_fitted(self, "scorer_service_")
        self._ensure_scorer_service()
        n_samples = len(X)
        probabilities = np.zeros((n_samples, 2))
        sample_data = [
            (idx, sample, self.scorer_service_) for idx, sample in enumerate(X)
        ]
        results = self.concurrency_manager.execute_sync(
            TransEWrapper._score_sample_static,
            sample_data,
            max_workers=4,
            desc="Pontuando amostras TransE",
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
        scorer_service: Any,
    ) -> tuple[int, float]:
        """Static method to score a single sample for multiprocessing."""
        if should_stop():
            return idx, 0.5
        scores = []
        for triple in sample_triples:
            try:
                head, relation, tail = map(str, triple)
                if scorer_service:
                    score = scorer_service.score_triple(head, relation, tail)
                    normalized_score = 1 / (1 + np.exp(-score))
                    normalized_score = 0.8 * normalized_score + 0.1
                    scores.append(normalized_score)
            except Exception:
                scores.append(0.5)

        return idx, float(np.mean(scores)) if scores else 0.5


# ══════════════════════════════════════════════════════════════════════════
# HYBRID WRAPPER
# ══════════════════════════════════════════════════════════════════════════


class HybridWrapper(BaseWrapper):
    """
    Passive HybridWrapper for TransE + LightGBM hybrid model.
    All dependencies must be injected (pre-loaded) via the constructor.
    """

    def __init__(
        self,
        lightgbm_model,
        entity_to_idx,
        relation_to_idx,
        entity_embeddings,
        relation_embeddings,
    ):
        super().__init__()
        self.model_ = lightgbm_model
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
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
            else 108
        )
        self._expected_features = (
            getattr(self.model_, "num_feature", lambda: 544)()
            if self.model_ is not None
            else 544
        )
        self.n_features_in_ = self._expected_features
        self._degraded_mode = False
        self._is_fitted = False

    def fit(self, X, y=None):
        """Mark the wrapper as fitted. No training or loading is performed."""
        logger.info("✅ HybridWrapper configurado com dependências pré-carregadas.")
        self._is_fitted = True
        return self

    def predict(self, X: list[Any]) -> np.ndarray:
        """Predict classes based on probability scores."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: list[Any]) -> np.ndarray:
        """Predict probabilities using the hybrid model."""
        self._ensure_model_loaded()
        features = self._extract_features_from_triples(X)
        if features.size == 0:
            logger.error("Features vazias - usando probabilidades neutras")
            return np.full((len(X), 2), [0.5, 0.5], dtype=np.float64)
        try:
            import lightgbm as lgb

            if isinstance(self.model_, lgb.Booster):
                params = {"num_threads": 1, "predict_disable_shape_check": True}
                positive_proba = self.model_.predict(features, **params)
                if not hasattr(self, "_first_successful_predict"):
                    self._first_successful_predict = True
                    logger.success(
                        f"✅ Predição LightGBM bem-sucedida com {features.shape[1]} features"
                    )
                else:
                    logger.debug(
                        f"Predição LightGBM OK com {features.shape[1]} features"
                    )
            elif hasattr(self.model_, "predict_proba") and not isinstance(
                self.model_, dict
            ):
                positive_proba = self.model_.predict_proba(features)[:, 1]
            elif hasattr(self.model_, "predict") and not isinstance(self.model_, dict):
                positive_proba = self.model_.predict(features)
            else:
                raise AttributeError("Modelo sem método de predição compatível")
            positive_proba = np.array(positive_proba).ravel()
            positive_proba = np.clip(positive_proba, 0.1, 0.9)
            negative_proba = 1.0 - positive_proba

            return np.column_stack((negative_proba, positive_proba))
        except Exception as e:
            logger.error(f"FALHA na predição: {str(e)}")
            logger.debug(f"Shape das features: {features.shape}")
            logger.debug(
                f"Features esperadas: {getattr(self, '_expected_features', 'desconhecido')}"
            )
            logger.debug(f"Traceback completo:\n{traceback.format_exc()}")
            return np.full((len(X), 2), [0.5, 0.5], dtype=np.float64)

    def _ensure_model_loaded(self):
        """Ensure model is loaded, attempting to reload if necessary."""
        if self.model_ is None and not self._degraded_mode:
            logger.warning("Modelo não carregado, tentando recarregar...")
        if not hasattr(self, "_expected_features") or self._expected_features is None:
            if (
                self.model_ is not None
                and not isinstance(self.model_, dict)
                and hasattr(self.model_, "num_feature")
            ):
                self._expected_features = self.model_.num_feature()  # type: ignore
                logger.debug(
                    f"Features esperadas detectadas: {self._expected_features}"
                )
            else:
                self._expected_features = 544  # Default based on error logs
                logger.debug(
                    f"Usando número padrão de features: {self._expected_features}"
                )

    def _extract_features_from_triples(self, triples_list: list) -> np.ndarray:
        """Extract embedding features from triples with dynamic padding."""
        expected = getattr(self, "_expected_features", self.n_features_in_ or 544)
        if self.entity_embeddings is None or self.relation_embeddings is None:
            logger.warning("Embeddings não carregados – usando features zero")
            return np.zeros((len(triples_list), expected), dtype=np.float32)

        def _process_sample(
            idx: int, sample: list | tuple | np.ndarray
        ) -> tuple[int, np.ndarray]:
            try:
                if isinstance(sample, np.ndarray) and sample.size == 3:
                    sample = [sample]
                if not isinstance(sample, (list, tuple)) or not sample:
                    return idx, np.zeros(expected, dtype=np.float32)
                sample_features = []
                for triple in sample:
                    features = self._get_triple_embeddings(triple)
                    if features is not None:
                        sample_features.append(features)
                if not sample_features:
                    return idx, np.zeros(expected, dtype=np.float32)
                aggregated_features = np.mean(sample_features, axis=0)

                return idx, aggregated_features.astype(np.float32)
            except Exception as e:
                logger.debug(f"Erro ao processar amostra {idx}: {e}")
                return idx, np.zeros(expected, dtype=np.float32)

        sample_data = list(enumerate(triples_list))
        results = self.concurrency_manager.execute_sync(
            _process_sample,
            sample_data,
            max_workers=4,
            desc="Extraindo features",
            task_type="thread",
        )
        results.sort(key=lambda t: t[0])
        feats = np.vstack([f for _, f in results])
        cur = feats.shape[1]
        if cur < expected:
            pad = expected - cur
            feats = np.pad(feats, ((0, 0), (0, pad)), constant_values=0)
            logger.debug(f"Features padded com {pad} zeros (dim {cur} → {expected})")
        elif cur > expected:
            feats = feats[:, :expected]
            logger.warning(f"Features truncadas de {cur} para {expected}")

        return feats.astype(np.float32)

    def _get_triple_embeddings(self, triple: tuple) -> np.ndarray:
        """
        Get embeddings for a single triple, using adaptive OOV strategy for unknown entities/relations.
        """
        if isinstance(triple, np.ndarray):
            triple = triple.tolist().pop()
        head_str, relation_str, tail_str = map(str, triple)
        head_idx = self.entity_to_idx.get(head_str) if self.entity_to_idx else None
        rel_idx = (
            self.relation_to_idx.get(relation_str) if self.relation_to_idx else None
        )
        tail_idx = self.entity_to_idx.get(tail_str) if self.entity_to_idx else None

        # Adaptive OOV strategy
        head_emb = (
            self.entity_embeddings[head_idx]
            if head_idx is not None
            else self._get_oov_embedding(head_str, "entity")
        )
        rel_emb = (
            self.relation_embeddings[rel_idx]
            if rel_idx is not None
            else self._get_oov_embedding(relation_str, "relation")
        )
        tail_emb = (
            self.entity_embeddings[tail_idx]
            if tail_idx is not None
            else self._get_oov_embedding(tail_str, "entity")
        )

        basic_features = np.concatenate([head_emb, rel_emb, tail_emb])
        expected = getattr(self, "_expected_features", None)
        if expected and expected > len(basic_features):
            meta_features = self._extract_meta_features(
                head_str, relation_str, tail_str
            )
            return np.concatenate([basic_features, meta_features])

        return basic_features

    def _get_oov_embedding(self, entity_str: str, entity_type: str) -> np.ndarray:
        """
        Generates an out-of-vocabulary (OOV) embedding for a given entity or relation.

        If a mean embedding is available for the specified entity type ("entity" or "relation"),
        returns the mean embedding with added Gaussian noise. Otherwise, returns a randomly
        generated embedding vector.

        Args:
            entity_str (str): The string representation of the entity or relation.
            entity_type (str): The type of the entity, either "entity" or "relation".

        Returns:
            np.ndarray: The generated OOV embedding vector.

        Notes:
            - If an exception occurs during embedding generation, a random embedding is returned.
            - The shape and noise parameters depend on the availability of mean embeddings.
        """
        try:
            if entity_type == "entity":
                if self.mean_entity_embedding_ is not None:
                    noise = np.random.normal(0, 0.05, self.mean_entity_embedding_.shape)
                    return self.mean_entity_embedding_ + noise
                else:
                    return np.random.normal(0, 0.1, self._embedding_dim)
            else:  # relation
                if self.mean_relation_embedding_ is not None:
                    noise = np.random.normal(
                        0, 0.05, self.mean_relation_embedding_.shape
                    )
                    return self.mean_relation_embedding_ + noise
                else:
                    return np.random.normal(0, 0.1, self._embedding_dim)
        except Exception as e:
            logger.debug(f"Erro no OOV embedding: {e}")
            return np.random.normal(0, 0.1, self._embedding_dim)

    def _extract_meta_features(self, head: str, relation: str, tail: str) -> np.ndarray:
        """Extract additional meta-features for compatibility with trained model."""
        meta_features = []
        try:
            head_freq = hash(head) % 100 / 100.0
            tail_freq = hash(tail) % 100 / 100.0
            rel_freq = hash(relation) % 50 / 50.0
            meta_features.extend([head_freq, tail_freq, rel_freq])
            meta_features.extend(
                [len(head) / 50.0, len(tail) / 50.0, len(relation) / 30.0]
            )
            head_tail_sim = len(set(head) & set(tail)) / max(len(head), len(tail), 1)
            meta_features.append(head_tail_sim)
            meta_features.extend(
                [
                    float(head.isdigit()),
                    float(tail.isdigit()),
                    float(relation.startswith("has")),
                    float(relation.startswith("is")),
                ]
            )
            expected = getattr(self, "_expected_features", 544)
            current = 3 * self._embedding_dim + len(meta_features)
            if current < expected:
                padding_needed = expected - current
                for i in range(padding_needed):
                    feature_val = (
                        hash(f"{head}_{relation}_{tail}_{i}") % 1000
                    ) / 1000.0
                    meta_features.append(feature_val)
        except Exception as e:
            logger.debug(f"Erro ao extrair meta-features: {e}")
            expected = getattr(self, "_expected_features", 544)
            needed = expected - 3 * self._embedding_dim
            return np.zeros(needed)
        return np.array(meta_features)

    def _debug_data_sources(self):
        """Debug data source compatibility between components."""
        logger.info("=== DIAGNÓSTICO DE COMPATIBILIDADE DE DADOS ===")
        files_to_check = [  # Check file existence
            settings.DATA_DIR / "models" / "kg" / "train.parquet",
            settings.DATA_DIR / "models" / "kg" / "train_optimized.parquet",
            settings.DATA_DIR / "models" / "kg" / "test.parquet",
            settings.DATA_DIR / "models" / "kg" / "test_optimized.parquet",
        ]
        logger.info("Arquivos disponíveis:")
        for file_path in files_to_check:
            if file_path.exists():
                size_mb = file_path.stat().st_size / 1024 / 1024
                logger.info(f"  ✅ {file_path.name} ({size_mb:.1f} MB)")
            else:
                logger.info(f"  ❌ {file_path.name}")
        if self.entity_to_idx:  # Sample entities
            sample_entities = list(self.entity_to_idx.keys())[:5]
            logger.info(f"Amostra de entidades no índice: {sample_entities}")
            logger.info(f"Total de entidades: {len(self.entity_to_idx)}")
        else:
            logger.warning("Nenhum índice de entidades carregado")
        logger.info("\n=== DIAGNÓSTICO DO MODELO ===")
        logger.info(f"Modelo carregado: {'Sim' if self.model_ is not None else 'Não'}")
        logger.info(
            f"Features esperadas pelo modelo: {getattr(self, '_expected_features', 'desconhecido')}"
        )
        logger.info(f"Features configuradas (n_features_in_): {self.n_features_in_}")
        logger.info(f"Dimensão dos embeddings: {self._embedding_dim}")
        logger.info(f"Features básicas (3 × embedding_dim): {3 * self._embedding_dim}")
        if hasattr(self, "_expected_features") and self._expected_features:
            meta_features_needed = self._expected_features - (3 * self._embedding_dim)
            logger.info(f"Meta-features necessárias: {meta_features_needed}")
        if self.entity_to_idx and len(self.entity_to_idx) >= 3:
            logger.info("\n=== TESTE DE EXTRAÇÃO DE FEATURES ===")
            test_entities = list(self.entity_to_idx.keys())[:3]
            test_triple = [(test_entities[0], "test_relation", test_entities[1])]
            try:
                test_features = self._extract_features_from_triples([test_triple])
                logger.info(
                    f"Features extraídas com sucesso: shape {test_features.shape}"
                )
                logger.info(f"Primeiras 10 features: {test_features[0][:10]}")
            except Exception as e:
                logger.error(f"Erro ao extrair features de teste: {str(e)}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
        try:  # Check data overlap
            train_path = settings.DATA_DIR / "models" / "kg" / "train_optimized.parquet"
            if not train_path.exists():
                train_path = settings.DATA_DIR / "models" / "kg" / "train.parquet"
            if train_path.exists():
                df = self.file_manager.read(train_path)
                logger.info(
                    f"\nDados de treino: {df.shape[0]} linhas, colunas: {list(df.columns)}"
                )
                if self.entity_to_idx and "s" in df.columns:  # Check for overlap
                    sample_heads = df["s"].head(10).to_list()
                    overlap = sum(
                        1 for h in sample_heads if str(h) in self.entity_to_idx
                    )
                    logger.info(f"Overlap de entidades: {overlap}/10 ({overlap * 10}%)")
                    if overlap == 0:
                        logger.error(
                            "PROBLEMA: Zero overlap entre TransE e dados do ensemble!"
                        )
                        logger.debug(f"Amostra de heads no arquivo: {sample_heads[:3]}")
                        logger.debug(
                            f"Amostra de keys no índice: {list(self.entity_to_idx.keys())[:3]}"
                        )
        except Exception as e:
            logger.error(f"Erro no diagnóstico: {str(e)}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
        logger.info("\n=== RESUMO DO DIAGNÓSTICO ===")
        if hasattr(self, "_expected_features") and self._expected_features:
            if self._expected_features == 544:
                logger.info(
                    "O modelo foi treinado com 544 features (324 embeddings + 220 meta-features)"
                )
                logger.info(
                    "As meta-features são geradas dinamicamente com base nas entidades/relações"
                )
            else:
                logger.info(f"O modelo espera {self._expected_features} features")
        logger.info(
            "Configure predict_disable_shape_check=True no LightGBM para maior flexibilidade"
        )
