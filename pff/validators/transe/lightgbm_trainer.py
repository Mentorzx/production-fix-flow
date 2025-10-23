from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence, cast

import joblib
import lightgbm as lgb
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pff import settings
from pff.utils import FileManager, logger

"""
TransE + LightGBM Hybrid Model Trainer

This module implements a hybrid approach combining TransE embeddings
with LightGBM for improved knowledge graph completion performance.
"""


class TransELightGBMTrainer:
    """
    Trainer for hybrid TransE + LightGBM model.

    This class combines TransE embeddings with LightGBM to create
    a powerful hybrid model for link prediction tasks.
    """

    def __init__(self, transe_manager):
        """
        Initialize the hybrid trainer.

        Args:
            transe_manager: Trained TransEManager instance
        """
        self.transe_manager = transe_manager
        self.file_manager = FileManager()
        self.lightgbm_model: lgb.Booster | None = None

        # Configuration
        self.embedding_dim = transe_manager.config["model"]["embedding_dim"]
        self.negative_ratio = 1  # Conservative negative sampling

        logger.info("✅ TransE+LightGBM Trainer inicializado")

    def create_lightgbm_dataset(
        self, data_path: Path | str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Builds the (X, y) set for LightGBM with optimized embedding handling."""
        data_path = Path(data_path)
        logger.info(f"📊 Criando dataset LightGBM de {data_path.name}...")
        if not hasattr(self.transe_manager, "node_embeddings"):
            embeddings: dict[str, Any] = self.extract_embeddings()
            self.transe_manager.node_embeddings = embeddings
        else:
            embeddings = self.transe_manager.node_embeddings
        entity_emb = embeddings.get("entity_embeddings")
        if entity_emb is None:
            entity_emb = embeddings.get("entity")
        relation_emb = embeddings.get("relation_embeddings")
        if relation_emb is None:
            relation_emb = embeddings.get("relation")
        if entity_emb is None or relation_emb is None:
            available_keys = list(embeddings.keys())
            raise KeyError(f"Embeddings ausentes. Chaves disponíveis: {available_keys}")
        df = self.file_manager.read(data_path)
        if {"s", "p", "o"}.issubset(df.columns):
            df = df.rename({"s": "head", "p": "relation", "o": "tail"})
        logger.debug(f"Dados carregados: {len(df):,} triplas")
        ent2idx = self.transe_manager.entity_to_idx
        rel2idx = self.transe_manager.relation_to_idx
        features: list[np.ndarray] = []
        meta: list[dict[str, str]] = []
        for row in df.iter_rows(named=True):
            h, r, t = map(str, (row["head"], row["relation"], row["tail"]))
            if h in ent2idx and t in ent2idx and r in rel2idx:
                h_vec = entity_emb[ent2idx[h]]
                t_vec = entity_emb[ent2idx[t]]
                r_vec = relation_emb[rel2idx[r]]
                concat = np.concatenate((h_vec, r_vec, t_vec), dtype=np.float32)
                delta = (h_vec + r_vec - t_vec).astype(np.float32)
                score = -float(np.linalg.norm(delta, ord=2))
                hadamard = h_vec * t_vec
                diff = np.abs(h_vec - t_vec)
                norms = np.array(
                    [
                        np.linalg.norm(h_vec),
                        np.linalg.norm(t_vec),
                        np.linalg.norm(r_vec),
                    ],
                    dtype=np.float32,
                )
                feat_vec = np.concatenate(
                    (concat, np.array([score], dtype=np.float32), hadamard, diff, norms)
                )
                features.append(feat_vec)
                meta.append({"head": h, "relation": r, "tail": t})
        if not features:
            raise ValueError("Nenhuma tripla válida encontrada!")
        X = np.vstack(features)
        y = np.ones(len(features), dtype=np.int8)

        logger.success(f"Features criadas: {len(X):,} válidas")
        return X, y, {"triples": meta}

    def extract_embeddings(self) -> dict[str, np.ndarray]:
        """Extract embeddings from TransE model with compatibility aliases."""
        logger.info("🔄 Extraindo embeddings do modelo TransE...")
        if self.transe_manager.model is None:
            raise RuntimeError("Modelo TransE não está carregado!")
        with torch.no_grad():
            entity_embeddings = (
                self.transe_manager.model.entity_embeddings.weight.cpu().numpy()
            )
            relation_embeddings = (
                self.transe_manager.model.relation_embeddings.weight.cpu().numpy()
            )
        logger.success(
            f"Embeddings extraídos: entities={entity_embeddings.shape}, relations={relation_embeddings.shape}"
        )
        embeddings_path = settings.OUTPUTS_DIR / "transe" / "node_embeddings.pkl"
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = {
            "entity_embeddings": entity_embeddings,
            "relation_embeddings": relation_embeddings,
            "entity": entity_embeddings,
            "relation": relation_embeddings,
        }
        self.file_manager.save(embeddings, embeddings_path)
        logger.debug(f"Embeddings salvos em: {embeddings_path}")

        return embeddings

    def _get_relation_embeddings(self) -> np.ndarray:
        """Get relation embeddings from TransE model."""
        with torch.no_grad():
            return self.transe_manager.model.relation_embeddings.weight.cpu().numpy()

    def generate_negative_samples(
        self,
        X_pos: np.ndarray,
        y_pos: np.ndarray,
        num_negatives_per_positive: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate negative samples for training.

        MODIFIED: Now reads negative sampling ratio from config
        """
        # Get ratio from config if not provided
        if num_negatives_per_positive is None:
            transe_config = self.file_manager.read(settings.CONFIG_DIR / "transe.yaml")
            lgb_training = transe_config.get("lightgbm", {}).get("training", {})
            num_negatives_per_positive = lgb_training.get(
                "negative_sampling_ratio", 1.0
            )

        logger.info(
            f"🔄 Gerando amostras negativas (ratio: {num_negatives_per_positive})..."
        )

        # Rest of the method remains the same...
        num_positives = len(X_pos)
        if num_negatives_per_positive is None:
            raise ValueError("num_negatives_per_positive não pode ser None")
        num_negatives = int(num_positives * num_negatives_per_positive)

        if num_negatives == 0:
            logger.warning("⚠️ Nenhuma amostra negativa gerada")
            return X_pos, y_pos

        # Get embeddings
        entity_embeddings = self.transe_manager.node_embeddings["entity"]
        relation_embeddings = self._get_relation_embeddings()

        num_entities = len(entity_embeddings)
        num_relations = len(relation_embeddings)

        X_neg = []

        # Generate random negative triples
        rng = np.random.RandomState(42)

        for _ in range(num_negatives):
            # Random entities and relation
            head_idx = rng.randint(0, num_entities)
            tail_idx = rng.randint(0, num_entities)
            rel_idx = rng.randint(0, num_relations)

            # Extract embeddings
            head_emb = entity_embeddings[head_idx]
            tail_emb = entity_embeddings[tail_idx]
            rel_emb = relation_embeddings[rel_idx]

            # Create same features as positive samples
            concat_features = np.concatenate([head_emb, rel_emb, tail_emb])
            transe_score = -np.linalg.norm(head_emb + rel_emb - tail_emb)
            hadamard = head_emb * tail_emb
            diff = np.abs(head_emb - tail_emb)
            head_norm = np.linalg.norm(head_emb)
            tail_norm = np.linalg.norm(tail_emb)
            rel_norm = np.linalg.norm(rel_emb)

            feature_vector = np.concatenate(
                [
                    concat_features,
                    [transe_score],
                    hadamard,
                    diff,
                    [head_norm, tail_norm, rel_norm],
                ]
            )

            X_neg.append(feature_vector)

        X_neg = np.array(X_neg)
        y_neg = np.zeros(num_negatives)

        # Combine positive and negative samples
        X_combined = np.vstack([X_pos, X_neg])
        y_combined = np.concatenate([y_pos, y_neg])

        # Shuffle
        shuffle_idx = rng.permutation(len(X_combined))
        X_combined = X_combined[shuffle_idx]
        y_combined = y_combined[shuffle_idx]

        pos_ratio = np.mean(y_combined)
        logger.info(
            f"✅ Dataset balanceado: {len(X_combined):,} amostras "
            f"({pos_ratio:.1%} positivas)"
        )

        return X_combined, y_combined

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> lgb.Booster:
        """Train LightGBM model with parameters from transe.yaml."""
        logger.info("🌲 Treinando LightGBM com parâmetros do transe.yaml...")
        transe_config = self.file_manager.read(settings.CONFIG_DIR / "transe.yaml")
        lgb_config = transe_config.get("lightgbm", {})
        params = lgb_config.get("params", {})
        if not params:
            raise ValueError(
                "❌ Parâmetros LightGBM ausentes em transe.yaml! "
                "Execute 'python -m pff config optimize' para gerar configuração completa."
            )
        params.setdefault("objective", "binary")
        params.setdefault("metric", "auc")
        params.setdefault("verbose", -1)
        params.setdefault("random_state", 42)
        training_config = lgb_config.get("training", {})
        num_boost_round = training_config.get("num_boost_round", 50)
        early_stopping_rounds = training_config.get("early_stopping_rounds", 5)
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        logger.debug(
            f"Parâmetros: lr={params.get('learning_rate')}, features={params.get('feature_fraction')}"
        )
        self.lightgbm_model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=10),
            ],
        )
        best_iter = self.lightgbm_model.best_iteration
        train_auc = self.lightgbm_model.best_score["train"]["auc"]
        val_auc = self.lightgbm_model.best_score["val"]["auc"]
        logger.success(f"Treinamento concluído: iteração {best_iter}")
        logger.info(f"AUC treino: {train_auc:.4f}, validação: {val_auc:.4f}")
        model_path = settings.OUTPUTS_DIR / "transe" / "lightgbm_model.bin"
        model_path.parent.mkdir(exist_ok=True)
        self.save_model(model_path)
        return self.lightgbm_model

    def evaluate_lightgbm(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """
        Evaluate LightGBM model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.lightgbm_model is None:
            raise RuntimeError("Modelo LightGBM não foi treinado!")

        logger.info("📊 Avaliando modelo LightGBM...")

        # Predict probabilities
        raw_pred = self.lightgbm_model.predict(
            X_test, num_iteration=self.lightgbm_model.best_iteration
        )
        y_pred_proba = self._to_dense_1d(raw_pred)

        # Ensure proper shape with binary predictions
        threshold = getattr(self, "optimal_thresh", 0.5)
        y_pred = (y_pred_proba > threshold).astype(int)

        # Calculate metrics
        metrics = {}

        try:
            metrics["auc"] = float(roc_auc_score(y_test, y_pred_proba))
        except Exception as e:
            logger.warning(f"Erro ao calcular AUC: {e}")
            metrics["auc"] = 0.0

        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))

        # Log metrics
        logger.info("📈 Métricas LightGBM:")
        for metric_name, value in metrics.items():
            logger.info(f"   {metric_name}: {value:.4f}")

        return metrics

    def _to_dense_1d(
        self, x: np.ndarray | sparse.spmatrix | list[sparse.spmatrix]
    ) -> np.ndarray:
        """
        Ensure dense 1-D output (shape: [n_samples]) for any type that
        LightGBM may return.

        • spmatrix  -> .toarray().ravel()
        • list[...] -> np.hstack(...).ravel()
        • ndarray   -> np.asarray(x).ravel()
        """
        if sparse.issparse(x):
            return cast(np.ndarray, x.toarray()).ravel()  # type: ignore
        if isinstance(x, Sequence) and x and sparse.issparse(x[0]):
            dense_parts = [cast(np.ndarray, m.toarray()) for m in x]  # type: ignore
            return np.hstack(dense_parts).ravel()
        return np.asarray(x).ravel()

    def train_hybrid_model(self) -> dict[str, float]:
        """
        Train the complete hybrid TransE + LightGBM model.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("🚀 INICIANDO TREINAMENTO HÍBRIDO TransE + LightGBM")
        logger.info("=" * 70)
        try:
            if self.transe_manager is None or self.transe_manager.model is None:
                raise RuntimeError("TransEManager/Modelo não está pronto!")
            embeddings = self.extract_embeddings()
            self.transe_manager.node_embeddings = embeddings
            train_path = settings.DATA_DIR / "models" / "kg" / "train_optimized.parquet"
            if not train_path.exists():
                train_path = settings.DATA_DIR / "models" / "kg" / "train.parquet"
            if not train_path.exists():
                raise FileNotFoundError(
                    f"Arquivo de treino não encontrado: {train_path}"
                )
            logger.info(f"📂 Usando arquivo de treino: {train_path}")
            X_pos, y_pos, _ = self.create_lightgbm_dataset(train_path)
            X_full, y_full = self.generate_negative_samples(
                X_pos, y_pos, num_negatives_per_positive=self.negative_ratio
            )
            X_trn, X_tmp, y_trn, y_tmp = train_test_split(
                X_full, y_full, test_size=0.30, random_state=42, stratify=y_full
            )
            X_val, X_tst, y_val, y_tst = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
            )
            logger.info("📊 Splits criados:")
            logger.info(f"   Treino:     {len(X_trn):,}  ({np.mean(y_trn):.1%} pos)")
            logger.info(f"   Validação:  {len(X_val):,}  ({np.mean(y_val):.1%} pos)")
            logger.info(f"   Teste:      {len(X_tst):,}  ({np.mean(y_tst):.1%} pos)")
            self.train_lightgbm(X_trn, y_trn, X_val, y_val)
            if self.lightgbm_model is None:
                raise RuntimeError("LightGBM falhou no treinamento!")
            if (
                hasattr(self.lightgbm_model, "best_iteration")
                and self.lightgbm_model.best_iteration
            ):
                raw_val = self.lightgbm_model.predict(
                    X_val, num_iteration=self.lightgbm_model.best_iteration
                )
            else:
                raw_val = self.lightgbm_model.predict(X_val)
            prob_val = self._to_dense_1d(raw_val)
            prec, rec, thr = precision_recall_curve(y_val, prob_val)
            f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)
            best_idx = np.nanargmax(f1_scores)
            self.optimal_thresh = float(thr[best_idx])
            logger.info(
                f"🎯 Limiar ótimo: {self.optimal_thresh:.3f} "
                f"(F1={f1_scores[best_idx]:.4f})"
            )
            metrics = self.evaluate_lightgbm(X_tst, y_tst)
            if hasattr(self.lightgbm_model, "feature_importance"):
                gain = self.lightgbm_model.feature_importance(importance_type="gain")
                top_indices = np.argsort(gain)[-10:][::-1]
                logger.info("\n🎯 Top-10 features mais importantes:")
                for rank, idx in enumerate(top_indices, 1):
                    logger.info(f"   {rank:2d}. Feature {idx}: {gain[idx]:.2f}")

            logger.success("✅ Treinamento híbrido concluído com sucesso!")
            return metrics
        except Exception as exc:
            logger.error(f"❌ Erro no treinamento híbrido: {exc}")
            import traceback

            traceback_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            logger.debug(f"Traceback completo:\n{traceback_str}")

            return {
                "auc": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for samples.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities
        """
        if self.lightgbm_model is None:
            raise RuntimeError("Modelo não treinado!")

        return np.asarray(
            self.lightgbm_model.predict(
                X, num_iteration=self.lightgbm_model.best_iteration
            )
        )

    def save_model(self, path: Path) -> None:
        """Save LightGBM model in native .bin format with separate metadata."""
        if self.lightgbm_model is None:
            raise RuntimeError("Modelo não treinado!")
        path = Path(path)
        model_bin = path.parent / "lightgbm_model.bin"
        metadata_pkl = path.parent / "lightgbm_metadata.pkl"
        self.lightgbm_model.save_model(str(model_bin))
        logger.success(f"Modelo LightGBM salvo em: {model_bin}")
        metadata = {
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.lightgbm_model.num_feature(),
            "entity_to_idx": self.transe_manager.entity_to_idx,
            "relation_to_idx": self.transe_manager.relation_to_idx,
            "optimal_thresh": getattr(self, "optimal_thresh", 0.5),
            "negative_ratio": self.negative_ratio,
        }
        joblib.dump(metadata, metadata_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Metadados salvos em: {metadata_pkl}")

    def load_model(self, path: Path) -> None:
        """Load LightGBM model from native .bin format with metadata."""
        path = Path(path)
        model_bin = path.parent / "lightgbm_model.bin"
        metadata_pkl = path.parent / "lightgbm_metadata.pkl"
        if not model_bin.exists():
            raise FileNotFoundError(f"Modelo LightGBM não encontrado: {model_bin}")
        if not metadata_pkl.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {metadata_pkl}")
        self.lightgbm_model = lgb.Booster(model_file=str(model_bin))
        logger.success(f"Modelo LightGBM carregado de: {model_bin}")
        metadata = joblib.load(metadata_pkl)
        self.embedding_dim = metadata["embedding_dim"]
        self.optimal_thresh = metadata.get("optimal_thresh", 0.5)
        self.negative_ratio = metadata.get("negative_ratio", 1.0)
        logger.info(f"Metadados carregados: {metadata_pkl}")
        logger.debug(f"Features esperadas: {metadata['feature_dim']}")
