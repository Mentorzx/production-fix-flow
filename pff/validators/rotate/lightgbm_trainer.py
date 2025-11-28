"""RotatE + LightGBM Hybrid Model Trainer.

This module implements a hybrid approach combining RotatE embeddings
with LightGBM for improved knowledge graph completion performance.

Design Patterns Applied:
    - **Strategy Pattern:** Embedding extraction strategy can be swapped.
    - **Template Method:** Training follows fixed flow with customizable steps.
    - **Adapter Pattern:** Adapts RotatE embeddings for LightGBM consumption.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import torch
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
from pff.config import ROTATE_CONFIG_PATH
from pff.utils import FileManager, logger


class RotatELightGBMTrainer:
    """Trainer for hybrid RotatE + LightGBM model.

    This class combines RotatE embeddings with LightGBM to create
    a powerful hybrid model for link prediction tasks.

    Attributes:
        rotate_manager: Trained RotatEManager or adapter instance.
        file_manager: FileManager for I/O operations.
        lightgbm_model: Trained LightGBM booster (after training).
        embedding_dim: Dimension of entity/relation embeddings.
        negative_ratio: Ratio of negative to positive samples.
    """

    def __init__(self, rotate_manager):
        """Initialize the hybrid trainer.

        Args:
            rotate_manager: Trained RotatEManager instance or compatible adapter.
                Must have: config, entity_to_idx, relation_to_idx, node_embeddings, model.
        """
        self.rotate_manager = rotate_manager
        self.file_manager = FileManager()
        self.lightgbm_model: lgb.Booster | None = None
        self.best_iteration_: int | None = None
        self.eval_history_: dict[str, dict[str, list[float]]] = {}

        # Configuration
        self.embedding_dim = rotate_manager.config.get("model", {}).get("embedding_dim", 256)
        self.negative_ratio = 1  # Conservative negative sampling

        logger.info("RotatE+LightGBM Trainer inicializado")

    def create_lightgbm_dataset(
        self, data_path: Path | str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Build the (X, y) dataset for LightGBM with optimized embedding handling.

        Args:
            data_path: Path to the training data file (parquet format).

        Returns:
            Tuple of (features, labels, metadata).

        Raises:
            KeyError: If embeddings are missing required keys.
        """
        data_path = Path(data_path)
        logger.info(f"Criando dataset LightGBM de {data_path.name}...")

        if not hasattr(self.rotate_manager, "node_embeddings"):
            embeddings: dict[str, Any] = self.extract_embeddings()
            self.rotate_manager.node_embeddings = embeddings
        else:
            embeddings = self.rotate_manager.node_embeddings

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

        ent2idx = self.rotate_manager.entity_to_idx
        rel2idx = self.rotate_manager.relation_to_idx

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
                        np.linalg.norm(r_vec),
                        np.linalg.norm(t_vec),
                    ],
                    dtype=np.float32,
                )

                feature_vec = np.concatenate(
                    [concat, delta, [score], hadamard, diff, norms]
                )
                features.append(feature_vec)
                meta.append({"head": h, "relation": r, "tail": t})

        X = np.array(features, dtype=np.float32)
        y = np.ones(len(X), dtype=np.int32)
        logger.success(f"Features criadas: {len(X):,} válidas")

        return X, y, {"triples": meta}

    def extract_embeddings(self) -> dict[str, np.ndarray]:
        """Extract embeddings from RotatE model with compatibility aliases.

        Returns:
            Dictionary with entity and relation embeddings.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        logger.info("Extraindo embeddings do modelo RotatE...")

        if self.rotate_manager.model is None:
            raise RuntimeError("Modelo RotatE não está carregado!")

        with torch.no_grad():
            # RotatE uses complex embeddings (real, imag parts)
            # We concatenate them for compatibility with downstream models
            entity_real, entity_imag = self.rotate_manager.model.get_entity_embeddings()
            entity_embeddings = torch.cat([entity_real, entity_imag], dim=-1).cpu().numpy()
            
            # Relation embeddings are phase angles in RotatE
            # Convert to (cos, sin) to match entity embedding dimensions
            relation_phases = self.rotate_manager.model.get_relation_phases().cpu().numpy()
            relation_real = np.cos(relation_phases)
            relation_imag = np.sin(relation_phases)
            relation_embeddings = np.concatenate([relation_real, relation_imag], axis=-1)

        logger.success(
            f"Embeddings extraídos: entities={entity_embeddings.shape}, "
            f"relations={relation_embeddings.shape}"
        )

        embeddings_path = settings.OUTPUTS_DIR / "rotate" / "node_embeddings.pkl"
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
        """Get relation embeddings from RotatE model.

        Returns:
            Relation embeddings array.
        """
        if hasattr(self.rotate_manager, "node_embeddings"):
            return self.rotate_manager.node_embeddings.get(
                "relation", self.rotate_manager.node_embeddings.get("relation_embeddings")
            )
        # RotatE uses phase angles for relations
        return self.rotate_manager.model.get_relation_phases().cpu().numpy()

    def _load_training_params(self) -> dict[str, Any]:
        """Load LightGBM training parameters from config.

        Returns:
            Dictionary with LightGBM training parameters.
        """
        try:
            rotate_config = self.file_manager.read(ROTATE_CONFIG_PATH)
            lgb_training = rotate_config.get("lightgbm", {}).get("training", {})
            return lgb_training
        except Exception:
            return {}

    def generate_negative_samples(
        self,
        positive_X: np.ndarray,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate negative samples by entity corruption.

        Args:
            positive_X: Positive sample features.
            metadata: Metadata with triple information.

        Returns:
            Tuple of (negative features, labels).
        """
        logger.info(f"Gerando amostras negativas por corrupção (ratio: {self.negative_ratio})...")

        if not hasattr(self.rotate_manager, "node_embeddings"):
            embeddings = self.extract_embeddings()
            self.rotate_manager.node_embeddings = embeddings
        else:
            embeddings = self.rotate_manager.node_embeddings

        entity_embeddings = embeddings.get("entity", embeddings.get("entity_embeddings"))
        relation_embeddings = embeddings.get("relation", embeddings.get("relation_embeddings"))

        ent2idx = self.rotate_manager.entity_to_idx
        rel2idx = self.rotate_manager.relation_to_idx

        # Load all known positives
        known_positives: set[tuple[int, int, int]] = set()
        for split in ["train", "valid", "test"]:
            split_path = settings.DATA_DIR / "models" / "kg" / f"{split}_optimized.parquet"
            if split_path.exists():
                df = self.file_manager.read(split_path)
                for row in df.iter_rows(named=True):
                    h, r, t = str(row.get("s", row.get("head"))), str(row.get("p", row.get("relation"))), str(row.get("o", row.get("tail")))
                    if h in ent2idx and r in rel2idx and t in ent2idx:
                        known_positives.add((ent2idx[h], rel2idx[r], ent2idx[t]))

        logger.info(f"Carregadas {len(known_positives):,} triplas positivas conhecidas (todos os splits)")

        triples_meta = metadata.get("triples", [])
        num_entities = len(ent2idx)
        entity_ids = list(ent2idx.values())

        negative_features = []
        num_failed = 0
        rng = np.random.default_rng(42)

        target_negatives = int(len(positive_X) * self.negative_ratio)
        logger.info(f"Corrompendo {len(triples_meta):,} triplas de treino para gerar {target_negatives:,} negativos")

        for i, triple_info in enumerate(triples_meta):
            if len(negative_features) >= target_negatives:
                break

            h_str, r_str, t_str = triple_info["head"], triple_info["relation"], triple_info["tail"]
            h_idx, r_idx, t_idx = ent2idx[h_str], rel2idx[r_str], ent2idx[t_str]

            for _ in range(self.negative_ratio * 2):
                if len(negative_features) >= target_negatives:
                    break

                corrupt_head = rng.random() < 0.5
                if corrupt_head:
                    new_ent = rng.choice(entity_ids)
                    candidate = (new_ent, r_idx, t_idx)
                else:
                    new_ent = rng.choice(entity_ids)
                    candidate = (h_idx, r_idx, new_ent)

                if candidate in known_positives:
                    num_failed += 1
                    continue

                h_vec = entity_embeddings[candidate[0]]
                t_vec = entity_embeddings[candidate[2]]
                r_vec = relation_embeddings[candidate[1]]

                concat = np.concatenate((h_vec, r_vec, t_vec), dtype=np.float32)
                delta = (h_vec + r_vec - t_vec).astype(np.float32)
                score = -float(np.linalg.norm(delta, ord=2))
                hadamard = h_vec * t_vec
                diff = np.abs(h_vec - t_vec)
                norms = np.array([np.linalg.norm(h_vec), np.linalg.norm(r_vec), np.linalg.norm(t_vec)], dtype=np.float32)

                feature_vec = np.concatenate([concat, delta, [score], hadamard, diff, norms])
                negative_features.append(feature_vec)

        neg_X = np.array(negative_features, dtype=np.float32)
        neg_y = np.zeros(len(neg_X), dtype=np.int32)

        logger.info(f"Geradas {len(neg_X):,} amostras negativas válidas, {num_failed:,} falharam")

        unique_count = len(set(map(tuple, neg_X[:, :10].tolist())))
        logger.info(f"  Unicidade: {100.0 * unique_count / max(1, len(neg_X)):.1f}%")

        return neg_X, neg_y

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> lgb.Booster:
        """Train LightGBM model with parameters from config/models/rotate.yaml.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Trained LightGBM Booster.
        """
        logger.info("Treinando LightGBM com parâmetros do config/models/rotate.yaml...")

        rotate_config = self.file_manager.read(ROTATE_CONFIG_PATH)
        lgb_config = rotate_config.get("lightgbm", {})

        if not lgb_config:
            logger.warning(
                "Parâmetros LightGBM ausentes em rotate.yaml! "
                "Usando valores padrão."
            )

        lgb_params = lgb_config.get("params", {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        })

        # SOTA: Try CUDA first (LightGBM 4.x+), fallback to CPU
        device_preference = lgb_config.get("device_preference", "auto")
        use_cuda = False
        
        if device_preference in ("auto", "cuda"):
            try:
                # Check if LightGBM has CUDA support by testing a mini-model
                import torch
                if torch.cuda.is_available():
                    # Create a small test to verify CUDA support in LightGBM
                    test_params = {"device": "cuda", "verbose": -1, "num_iterations": 1}
                    test_data = lgb.Dataset([[0, 1], [1, 0]], label=[0, 1])
                    try:
                        test_model = lgb.train(test_params, test_data, num_boost_round=1)
                        del test_model, test_data
                        lgb_params["device"] = "cuda"
                        # gpu_use_dp=True for deterministic GPU behavior; max_bin=63 is GPU-optimal histogram binning
                        lgb_params["gpu_use_dp"] = True
                        lgb_params.setdefault("max_bin", 63)
                        use_cuda = True
                        logger.info("LightGBM configurado para CUDA (GPU).")
                    except Exception as cuda_err:
                        logger.warning(f"LightGBM CUDA not available in this build, falling back to CPU: {cuda_err}")
                        use_cuda = False
            except Exception:
                use_cuda = False
        
        if not use_cuda:
            lgb_params["device"] = "cpu"
            lgb_params["num_threads"] = max(1, (os.cpu_count() or 1) - 1)
            logger.info(f"LightGBM executara em CPU com {lgb_params['num_threads']} threads.")

        train_params = lgb_config.get("training", {})
        num_boost_round = int(train_params.get("num_boost_round", 100))
        early_stopping_rounds = int(train_params.get("early_stopping_rounds", 5))

        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        self.eval_history_ = {}

        model = lgb.train(
            lgb_params,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=10),
                lgb.record_evaluation(self.eval_history_),
            ],
        )

        self.best_iteration_ = model.best_iteration
        return model

    def save_model(self, output_dir: Path | str | None = None) -> Path:
        """Save trained LightGBM model.

        Args:
            output_dir: Output directory (uses default if None).

        Returns:
            Path to saved model.

        Raises:
            RuntimeError: If model is not trained.
        """
        if self.lightgbm_model is None:
            raise RuntimeError("Modelo não treinado. Execute train_hybrid_model primeiro.")

        model_path = Path(output_dir) if output_dir else (settings.OUTPUTS_DIR / "rotate")
        model_path.mkdir(parents=True, exist_ok=True)

        model_file = model_path / "lightgbm_model.bin"
        self.lightgbm_model.save_model(str(model_file))

        metadata = {
            "embedding_dim": self.embedding_dim,
            "negative_ratio": self.negative_ratio,
            "best_iteration": self.best_iteration_,
            "eval_history": self.eval_history_,
            "entity_to_idx": self.rotate_manager.entity_to_idx,
            "relation_to_idx": self.rotate_manager.relation_to_idx,
        }

        metadata_file = model_path / "lightgbm_metadata.pkl"
        self.file_manager.save(metadata, metadata_file)

        logger.info(f"Modelo salvo em: {model_file}")
        return model_file

    def train_hybrid_model(self, force_retrain: bool = False) -> dict[str, float]:
        """Train the complete hybrid RotatE + LightGBM model.

        Args:
            force_retrain: If True, ignore existing model and retrain from scratch.

        Returns:
            Dictionary with training metrics.
        """
        # Check if LightGBM model already exists
        model_path = settings.OUTPUTS_DIR / "rotate" / "lightgbm_model.bin"
        metrics_path = settings.OUTPUTS_DIR / "rotate" / "lightgbm_metrics.json"
        
        if not force_retrain and model_path.exists() and metrics_path.exists():
            cached_metrics = self.file_manager.read(metrics_path)
            logger.info(
                f"Modelo LightGBM ja existe em {model_path}. "
                f"Pulando treinamento... (AUC={cached_metrics.get('val_auc', 0):.4f})"
            )
            # Load existing model
            self.lightgbm_model = lgb.Booster(model_file=str(model_path))
            return cached_metrics

        logger.info("INICIANDO TREINAMENTO HÍBRIDO RotatE + LightGBM")
        logger.info("=" * 70)

        try:
            if self.rotate_manager is None or self.rotate_manager.model is None:
                raise RuntimeError("RotatEManager/Modelo não está pronto!")

            embeddings = self.extract_embeddings()
            self.rotate_manager.node_embeddings = embeddings

            # Try multiple possible training file locations
            train_path = settings.DATA_DIR / "models" / "kg" / "train_optimized.parquet"
            if not train_path.exists():
                train_path = settings.DATA_DIR / "models" / "kg" / "train.parquet"
            if not train_path.exists():
                train_path = settings.OUTPUTS_DIR / "pyclause" / "train.homogenized.parquet"
            if not train_path.exists():
                raise FileNotFoundError(
                    f"Training data not found. Tried: train_optimized.parquet, "
                    f"train.parquet, train.homogenized.parquet"
                )
            logger.info(f"Usando arquivo de treino: {train_path}")

            X_pos, y_pos, meta = self.create_lightgbm_dataset(train_path)
            X_neg, y_neg = self.generate_negative_samples(X_pos, meta)

            X = np.vstack([X_pos, X_neg])
            y = np.concatenate([y_pos, y_neg])

            logger.info(f"Dataset balanceado: {len(X):,} amostras ({100 * y.mean():.1f}% positivas)")

            # P3: Check if we should use true validation split from valid_optimized.parquet
            rotate_config = self.file_manager.read(ROTATE_CONFIG_PATH)
            lgb_training_config = rotate_config.get("lightgbm", {}).get("training", {})
            use_true_validation_split = bool(lgb_training_config.get("use_true_validation_split", False))

            if use_true_validation_split:
                # P3: Use valid_optimized.parquet for validation (more honest evaluation)
                val_path = settings.DATA_DIR / "models" / "kg" / "valid_optimized.parquet"
                if not val_path.exists():
                    val_path = settings.DATA_DIR / "models" / "kg" / "valid.parquet"

                if val_path.exists():
                    logger.info(f"Usando split de validacao real de: {val_path}")
                    X_val_pos, _, val_meta = self.create_lightgbm_dataset(val_path)
                    X_val_neg, y_val_neg = self.generate_negative_samples(X_val_pos, val_meta)

                    X_val = np.vstack([X_val_pos, X_val_neg])
                    y_val = np.concatenate([np.ones(len(X_val_pos)), y_val_neg])

                    # Shuffle validation set
                    val_indices = np.random.default_rng(42).permutation(len(X_val))
                    X_val = X_val[val_indices]
                    y_val = y_val[val_indices]

                    X_train = X
                    y_train = y
                    logger.info(f"Splits criados (validacao real):")
                    logger.info(f"  Treino:     {len(X_train):,}  ({100 * y_train.mean():.1f}% pos)")
                    logger.info(f"  Validação:  {len(X_val):,}  ({100 * y_val.mean():.1f}% pos)")
                else:
                    logger.warning(f"Validation file not found at {val_path}, falling back to train_test_split")
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    logger.info(f"Splits criados (fallback train_test_split):")
                    logger.info(f"  Treino:     {len(X_train):,}  ({100 * y_train.mean():.1f}% pos)")
                    logger.info(f"  Validação:  {len(X_val):,}  ({100 * y_val.mean():.1f}% pos)")
            else:
                # Original behavior: split from train + negatives
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                logger.info(f"Splits criados:")
                logger.info(f"  Treino:     {len(X_train):,}  ({100 * y_train.mean():.1f}% pos)")
                logger.info(f"  Validação:  {len(X_val):,}  ({100 * y_val.mean():.1f}% pos)")

            self.lightgbm_model = self._train_lightgbm(X_train, y_train, X_val, y_val)

            y_pred = self.lightgbm_model.predict(X_val)
            y_pred_binary = (y_pred > 0.5).astype(int)

            metrics = {
                "val_auc": float(roc_auc_score(y_val, y_pred)),
                "val_accuracy": float(accuracy_score(y_val, y_pred_binary)),
                "val_f1": float(f1_score(y_val, y_pred_binary)),
                "val_precision": float(precision_score(y_val, y_pred_binary)),
                "val_recall": float(recall_score(y_val, y_pred_binary)),
            }

            logger.success("Treinamento híbrido concluído!")
            logger.info(f"  AUC: {metrics['val_auc']:.4f}")
            logger.info(f"  F1:  {metrics['val_f1']:.4f}")

            self.save_model()
            
            # Save metrics for future skip checks
            FileManager.save(metrics, metrics_path)

            return metrics

        except Exception as e:
            logger.exception(f"Hybrid training error: {e}")
            raise
