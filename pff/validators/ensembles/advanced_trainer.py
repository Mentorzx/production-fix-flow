from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pff import settings
from pff.utils import logger
from pff.utils.file_manager import FileManager

from .ensemble_wrappers import HybridWrapper, ProbaTransformer, SymbolicFeatureExtractor
from .oov_solution_config import OOVAwareEnsembleManager


class OutOfFoldFeatureUnion(BaseEstimator, TransformerMixin):
    """Wrap a ``FeatureUnion`` to generate out-of-fold training features."""

    def __init__(
        self,
        base_union: FeatureUnion,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.base_union = base_union
        self.n_splits = max(2, n_splits)
        self.shuffle = shuffle
        self.random_state = random_state
        self._trained_union: FeatureUnion | None = None
        self._oof_features: np.ndarray | None = None

    def fit(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("y é obrigatório para gerar features out-of-fold")

        X_array = np.asarray(X, dtype=object)
        y_array = np.asarray(y)
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )

        oof_features: np.ndarray | None = None
        for train_idx, val_idx in splitter.split(X_array, y_array):
            fold_union = clone(self.base_union)
            fold_union.fit(X_array[train_idx], y_array[train_idx])
            fold_features = self._ensure_dense(fold_union.transform(X_array[val_idx]))
            if oof_features is None:
                oof_features = np.zeros(
                    (len(X_array), fold_features.shape[1]),
                    dtype=fold_features.dtype if hasattr(fold_features, "dtype") else np.float32,
                )
            oof_features[val_idx] = fold_features

        if oof_features is None:
            raise RuntimeError("Falha ao gerar features out-of-fold do FeatureUnion")

        self._oof_features = oof_features
        self._trained_union = clone(self.base_union)
        self._trained_union.fit(X_array, y_array)
        return self

    def transform(self, X):
        if self._trained_union is None:
            raise RuntimeError("Transformer não treinado. Chame fit primeiro.")
        features = self._trained_union.transform(X)
        return self._ensure_dense(features)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        if self._oof_features is None:
            raise RuntimeError("Transformer não possui features OOF disponíveis.")
        return self._oof_features

    @staticmethod
    def _ensure_dense(features) -> np.ndarray:
        if sparse.issparse(features):
            dense = features.toarray()
        else:
            dense = np.asarray(features)
        if dense.ndim == 1:
            dense = dense.reshape(-1, 1)
        return dense.astype(np.float32, copy=False)

class AdvancedEnsembleTrainer:
    """
    Main orchestrator for training the Hybrid Stacking Ensemble.

    Architecture:
    - Layer 1: Base models (TransE + AnyBURL + LightGBM)
    - Layer 2: Meta-learner (XGBoost) that combines the predictions
    """

    def __init__(
        self,
        neural_model_path: str,
        rules_path: str,
        lightgbm_model_path: str,
        output_dir: Path | None = None,
        force_symbolic_contribution: bool = False,
    ):
        """
        Initialize the trainer with the paths to the pre-trained models.

        Args:
            neural_model_path: Path to the TransE model
            rules_path: Path to AnyBURL rules
            lightgbm_model_path: Path to the LightGBM model
            output_dir: Directory to save artifacts
        """
        self.neural_model_path = Path(neural_model_path)
        self.rules_path = Path(rules_path)
        self.lightgbm_model_path = Path(lightgbm_model_path)
        self.output_dir = output_dir or settings.OUTPUTS_DIR / "ensemble"

        self.ensemble_model = None
        self.metrics_history = []
        self.optimal_threshold = 0.5

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("🚀 AdvancedEnsembleTrainer inicializado")
        logger.info(f"📁 Diretório de saída: {self.output_dir}")

        self.force_symbolic_contribution = force_symbolic_contribution

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        meta_params: dict | None = None,
    ) -> Pipeline:
        logger.info("🏗️ Construindo o pipeline do Stacking Ensemble...")
        logger.info(
            "🎯 Configurando XGBoost para balancear features contínuas vs binárias..."
        )
        self.oov_manager = OOVAwareEnsembleManager()

        try:
            from .ensemble_wrappers import _coerce_mapping_df

            logger.info("🔄 Carregando dependências para o HybridWrapper...")

            if not self.lightgbm_model_path.exists():
                raise FileNotFoundError(f"LightGBM model not found: {self.lightgbm_model_path}")

            # Load TransE mappings
            entity_map_path = settings.OUTPUTS_DIR / "transe" / "transe_entity_map.parquet"
            ent_map_df = FileManager().read(entity_map_path)
            entity_to_idx = _coerce_mapping_df(ent_map_df)

            rel_map_path = settings.OUTPUTS_DIR / "transe" / "transe_relation_map.parquet"
            rel_map_df = FileManager().read(rel_map_path)
            relation_to_idx = _coerce_mapping_df(rel_map_df)
            embeddings_data = joblib.load(
                settings.OUTPUTS_DIR / "transe" / "node_embeddings.pkl"
            )
            entity_embeddings = embeddings_data["entity_embeddings"]
            relation_embeddings = embeddings_data["relation_embeddings"]
            logger.success(
                "✅ Todas as dependências do HybridWrapper foram carregadas com sucesso."
            )
        except Exception as e:
            logger.error(
                f"Falha crítica ao carregar dependências do modelo. Abortando. Erro: {e}"
            )
            raise

        hybrid_predictor = HybridWrapper(
            lightgbm_model=None,
            entity_to_idx=entity_to_idx,
            relation_to_idx=relation_to_idx,
            entity_embeddings=entity_embeddings,
            relation_embeddings=relation_embeddings,
            lightgbm_model_path=self.lightgbm_model_path,
        )

        # Load ensemble config FIRST to get min_confidence_threshold
        ensemble_config = self._load_ensemble_config()
        training_config = ensemble_config.get("training", {})
        cv_folds = int(training_config.get("cv_folds", 5))
        shuffle_folds = bool(training_config.get("shuffle", True))
        random_state = training_config.get("random_state", 42)

        # Extract min_confidence_threshold from config (default 0.05)
        symbolic_config = {}
        for model in ensemble_config.get("base_models", []):
            if model.get("type") == "symbolic":
                symbolic_config = model.get("params", {})
                break

        min_confidence_threshold = symbolic_config.get("min_confidence_threshold", 0.05)
        logger.info(f"🔍 Using min_confidence_threshold: {min_confidence_threshold} from config")

        # Numba accelerator removed because it does incorrect matching (0% sparsity)
        # Solution: Use business_service for matching (correct) + rule_indexing (10-100× speedup)
        # Result: Centralized logic in business_service, fast via indexing
        logger.info("🏗️ Architecture: business_service matching + rule indexing (centralized & fast)")

        max_rules_per_predicate = int(symbolic_config.get("max_rules_per_predicate", 250))
        symbolic_common_kwargs = {
            "rules_path": self.rules_path,
            "min_confidence_threshold": min_confidence_threshold,
            "enable_numba": True,
            "enable_rule_indexing": True,
            "max_rules_per_predicate": max_rules_per_predicate,
        }

        if self.force_symbolic_contribution:
            logger.info("⚖️ Modo de contribuição forçada ATIVADO")
            symbolic_extractor = SymbolicFeatureExtractor(
                enable_grouping=True,
                n_groups=50,
                boost_factor=1.0,
                **symbolic_common_kwargs,
            )
        else:
            logger.info("⚖️ Modo de contribuição balanceada (sem forçar)")
            symbolic_extractor = SymbolicFeatureExtractor(
                enable_grouping=False,
                **symbolic_common_kwargs,
            )
        logger.info("⚖️ Configurando parâmetros balanceados do XGBoost...")
        yaml_meta_params = ensemble_config.get("meta_learner", {}).get("params", {})
        if self.force_symbolic_contribution:
            # yaml_meta_params.update({
            #     "max_depth": 2,
            #     "min_child_weight": 0.01,
            #     "gamma": 0.0001,
            #     "colsample_bytree": 0.9,
            #     "learning_rate": 0.02,
            #     "reg_alpha": 0.001,
            #     "reg_lambda": 0.001,
            # })
            logger.info("📊 XGBoost usando parâmetros balanceados do YAML (modo simbólico)")
        else:
            logger.info("📊 XGBoost usando parâmetros padrão do YAML")

        balanced_meta_params = {
            "n_estimators": yaml_meta_params.get("n_estimators", 100),  # Reduced from 400 to prevent overfitting
            "max_depth": yaml_meta_params.get("max_depth", 3),        # Reduced from 4 for shallow trees on sparse features
            "learning_rate": yaml_meta_params.get("learning_rate", 0.01),   # Keep low for stability
            "colsample_bytree": yaml_meta_params.get("colsample_bytree", 0.3),  # Reduced from 0.4 to limit sparse feature sampling
            "colsample_bylevel": yaml_meta_params.get("colsample_bylevel", 0.4),  # Reduced from 0.6
            "colsample_bynode": yaml_meta_params.get("colsample_bynode", 0.6),  # Reduced from 0.8
            "reg_alpha": yaml_meta_params.get("reg_alpha", 0.5),      # INCREASED from 0.005 - strong L1 for feature selection
            "reg_lambda": yaml_meta_params.get("reg_lambda", 5.0),       # INCREASED from 0.05 - strong L2 for weight shrinkage
            "min_child_weight": yaml_meta_params.get("min_child_weight", 20), # INCREASED from 5 to prevent splits on rare patterns
            "gamma": yaml_meta_params.get("gamma", 0.1),            # INCREASED from 0.005 for minimum loss reduction
            "subsample": yaml_meta_params.get("subsample", 0.7),         # Reduced from 0.9 for robustness
            "scale_pos_weight": yaml_meta_params.get("scale_pos_weight", 1.0),
            "tree_method": yaml_meta_params.get("tree_method", "hist"),
            "objective": "binary:logistic",
            "eval_metric": yaml_meta_params.get("eval_metric", ["logloss", "aucpr"]),
            "use_label_encoder": False,
            "random_state": yaml_meta_params.get("random_state", 42),
            "n_jobs": -1,
        }

        logger.info(f"⚖️ STRONG REGULARIZATION to prevent overfitting on sparse features (Fix for 0.66% activation bug):")
        logger.info(f"   - n_estimators: {balanced_meta_params['n_estimators']} (reduced to prevent overfitting)")
        logger.info(f"   - max_depth: {balanced_meta_params['max_depth']} (shallow trees for sparse features)")
        logger.info(f"   - learning_rate: {balanced_meta_params['learning_rate']} (stable learning)")
        logger.info(f"   - colsample_bytree: {balanced_meta_params['colsample_bytree']} (reduced sparse feature sampling)")
        logger.info(f"   - reg_alpha (L1): {balanced_meta_params['reg_alpha']} (STRONG - feature selection)")
        logger.info(f"   - reg_lambda (L2): {balanced_meta_params['reg_lambda']} (STRONG - weight shrinkage)")
        logger.info(f"   - min_child_weight: {balanced_meta_params['min_child_weight']} (prevent splits on rare patterns)")
        logger.info(f"   - gamma: {balanced_meta_params['gamma']} (minimum loss reduction required)")
        logger.info(f"   - subsample: {balanced_meta_params['subsample']} (row sampling for robustness)")

        early_stopping_rounds = yaml_meta_params.get("early_stopping_rounds")
        if early_stopping_rounds:
            if X_val is None or y_val is None:
                logger.info("🔄 Criando split de validação para early stopping...")
                from sklearn.model_selection import train_test_split

                X_train_split, X_val_split, y_train_split, y_val_split = (
                    train_test_split(
                        X_train,
                        y_train,
                        test_size=0.2,
                        random_state=42,
                        stratify=y_train,
                    )
                )
                X_train, X_val = X_train_split, X_val_split
                y_train, y_val = y_train_split, y_val_split
            balanced_meta_params["early_stopping_rounds"] = early_stopping_rounds
            logger.info(
                f"✅ Early stopping configurado: {early_stopping_rounds} rounds"
            )
        else:
            logger.info("📊 Treinando sem early stopping")
        if meta_params:
            meta_params.update({
                "importance_type": "gain",
                "feature_importance_output": True
            })
            balanced_meta_params.update(meta_params)
        logger.info("📊 Parâmetros XGBoost configurados:")
        logger.info(
            f"   - max_depth: {balanced_meta_params['max_depth']} (árvores rasas)"
        )
        logger.info(
            f"   - colsample_bytree: {balanced_meta_params['colsample_bytree']} (sampling reduzido)"
        )
        logger.info(
            f"   - reg_alpha: {balanced_meta_params['reg_alpha']} (L1 para esparsas)"
        )
        logger.info(
            f"   - subsample: {balanced_meta_params['subsample']} (reduz bias híbrido)"
        )     
        meta_learner = Pipeline([
            ('scaler', StandardScaler()),
            ('xgboost', XGBClassifier(**balanced_meta_params))
        ])
        
        hybrid_pipe = Pipeline([("hybrid", ProbaTransformer(hybrid_predictor))])
        combined_features = FeatureUnion(
            [
                ("hybrid_pred", hybrid_pipe),
                ("symbolic_rules", symbolic_extractor),
            ]
        )
        features_union = OutOfFoldFeatureUnion(
            combined_features,
            n_splits=cv_folds,
            shuffle=shuffle_folds,
            random_state=random_state,
        )
        self.ensemble_model = Pipeline(
            [("features", features_union), ("meta_learner", meta_learner)]
        )
        logger.info(
            f"🔁 Stacking OOF configurado com {cv_folds} folds (shuffle={shuffle_folds})."
        )
        logger.info("=" * 80)
        logger.info("🔍 DIAGNOSTIC: Validando features simbólicas antes do treinamento")
        logger.info("=" * 80)
        try:
            X_sample = X_train[:10] if len(X_train) > 10 else X_train
            feature_union_step = self.ensemble_model.named_steps["features"]
            base_union = getattr(feature_union_step, "base_union", feature_union_step)
            symbolic_transformer = base_union.transformer_list[1][1]
            logger.info("🔧 Treinando symbolic_transformer para carregar regras...")
            symbolic_transformer.fit(X_sample)
            if hasattr(symbolic_transformer, "rules_"):
                logger.info(f"✅ Regras carregadas: {len(symbolic_transformer.rules_)}")
            else:
                logger.error("❌ CRITICAL: self.rules_ não existe no transformer!")

            if hasattr(symbolic_transformer, "transform"):
                logger.info(f"🧪 Testando transform() com {len(X_sample)} amostras...")
                symbolic_features = symbolic_transformer.transform(X_sample)

                logger.info(f"📐 Shape das features simbólicas: {symbolic_features.shape}")

                if len(symbolic_features.shape) > 1:
                    self.n_rules = symbolic_features.shape[1]
                    n_rules = self.n_rules  # Para compatibilidade local
                    logger.info(f"📊 Número de features (regras ou grupos): {n_rules}")
                else:
                    logger.error("❌ Features simbólicas têm dimensão errada!")

                if symbolic_features.size > 0:
                    total_non_zero = np.count_nonzero(symbolic_features)
                    total_elements = symbolic_features.size
                    sparsity_pct = (total_non_zero / total_elements) * 100

                    logger.info(
                        f"📊 Sparsidade: {total_non_zero:,}/{total_elements:,} "
                        f"({sparsity_pct:.2f}%) não-zero"
                    )

                    if total_non_zero == 0:
                        logger.error("❌ PROBLEMA CRÍTICO: Todas as features simbólicas são ZERO!")
                        logger.error("   Possíveis causas:")
                        logger.error("   1. min_confidence_threshold muito alto (filtrou todas as regras)")
                        logger.error("   2. Regras não aplicáveis às amostras de treino")
                        logger.error("   3. Erro no parsing das regras")
                        logger.error("   4. Formato das amostras incompatível com validação")
                    elif sparsity_pct < 1.0:
                        logger.warning(
                            f"⚠️ Features MUITO esparsas ({sparsity_pct:.2f}% não-zero)"
                        )
                        logger.warning("   Ensemble pode ignorar features simbólicas")
                    elif sparsity_pct < 5.0:
                        logger.warning(
                            f"⚠️ Features esparsas ({sparsity_pct:.2f}% não-zero)"
                        )
                    else:
                        logger.success(
                            f"✅ Features simbólicas OK ({sparsity_pct:.2f}% não-zero)"
                        )

                    # Sample-level analysis
                    active_per_sample = np.sum(symbolic_features > 0, axis=1)
                    logger.info(
                        f"📊 Regras ativas por amostra: "
                        f"min={active_per_sample.min()}, "
                        f"max={active_per_sample.max()}, "
                        f"mean={active_per_sample.mean():.1f}, "
                        f"median={np.median(active_per_sample):.1f}"
                    )

                    if active_per_sample.max() == 0:
                        logger.error("❌ Nenhuma amostra tem regras ativas!")

                else:
                    logger.error("❌ PROBLEMA: Features simbólicas vazias (size=0)!")
            else:
                logger.error(
                    "❌ PROBLEMA: SymbolicFeatureExtractor não tem método transform!"
                )
        except Exception as e:
            logger.error(f"❌ Erro no diagnóstico das features simbólicas: {e}")
            import traceback

            logger.error(traceback.format_exc())

        logger.info("=" * 80)
        logger.info("🎯 Treinando o Stacking Ensemble...")
        start_time = datetime.now()
        if early_stopping_rounds and X_val is not None:
            logger.info("🛑 Treinando com early stopping...")
            
            # Step 1: Extract features using FeatureUnion
            feature_transformer = self.ensemble_model.named_steps["features"]
            X_train_features = feature_transformer.fit_transform(X_train, y_train)
            X_val_features = feature_transformer.transform(X_val)
            
            # Step 2: Get scaler and XGBoost from meta_learner pipeline
            if isinstance(meta_learner, Pipeline):
                scaler = meta_learner.named_steps['scaler']
                xgb_model = meta_learner.named_steps['xgboost']
                
                # Step 3: Fit scaler on training features
                X_train_scaled = scaler.fit_transform(X_train_features)
                X_val_scaled = scaler.transform(X_val_features)
                
                # Step 4: Train XGBoost with scaled features
                xgb_model.fit(
                    X_train_scaled,
                    y_train,
                    eval_set=[(X_val_scaled, y_val)],  # Now both are scaled!
                    verbose=False,
                )
            else:
                # Backwards compatibility (no scaler)
                meta_learner.fit(
                    X_train_features,
                    y_train,
                    eval_set=[(X_val_features, y_val)],
                    verbose=False,
                )
        else:
            # Use full pipeline (scaler gets fitted automatically)
            self.ensemble_model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        logger.success(f"✅ Treinamento concluído em {train_time:.2f} segundos")
        self._validate_feature_balance()
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, prefix="validation")
            logger.info("📊 Métricas de validação:")
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   - {key}: {value:.4f}")
                else:
                    logger.info(f"   - {key}: {value}")

            threshold, threshold_metrics = self.calibrate_threshold(X_val, y_val, metric="f1")
            logger.info(f"🎯 Threshold otimizado: {threshold:.3f}")
            logger.info("📊 Métricas com threshold calibrado:")
            for key, value in threshold_metrics.items():
                if key != "threshold":
                    logger.info(f"   - {key}: {value:.4f}")

        return self.ensemble_model

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, prefix: str = "test"
    ) -> dict:
        """
        Evaluate the ensemble model on the test data.

        Args:
            X_test: Test features
            y_test: Test labels
            prefix: Prefix for the metrics

        Returns:
            Dictionary with metrics
        """
        if self.ensemble_model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        logger.info(f"📊 Avaliando modelo no conjunto {prefix}...")
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        metrics = {
            f"{prefix}_accuracy": accuracy_score(y_test, y_pred),
            f"{prefix}_precision": precision_score(y_test, y_pred, average="weighted"),
            f"{prefix}_recall": recall_score(y_test, y_pred, average="weighted"),
            f"{prefix}_f1_score": f1_score(y_test, y_pred, average="weighted"),
            f"{prefix}_auc_roc": roc_auc_score(y_test, y_pred_proba),
        }
        cm = confusion_matrix(y_test, y_pred)
        metrics[f"{prefix}_confusion_matrix"] = cm.tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics[f"{prefix}_classification_report"] = report
        return metrics

    def calibrate_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1"
    ) -> tuple[float, dict]:
        """
        Calibrate decision threshold to optimize a specific metric.

        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')

        Returns:
            Tuple of (best_threshold, metrics_at_threshold)
        """
        if self.ensemble_model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")

        logger.info(f"🎯 Calibrando threshold para otimizar {metric}...")

        y_pred_proba = self.ensemble_model.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0.1, 0.9, 81)
        best_score = -float("inf")
        best_threshold = 0.5
        best_metrics = {}

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_val, y_pred, average="weighted")
            elif metric == "precision":
                score = precision_score(y_val, y_pred, average="weighted")
            elif metric == "recall":
                score = recall_score(y_val, y_pred, average="weighted")
            elif metric == "accuracy":
                score = accuracy_score(y_val, y_pred)
            else:
                raise ValueError(f"Metric '{metric}' not supported")

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    "threshold": threshold,
                    "f1": f1_score(y_val, y_pred, average="weighted"),
                    "precision": precision_score(y_val, y_pred, average="weighted"),
                    "recall": recall_score(y_val, y_pred, average="weighted"),
                    "accuracy": accuracy_score(y_val, y_pred),
                }

        logger.success(
            f"✅ Melhor threshold: {best_threshold:.3f} "
            f"(F1={best_metrics['f1']:.4f}, "
            f"Acc={best_metrics['accuracy']:.4f})"
        )

        self.optimal_threshold = best_threshold
        return best_threshold, best_metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """
        Perform stratified cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"🔄 Iniciando validação cruzada com {cv} folds...")
        if self.ensemble_model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "roc_auc": "roc_auc",
        }
        cv_results = cross_validate(
            self.ensemble_model,
            X,
            y,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        results = {}
        for metric in scoring.keys():
            train_scores = cv_results[f"train_{metric}"]
            test_scores = cv_results[f"test_{metric}"]
            results[f"{metric}_train_mean"] = np.mean(train_scores)
            results[f"{metric}_train_std"] = np.std(train_scores)
            results[f"{metric}_test_mean"] = np.mean(test_scores)
            results[f"{metric}_test_std"] = np.std(test_scores)

        return results

    def _load_ensemble_config(self) -> dict:
        """
        Loads the ensemble configuration from the 'ensemble.yaml' file.

        Attempts to read the configuration file located in the directory specified by
        `settings.CONFIG_DIR`. If the file cannot be loaded for any reason, logs a warning
        and returns an empty dictionary.

        Returns:
            dict: The contents of the ensemble configuration file as a dictionary,
                  or an empty dictionary if loading fails.
        """
        try:
            config_path = settings.CONFIG_DIR / "ensemble.yaml"
            return FileManager().read(config_path)
        except Exception as e:
            logger.warning(f"Erro ao carregar ensemble.yaml: {e}")
            return {}

    def _validate_feature_balance(self):
        """
        Validates the balance of feature importances between hybrid and symbolic features in the ensemble model's meta-learner.
        This method checks if the ensemble model is present and retrieves the feature importances from the meta-learner step.
        It calculates the contribution of the first feature (assumed to be hybrid) and the remaining features (assumed to be symbolic).
        Logs the percentage contributions of both hybrid and symbolic features, and evaluates if the symbolic contribution meets
        the target threshold (>= 15%). Provides suggestions for improving symbolic contribution if it is too low.
        Logs:
            - Info: Contributions of hybrid and symbolic features.
            - Success: If symbolic contribution is >= 15%.
            - Warning: If symbolic contribution is between 5% and 15%.
            - Error: If symbolic contribution is < 5%, along with improvement suggestions.
        Handles and logs any exceptions that occur during validation.
        """
        if not self.ensemble_model:
            return
        try:
            meta_learner = self.ensemble_model.named_steps["meta_learner"]
            
            if isinstance(meta_learner, Pipeline):
                xgb_model = meta_learner.named_steps['xgboost']
                importances = xgb_model.feature_importances_
            else:
                importances = meta_learner.feature_importances_
            hybrid_importance = importances[0] if len(importances) > 0 else 0.0
            symbolic_importance = (
                np.sum(importances[1:]) if len(importances) > 1 else 0.0
            )
            total_importance = hybrid_importance + symbolic_importance
            if total_importance > 0:
                hybrid_contrib = hybrid_importance / total_importance
                symbolic_contrib = symbolic_importance / total_importance
                logger.info("⚖️ VALIDAÇÃO DE BALANCEAMENTO:")
                logger.info(f"   🔍 Contribuição Híbrida: {hybrid_contrib:.2%}")
                logger.info(f"   🧠 Contribuição Simbólica: {symbolic_contrib:.2%}")
                if symbolic_contrib >= 0.15:
                    logger.success(
                        f"✅ Balanceamento FUNCIONANDO! Simbólico: {symbolic_contrib:.2%}"
                    )
                elif symbolic_contrib >= 0.05:
                    logger.warning(
                        f"⚠️ Balanceamento PARCIAL. Simbólico: {symbolic_contrib:.2%}"
                    )
                else:
                    logger.error(
                        f"❌ Balanceamento FALHOU. Simbólico: {symbolic_contrib:.2%}"
                    )
                    logger.info("💡 Sugestões:")
                    logger.info("   - Reduzir colsample_bytree (ex: 0.2)")
                    logger.info("   - Reduzir max_depth (ex: 2)")
                    logger.info("   - Aumentar reg_alpha (ex: 0.1)")
        except Exception as e:
            logger.error(f"Erro na validação de balanceamento: {e}")

    def _apply_sample_weighting(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        features = self.ensemble_model.named_steps["features"].transform(X)
        if features.shape[1] > 1:
            symbolic_features = features[:, 1:]
            n_active = np.sum(symbolic_features > 0, axis=1)
            weights = 1.0 + np.log1p(n_active) * 0.5
            weights = weights / np.mean(weights)

            logger.info(f"📊 Pesos: min={weights.min():.2f}, max={weights.max():.2f}")
            return weights

        return np.ones(len(X))

    def _save_final_metrics_report(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Generates and saves a comprehensive final metrics report for the ensemble model using the provided test data.
        This method evaluates the trained ensemble model on the given test set, extracts feature importances from the meta-learner,
        calculates the contributions of hybrid and symbolic features, and compiles a detailed report including model information,
        performance metrics, feature balance, confusion matrix, and classification report. The report is saved as a JSON file in the
        output directory, and key metrics are logged for review.
        Args:
            X_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): True labels for the test set.
        Returns:
            None
        """
        logger.info("📊 Gerando relatório de métricas final...")
        if not self.ensemble_model:
            logger.error(
                "Modelo de ensemble não treinado. Não é possível gerar relatório."
            )
            return
        features_step = self.ensemble_model.named_steps["features"]
        feature_union = getattr(features_step, "base_union", features_step)
        meta_learner = self.ensemble_model.named_steps["meta_learner"]
        
        if isinstance(meta_learner, Pipeline):
            xgb_model = meta_learner.named_steps['xgboost']
            importances = xgb_model.feature_importances_
        else:
            xgb_model = meta_learner  # For backwards compatibility
            importances = meta_learner.feature_importances_

        n_features = len(importances)
        feature_names = ["hybrid_probability"]

        symbolic_transformer = feature_union.transformer_list[1][1]
        num_symbolic_features = n_features - 1

        if num_symbolic_features > 0:
            if hasattr(symbolic_transformer, "enable_grouping") and symbolic_transformer.enable_grouping:
                feature_names.extend([f"symbolic_group_{i}" for i in range(num_symbolic_features)])
                logger.info(
                    f"📋 Total de features: 1 híbrida + {num_symbolic_features} grupos simbólicos "
                    f"(agrupadas de {len(getattr(symbolic_transformer, 'rules_', []))} regras)"
                )
            else:
                feature_names.extend([f"rule_{i}" for i in range(num_symbolic_features)])
                logger.info(
                    f"📋 Total de features: 1 híbrida + {num_symbolic_features} regras simbólicas"
                )
        else:
            logger.warning("Nenhuma feature simbólica detectada no modelo.")
        if len(importances) != len(feature_names):
            logger.warning(
                f"Descompasso: {len(importances)} importâncias vs {len(feature_names)} nomes"
            )
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]
        feature_importance_list = sorted(
            [(name, float(imp)) for name, imp in zip(feature_names, importances)],
            key=lambda x: x[1],
            reverse=True,
        )
        final_metrics = _convert_numpy_types(
            self.evaluate(X_test, y_test, prefix="test")
        )
        hybrid_contribution = float(importances[0]) if len(importances) > 0 else 0.0
        symbolic_total_contribution = (
            float(np.sum(importances[1:])) if len(importances) > 1 else 0.0
        )
        total_contribution = hybrid_contribution + symbolic_total_contribution
        if total_contribution > 0:
            hybrid_contribution_pct = hybrid_contribution / total_contribution
            symbolic_contribution_pct = symbolic_total_contribution / total_contribution
        else:
            hybrid_contribution_pct = 0.0
            symbolic_contribution_pct = 0.0
        report = {
            "model_info": {
                "type": "Balanced Hybrid Neuro-Symbolic Stacking Ensemble",
                "components": {
                    "base_models": ["TransE", "AnyBURL", "LightGBM"],
                    "meta_learner": "XGBoost (Balanced for Binary Features)",
                },
                "training_date": datetime.now().isoformat(),
                "total_features": int(len(feature_names)),
                "xgboost_config": {
                    "max_depth": int(xgb_model.max_depth),
                    "colsample_bytree": float(xgb_model.colsample_bytree),
                    "reg_alpha": float(xgb_model.reg_alpha),
                    "subsample": float(xgb_model.subsample),
                },
            },
            "Ensemble_Final": {
                "accuracy": final_metrics.get("test_accuracy", 0),
                "precision": final_metrics.get("test_precision", 0),
                "recall": final_metrics.get("test_recall", 0),
                "f1_score": final_metrics.get("test_f1_score", 0),
                "auc_roc": final_metrics.get("test_auc_roc", 0),
            },
            "Feature_Balance": {
                "top_20_features": feature_importance_list[:20],
                "hybrid_contribution": float(hybrid_contribution_pct),
                "symbolic_total_contribution": float(symbolic_contribution_pct),
                "contribution_ratio": {
                    "hybrid": f"{hybrid_contribution_pct:.2%}",
                    "symbolic": f"{symbolic_contribution_pct:.2%}",
                },
                "balance_status": (
                    "BALANCED"
                    if symbolic_contribution_pct >= 0.15
                    else (
                        "PARTIAL" if symbolic_contribution_pct >= 0.05 else "IMBALANCED"
                    )
                ),
                "symbolic_rules_count": int(getattr(self, 'n_rules', 0)),
            },
            "confusion_matrix": final_metrics.get("test_confusion_matrix", []),
            "classification_report": final_metrics.get(
                "test_classification_report", {}
            ),
        }
        out_path = self.output_dir / "metrics_all.json"
        report = _convert_numpy_types(report)
        FileManager().save(report, out_path)
        logger.success(f"✅ Relatório de métricas final salvo em {out_path}")
        logger.info(f"📈 F1-Score Final: {report['Ensemble_Final']['f1_score']:.4f}")
        logger.info(f"🔍 Contribuição do modelo híbrido: {hybrid_contribution_pct:.2%}")
        logger.info(
            f"📋 Contribuição das regras simbólicas: {symbolic_contribution_pct:.2%}"
        )
        logger.info(
            f"⚖️ Status do balanceamento: {report['Feature_Balance']['balance_status']}"
        )
        logger.info("=" * 80)
        logger.info("🏆 Top 10 features mais importantes (with names):")
        logger.info("=" * 80)
        for idx, (name, importance) in enumerate(feature_importance_list[:10], 1):
            logger.info(f"   {idx:2d}. {name:40s}: {importance:10.2f}")
        
        logger.info("=" * 80)
        logger.info(f"   Features declared: {len(feature_names)}")
        logger.info(f"   Importances shape: {len(importances)}")
        
        if len(feature_names) == len(importances):
            logger.success("   ✅ Feature mapping is CONSISTENT")
        else:
            logger.error(f"   ❌ MISMATCH: {len(feature_names)} names vs {len(importances)} importances")
            logger.info("   🔧 Adjusted to minimum length above")
        logger.info("=" * 80)

    def save_model(self, filename: str = "stacking_model_advanced.joblib"):
        """
        Save the trained model.

        Args:
            filename: File name
        """
        if self.ensemble_model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        model_path = self.output_dir / filename
        # Safe dump: strip non-picklable attrs
        try:
            if hasattr(self.ensemble_model, 'named_steps') and 'features' in self.ensemble_model.named_steps:
                feats = self.ensemble_model.named_steps['features']
                for name, step in getattr(feats, 'transformer_list', []):
                    for attr in dir(step):
                        if attr.startswith('_') and 'nvml' in attr:
                            try:
                                setattr(step, attr, None)
                            except Exception:
                                pass
        except Exception:
            pass
        joblib.dump(self.ensemble_model, model_path)
        logger.success(f"💾 Modelo salvo em {model_path}")
        metadata = {
            "model_type": "Hybrid Stacking Ensemble",
            "saved_at": datetime.now().isoformat(),
            "components": {
                "neural_model": self.neural_model_path,
                "rules": self.rules_path,
                "lightgbm_model": self.lightgbm_model_path,
            },
        }
        metadata_path = self.output_dir / "model_metadata.json"
        FileManager().save(metadata, metadata_path)

    def run_ensemble_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        perform_cv: bool = True,
    ) -> dict:
        """
        Run the complete training and evaluation pipeline.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            perform_cv: Whether to perform cross-validation

        Returns:
            Dictionary with all results
        """
        logger.info("🚀 Iniciando pipeline completo do Ensemble...")
        results = {}
        self.train(X_train, y_train)
        if perform_cv:
            cv_results = self.cross_validate(X_train, y_train)
            results["cross_validation"] = cv_results
            logger.info(
                f"📊 CV F1-Score: {cv_results['f1_test_mean']:.4f} ± {cv_results['f1_test_std']:.4f}"
            )
        test_metrics = self.evaluate(X_test, y_test)
        results["test_metrics"] = test_metrics
        self._save_final_metrics_report(X_test, y_test)
        self.save_model()
        logger.success("✅ Pipeline completo executado com sucesso!")
        return results


async def run_standalone_ensemble_pipeline() -> dict:
    logger.info("🚀 Orquestrando pipeline de ensemble autônomo...")
    try:
        from .data_loader import EnsembleDataLoader

        data_loader = EnsembleDataLoader()
        X_train, y_train, X_test, y_test = data_loader.load_ensemble_data()
        X_train_np = np.asarray(X_train)
        y_train_np = np.asarray(y_train)
        X_test_np = np.asarray(X_test)
        y_test_np = np.asarray(y_test)
    except Exception as e:
        logger.exception(f"Falha ao carregar os dados para o ensemble: {e}")
        return {"status": "failed", "error": "data_loading_failed"}

    trainer = AdvancedEnsembleTrainer(
        neural_model_path=str(settings.OUTPUTS_DIR / "transe"),
        rules_path=str(settings.OUTPUTS_DIR / "pyclause" / "rules_anyburl.tsv"),
        lightgbm_model_path=str(settings.OUTPUTS_DIR / "transe" / "lightgbm_model.bin"),
        force_symbolic_contribution=True,
    )
    results = trainer.run_ensemble_pipeline(
        X_train=X_train_np,
        y_train=y_train_np,
        X_test=X_test_np,
        y_test=y_test_np,
        perform_cv=False,
    )

    return results


def _convert_numpy_types(obj):
    """Recursivamente converte tipos NumPy para tipos Python nativos."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    return obj
