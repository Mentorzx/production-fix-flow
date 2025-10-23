"""
Hyperparameter Tuning for Stacking Ensemble
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from xgboost import XGBClassifier

from pff import settings
from pff.utils import FileManager, logger

from .ensemble_wrappers import HybridWrapper, ProbaTransformer, SymbolicFeatureExtractor


class HyperparameterTuner:
    """
    Hyperparameter optimizer for the Stacking Ensemble.
    Supports Grid Search, Random Search, and Optuna.
    """

    def __init__(
        self,
        neural_model_path: str,
        rules_path: str,
        lightgbm_model_path: str,
        output_dir: Path | None = None,
    ):
        """
        Initialize the tuner with the model paths.

        Args:
            neural_model_path: Path to the TransE model
            rules_path: Path to AnyBURL rules
            lightgbm_model_path: Path to the LightGBM model
            output_dir: Directory to save results
        """
        self.neural_model_path = neural_model_path
        self.rules_path = rules_path
        self.lightgbm_model_path = lightgbm_model_path
        self.output_dir = output_dir or settings.OUTPUTS_DIR / "tuning"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scorer = make_scorer(f1_score, average="weighted")
        logger.info("🔧 HyperparameterTuner inicializado")

    def create_ensemble_pipeline(self, **params) -> Pipeline:
        """
        Create the ensemble pipeline with the specified parameters.

        Args:
            **params: Parameters for the pipeline components

        Returns:
            Configured pipeline
        """
        symbolic_threshold = params.get("symbolic_min_confidence_threshold", 0.05)
        ensemble_config = self._load_ensemble_config()
        yaml_meta_params = ensemble_config.get("meta_learner", {}).get("params", {})
        meta_params = {
            "n_estimators": params.get(
                "meta_n_estimators", yaml_meta_params.get("n_estimators", 250)
            ),
            "max_depth": params.get(
                "meta_max_depth", yaml_meta_params.get("max_depth", 5)
            ),
            "learning_rate": params.get(
                "meta_learning_rate", yaml_meta_params.get("learning_rate", 0.05)
            ),
            "subsample": params.get(
                "meta_subsample", yaml_meta_params.get("subsample", 0.8)
            ),
            "colsample_bytree": params.get(
                "meta_colsample_bytree", yaml_meta_params.get("colsample_bytree", 0.8)
            ),
            "min_child_weight": params.get(
                "meta_min_child_weight", yaml_meta_params.get("min_child_weight", 3)
            ),
            "gamma": params.get("meta_gamma", yaml_meta_params.get("gamma", 0.1)),
            "objective": "binary:logistic",
            "eval_metric": yaml_meta_params.get("eval_metric", ["logloss", "aucpr"]),
            "use_label_encoder": False,
            "random_state": yaml_meta_params.get("random_state", 42),
            "n_jobs": -1,
        }
        hybrid_predictor = HybridWrapper(hybrid_model_path=self.neural_model_path)
        symbolic_extractor = SymbolicFeatureExtractor(
            rules_path=self.rules_path, min_confidence_threshold=symbolic_threshold
        )
        meta_learner = XGBClassifier(**meta_params)
        hybrid_pipe = Pipeline([("hybrid", ProbaTransformer(hybrid_predictor))])
        symbolic_pipe = Pipeline([("symbolic", symbolic_extractor)])
        combined_features = FeatureUnion(
            [("hybrid_pred", hybrid_pipe), ("symbolic_rules", symbolic_pipe)]
        )
        pipeline = Pipeline(
            [("features", combined_features), ("meta_learner", meta_learner)]
        )

        return pipeline

    def get_parameter_grid(self, search_type: str = "medium") -> dict[str, list]:
        """
        Returns the parameter grid for search.

        Args:
            search_type: Type of search ('quick', 'medium', 'extensive')

        Returns:
            Dictionary with the parameter grid
        """
        if search_type == "quick":
            param_grid = {
                "features__symbolic_rules__symbolic__min_confidence_threshold": [
                    0.05,
                    0.1,
                ],
                "meta_learner__n_estimators": [100, 200],
                "meta_learner__max_depth": [3, 5],
                "meta_learner__learning_rate": [0.05, 0.1],
            }
        elif search_type == "medium":
            param_grid = {
                "features__symbolic_rules__symbolic__min_confidence_threshold": [
                    0.01,
                    0.05,
                    0.1,
                    0.15,
                ],
                "meta_learner__n_estimators": [100, 200, 300],
                "meta_learner__max_depth": [3, 5, 7],
                "meta_learner__learning_rate": [0.01, 0.05, 0.1],
                "meta_learner__subsample": [0.7, 0.8, 0.9],
                "meta_learner__colsample_bytree": [0.7, 0.8, 0.9],
            }

        else:  # extensive
            param_grid = {
                "features__symbolic_rules__symbolic__min_confidence_threshold": [
                    0.001,
                    0.01,
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                ],
                "meta_learner__n_estimators": [50, 100, 200, 300, 500],
                "meta_learner__max_depth": [3, 5, 7, 10, 15],
                "meta_learner__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3],
                "meta_learner__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "meta_learner__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "meta_learner__min_child_weight": [1, 3, 5, 7],
                "meta_learner__gamma": [0, 0.1, 0.2, 0.3],
            }

        return param_grid

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search_type: str = "medium",
        cv: int = 5,
    ) -> dict:
        """
        Performs Grid Search for hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            search_type: Type of search
            cv: Number of folds for cross-validation

        Returns:
            Search results
        """
        logger.info(f"🔍 Iniciando Grid Search ({search_type})...")
        pipeline = self.create_ensemble_pipeline()
        param_grid = self.get_parameter_grid(search_type)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.scorer,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
        )
        start_time = datetime.now()
        grid_search.fit(X_train, y_train)
        search_time = (datetime.now() - start_time).total_seconds()
        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "search_time": search_time,
            "cv_results": grid_search.cv_results_,
        }
        self._save_results(results, "grid_search_results.json")
        logger.success(f"✅ Grid Search concluído em {search_time:.2f}s")
        logger.info(f"🏆 Melhor score: {results['best_score']:.4f}")

        return results

    def random_search(
        self, X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 50, cv: int = 5
    ) -> dict:
        """
        Performs Random Search for hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            n_iter: Number of iterations
            cv: Number of folds

        Returns:
            Search results
        """
        logger.info(f"🎲 Iniciando Random Search ({n_iter} iterações)...")
        pipeline = self.create_ensemble_pipeline()
        param_distributions = {
            "features__symbolic_rules__symbolic__min_confidence_threshold": np.linspace(
                0.001, 0.2, 20
            ),
            "meta_learner__n_estimators": [50, 100, 200, 300, 500],
            "meta_learner__max_depth": [3, 5, 7, 10, 15, 20],
            "meta_learner__learning_rate": np.logspace(-3, -0.5, 20),
            "meta_learner__subsample": np.linspace(0.6, 1.0, 10),
            "meta_learner__colsample_bytree": np.linspace(0.6, 1.0, 10),
            "meta_learner__min_child_weight": [1, 3, 5, 7, 10],
            "meta_learner__gamma": np.linspace(0, 0.5, 10),
        }
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scorer,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            n_jobs=-1,
            verbose=2,
            random_state=42,
            return_train_score=True,
        )
        start_time = datetime.now()
        random_search.fit(X_train, y_train)
        search_time = (datetime.now() - start_time).total_seconds()
        results = {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "search_time": search_time,
            "n_iterations": n_iter,
            "cv_results": random_search.cv_results_,
        }
        self._save_results(results, "random_search_results.json")
        logger.success(f"✅ Random Search concluído em {search_time:.2f}s")
        logger.info(f"🏆 Melhor score: {results['best_score']:.4f}")

        return results

    def optuna_search(
        self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 100, cv: int = 5
    ) -> dict:
        """
        Uses Optuna for Bayesian hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of trials
            cv: Number of folds

        Returns:
            Optimization results
        """
        logger.info(f"🧠 Iniciando otimização com Optuna ({n_trials} trials)...")

        def objective(trial):
            params = {
                "symbolic_min_confidence_threshold": trial.suggest_float(
                    "symbolic_threshold", 0.001, 0.2, log=True
                ),
                "meta_n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "meta_max_depth": trial.suggest_int("max_depth", 3, 20),
                "meta_learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                "meta_subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "meta_colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "meta_min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "meta_gamma": trial.suggest_float("gamma", 0, 0.5),
            }
            pipeline = self.create_ensemble_pipeline(**params)
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring=self.scorer,
                n_jobs=-1,
            )

            return scores.mean()

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        start_time = datetime.now()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        search_time = (datetime.now() - start_time).total_seconds()
        best_params = {
            "features__symbolic_rules__symbolic__min_confidence_threshold": study.best_params[
                "symbolic_threshold"
            ],
            "meta_learner__n_estimators": study.best_params["n_estimators"],
            "meta_learner__max_depth": study.best_params["max_depth"],
            "meta_learner__learning_rate": study.best_params["learning_rate"],
            "meta_learner__subsample": study.best_params["subsample"],
            "meta_learner__colsample_bytree": study.best_params["colsample_bytree"],
            "meta_learner__min_child_weight": study.best_params["min_child_weight"],
            "meta_learner__gamma": study.best_params["gamma"],
        }
        results = {
            "best_params": best_params,
            "best_score": study.best_value,
            "search_time": search_time,
            "n_trials": n_trials,
            "study_stats": {
                "n_trials": len(study.trials),
                "n_complete": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                ),
                "n_pruned": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                ),
            },
        }
        self._save_results(results, "optuna_search_results.json")
        study_path = self.output_dir / "optuna_study.pkl"
        joblib.dump(study, study_path)
        logger.success(f"✅ Optuna Search concluído em {search_time:.2f}s")
        logger.info(f"🏆 Melhor score: {results['best_score']:.4f}")

        return results

    def compare_methods(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Compares different optimization methods.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Comparison of results
        """
        logger.info("📊 Comparando métodos de otimização...")
        comparison = {}
        logger.info("1️⃣ Grid Search...")
        grid_results = self.grid_search(X_train, y_train, search_type="quick")
        comparison["grid_search"] = {
            "best_score": grid_results["best_score"],
            "time": grid_results["search_time"],
        }
        logger.info("2️⃣ Random Search...")
        random_results = self.random_search(X_train, y_train, n_iter=30)
        comparison["random_search"] = {
            "best_score": random_results["best_score"],
            "time": random_results["search_time"],
        }
        logger.info("3️⃣ Optuna...")
        optuna_results = self.optuna_search(X_train, y_train, n_trials=50)
        comparison["optuna"] = {
            "best_score": optuna_results["best_score"],
            "time": optuna_results["search_time"],
        }
        best_method = max(comparison.items(), key=lambda x: x[1]["best_score"])
        comparison["best_method"] = best_method[0]
        comparison["best_overall_score"] = best_method[1]["best_score"]
        self._save_results(comparison, "methods_comparison.json")
        logger.success("✅ Comparação concluída!")
        logger.info(
            f"🏆 Melhor método: {comparison['best_method']} (Score: {comparison['best_overall_score']:.4f})"
        )

        return comparison

    def _save_results(self, results: dict, filename: str):
        """
        Save the optimization results.

        Args:
            results: Results to save
            filename: File name
        """

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        clean_results = convert_numpy(results)
        output_path = self.output_dir / filename
        FileManager().save(clean_results, output_path)
        logger.info(f"💾 Resultados salvos em {output_path}")


if __name__ == "__main__":
    """
    Example main for running hyperparameter tuning without DataManager.
    Loads data from CSV or .npy, initializes the tuner, and runs grid search.
    """

    import argparse
    import sys

    import polars as pl

    from pff.config import settings

    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for the stacking ensemble."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV or .npy file with features and target",
    )
    parser.add_argument(
        "--target", type=str, default="target", help="Target column name (for CSV)"
    )
    parser.add_argument(
        "--rules_path",
        type=str,
        default=str(settings.OUTPUTS_DIR / "pyclause" / "rules_anyburl.tsv"),
        help="Path to the generated AnyBURL rules .tsv file.",
    )
    parser.add_argument(
        "--hybrid_model_path",
        type=str,
        default=str(settings.OUTPUTS_DIR / "transe" / "lightgbm_model.bin"),
        help="Path to the trained Hybrid LightGBM model .bin file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--search_type",
        type=str,
        default="medium",
        choices=["quick", "medium", "extensive"],
        help="Grid search type",
    )
    args = parser.parse_args()

    try:
        if args.data.endswith(".csv"):
            df = pl.read_csv(args.data)
            if args.target not in df.columns:
                raise ValueError(f"Target column '{args.target}' not found in CSV.")
            X_train = df.drop(args.target).to_numpy()
            y_train = df[args.target].to_numpy()
        elif args.data.endswith(".npy"):
            data = np.load(args.data, allow_pickle=True)
            if isinstance(data, dict) and "X" in data and "y" in data:
                X_train = data["X"]
                y_train = data["y"]
            else:
                raise ValueError(
                    'Numpy file must contain a dict with keys "X" and "y".'
                )
        else:
            logger.error("Formato de arquivo de dados não suportado. Use CSV ou NPY.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Erro ao carregar os dados: {e}")
        sys.exit(1)

    tuner = HyperparameterTuner(
        neural_model_path=args.hybrid_model_path,
        rules_path=args.rules_path,
        lightgbm_model_path=args.hybrid_model_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    logger.info("🚀 Iniciando grid search...")
    results = tuner.grid_search(X_train, y_train, search_type=args.search_type)
    logger.success("🏁 Tuning finalizado!")
