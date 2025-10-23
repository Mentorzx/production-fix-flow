from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


class KGCMetrics:
    """
    KGCMetrics
    A utility class providing static methods to compute common evaluation metrics for Knowledge Graph Completion (KGC) tasks.
    Includes ranking metrics such as Mean Reciprocal Rank (MRR), Hits@K, and NDCG, as well as standard classification metrics
    like AUC-ROC, Average Precision, and accuracy. Designed to work with numpy arrays for ground truth labels and predicted scores.
    Methods:
        mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float
        hits_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float
            Computes the Hits@K metric, representing the proportion of cases where at least one relevant item is ranked in the top-K predictions.
        calculate_all_metrics(
            y_true: np.ndarray,
            y_scores: np.ndarray,
            y_pred: np.ndarray | None = None
        ) -> dict[str, float]
            Calculates a comprehensive set of KGC metrics, including MRR, Hits@K, NDCG, AUC-ROC, Average Precision, and accuracy,
            returning them in a dictionary.
    """

    @staticmethod
    def mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Calculates the Mean Reciprocal Rank (MRR) for a single set of predictions.
        The MRR is a metric commonly used for evaluating systems that return a list of possible responses to a query, ordered by probability of correctness. It is the average of the reciprocal ranks of the first relevant item (label == 1) in the predicted ranking.
        Args:
            y_true (np.ndarray): Binary ground truth labels (1 for relevant, 0 for not relevant).
            y_scores (np.ndarray): Predicted scores for each item, where higher scores indicate higher relevance.
        Returns:
            float: The mean reciprocal rank for the given predictions. Returns 0.0 if there are no relevant items.
        """
        sorted_indices = np.argsort(-y_scores)
        sorted_labels = y_true[sorted_indices]
        reciprocal_ranks = []
        for i, label in enumerate(sorted_labels):
            if label == 1:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    @staticmethod
    def hits_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
        """
        Calculates the Hits@K metric for a single prediction.
        The Hits@K metric evaluates whether at least one of the top K predictions is a positive instance (label 1).
        It is commonly used in information retrieval and knowledge graph completion tasks.
        Args:
            y_true (np.ndarray): Array of true binary labels (0 or 1) for each candidate.
            y_scores (np.ndarray): Array of predicted scores for each candidate.
            k (int, optional): The number of top elements to consider. Defaults to 10.
        Returns:
            float: 1.0 if at least one of the top K predictions is a positive instance, 0.0 otherwise.
        """
        sorted_indices = np.argsort(-y_scores)
        top_k_indices = sorted_indices[:k]
        top_k_labels = y_true[top_k_indices]

        return float(np.any(top_k_labels == 1))

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Calculate a set of evaluation metrics for knowledge graph completion or binary classification tasks.
        Parameters:
            y_true (np.ndarray): Ground truth binary labels (0 or 1).
            y_scores (np.ndarray): Predicted scores or probabilities for the positive class.
            y_pred (np.ndarray | None, optional): Predicted binary labels. If None, will be computed from y_scores using a threshold of 0.5.
        Returns:
            dict[str, float]: Dictionary containing the following metrics:
                - "mrr": Mean Reciprocal Rank.
                - "hits@1": Hits@1 (proportion of correct results in top 1).
                - "hits@3": Hits@3 (proportion of correct results in top 3).
                - "hits@10": Hits@10 (proportion of correct results in top 10).
                - "auc_roc": Area Under the ROC Curve.
                - "auc_pr": Area Under the Precision-Recall Curve (Average Precision).
                - "ndcg@10": Normalized Discounted Cumulative Gain at rank 10.
                - "accuracy": Classification accuracy.
        Notes:
            - If ranking metrics or AUC metrics fail, default values are used and a warning is printed.
            - For NDCG, input arrays are reshaped to match the expected format.
        """
        if y_pred is None:
            y_pred = (y_scores > 0.5).astype(int)
        metrics = {}
        try:
            metrics["mrr"] = KGCMetrics.mean_reciprocal_rank(y_true, y_scores)
            metrics["hits@1"] = KGCMetrics.hits_at_k(y_true, y_scores, k=1)
            metrics["hits@3"] = KGCMetrics.hits_at_k(y_true, y_scores, k=3)
            metrics["hits@10"] = KGCMetrics.hits_at_k(y_true, y_scores, k=10)
        except Exception as e:
            print(f"Warning: Ranking metrics failed: {e}")
            metrics.update({"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0})
        try:
            if len(np.unique(y_true)) > 1:
                metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
                metrics["auc_pr"] = average_precision_score(y_true, y_scores)
            else:
                metrics["auc_roc"] = 0.5
                metrics["auc_pr"] = np.mean(y_true)
        except Exception as e:
            print(f"Warning: AUC metrics failed: {e}")
            metrics.update({"auc_roc": 0.5, "auc_pr": 0.5})
        try:
            y_true_ndcg = y_true.reshape(1, -1)
            y_scores_ndcg = y_scores.reshape(1, -1)
            metrics["ndcg@10"] = ndcg_score(y_true_ndcg, y_scores_ndcg, k=10)
        except Exception as e:
            print(f"Warning: NDCG metric failed: {e}")
            metrics["ndcg@10"] = 0.0
        accuracy = np.mean(y_true == y_pred)
        metrics["accuracy"] = float(accuracy)

        return metrics


class KGCEvaluator:
    """
    KGCEvaluator is a utility class for evaluating Knowledge Graph Completion (KGC) models with detailed metrics and analyses.
    Attributes:
        model: The trained model to be evaluated. Must implement `predict` and `predict_proba` methods.
        metrics_history: List of dictionaries containing historical KGC metric results from previous evaluations.
    Methods:
        __init__(model):
            Initializes the evaluator with a given model.
        evaluate_detailed(X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
            Performs a detailed evaluation of the model on the provided test data, computing KGC-specific metrics,
            per-class analysis, and confidence analysis. Returns a dictionary with all results and prints a detailed report.
        _analyze_by_class(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict[str, Any]:
            Computes statistics for each class in the test set, including count, mean/standard deviation of scores,
            accuracy, and score range.
        _analyze_confidence(y_scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
            Analyzes prediction confidence by dividing samples into quartiles based on predicted scores,
            reporting count, accuracy, and mean score for each confidence range.
        _print_detailed_report(metrics: dict[str, float], class_analysis: dict[str, Any]) -> None:
            Prints a comprehensive evaluation report including ranking metrics, classification metrics,
            and per-class analysis.
    """

    def __init__(self, model):
        self.model = model
        self.metrics_history = []

    def evaluate_detailed(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, Any]:
        """
        Performs a detailed evaluation of the model on the provided test data.
        """
        print("🔍 Executando avaliação detalhada para KGC...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        y_scores = y_proba[:, 1]
        kgc_metrics = KGCMetrics.calculate_all_metrics(y_test, y_scores, y_pred)
        class_analysis = self._analyze_by_class(y_test, y_pred, y_scores)
        confidence_analysis = self._analyze_confidence(y_scores, y_test)
        results = {
            "kgc_metrics": kgc_metrics,
            "class_analysis": class_analysis,
            "confidence_analysis": confidence_analysis,
            "predictions": y_pred,
            "probabilities": y_proba,
            "scores": y_scores,
        }
        self.metrics_history.append(kgc_metrics)
        self._print_detailed_report(kgc_metrics, class_analysis)

        return results

    def _analyze_by_class(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
    ) -> dict[str, Any]:
        """
        Analyzes prediction results by class, computing statistics for each unique class label.
        Parameters:
            y_true (np.ndarray): Array of true class labels.
            y_pred (np.ndarray): Array of predicted class labels.
            y_scores (np.ndarray): Array of prediction scores corresponding to each sample.
        Returns:
            dict[str, Any]: A dictionary where each key is of the form 'class_{cls}' for each unique class label,
                and each value is a dictionary containing:
                    - 'count': Number of samples for the class.
                    - 'mean_score': Mean of the prediction scores for the class.
                    - 'std_score': Standard deviation of the prediction scores for the class.
                    - 'accuracy': Accuracy of predictions for the class.
                    - 'min_score': Minimum prediction score for the class.
                    - 'max_score': Maximum prediction score for the class.
        """
        unique_classes = np.unique(y_true)
        analysis = {}
        for cls in unique_classes:
            mask = y_true == cls
            cls_scores = y_scores[mask]
            cls_pred = y_pred[mask]
            cls_true = y_true[mask]
            analysis[f"class_{cls}"] = {
                "count": int(np.sum(mask)),
                "mean_score": float(np.mean(cls_scores)),
                "std_score": float(np.std(cls_scores)),
                "accuracy": float(np.mean(cls_pred == cls_true)),
                "min_score": float(np.min(cls_scores)),
                "max_score": float(np.max(cls_scores)),
            }

        return analysis

    def _analyze_confidence(
        self, y_scores: np.ndarray, y_true: np.ndarray
    ) -> dict[str, float]:
        """
        Analyzes prediction confidence by dividing the score distribution into quartiles and computing statistics for each range.
        Parameters:
            y_scores (np.ndarray): Array of predicted confidence scores, expected to be in the range [0, 1].
            y_true (np.ndarray): Array of true binary labels (0 or 1), same shape as y_scores.
        Returns:
            dict[str, float]: A dictionary with keys for each confidence range ("low_confidence", "medium_low_confidence",
            "medium_high_confidence", "high_confidence"). Each value is a dictionary containing:
                - "count": Number of samples in the range.
                - "accuracy": Accuracy of predictions (thresholded at 0.5) within the range.
                - "mean_score": Mean confidence score within the range.
        """
        quartiles = np.percentile(y_scores, [25, 50, 75])
        analysis = {}
        ranges = [
            ("low_confidence", 0.0, quartiles[0]),
            ("medium_low_confidence", quartiles[0], quartiles[1]),
            ("medium_high_confidence", quartiles[1], quartiles[2]),
            ("high_confidence", quartiles[2], 1.0),
        ]
        for name, low, high in ranges:
            mask = (y_scores >= low) & (y_scores <= high)
            if np.any(mask):
                subset_scores = y_scores[mask]
                subset_true = y_true[mask]
                subset_pred = (subset_scores > 0.5).astype(int)
                analysis[name] = {
                    "count": int(np.sum(mask)),
                    "accuracy": float(np.mean(subset_pred == subset_true)),
                    "mean_score": float(np.mean(subset_scores)),
                }
            else:
                analysis[name] = {"count": 0, "accuracy": 0.0, "mean_score": 0.0}

        return analysis

    def _print_detailed_report(
        self, metrics: dict[str, float], class_analysis: dict[str, Any]
    ) -> None:
        """
        Prints a detailed evaluation report for Knowledge Graph Completion (KGC) metrics and per-class analysis.
        Args:
            metrics (dict[str, float]): Dictionary containing overall evaluation metrics such as MRR, Hits@K, NDCG@10, accuracy, AUC-ROC, and AUC-PR.
            class_analysis (dict[str, Any]): Dictionary containing per-class analysis with statistics like sample count, accuracy, mean score, and standard deviation of scores.
        Returns:
            None
        """
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO DETALHADO DE AVALIAÇÃO KGC")
        print("=" * 60)
        print("\n🎯 Métricas de Ranking (KGC):")
        print(f"   MRR:        {metrics['mrr']:.4f}")
        print(f"   Hits@1:     {metrics['hits@1']:.4f}")
        print(f"   Hits@3:     {metrics['hits@3']:.4f}")
        print(f"   Hits@10:    {metrics['hits@10']:.4f}")
        print(f"   NDCG@10:    {metrics['ndcg@10']:.4f}")
        print("\n📈 Métricas de Classificação:")
        print(f"   Accuracy:   {metrics['accuracy']:.4f}")
        print(f"   AUC-ROC:    {metrics['auc_roc']:.4f}")
        print(f"   AUC-PR:     {metrics['auc_pr']:.4f}")
        print("\n📋 Análise por Classe:")
        for class_name, stats in class_analysis.items():
            print(
                f"   {class_name}: {stats['count']} samples, "
                f"acc={stats['accuracy']:.3f}, "
                f"mean_score={stats['mean_score']:.3f}±{stats['std_score']:.3f}"
            )
        print("=" * 60 + "\n")
