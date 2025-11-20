"""
Adaptive Learning Module for AnyBURL & PyClause

Implements intelligent parameter adaptation based on:
- Real-time performance metrics
- Rule quality analysis
- Ranking score feedback
- Dynamic threshold adjustment
- Multi-objective optimization

Author: PFF Team
Date: 2025-11-04
Version: 1.0.0
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pff.utils import logger
from pff.utils.core.file_manager import FileManager


class PerformanceMetrics:
    """Track and analyze performance metrics."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_file = output_dir / "performance_metrics.json"
        self.metrics_history = self._load_metrics_history()

    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load historical performance metrics."""
        if self.metrics_file.exists():
            try:
                return FileManager.read(self.metrics_file)
            except Exception:
                pass
        return []

    def record_trial(self, trial_params: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Record trial performance metrics."""
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'parameters': trial_params,
            'metrics': results,
        }
        self.metrics_history.append(metric_entry)

        FileManager.save(self.metrics_history, self.metrics_file, indent=2)

    def get_best_parameters(self, metric_name: str, maximize: bool = True) -> Dict[str, Any]:
        """Get best parameters for specific metric."""
        if not self.metrics_history:
            return {}

        sorted_trials = sorted(
            self.metrics_history,
            key=lambda t: t['metrics'].get(metric_name, 0),
            reverse=maximize,
        )

        return sorted_trials[0]['parameters'] if sorted_trials else {}

    def analyze_parameter_impact(self) -> Dict[str, float]:
        """Analyze impact of different parameters on performance."""
        if len(self.metrics_history) < 3:
            return {}

        impact_scores = {}

        for param_name in self.metrics_history[0]['parameters'].keys():
            try:
                values = [t['parameters'][param_name] for t in self.metrics_history]
                scores = [t['metrics'].get('mrr', 0) for t in self.metrics_history]

                correlation = self._calculate_correlation(values, scores)
                impact_scores[param_name] = abs(correlation) if correlation else 0

            except Exception:
                impact_scores[param_name] = 0

        return impact_scores

    @staticmethod
    def _calculate_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return 0

        return numerator / denominator


class AdaptiveParameterTuner:
    """Adapt parameters based on real-time feedback."""

    def __init__(self, metrics_tracker: PerformanceMetrics):
        self.metrics = metrics_tracker

    def suggest_next_parameters(
        self,
        current_params: Dict[str, Any],
        optimization_target: str = 'mrr',
    ) -> Dict[str, Any]:
        """
        Suggest next parameter configuration based on history.

        Args:
            current_params: Current parameters
            optimization_target: Target metric to optimize

        Returns:
            Suggested parameters
        """
        suggested = current_params.copy()

        best_params = self.metrics.get_best_parameters(optimization_target)

        if best_params:
            for param_name, best_value in best_params.items():
                if param_name in current_params:
                    current_value = current_params[param_name]

                    if isinstance(current_value, (int, float)):
                        adaptation_factor = 0.1
                        suggested[param_name] = current_value + adaptation_factor * (best_value - current_value)

                        if param_name.startswith('THRESHOLD_') or param_name.endswith('_WEIGHT'):
                            suggested[param_name] = max(0.001, min(1.0, suggested[param_name]))

                        logger.info(f"   {param_name}: {current_value:.4f} → {suggested[param_name]:.4f}")

        return suggested


class RuleQualityAnalyzer:
    """Analyze quality of learned rules."""

    @staticmethod
    def analyze_rule_distribution(rules_path: Path) -> Dict[str, float]:
        """
        Analyze distribution and quality of rules.

        Args:
            rules_path: Path to rules file

        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'rule_count': 0,
            'avg_confidence': 0,
            'cyclic_ratio': 0,
            'length_distribution': {},
        }

        if not rules_path.exists():
            return metrics

        try:
            confidences = []
            cyclic_count = 0
            length_counts = {}

            # Load rules using FileManager (assuming text content)
            content = FileManager.read(rules_path)
            # If content is bytes (e.g. unknown extension), decode it
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # If FileManager returned a DataFrame (e.g. .tsv), we might need to handle it differently
            # But assuming it's a text file for rules as per original code structure
            if hasattr(content, 'splitlines'):
                lines = content.splitlines()
            else:
                # Fallback if it's not string-like
                lines = []

            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    confidence = float(parts[1])
                    confidences.append(confidence)

                    rule_body = parts[0]
                    rule_length = rule_body.count(' ') + 1

                    length_counts[rule_length] = length_counts.get(rule_length, 0) + 1

                    if 'INV' in rule_body or rule_length > 2:
                        cyclic_count += 1

            metrics['rule_count'] = len(confidences)
            metrics['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0
            metrics['cyclic_ratio'] = cyclic_count / metrics['rule_count'] if metrics['rule_count'] > 0 else 0
            metrics['length_distribution'] = length_counts

        except Exception as e:
            logger.warning(f"Could not analyze rule quality: {e}")

        return metrics

    @staticmethod
    def suggest_rule_optimizations(
        rule_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Suggest optimizations based on rule analysis.

        Args:
            rule_metrics: Rule quality metrics

        Returns:
            Suggested optimizations
        """
        suggestions = {}

        if rule_metrics.get('avg_confidence', 0) < 0.05:
            suggestions['THRESHOLD_CONFIDENCE'] = max(0.001, rule_metrics['avg_confidence'] * 0.5)
            logger.info("   Low confidence detected, reducing threshold")

        cyclic_ratio = rule_metrics.get('cyclic_ratio', 0)
        if cyclic_ratio > 0.7:
            suggestions['MAX_LENGTH_CYCLIC'] = max(2, rule_metrics.get('length_distribution', {}).get(3, 0) // 10)
            suggestions['EXCLUDE_AC2_RULES'] = True
            logger.info(f"   High cyclic ratio ({cyclic_ratio:.2f}), limiting cyclic rules")

        elif cyclic_ratio < 0.3:
            suggestions['MAX_LENGTH_CYCLIC'] = min(4, suggestions.get('MAX_LENGTH_CYCLIC', 3) + 1)
            logger.info(f"   Low cyclic ratio ({cyclic_ratio:.2f}), increasing cyclic length")

        rule_count = rule_metrics.get('rule_count', 0)
        if rule_count < 100:
            suggestions['SAMPLE_SIZE'] = min(1000, suggestions.get('SAMPLE_SIZE', 400) * 1.5)
            logger.info(f"   Low rule count ({rule_count}), increasing sample size")

        elif rule_count > 5000:
            suggestions['THRESHOLD_CONFIDENCE'] = suggestions.get('THRESHOLD_CONFIDENCE', 0.03) * 1.2
            logger.info(f"   High rule count ({rule_count}), increasing confidence")

        return suggestions


class RankingScoreAnalyzer:
    """Analyze ranking scores and suggest improvements."""

    @staticmethod
    def analyze_ranking_performance(ranking_file: Path) -> Dict[str, float]:
        """
        Analyze ranking performance metrics.

        Args:
            ranking_file: Path to ranking results

        Returns:
            Performance metrics
        """
        metrics = {
            'top1_accuracy': 0,
            'top10_coverage': 0,
            'mean_rank': float('inf'),
            'mrr': 0,
        }

        if not ranking_file.exists():
            return metrics

        try:
            rankings = FileManager.read(ranking_file)

            correct_predictions = 0
            in_top10 = 0
            total_ranks = []

            for query_id, predictions in rankings.items():
                if 'correct' in predictions:
                    correct_entity = predictions['correct']
                    predicted_ranking = predictions.get('ranking', {})

                    if correct_entity in predicted_ranking:
                        rank = predicted_ranking[correct_entity]
                        total_ranks.append(rank)

                        if rank == 1:
                            correct_predictions += 1

                        if rank <= 10:
                            in_top10 += 1

            total_queries = len(rankings)

            if total_queries > 0:
                metrics['top1_accuracy'] = correct_predictions / total_queries
                metrics['top10_coverage'] = in_top10 / total_queries
                metrics['mean_rank'] = sum(total_ranks) / len(total_ranks)
                metrics['mrr'] = sum(1 / r for r in total_ranks) / len(total_ranks)

        except Exception as e:
            logger.warning(f"Could not analyze ranking performance: {e}")

        return metrics

    @staticmethod
    def suggest_ranking_optimizations(
        ranking_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Suggest ranking optimizations based on performance.

        Args:
            ranking_metrics: Ranking performance metrics

        Returns:
            Suggested optimizations
        """
        suggestions = {}

        mrr = ranking_metrics.get('mrr', 0)
        top1_acc = ranking_metrics.get('top1_accuracy', 0)

        if mrr < 0.3:
            suggestions['aggregation_function'] = 'noisyor'
            suggestions['filter_w_data'] = True
            logger.info("   Low MRR, using conservative aggregation")

        elif mrr > 0.6:
            suggestions['aggregation_function'] = 'maxplus'
            logger.info("   High MRR, using aggressive aggregation")

        top1_accuracy = ranking_metrics.get('top1_accuracy', 0)
        if top1_accuracy < 0.1:
            suggestions['tie_handling'] = 'frequency'
            logger.info("   Low top-1 accuracy, using frequency-based tie handling")

        mean_rank = ranking_metrics.get('mean_rank', float('inf'))
        if mean_rank > 100:
            suggestions['num_threads'] = max(1, suggestions.get('num_threads', 1) * 2)
            logger.info(f"   High mean rank ({mean_rank:.1f}), increasing threads")

        return suggestions


class SelfTuningOptimizer:
    """Self-tuning optimizer for AnyBURL & PyClause pipeline."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_tracker = PerformanceMetrics(output_dir)
        self.param_tuner = AdaptiveParameterTuner(self.metrics_tracker)

    def optimize_with_feedback(
        self,
        pipeline_config: Dict[str, Any],
        previous_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Optimize pipeline parameters based on previous results.

        Args:
            pipeline_config: Current pipeline configuration
            previous_results: Results from previous run

        Returns:
            Optimized configuration
        """
        optimized = pipeline_config.copy()

        if previous_results:
            self.metrics_tracker.record_trial(
                pipeline_config.get('anyburl', {}),
                previous_results,
            )

            suggested_anyburl = self.param_tuner.suggest_next_parameters(
                pipeline_config.get('anyburl', {}),
                optimization_target='mrr',
            )

            if suggested_anyburl != pipeline_config.get('anyburl', {}):
                optimized['anyburl'] = suggested_anyburl
                logger.info(" Adapted AnyBURL parameters based on feedback")

        return optimized

    def analyze_and_adapt(
        self,
        rules_path: Path,
        ranking_file: Path,
        current_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze results and adapt parameters.

        Args:
            rules_path: Path to learned rules
            ranking_file: Path to ranking results
            current_config: Current configuration

        Returns:
            Adapted configuration
        """
        adapted = current_config.copy()

        rule_metrics = RuleQualityAnalyzer.analyze_rule_distribution(rules_path)
        ranking_metrics = RankingScoreAnalyzer.analyze_ranking_performance(ranking_file)

        logger.info(" Rule Quality Analysis:")
        logger.info(f"   Rules: {rule_metrics.get('rule_count', 0)}")
        logger.info(f"   Avg Confidence: {rule_metrics.get('avg_confidence', 0):.4f}")
        logger.info(f"   Cyclic Ratio: {rule_metrics.get('cyclic_ratio', 0):.2f}")

        rule_suggestions = RuleQualityAnalyzer.suggest_rule_optimizations(rule_metrics)

        logger.info(" Ranking Analysis:")
        logger.info(f"   MRR: {ranking_metrics.get('mrr', 0):.4f}")
        logger.info(f"   Top-1 Accuracy: {ranking_metrics.get('top1_accuracy', 0):.4f}")
        logger.info(f"   Mean Rank: {ranking_metrics.get('mean_rank', float('inf')):.1f}")

        ranking_suggestions = RankingScoreAnalyzer.suggest_ranking_optimizations(ranking_metrics)

        if 'anyburl' in adapted and rule_suggestions:
            adapted['anyburl'].update(rule_suggestions)
            logger.info(" Adapted AnyBURL parameters based on rule analysis")

        if 'pyclause' in adapted and ranking_suggestions:
            pyclause_config = adapted['pyclause'].copy()
            ranking_handler = pyclause_config.get('ranking_handler', {}).copy()

            for key, value in ranking_suggestions.items():
                if key in ['aggregation_function', 'tie_handling']:
                    ranking_handler[key] = value
                else:
                    pyclause_config[key] = value

            pyclause_config['ranking_handler'] = ranking_handler
            adapted['pyclause'] = pyclause_config
            logger.info(" Adapted PyClause parameters based on ranking analysis")

        return adapted


if __name__ == "__main__":
    logger.info("Adaptive Learning Module for AnyBURL & PyClause")
    logger.info("=" * 60)
