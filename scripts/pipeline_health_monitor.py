#!/usr/bin/env python3
"""
Pipeline Health Monitor
Monitors ML pipeline health metrics and provides alerts for critical issues.

Based on LOGS_ANALYSIS.md corrections:
1. Overfitting detection (violation percentages)
2. Feature 324 presence validation
3. Numba vs fallback consistency
4. Model balance (symbolic vs hybrid contributions)
"""

import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline_health.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Represents a single health metric with thresholds."""
    name: str
    value: float
    unit: str
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    is_critical: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def is_healthy(self) -> bool:
        """Check if metric is within acceptable range."""
        if self.min_threshold is not None and self.value < self.min_threshold:
            return False
        if self.max_threshold is not None and self.value > self.max_threshold:
            return False
        return True

    def get_status(self) -> str:
        """Get health status."""
        if not self.is_healthy():
            return "CRITICAL" if self.is_critical else "WARNING"
        return "HEALTHY"

@dataclass
class PipelineAlert:
    """Represents a pipeline health alert."""
    severity: str  # CRITICAL, WARNING, INFO
    message: str
    metric_name: str
    value: float
    threshold: Optional[float]
    timestamp: datetime = None
    resolved: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PipelineHealthMonitor:
    """Comprehensive pipeline health monitoring."""

    def __init__(self, log_dir: str = "logs", output_dir: str = "outputs"):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: List[PipelineAlert] = []
        self.health_metrics: Dict[str, HealthMetric] = {}

        # Thresholds from LOGS_ANALYSIS.md
        self.thresholds = {
            'violation_percentage': 200.0,  # Max 200% violations
            'symbolic_contribution': 85.0,  # Max 85% symbolic contribution
            'hybrid_contribution': 15.0,    # Min 15% hybrid contribution
            'f1_score': 0.65,               # Min F1-Score
            'feature_324_importance': 0.01, # Min importance for feature 324
            'numba_success_rate': 0.95,     # Min 95% Numba success
        }

    def parse_latest_log(self) -> Optional[Path]:
        """Find and return the latest log file."""
        log_files = list(self.log_dir.glob("*.log"))
        if not log_files:
            logger.warning("No log files found")
            return None

        # Prefer the most recent date-prefixed log file
        date_logs = [f for f in log_files if re.match(r'\d{4}-\d{2}-\d{2}\.log', f.name)]
        if date_logs:
            latest_log = max(date_logs, key=lambda f: f.stat().st_mtime)
        else:
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)

        logger.info(f"Analyzing log file: {latest_log}")
        return latest_log

    def extract_log_metrics(self, log_file: Path) -> Dict[str, Any]:
        """Extract metrics from log file using regex patterns."""
        metrics = {
            'violation_percentages': [],
            'numba_successes': 0,
            'numba_failures': 0,
            'fallback_usages': 0,
            'features_shape': None,
            'feature_importance': {},
            'ensemble_results': {},
            'processing_times': [],
            'error_count': 0
        }

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Violation percentages
            violation_pattern = r'Feature stats: (\d+)/(\d+) violations \(([\d.]+) = ([\d.]+)%\)'
            for match in re.finditer(violation_pattern, content):
                violations, total, ratio, percentage = match.groups()
                metrics['violation_percentages'].append({
                    'violations': int(violations),
                    'total': int(total),
                    'ratio': float(ratio),
                    'percentage': float(percentage)
                })

            # Numba success/failure
            metrics['numba_successes'] = len(re.findall(r'Numba: batch-parallel succeeded', content))
            metrics['numba_failures'] = len(re.findall(r'Numba.*failed|Numba.*error', content))
            metrics['fallback_usages'] = len(re.findall(r'Usando fallback|fallback.*para', content))

            # Features shape
            shape_pattern = r'Features binárias calculadas: shape=\((\d+), (\d+)\)'
            shape_match = re.search(shape_pattern, content)
            if shape_match:
                metrics['features_shape'] = (int(shape_match.group(1)), int(shape_match.group(2)))

            # Feature importance
            importance_pattern = r'(\d+\.\w+):\s+([\d.]+)'
            importance_matches = re.findall(importance_pattern, content)
            for feature, importance in importance_matches:
                metrics['feature_importance'][feature] = float(importance)

            # Ensemble results
            f1_pattern = r'F1-Score Final: ([\d.]+)'
            f1_match = re.search(f1_pattern, content)
            if f1_match:
                metrics['ensemble_results']['f1_score'] = float(f1_match.group(1))

            symbolic_pattern = r'Contribuição das regras simbólicas: ([\d.]+)%'
            symbolic_match = re.search(symbolic_pattern, content)
            if symbolic_match:
                metrics['ensemble_results']['symbolic_contribution'] = float(symbolic_match.group(1))

            hybrid_pattern = r'Contribuição do modelo híbrido: ([\d.]+)%'
            hybrid_match = re.search(hybrid_pattern, content)
            if hybrid_match:
                metrics['ensemble_results']['hybrid_contribution'] = float(hybrid_match.group(1))

            # Processing times
            time_pattern = r'Processed (\d+)/(\d+) samples.*?([\d.]+)s'
            for match in re.finditer(time_pattern, content, re.DOTALL):
                processed, total, time_taken = match.groups()
                metrics['processing_times'].append(float(time_taken))

            # Error count
            metrics['error_count'] = len(re.findall(r'ERROR|CRITICAL|Exception|Traceback', content))

        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {e}")

        return metrics

    def check_model_metadata(self) -> Dict[str, Any]:
        """Check model metadata for additional metrics."""
        metadata_file = self.output_dir / "ensemble" / "model_metadata.json"
        metadata = {}

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    logger.info("Model metadata loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model metadata: {e}")

        return metadata

    def calculate_health_metrics(self, log_metrics: Dict[str, Any],
                               metadata: Dict[str, Any]) -> List[HealthMetric]:
        """Calculate health metrics from extracted data."""
        health_metrics = []

        # 1. Violation Percentage (CRITICAL from LOGS_ANALYSIS.md)
        if log_metrics['violation_percentages']:
            avg_violation_pct = np.mean([v['percentage'] for v in log_metrics['violation_percentages']])
            health_metrics.append(HealthMetric(
                name="violation_percentage",
                value=avg_violation_pct,
                unit="%",
                max_threshold=self.thresholds['violation_percentage'],
                is_critical=True
            ))

        # 2. Numba Success Rate
        total_numba = log_metrics['numba_successes'] + log_metrics['numba_failures']
        if total_numba > 0:
            numba_success_rate = log_metrics['numba_successes'] / total_numba
            health_metrics.append(HealthMetric(
                name="numba_success_rate",
                value=numba_success_rate,
                unit="ratio",
                min_threshold=self.thresholds['numba_success_rate']
            ))

        # 3. Model Balance (Symbolic vs Hybrid)
        ensemble = log_metrics['ensemble_results']
        if 'symbolic_contribution' in ensemble:
            health_metrics.append(HealthMetric(
                name="symbolic_contribution",
                value=ensemble['symbolic_contribution'],
                unit="%",
                max_threshold=self.thresholds['symbolic_contribution'],
                is_critical=True
            ))

        if 'hybrid_contribution' in ensemble:
            health_metrics.append(HealthMetric(
                name="hybrid_contribution",
                value=ensemble['hybrid_contribution'],
                unit="%",
                min_threshold=self.thresholds['hybrid_contribution']
            ))

        # 4. F1-Score
        if 'f1_score' in ensemble:
            health_metrics.append(HealthMetric(
                name="f1_score",
                value=ensemble['f1_score'],
                unit="score",
                min_threshold=self.thresholds['f1_score']
            ))

        # 5. Feature 324 Presence
        feature_324_importance = 0.0
        for feature, importance in log_metrics['feature_importance'].items():
            if '324' in str(feature):
                feature_324_importance = max(feature_324_importance, importance)

        health_metrics.append(HealthMetric(
            name="feature_324_importance",
            value=feature_324_importance,
            unit="importance",
            min_threshold=self.thresholds['feature_324_importance']
        ))

        # 6. Processing Performance
        if log_metrics['processing_times']:
            avg_time = np.mean(log_metrics['processing_times'])
            health_metrics.append(HealthMetric(
                name="avg_processing_time",
                value=avg_time,
                unit="seconds",
                max_threshold=120.0  # 2 minutes max
            ))

        # 7. Error Count
        health_metrics.append(HealthMetric(
            name="error_count",
            value=log_metrics['error_count'],
            unit="count",
            max_threshold=5.0,
            is_critical=log_metrics['error_count'] > 10
        ))

        return health_metrics

    def generate_alerts(self, metrics: List[HealthMetric]) -> List[PipelineAlert]:
        """Generate alerts for unhealthy metrics."""
        alerts = []

        for metric in metrics:
            if not metric.is_healthy():
                # Determine severity
                if metric.is_critical or metric.name in ['violation_percentage', 'symbolic_contribution']:
                    severity = "CRITICAL"
                elif metric.name in ['hybrid_contribution', 'feature_324_importance']:
                    severity = "WARNING"
                else:
                    severity = "WARNING"

                # Create appropriate message
                if metric.min_threshold is not None and metric.value < metric.min_threshold:
                    message = f"{metric.name} too low: {metric.value:.3f} {metric.unit} (min: {metric.min_threshold})"
                    threshold = metric.min_threshold
                elif metric.max_threshold is not None and metric.value > metric.max_threshold:
                    message = f"{metric.name} too high: {metric.value:.3f} {metric.unit} (max: {metric.max_threshold})"
                    threshold = metric.max_threshold
                else:
                    message = f"{metric.name} unhealthy: {metric.value:.3f} {metric.unit}"
                    threshold = None

                alerts.append(PipelineAlert(
                    severity=severity,
                    message=message,
                    metric_name=metric.name,
                    value=metric.value,
                    threshold=threshold
                ))

        return alerts

    def store_metrics(self, metrics: List[HealthMetric]):
        """Store metrics in history."""
        for metric in metrics:
            self.metrics_history[metric.name].append(metric)
            self.health_metrics[metric.name] = metric

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'metrics_count': len(self.health_metrics),
            'critical_alerts': 0,
            'warning_alerts': 0,
            'metrics': {},
            'alerts': [],
            'recommendations': [],
            'trends': {}
        }

        # Process metrics
        for name, metric in self.health_metrics.items():
            report['metrics'][name] = {
                'value': metric.value,
                'unit': metric.unit,
                'status': metric.get_status(),
                'threshold': {
                    'min': metric.min_threshold,
                    'max': metric.max_threshold
                }
            }

            if metric.get_status() == 'CRITICAL':
                report['critical_alerts'] += 1
            elif metric.get_status() == 'WARNING':
                report['warning_alerts'] += 1

        # Process alerts
        report['alerts'] = [
            {
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in self.alerts
        ]

        # Determine overall status
        if report['critical_alerts'] > 0:
            report['overall_status'] = 'CRITICAL'
        elif report['warning_alerts'] > 0:
            report['overall_status'] = 'WARNING'

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()

        # Calculate trends
        report['trends'] = self._calculate_trends()

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on current issues."""
        recommendations = []

        metrics = self.health_metrics

        # Overfitting recommendations
        if 'violation_percentage' in metrics and metrics['violation_percentage'].value > 200:
            recommendations.append(
                "🔴 CRITICAL: Overfitting detected! "
                "Consider increasing min_confidence_threshold in transformers.py "
                f"(current violation rate: {metrics['violation_percentage'].value:.1f}%)"
            )

        # Model balance recommendations
        if 'symbolic_contribution' in metrics and metrics['symbolic_contribution'].value > 85:
            recommendations.append(
                "🔴 CRITICAL: Model too dependent on symbolic rules! "
                "Consider adjusting XGBoost hyperparameters in advanced_trainer.py "
                f"(current symbolic: {metrics['symbolic_contribution'].value:.1f}%)"
            )

        if 'hybrid_contribution' in metrics and metrics['hybrid_contribution'].value < 15:
            recommendations.append(
                "⚠️ WARNING: Hybrid model contribution too low! "
                "Increase colsample_bytree and max_depth in XGBoost parameters "
                f"(current hybrid: {metrics['hybrid_contribution'].value:.1f}%)"
            )

        # Feature 324 recommendations
        if 'feature_324_importance' in metrics and metrics['feature_324_importance'].value < 0.01:
            recommendations.append(
                "⚠️ WARNING: Feature 324 not found or low importance! "
                "Check feature engineering pipeline and indexing "
                f"(current importance: {metrics['feature_324_importance'].value:.4f})"
            )

        # Numba recommendations
        if 'numba_success_rate' in metrics and metrics['numba_success_rate'].value < 0.95:
            recommendations.append(
                "⚠️ WARNING: Numba acceleration issues detected! "
                "Check for serialization issues and fallback logic "
                f"(success rate: {metrics['numba_success_rate'].value:.1%})"
            )

        # Performance recommendations
        if 'avg_processing_time' in metrics and metrics['avg_processing_time'].value > 120:
            recommendations.append(
                "⚠️ WARNING: Processing time too high! "
                "Consider enabling Numba acceleration or optimizing batch sizes "
                f"(current time: {metrics['avg_processing_time'].value:.1f}s)"
            )

        # General recommendations
        if not recommendations:
            recommendations.append("✅ All metrics healthy! Pipeline operating normally.")

        return recommendations

    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate trends for metrics with history."""
        trends = {}

        for name, history in self.metrics_history.items():
            if len(history) >= 2:
                recent_values = [m.value for m in list(history)[-5:]]  # Last 5 values
                if len(recent_values) >= 2:
                    # Simple trend calculation
                    recent_avg = np.mean(recent_values[-3:])  # Last 3
                    older_avg = np.mean(recent_values[:-3]) if len(recent_values) > 3 else recent_values[0]

                    change_pct = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0

                    if abs(change_pct) < 5:
                        trends[name] = "STABLE"
                    elif change_pct > 0:
                        trends[name] = "IMPROVING" if name in ['f1_score', 'hybrid_contribution'] else "DEGRADING"
                    else:
                        trends[name] = "IMPROVING" if name in ['violation_percentage', 'error_count'] else "DEGRADING"

        return trends

    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save health report to file."""
        report_file = self.output_dir / "pipeline_health_report.json"

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Health report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        return report_file

    def print_summary(self, report: Dict[str, Any]):
        """Print health summary to console."""
        print("\n" + "="*80)
        print(f"🏥 PIPELINE HEALTH REPORT - {report['timestamp']}")
        print("="*80)

        # Overall status
        status_emoji = {"HEALTHY": "✅", "WARNING": "⚠️", "CRITICAL": "🔴"}
        print(f"Overall Status: {status_emoji[report['overall_status']]} {report['overall_status']}")
        print(f"Metrics: {report['metrics_count']} | Critical Alerts: {report['critical_alerts']} | Warnings: {report['warning_alerts']}")

        # Key metrics
        print("\n📊 KEY METRICS:")
        print("-" * 40)
        key_metrics = ['violation_percentage', 'symbolic_contribution', 'hybrid_contribution', 'f1_score', 'feature_324_importance']

        for metric_name in key_metrics:
            if metric_name in report['metrics']:
                metric = report['metrics'][metric_name]
                status_emoji = {"HEALTHY": "✅", "WARNING": "⚠️", "CRITICAL": "🔴"}
                trend = report['trends'].get(metric_name, "")
                trend_symbol = {"IMPROVING": "📈", "DEGRADING": "📉", "STABLE": "➡️"}.get(trend, "")

                print(f"{metric_name:25}: {metric['value']:8.3f} {metric['unit']:10} "
                      f"{status_emoji[metric['status']]} {metric['status']:8} {trend_symbol}")

        # Alerts
        if report['alerts']:
            print(f"\n🚨 ALERTS ({len(report['alerts'])}):")
            print("-" * 40)
            for alert in report['alerts'][:5]:  # Show first 5
                severity_emoji = {"CRITICAL": "🔴", "WARNING": "⚠️", "INFO": "ℹ️"}
                print(f"{severity_emoji[alert['severity']]} {alert['message']}")

            if len(report['alerts']) > 5:
                print(f"... and {len(report['alerts']) - 5} more alerts")

        # Recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            print("-" * 40)
            for rec in report['recommendations']:
                print(f"• {rec}")

        print("\n" + "="*80)

    def run_health_check(self) -> Dict[str, Any]:
        """Run complete health check pipeline."""
        logger.info("Starting pipeline health check...")

        # 1. Parse latest log file
        log_file = self.parse_latest_log()
        if log_file is None:
            logger.error("No log files found for analysis")
            return {}

        # 2. Extract metrics
        log_metrics = self.extract_log_metrics(log_file)
        logger.info(f"Extracted {len(log_metrics)} metric categories from log")

        # 3. Check model metadata
        metadata = self.check_model_metadata()

        # 4. Calculate health metrics
        health_metrics = self.calculate_health_metrics(log_metrics, metadata)
        logger.info(f"Calculated {len(health_metrics)} health metrics")

        # 5. Generate alerts
        self.alerts = self.generate_alerts(health_metrics)
        logger.info(f"Generated {len(self.alerts)} alerts")

        # 6. Store metrics
        self.store_metrics(health_metrics)

        # 7. Generate report
        report = self.generate_health_report()

        # 8. Save report
        self.save_report(report)

        # 9. Print summary
        self.print_summary(report)

        return report

def main():
    """Main execution function."""
    print("🏥 Pipeline Health Monitor")
    print("Based on LOGS_ANALYSIS.md corrections")
    print("=" * 50)

    # Initialize monitor
    monitor = PipelineHealthMonitor()

    # Run health check
    try:
        report = monitor.run_health_check()

        # Exit with appropriate code
        if report.get('overall_status') == 'CRITICAL':
            print("\n🔴 CRITICAL ISSUES DETECTED - Immediate action required!")
            sys.exit(2)
        elif report.get('overall_status') == 'WARNING':
            print("\n⚠️ WARNINGS DETECTED - Attention needed")
            sys.exit(1)
        else:
            print("\n✅ Pipeline healthy - No action needed")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        print(f"\n❌ Health check failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()