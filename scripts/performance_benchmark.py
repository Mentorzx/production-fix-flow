#!/usr/bin/env python3
"""
Performance Benchmark System
Comprehensive benchmarking to compare pipeline performance before/after corrections.

Measures:
1. Runtime performance (execution time, throughput)
2. Memory usage (peak, average)
3. Model quality metrics (F1, precision, recall, AUC)
4. Resource utilization (CPU, GPU, disk I/O)
5. System health metrics (violation %, model balance)
"""

import sys
import json
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import subprocess
import threading
import yaml
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""
    n_runs: int = 3
    warmup_runs: int = 1
    monitoring_interval: float = 1.0  # seconds
    output_dir: str = "outputs/benchmarks"
    baseline_file: str = "outputs/benchmarks/baseline_results.json"
    create_visualizations: bool = True
    detailed_profiling: bool = True
    memory_profiling: bool = True

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single run."""
    run_id: int
    start_time: datetime
    end_time: datetime
    duration: float  # seconds
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_recv_mb: float
    network_io_sent_mb: float
    model_metrics: Dict[str, float]
    system_health: Dict[str, float]
    success: bool
    error_message: Optional[str] = None

@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    config: BenchmarkConfig
    timestamp: datetime
    baseline_version: str
    current_version: str
    runs: List[PerformanceMetrics]
    summary: Dict[str, Any]
    comparison: Optional[Dict[str, Any]] = None

class ResourceMonitor:
    """Monitor system resources during benchmark execution."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics = {
            'memory': [],
            'cpu': [],
            'disk_read': [],
            'disk_write': [],
            'network_recv': [],
            'network_sent': [],
            'timestamps': []
        }
        self.process = psutil.Process()

    def start(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop."""
        # Get initial disk I/O counters
        initial_disk_io = psutil.disk_io_counters()
        initial_net_io = psutil.net_io_counters()

        while self.monitoring:
            try:
                # Memory
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.metrics['memory'].append(memory_mb)

                # CPU
                cpu_percent = self.process.cpu_percent()
                self.metrics['cpu'].append(cpu_percent)

                # Disk I/O
                current_disk_io = psutil.disk_io_counters()
                if initial_disk_io and current_disk_io:
                    disk_read_mb = (current_disk_io.read_bytes - initial_disk_io.read_bytes) / 1024 / 1024
                    disk_write_mb = (current_disk_io.write_bytes - initial_disk_io.write_bytes) / 1024 / 1024
                    self.metrics['disk_read'].append(disk_read_mb)
                    self.metrics['disk_write'].append(disk_write_mb)

                # Network I/O
                current_net_io = psutil.net_io_counters()
                if initial_net_io and current_net_io:
                    net_recv_mb = (current_net_io.bytes_recv - initial_net_io.bytes_recv) / 1024 / 1024
                    net_sent_mb = (current_net_io.bytes_sent - initial_net_io.bytes_sent) / 1024 / 1024
                    self.metrics['network_recv'].append(net_recv_mb)
                    self.metrics['network_sent'].append(net_sent_mb)

                self.metrics['timestamps'].append(datetime.now())

                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"❌ Error in resource monitoring: {e}")
                time.sleep(self.interval)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from monitoring data."""
        if not self.metrics['memory']:
            return {}

        return {
            'peak_memory_mb': max(self.metrics['memory']),
            'avg_memory_mb': np.mean(self.metrics['memory']),
            'peak_cpu_percent': max(self.metrics['cpu']),
            'avg_cpu_percent': np.mean(self.metrics['cpu']),
            'disk_io_read_mb': max(self.metrics['disk_read']) if self.metrics['disk_read'] else 0,
            'disk_io_write_mb': max(self.metrics['disk_write']) if self.metrics['disk_write'] else 0,
            'network_io_recv_mb': max(self.metrics['network_recv']) if self.metrics['network_recv'] else 0,
            'network_io_sent_mb': max(self.metrics['network_sent']) if self.metrics['network_sent'] else 0
        }

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    @contextmanager
    def resource_monitoring(self):
        """Context manager for resource monitoring."""
        monitor = ResourceMonitor(self.config.monitoring_interval)
        monitor.start()
        try:
            yield monitor
        finally:
            monitor.stop()

    def extract_model_metrics(self, log_file: Path) -> Dict[str, float]:
        """Extract model quality metrics from log file."""
        metrics = {}
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract various metrics using regex
            import re

            # F1-Score
            f1_match = re.search(r'F1-Score Final: ([\d.]+)', content)
            if f1_match:
                metrics['f1_score'] = float(f1_match.group(1))

            # Precision and Recall (if available)
            precision_match = re.search(r'Precision: ([\d.]+)', content)
            if precision_match:
                metrics['precision'] = float(precision_match.group(1))

            recall_match = re.search(r'Recall: ([\d.]+)', content)
            if recall_match:
                metrics['recall'] = float(recall_match.group(1))

            # AUC
            auc_match = re.search(r'AUC: ([\d.]+)', content)
            if auc_match:
                metrics['auc'] = float(auc_match.group(1))

            # Symbolic/Hybrid contributions
            symbolic_match = re.search(r'Contribuição das regras simbólicas: ([\d.]+)%', content)
            if symbolic_match:
                metrics['symbolic_contribution'] = float(symbolic_match.group(1))

            hybrid_match = re.search(r'Contribuição do modelo híbrido: ([\d.]+)%', content)
            if hybrid_match:
                metrics['hybrid_contribution'] = float(hybrid_match.group(1))

            # Violation percentage
            violation_matches = re.findall(r'violations.*?([\d.]+)%', content)
            if violation_matches:
                # Average of all violation percentages
                violations = [float(v) for v in violation_matches]
                metrics['avg_violation_percentage'] = np.mean(violations)
                metrics['max_violation_percentage'] = max(violations)

            # Processing time
            time_matches = re.findall(r'Processed.*samples.*?([\d.]+)s', content)
            if time_matches:
                times = [float(t) for t in time_matches]
                metrics['avg_processing_time'] = np.mean(times)

            # Number of features
            feature_match = re.search(r'Features.*shape=\((\d+),\s*(\d+)\)', content)
            if feature_match:
                metrics['n_samples'] = int(feature_match.group(1))
                metrics['n_features'] = int(feature_match.group(2))

        except Exception as e:
            logger.error(f"❌ Error extracting model metrics: {e}")

        return metrics

    def extract_system_health(self, log_file: Path) -> Dict[str, float]:
        """Extract system health metrics from log file."""
        health = {}
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            import re

            # Count errors
            error_count = len(re.findall(r'ERROR|CRITICAL', content))
            health['error_count'] = error_count

            # Count warnings
            warning_count = len(re.findall(r'WARNING', content))
            health['warning_count'] = warning_count

            # Numba success rate
            numba_successes = len(re.findall(r'Numba.*succeeded|Using Numba JIT', content))
            numba_failures = len(re.findall(r'Numba.*failed|Numba.*error', content))
            total_numba = numba_successes + numba_failures

            if total_numba > 0:
                health['numba_success_rate'] = numba_successes / total_numba
            else:
                health['numba_success_rate'] = 1.0  # Assume success if no Numba activity

            # Feature 324 importance
            feature_324_found = len(re.findall(r'324', content)) > 0
            health['feature_324_detected'] = 1.0 if feature_324_found else 0.0

        except Exception as e:
            logger.error(f"❌ Error extracting system health: {e}")

        return health

    def run_pipeline_benchmark(self, run_id: int) -> Optional[PerformanceMetrics]:
        """Run a single pipeline benchmark."""
        logger.info(f"🚀 Running pipeline benchmark {run_id}/{self.config.n_runs}")

        start_time = datetime.now()
        success = False
        error_message = None

        # Create log file for this run
        log_file = self.output_dir / f"benchmark_run_{run_id:02d}.log"

        try:
            with self.resource_monitoring() as monitor:
                # Run the pipeline command
                cmd = ["pff", "learn", "kg"]

                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=Path.cwd()
                    )

                # Wait for completion
                return_code = process.wait()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if return_code == 0:
                success = True
                logger.info(f"✅ Run {run_id} completed successfully in {duration:.2f}s")
            else:
                error_message = f"Pipeline failed with return code {return_code}"
                logger.error(f"❌ Run {run_id} failed: {error_message}")

            # Extract metrics
            resource_summary = monitor.get_summary()
            model_metrics = self.extract_model_metrics(log_file)
            system_health = self.extract_system_health(log_file)

            return PerformanceMetrics(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                peak_memory_mb=resource_summary.get('peak_memory_mb', 0),
                avg_memory_mb=resource_summary.get('avg_memory_mb', 0),
                peak_cpu_percent=resource_summary.get('peak_cpu_percent', 0),
                avg_cpu_percent=resource_summary.get('avg_cpu_percent', 0),
                disk_io_read_mb=resource_summary.get('disk_io_read_mb', 0),
                disk_io_write_mb=resource_summary.get('disk_io_write_mb', 0),
                network_io_recv_mb=resource_summary.get('network_io_recv_mb', 0),
                network_io_sent_mb=resource_summary.get('network_io_sent_mb', 0),
                model_metrics=model_metrics,
                system_health=system_health,
                success=success,
                error_message=error_message
            )

        except Exception as e:
            end_time = datetime.now()
            error_message = str(e)
            logger.error(f"❌ Run {run_id} failed with exception: {error_message}")

            # Try to get partial resource data
            try:
                resource_summary = monitor.get_summary()
            except:
                resource_summary = {}

            return PerformanceMetrics(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                peak_memory_mb=resource_summary.get('peak_memory_mb', 0),
                avg_memory_mb=resource_summary.get('avg_memory_mb', 0),
                peak_cpu_percent=resource_summary.get('peak_cpu_percent', 0),
                avg_cpu_percent=resource_summary.get('avg_cpu_percent', 0),
                disk_io_read_mb=resource_summary.get('disk_io_read_mb', 0),
                disk_io_write_mb=resource_summary.get('disk_io_write_mb', 0),
                network_io_recv_mb=resource_summary.get('network_io_recv_mb', 0),
                network_io_sent_mb=resource_summary.get('network_io_sent_mb', 0),
                model_metrics={},
                system_health={},
                success=False,
                error_message=error_message
            )

    def run_full_benchmark(self) -> BenchmarkResult:
        """Run the complete performance benchmark."""
        logger.info("🏁 Starting comprehensive performance benchmark...")

        # Get version information
        try:
            git_result = subprocess.run(['git', 'describe', '--tags'],
                                      capture_output=True, text=True)
            current_version = git_result.stdout.strip() or "unknown"
        except:
            current_version = "unknown"

        timestamp = datetime.now()
        all_runs = []

        # Warmup runs
        logger.info(f"🔥 Running {self.config.warmup_runs} warmup runs...")
        for i in range(self.config.warmup_runs):
            warmup_run = self.run_pipeline_benchmark(f"warmup_{i+1}")
            if warmup_run:
                logger.info(f"Warmup {i+1} completed in {warmup_run.duration:.2f}s")

        # Main benchmark runs
        logger.info(f"📊 Running {self.config.n_runs} benchmark runs...")
        for i in range(self.config.n_runs):
            run = self.run_pipeline_benchmark(i + 1)
            if run:
                all_runs.append(run)
            else:
                logger.error(f"❌ Run {i+1} failed completely")

        # Calculate summary statistics
        successful_runs = [r for r in all_runs if r.success]

        if not successful_runs:
            raise ValueError("No successful benchmark runs completed")

        summary = self.calculate_summary_statistics(successful_runs)

        # Load baseline for comparison
        baseline_result = self.load_baseline()
        comparison = None
        if baseline_result:
            comparison = self.compare_with_baseline(summary, baseline_result)

        result = BenchmarkResult(
            config=self.config,
            timestamp=timestamp,
            baseline_version=baseline_result.get('version', 'unknown') if baseline_result else 'none',
            current_version=current_version,
            runs=all_runs,
            summary=summary,
            comparison=comparison
        )

        # Save results
        self.save_results(result)

        # Create visualizations
        if self.config.create_visualizations:
            self.create_visualizations(result)

        logger.info("✅ Performance benchmark completed successfully!")
        return result

    def calculate_summary_statistics(self, runs: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics from benchmark runs."""
        if not runs:
            return {}

        # Performance metrics
        durations = [r.duration for r in runs]
        peak_memories = [r.peak_memory_mb for r in runs]
        avg_memories = [r.avg_memory_mb for r in runs]
        peak_cpus = [r.peak_cpu_percent for r in runs]

        # Model quality metrics
        model_metrics = {}
        for metric_name in ['f1_score', 'precision', 'recall', 'auc', 'symbolic_contribution',
                          'hybrid_contribution', 'avg_violation_percentage']:
            values = [r.model_metrics.get(metric_name) for r in runs if metric_name in r.model_metrics]
            if values and all(v is not None for v in values):
                model_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        # System health metrics
        health_metrics = {}
        for metric_name in ['error_count', 'warning_count', 'numba_success_rate', 'feature_324_detected']:
            values = [r.system_health.get(metric_name, 0) for r in runs]
            if values:
                health_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        return {
            'performance': {
                'duration_mean': np.mean(durations),
                'duration_std': np.std(durations),
                'peak_memory_mean': np.mean(peak_memories),
                'peak_memory_std': np.std(peak_memories),
                'avg_memory_mean': np.mean(avg_memories),
                'peak_cpu_mean': np.mean(peak_cpus),
                'successful_runs': len(runs),
                'total_runs': self.config.n_runs
            },
            'model_metrics': model_metrics,
            'system_health': health_metrics
        }

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline results for comparison."""
        baseline_path = Path(self.config.baseline_file)
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    baseline = json.load(f)
                logger.info(f"✅ Loaded baseline from {baseline_path}")
                return baseline
            except Exception as e:
                logger.error(f"❌ Error loading baseline: {e}")
        return None

    def compare_with_baseline(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline."""
        comparison = {
            'performance_changes': {},
            'model_metric_changes': {},
            'health_changes': {},
            'overall_improvement': 0.0
        }

        # Performance comparison
        perf_current = current['performance']
        perf_baseline = baseline.get('summary', {}).get('performance', {})

        for metric in ['duration_mean', 'peak_memory_mean', 'peak_cpu_mean']:
            if metric in perf_current and metric in perf_baseline:
                current_val = perf_current[metric]
                baseline_val = perf_baseline[metric]

                if metric == 'duration_mean':
                    # Lower is better
                    change_pct = ((baseline_val - current_val) / baseline_val) * 100
                else:
                    # For CPU/memory, lower is also better
                    change_pct = ((baseline_val - current_val) / baseline_val) * 100

                comparison['performance_changes'][metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'change_percent': change_pct,
                    'improved': change_pct > 0
                }

        # Model metrics comparison
        model_current = current['model_metrics']
        model_baseline = baseline.get('summary', {}).get('model_metrics', {})

        for metric in model_current:
            if metric in model_baseline:
                current_mean = model_current[metric]['mean']
                baseline_mean = model_baseline[metric]['mean']

                # Most metrics should improve (higher is better), except violation percentage
                if 'violation' in metric:
                    change_pct = ((baseline_mean - current_mean) / baseline_mean) * 100
                    improved = change_pct > 0
                else:
                    change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
                    improved = change_pct > 0

                comparison['model_metric_changes'][metric] = {
                    'baseline': baseline_mean,
                    'current': current_mean,
                    'change_percent': change_pct,
                    'improved': improved
                }

        # Calculate overall improvement score
        all_changes = []
        for category in ['performance_changes', 'model_metric_changes']:
            for metric, data in comparison[category].items():
                if data['improved']:
                    all_changes.append(abs(data['change_percent']))
                else:
                    all_changes.append(-abs(data['change_percent']))

        if all_changes:
            comparison['overall_improvement'] = np.mean(all_changes)

        return comparison

    def save_results(self, result: BenchmarkResult):
        """Save benchmark results to file."""
        # Save as JSON
        results_file = self.output_dir / f"benchmark_results_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to serializable format
        result_dict = asdict(result)
        result_dict['timestamp'] = result.timestamp.isoformat()

        # Convert run times to ISO format
        for run in result_dict['runs']:
            run['start_time'] = run['start_time']
            run['end_time'] = run['end_time']

        with open(results_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        # Also save as latest baseline
        baseline_file = Path(self.config.baseline_file)
        baseline_file.parent.mkdir(parents=True, exist_ok=True)

        baseline_data = {
            'version': result.current_version,
            'timestamp': result.timestamp.isoformat(),
            'summary': result.summary,
            'config': asdict(result.config)
        }

        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)

        logger.info(f"💾 Results saved to {results_file}")
        logger.info(f"📊 Baseline updated to {baseline_file}")

    def create_visualizations(self, result: BenchmarkResult):
        """Create performance visualization charts."""
        if not self.config.create_visualizations:
            return

        successful_runs = [r for r in result.runs if r.success]
        if not successful_runs:
            logger.warning("⚠️ No successful runs to visualize")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Performance Benchmark - {result.current_version}', fontsize=16, fontweight='bold')

        # 1. Runtime comparison
        run_times = [r.duration for r in successful_runs]
        axes[0, 0].bar(range(1, len(run_times) + 1), run_times, color='skyblue', alpha=0.7)
        axes[0, 0].axhline(y=np.mean(run_times), color='red', linestyle='--', label=f'Mean: {np.mean(run_times):.2f}s')
        axes[0, 0].set_xlabel('Run Number')
        axes[0, 0].set_ylabel('Duration (seconds)')
        axes[0, 0].set_title('Runtime Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Memory usage
        memory_usage = [r.peak_memory_mb for r in successful_runs]
        axes[0, 1].bar(range(1, len(memory_usage) + 1), memory_usage, color='lightgreen', alpha=0.7)
        axes[0, 1].axhline(y=np.mean(memory_usage), color='red', linestyle='--', label=f'Mean: {np.mean(memory_usage):.1f}MB')
        axes[0, 1].set_xlabel('Run Number')
        axes[0, 1].set_ylabel('Peak Memory (MB)')
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Model quality metrics
        model_metrics = result.summary['model_metrics']
        if model_metrics:
            metric_names = list(model_metrics.keys())
            metric_means = [model_metrics[m]['mean'] for m in metric_names]
            metric_stds = [model_metrics[m]['std'] for m in metric_names]

            x_pos = np.arange(len(metric_names))
            axes[0, 2].bar(x_pos, metric_means, yerr=metric_stds, capsize=5, color='lightcoral', alpha=0.7)
            axes[0, 2].set_xlabel('Metric')
            axes[0, 2].set_ylabel('Value')
            axes[0, 2].set_title('Model Quality Metrics')
            axes[0, 2].set_xticks(x_pos)
            axes[0, 2].set_xticklabels([m.replace('_', ' ') for m in metric_names], rotation=45, ha='right')
            axes[0, 2].grid(True, alpha=0.3)

        # 4. CPU usage
        cpu_usage = [r.peak_cpu_percent for r in successful_runs]
        axes[1, 0].bar(range(1, len(cpu_usage) + 1), cpu_usage, color='orange', alpha=0.7)
        axes[1, 0].axhline(y=np.mean(cpu_usage), color='red', linestyle='--', label=f'Mean: {np.mean(cpu_usage):.1f}%')
        axes[1, 0].set_xlabel('Run Number')
        axes[1, 0].set_ylabel('Peak CPU Usage (%)')
        axes[1, 0].set_title('CPU Utilization')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Comparison with baseline (if available)
        if result.comparison:
            perf_changes = result.comparison['performance_changes']
            if perf_changes:
                metrics = list(perf_changes.keys())
                changes = [perf_changes[m]['change_percent'] for m in metrics]
                colors = ['green' if perf_changes[m]['improved'] else 'red' for m in metrics]

                axes[1, 1].bar(range(len(metrics)), changes, color=colors, alpha=0.7)
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 1].set_xlabel('Metric')
                axes[1, 1].set_ylabel('Change (%)')
                axes[1, 1].set_title('Performance Change vs Baseline')
                axes[1, 1].set_xticks(range(len(metrics)))
                axes[1, 1].set_xticklabels([m.replace('_', ' ') for m in metrics], rotation=45, ha='right')
                axes[1, 1].grid(True, alpha=0.3)

        # 6. Summary statistics table
        axes[1, 2].axis('off')
        summary_text = "Summary Statistics:\\n\\n"
        summary_text += f"Successful Runs: {len(successful_runs)}/{result.config.n_runs}\\n"
        summary_text += f"Avg Duration: {result.summary['performance']['duration_mean']:.2f}s\\n"
        summary_text += f"Avg Memory: {result.summary['performance']['peak_memory_mean']:.1f}MB\\n"

        if 'f1_score' in result.summary['model_metrics']:
            f1_mean = result.summary['model_metrics']['f1_score']['mean']
            summary_text += f"F1-Score: {f1_mean:.4f}\\n"

        if 'avg_violation_percentage' in result.summary['model_metrics']:
            violation_mean = result.summary['model_metrics']['avg_violation_percentage']['mean']
            summary_text += f"Violation %: {violation_mean:.1f}%\\n"

        if result.comparison:
            overall_improvement = result.comparison['overall_improvement']
            summary_text += f"\\nOverall Improvement: {overall_improvement:+.1f}%"

        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        chart_path = self.output_dir / f"benchmark_chart_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Visualization saved to {chart_path}")

    def print_summary(self, result: BenchmarkResult):
        """Print comprehensive benchmark summary."""
        print("\\n" + "="*80)
        print("🏁 PERFORMANCE BENCHMARK RESULTS")
        print("="*80)

        print(f"Version: {result.current_version}")
        print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runs: {len(result.runs)} total, {len([r for r in result.runs if r.success])} successful")

        if result.comparison:
            print(f"Baseline: {result.baseline_version}")
            improvement = result.comparison['overall_improvement']
            improvement_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"Overall Improvement: {improvement_emoji} {improvement:+.1f}%")

        print(f"\\n⚡ PERFORMANCE METRICS:")
        perf = result.summary['performance']
        print(f"   - Duration: {perf['duration_mean']:.2f}s ± {perf['duration_std']:.2f}s")
        print(f"   - Peak Memory: {perf['peak_memory_mean']:.1f}MB ± {perf['peak_memory_std']:.1f}MB")
        print(f"   - Peak CPU: {perf['peak_cpu_mean']:.1f}%")

        print(f"\\n🎯 MODEL QUALITY:")
        model_metrics = result.summary['model_metrics']
        for metric, stats in model_metrics.items():
            print(f"   - {metric.replace('_', ' ').title()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        print(f"\\n🏥 SYSTEM HEALTH:")
        health = result.summary['system_health']
        for metric, stats in health.items():
            print(f"   - {metric.replace('_', ' ').title()}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        if result.comparison:
            print(f"\\n📊 COMPARISON VS BASELINE:")
            for category, changes in result.comparison.items():
                if category == 'overall_improvement':
                    continue
                print(f"   {category.replace('_', ' ').title()}:")
                for metric, data in changes.items():
                    if isinstance(data, dict) and 'change_percent' in data:
                        emoji = "📈" if data['improved'] else "📉"
                        print(f"     - {metric}: {emoji} {data['change_percent']:+.1f}%")

        print("\\n" + "="*80)

def main():
    """Main execution function."""
    print("🏁 Performance Benchmark System")
    print("Comprehensive benchmarking to compare before/after corrections")
    print("=" * 70)

    # Configuration
    config = BenchmarkConfig(
        n_runs=3,                    # Number of benchmark runs
        warmup_runs=1,               # Warmup runs before actual benchmark
        monitoring_interval=0.5,     # Resource monitoring interval
        output_dir="outputs/benchmarks",
        create_visualizations=True,
        detailed_profiling=True,
        memory_profiling=True
    )

    # Create benchmark system
    benchmark = PerformanceBenchmark(config)

    try:
        # Run benchmark
        result = benchmark.run_full_benchmark()

        # Print summary
        benchmark.print_summary(result)

        print(f"\\n📁 Results saved to: {benchmark.output_dir}")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\\n⚠️ Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()