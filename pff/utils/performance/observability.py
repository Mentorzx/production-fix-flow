"""
Production-grade observability for ML pipelines.

Provides structured logging, metrics, tracing, and monitoring capabilities
for distributed ML systems using Ray 3.0+.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any
from contextlib import contextmanager
from pathlib import Path

from pff.utils import logger


class MetricsCollector:
    """Collects and exports metrics for ML pipeline monitoring."""

    def __init__(self, experiment_name: str = "pff_experiment") -> None:
        self.experiment_name = experiment_name
        self.metrics: dict[str, Any] = {}
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Record a metric value."""
        timestamp = time.time()
        metric_data = {
            "value": value,
            "timestamp": timestamp,
            "step": step,
        }

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(metric_data)

        logger.info(
            f"Metric: {name} = {value:.6f}",
            extra={
                "metric_name": name,
                "metric_value": value,
                "step": step,
            }
        )

    def record_training_metrics(
        self,
        epoch: int,
        loss: float,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        """Record training epoch metrics."""
        self.record_metric("train_loss", loss, step=epoch)

        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                self.record_metric(f"val_{metric_name}", metric_value, step=epoch)

        logger.info(
            f"Epoch {epoch}: Loss={loss:.4f}",
            extra={
                "epoch": epoch,
                "train_loss": loss,
                "val_metrics": val_metrics or {},
            }
        )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all collected metrics."""
        summary = {
            "experiment_name": self.experiment_name,
            "total_duration": time.time() - self.start_time,
            "metrics": self.metrics,
        }
        return summary


class DistributedDebugger:
    """Distributed debugging utilities for Ray-based training."""

    def __init__(self, enable_debugging: bool = False) -> None:
        self.enable_debugging = enable_debugging
        self.logger = logger

    def enable_ray_dashboard(self) -> None:
        """Enable Ray dashboard for monitoring."""
        os.environ["RAY_DASHBOARD_ENABLE"] = "1"
        os.environ["RAY_METRICS_ENABLE"] = "1"
        os.environ["RAY_METRICS_EXPORT_INTERVAL_MS"] = "5000"

        self.logger.info("Ray dashboard enabled at http://localhost:8265")
        self.logger.info("Use 'ray status' to check cluster state")

    def enable_debugpy(self, port: int = 5678) -> None:
        """Enable debugpy for remote debugging."""
        if not self.enable_debugging:
            return

        os.environ["RAY_DEBUGPY_ENABLE"] = "1"
        os.environ["RAY_DEBUGPY_PORT"] = str(port)

        self.logger.info(f"Debugpy enabled on port {port}")
        self.logger.info("Attach VS Code to debug Ray workers")

    def monitor_fault_tolerance(self) -> None:
        """Configure fault tolerance monitoring."""
        os.environ["RAY_FAULT_TOLERANCE_ENABLED"] = "1"
        os.environ["RAY_CHECKPOINT_FREQUENCY"] = "5"

        self.logger.info("Fault tolerance monitoring enabled")


class ObservabilityManager:
    """Centralized observability management for production ML systems."""

    def __init__(
        self,
        experiment_name: str = "pff_experiment",
        enable_debugging: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.enable_debugging = enable_debugging
        self.logger = logger
        self.correlation_id = str(uuid.uuid4())
        self.metrics_collector = MetricsCollector(experiment_name)
        self.debugger = DistributedDebugger(enable_debugging)

        self._setup_structured_logging()
        self._setup_metrics_export()
        self._setup_debugging()

    def _setup_structured_logging(self) -> None:
        """Setup structured logging with correlation IDs."""
        self.logger.debug("Setting up structured logging")

        self.logger.debug(
            f"Correlation ID: {self.correlation_id}",
            extra={"correlation_id": self.correlation_id}
        )

        self.logger.debug("Structured logging configured")

    def _setup_metrics_export(self) -> None:
        """Setup metrics export for monitoring."""
        self.logger.debug("Setting up metrics export")

        os.environ["RAY_METRICS_ENABLE"] = "1"
        os.environ["RAY_METRICS_EXPORT_INTERVAL_MS"] = "5000"

        os.environ["RAY_PROMETHEUS_ENABLE"] = "1"

        self.logger.debug("Metrics export configured")

    def _setup_debugging(self) -> None:
        """Setup distributed debugging."""
        if self.enable_debugging:
            self.logger.info("Configurando depuração distribuída")
            self.debugger.enable_ray_dashboard()
            self.debugger.enable_debugpy()
            self.debugger.monitor_fault_tolerance()
            self.logger.success("Depuração distribuída configurada")
        else:
            if not getattr(self, "_debug_notice_logged", False):
                self.logger.info("Depuração desativada (defina enable_debugging=True para habilitar)")
                self._debug_notice_logged = True

    @contextmanager
    def track_execution(self, operation_name: str, **kwargs):
        """Context manager to track operation execution time."""
        start_time = time.time()
        self.logger.info(
            f"Iniciando operação: {operation_name}",
            extra={
                "operation": operation_name,
                "metadata": kwargs,
                "correlation_id": self.correlation_id,
            }
        )

        try:
            yield
            duration = time.time() - start_time
            self.logger.info(
                f"Operação concluída: {operation_name} ({duration:.2f}s)",
                extra={
                    "operation": operation_name,
                    "duration": duration,
                    "status": "success",
                    "correlation_id": self.correlation_id,
                }
            )
            self.record_metric(f"operation_{operation_name}_duration", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Failed operation: {operation_name} ({duration:.2f}s) - {e}",
                extra={
                    "operation": operation_name,
                    "duration": duration,
                    "status": "error",
                    "error": str(e),
                    "correlation_id": self.correlation_id,
                }
            )
            raise

    def record_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Record a metric."""
        self.metrics_collector.record_metric(name, value, step)

    def record_training_metrics(
        self,
        epoch: int,
        loss: float,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        """Record training metrics."""
        self.metrics_collector.record_training_metrics(epoch, loss, val_metrics)

    def get_correlation_id(self) -> str:
        """Get current correlation ID."""
        return self.correlation_id

    def export_metrics(self, output_path: Path | None = None) -> dict[str, Any]:
        """Export all collected metrics."""
        summary = self.metrics_collector.get_metrics_summary()

        if output_path:
            from pff.utils import FileManager
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            FileManager().save(summary, output_path)

            self.logger.info(f"Metrics exported to {output_path}")

        return summary

    def get_observability_status(self) -> dict[str, Any]:
        """Get current observability status."""
        return {
            "correlation_id": self.correlation_id,
            "experiment_name": self.experiment_name,
            "debugging_enabled": self.enable_debugging,
            "ray_dashboard_url": "http://localhost:8265" if os.getenv("RAY_DASHBOARD_ENABLE") else None,
            "metrics_export_enabled": os.getenv("RAY_METRICS_ENABLE") == "1",
        }


# Global observability manager instance
_observability_manager: ObservabilityManager | None = None


def get_observability_manager(
    experiment_name: str = "pff_experiment",
    enable_debugging: bool = False,
) -> ObservabilityManager:
    """Get global observability manager instance (singleton)."""
    global _observability_manager

    if _observability_manager is None:
        _observability_manager = ObservabilityManager(
            experiment_name=experiment_name,
            enable_debugging=enable_debugging,
        )

    return _observability_manager


def setup_observability(
    experiment_name: str = "pff_experiment",
    enable_debugging: bool = False,
) -> ObservabilityManager:
    """Setup production observability."""
    return get_observability_manager(
        experiment_name=experiment_name,
        enable_debugging=enable_debugging,
    )
