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

import asyncio

from pff import settings
from pff.config import TRAINING_METRICS_CONFIG_PATH
from pff.utils import FileManager, logger
from pff.db.repositories.training_metrics import TrainingMetricsRepository


class MetricsCollector:
    """Collects and exports metrics for ML pipeline monitoring."""

    def __init__(
        self,
        experiment_name: str = "pff_experiment",
        model_name: str | None = None,
        training_metrics_repo: TrainingMetricsRepository | None = None,
        enable_db_metrics: bool = False,
        default_split: str = "train",
        log_interval: int = 1,
    ) -> None:
        self.experiment_name = experiment_name
        self.metrics: dict[str, Any] = {}
        self.start_time = time.time()
        self.model_name = model_name
        self.training_metrics_repo = training_metrics_repo
        self.enable_db_metrics = enable_db_metrics
        self.default_split = default_split
        self.log_interval = max(1, int(log_interval))

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

        if step is None or (step % self.log_interval == 0):
            logger.info(
                f"Metrica: {name} = {value:.6f}",
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
            f"Epoca {epoch}: perda={loss:.4f}",
            extra={
                "epoch": epoch,
                "train_loss": loss,
                "val_metrics": val_metrics or {},
            }
        )

        if self.enable_db_metrics and self.training_metrics_repo is not None and self.model_name:
            self._persist_training_metrics(epoch, loss, val_metrics or {})

    def _persist_training_metrics(
        self,
        epoch: int,
        loss: float,
        val_metrics: dict[str, float],
    ) -> None:
        """Persist training metrics to PostgreSQL when enabled."""
        async def _persist() -> None:
            await self.training_metrics_repo.log_metric(
                model_name=self.model_name,
                metric_name="loss",
                metric_value=loss,
                epoch=epoch,
                split=self.default_split,
            )
            if val_metrics:
                await self.training_metrics_repo.log_epoch_metrics(
                    model_name=self.model_name,
                    epoch=epoch,
                    metrics=val_metrics,
                    split="valid",
                )

        self._run_async(_persist())

    @staticmethod
    def _run_async(coro: asyncio.Future) -> None:
        """Execute coroutine in running loop or start a new one."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)

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

        self.logger.info("Dashboard Ray habilitado em http://localhost:8265")
        self.logger.info("Use 'ray status' para verificar o estado do cluster")

    def enable_debugpy(self, port: int = 5678) -> None:
        """Enable debugpy for remote debugging."""
        if not self.enable_debugging:
            return

        os.environ["RAY_DEBUGPY_ENABLE"] = "1"
        os.environ["RAY_DEBUGPY_PORT"] = str(port)

        self.logger.info(f"Debugpy habilitado na porta {port}")
        self.logger.info("Conecte o VS Code para depurar workers Ray")

    def monitor_fault_tolerance(self) -> None:
        """Configure fault tolerance monitoring."""
        os.environ["RAY_FAULT_TOLERANCE_ENABLED"] = "1"
        os.environ["RAY_CHECKPOINT_FREQUENCY"] = "5"

        self.logger.info("Monitoramento de tolerancia a falhas habilitado")


class ObservabilityManager:
    """Centralized observability management for production ML systems."""

    def __init__(
        self,
        experiment_name: str = "pff_experiment",
        enable_debugging: bool = False,
        correlation_id: str | None = None,
        model_name: str | None = None,
        enable_db_metrics: bool | None = None,
        training_metrics_repo: TrainingMetricsRepository | None = None,
        training_metrics_config_path: Path | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.enable_debugging = enable_debugging
        self.logger = logger
        self.correlation_id = correlation_id or str(uuid.uuid4())

        self.metrics_collector = MetricsCollector(
            experiment_name,
            model_name=model_name,
            training_metrics_repo=self._resolve_training_metrics_repo(
                training_metrics_repo,
                enable_db_metrics,
                training_metrics_config_path,
                model_name,
            ),
            enable_db_metrics=self._resolve_enable_db_metrics(
                enable_db_metrics, training_metrics_config_path
            ),
            default_split=self._resolve_default_split(training_metrics_config_path),
            log_interval=self._resolve_log_interval(training_metrics_config_path),
        )
        self.debugger = DistributedDebugger(enable_debugging)

        self._setup_structured_logging()
        self._setup_metrics_export()
        self._setup_debugging()

    def _resolve_training_metrics_repo(
        self,
        repo: TrainingMetricsRepository | None,
        enable_db_metrics: bool | None,
        config_path: Path | None,
        model_name: str | None,
    ) -> TrainingMetricsRepository | None:
        """Resolve repository instance based on configuration."""
        should_enable = self._resolve_enable_db_metrics(enable_db_metrics, config_path)
        if not should_enable or repo is not None:
            return repo
        try:
            return TrainingMetricsRepository()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(f"Failed to initialize TrainingMetricsRepository: {exc}")
            return None

    def _resolve_enable_db_metrics(
        self,
        enable_db_metrics: bool | None,
        config_path: Path | None,
    ) -> bool:
        """Decide whether DB metrics logging is enabled."""
        if enable_db_metrics is not None:
            return enable_db_metrics
        cfg = self._load_training_metrics_config(config_path)
        return bool(cfg.get("log_to_postgres", False))

    def _resolve_default_split(self, config_path: Path | None) -> str:
        """Return configured default split for training metrics."""
        cfg = self._load_training_metrics_config(config_path)
        return str(cfg.get("default_split", "train"))

    def _load_training_metrics_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load training metrics configuration from YAML."""
        resolved_path = config_path or TRAINING_METRICS_CONFIG_PATH
        if not resolved_path.exists():
            return {}
        try:
            content = FileManager().read(resolved_path)
            return content or {}
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(f"Failed to read training metrics config: {exc}")
            return {}

    def _resolve_log_interval(self, config_path: Path | None) -> int:
        """Resolve log interval for metrics to reduce I/O overhead."""
        cfg = self._load_training_metrics_config(config_path)
        return int(cfg.get("log_interval", 1))

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

            self.logger.info(f"Metricas exportadas para {output_path}")

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
    correlation_id: str | None = None,
    model_name: str | None = None,
    enable_db_metrics: bool | None = None,
    training_metrics_repo: TrainingMetricsRepository | None = None,
    training_metrics_config_path: Path | None = None,
) -> ObservabilityManager:
    """Get global observability manager instance (singleton)."""
    global _observability_manager

    if _observability_manager is None:
        _observability_manager = ObservabilityManager(
            experiment_name=experiment_name,
            enable_debugging=enable_debugging,
            correlation_id=correlation_id,
            model_name=model_name,
            enable_db_metrics=enable_db_metrics,
            training_metrics_repo=training_metrics_repo,
            training_metrics_config_path=training_metrics_config_path,
        )

    return _observability_manager


def setup_observability(
    experiment_name: str = "pff_experiment",
    enable_debugging: bool = False,
    correlation_id: str | None = None,
    model_name: str | None = None,
    enable_db_metrics: bool | None = None,
    training_metrics_repo: TrainingMetricsRepository | None = None,
    training_metrics_config_path: Path | None = None,
) -> ObservabilityManager:
    """Setup production observability."""
    return get_observability_manager(
        experiment_name=experiment_name,
        enable_debugging=enable_debugging,
        correlation_id=correlation_id,
        model_name=model_name,
        enable_db_metrics=enable_db_metrics,
        training_metrics_repo=training_metrics_repo,
        training_metrics_config_path=training_metrics_config_path,
    )
