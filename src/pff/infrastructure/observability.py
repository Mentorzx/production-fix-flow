"""
Production-grade observability for ML pipelines.

Provides structured logging, metrics, tracing, and monitoring capabilities
for distributed ML systems using Ray 3.0+.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl

from pff.infrastructure.persistence.db.repositories.training_metrics import (
    TrainingMetricsRepository,
)
from pff.shared import FileManager, logger
from pff.shared.core.config import TRAINING_METRICS_CONFIG_PATH


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
                },
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
            },
        )

        if (
            self.enable_db_metrics
            and self.training_metrics_repo is not None
            and self.model_name
        ):
            self._persist_training_metrics(epoch, loss, val_metrics or {})

    def _persist_training_metrics(
        self,
        epoch: int,
        loss: float,
        val_metrics: dict[str, float],
    ) -> None:
        """Persist training metrics to PostgreSQL when enabled."""
        repo = self.training_metrics_repo
        model_name = self.model_name

        if repo is None or model_name is None:
            return

        async def _persist() -> None:
            try:
                await repo.log_metric(
                    model_name=model_name,
                    metric_name="loss",
                    metric_value=loss,
                    epoch=epoch,
                    split=self.default_split,
                )
                if val_metrics:
                    await repo.log_epoch_metrics(
                        model_name=model_name,
                        epoch=epoch,
                        metrics=val_metrics,
                        split="valid",
                    )
            except Exception as exc:
                error_name = type(exc).__name__
                if (
                    "TooManyConnections" in error_name
                    or "connection" in str(exc).lower()
                ):
                    logger.warning(
                        f"DB metrics persistence skipped (connection exhausted): {error_name}"
                    )
                else:
                    logger.debug(f"DB metrics persistence failed: {exc}")

        self._run_async(_persist())

    @staticmethod
    def _run_async(coro: Any) -> None:
        """Execute coroutine in running loop or start a new one.

        Note: When called from sync context without a running loop, asyncio.run()
        creates a temporary event loop. This can cause connection pool mismatches
        in heavy async/sync mixed workloads. The calling code should handle
        connection errors gracefully.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                from pff.shared.acceleration.asyncio_runner import run_coroutine_sync

                run_coroutine_sync(coro)
            except Exception as exc:
                logger.debug(f"Async execution in sync context failed: {exc}")
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

    def __init__(
        self,
        enable_debugging: bool = False,
        *,
        dashboard_url: str = "http://localhost:8265",
        debugpy_port: int = 5678,
        checkpoint_frequency: int = 5,
    ) -> None:
        self.enable_debugging = enable_debugging
        self.logger = logger
        self.dashboard_url = dashboard_url
        self.debugpy_port = debugpy_port
        self.checkpoint_frequency = checkpoint_frequency

    def enable_ray_dashboard(self) -> None:
        """Enable Ray dashboard for monitoring."""
        os.environ["RAY_DASHBOARD_ENABLE"] = "1"
        os.environ["RAY_METRICS_ENABLE"] = "1"
        os.environ["RAY_METRICS_EXPORT_INTERVAL_MS"] = os.environ.get(
            "RAY_METRICS_EXPORT_INTERVAL_MS", "5000"
        )
        os.environ["RAY_DASHBOARD_URL"] = self.dashboard_url

        self.logger.info(f"Dashboard Ray habilitado em {self.dashboard_url}")
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
        os.environ["RAY_CHECKPOINT_FREQUENCY"] = str(self.checkpoint_frequency)

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
        self._ray_settings = self._resolve_ray_settings()

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
        self.debugger = DistributedDebugger(
            enable_debugging,
            dashboard_url=self._ray_settings["dashboard_url"],
            debugpy_port=self._ray_settings["debugpy_port"],
            checkpoint_frequency=self._ray_settings["checkpoint_frequency"],
        )

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
        except Exception as exc:
            self.logger.warning(
                f"Failed to initialize TrainingMetricsRepository: {exc}"
            )
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
            content = FileManager().read(resolved_path, return_native=True)
            return content or {}
        except Exception as exc:
            self.logger.warning(f"Failed to read training metrics config: {exc}")
            return {}

    def _resolve_log_interval(self, config_path: Path | None) -> int:
        """Resolve log interval for metrics to reduce I/O overhead."""
        cfg = self._load_training_metrics_config(config_path)
        return int(cfg.get("log_interval", 1))

    def _resolve_ray_settings(self) -> dict[str, Any]:
        """Resolve Ray observability settings from environment with defaults."""
        return {
            "metrics_enable": _parse_bool(os.getenv("RAY_METRICS_ENABLE"), True),
            "metrics_export_interval_ms": _parse_int(
                os.getenv("RAY_METRICS_EXPORT_INTERVAL_MS"), 5000
            ),
            "prometheus_enable": _parse_bool(os.getenv("RAY_PROMETHEUS_ENABLE"), True),
            "dashboard_url": os.getenv("RAY_DASHBOARD_URL") or "http://localhost:8265",
            "debugpy_port": _parse_int(os.getenv("RAY_DEBUGPY_PORT"), 5678),
            "checkpoint_frequency": _parse_int(
                os.getenv("RAY_CHECKPOINT_FREQUENCY"), 5
            ),
        }

    def _setup_structured_logging(self) -> None:
        """Setup structured logging with correlation IDs."""
        self.logger.debug("Setting up structured logging")

        self.logger.debug(
            f"Correlation ID: {self.correlation_id}",
            extra={"correlation_id": self.correlation_id},
        )

        self.logger.debug("Structured logging configured")

    def _setup_metrics_export(self) -> None:
        """Setup metrics export for monitoring."""
        self.logger.debug("Setting up metrics export")

        metrics_enable = "1" if self._ray_settings["metrics_enable"] else "0"
        os.environ["RAY_METRICS_ENABLE"] = metrics_enable
        os.environ["RAY_METRICS_EXPORT_INTERVAL_MS"] = str(
            self._ray_settings["metrics_export_interval_ms"]
        )

        prometheus_enable = "1" if self._ray_settings["prometheus_enable"] else "0"
        os.environ["RAY_PROMETHEUS_ENABLE"] = prometheus_enable

        self.logger.debug("Metrics export configured")

    def _setup_debugging(self) -> None:
        """Setup distributed debugging."""
        if self.enable_debugging:
            self.logger.info("Configurando depuração distribuída")
            self.debugger.enable_ray_dashboard()
            self.debugger.enable_debugpy(port=self._ray_settings["debugpy_port"])
            self.debugger.monitor_fault_tolerance()
            self.logger.success("Depuração distribuída configurada")
        elif not getattr(self, "_debug_notice_logged", False):
            self.logger.info(
                "Depuração desativada (defina enable_debugging=True para habilitar)"
            )
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
            },
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
                },
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
                },
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
            from pff.shared import FileManager

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix.lower() == ".parquet":
                df = (
                    pl.DataFrame([summary])
                    if isinstance(summary, dict)
                    else pl.DataFrame(summary)
                )
                FileManager().save(df, output_path)
            else:
                FileManager().save(summary, output_path)

            self.logger.info(f"Metricas exportadas para {output_path}")

        return summary

    def get_observability_status(self) -> dict[str, Any]:
        """Get current observability status."""
        return {
            "correlation_id": self.correlation_id,
            "experiment_name": self.experiment_name,
            "debugging_enabled": self.enable_debugging,
            "ray_dashboard_url": (
                self._ray_settings["dashboard_url"]
                if os.getenv("RAY_DASHBOARD_ENABLE")
                else None
            ),
            "metrics_export_enabled": os.getenv("RAY_METRICS_ENABLE") == "1",
        }


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
