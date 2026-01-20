"""Artifact manager for DSLFM-only HPO trials."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger

from .postgres_store import HpoPostgresStore


class TrialArtifactManager:
    """Persist trial results and metadata for DSLFM HPO."""

    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        study_name: str | None = None,
        store: HpoPostgresStore | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.study_name = study_name
        self.store = store
        self.file_manager = file_manager or FileManager()

    def record_result(self, trial_number: int, payload: dict[str, Any]) -> None:
        """Save trial payload to disk if a base_dir is configured."""
        if self.store is None or not self.study_name:
            raise ValueError("HPO trial artifacts require a Postgres store and study name")
        try:
            run_coroutine_sync(
                self.store.upsert_trial_result(self.study_name, trial_number, payload)
            )
            logger.debug(f"trial_artifacts_saved_backend=postgres trial={trial_number}")
        except Exception as exc:
            logger.warning(
                f"Failed to save trial artifacts to Postgres: trial={trial_number} error={exc}"
            )

    def list_metrics(self) -> list[dict[str, Any]]:
        """Load all stored metrics for completed trials."""
        if self.store is None or not self.study_name:
            raise ValueError("HPO trial metrics require a Postgres store and study name")
        try:
            return run_coroutine_sync(self.store.list_trial_metrics(self.study_name))
        except Exception as exc:
            logger.warning(f"Failed to load metrics from Postgres: {exc}")
            return []

    def load_all_results(self) -> list[dict[str, Any]]:
        """Load every stored trial payload."""
        if self.store is None or not self.study_name:
            raise ValueError("HPO trial results require a Postgres store and study name")
        try:
            return run_coroutine_sync(self.store.load_all_results(self.study_name))
        except Exception as exc:
            logger.warning(f"Failed to load trial results from Postgres: {exc}")
            return []
