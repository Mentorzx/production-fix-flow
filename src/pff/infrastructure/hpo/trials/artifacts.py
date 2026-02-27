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
        """Execute init.



        Args:

            base_dir: Optional input value.

            study_name: Optional input value.

            store: Optional input value.

            file_manager: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.base_dir = base_dir
        self.study_name = study_name
        self.store = store
        self.file_manager = file_manager or FileManager()

    def _local_trials_dir(self) -> Path | None:
        if self.base_dir is None or not self.study_name:
            return None
        return Path(self.base_dir) / self.study_name / "trials"

    def _local_trial_path(self, trial_number: int) -> Path | None:
        trials_dir = self._local_trials_dir()
        if trials_dir is None:
            return None
        return trials_dir / f"trial_{int(trial_number):06d}.json"

    def record_result(self, trial_number: int, payload: dict[str, Any]) -> None:
        """Save trial payload to disk if a base_dir is configured."""
        if self.store is not None and self.study_name:
            try:
                run_coroutine_sync(
                    self.store.upsert_trial_result(
                        self.study_name, trial_number, payload
                    )
                )
                logger.debug(
                    f"trial_artifacts_saved_backend=postgres trial={trial_number}"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to save trial artifacts to Postgres: trial={trial_number} error={exc}"
                )

        local_path = self._local_trial_path(trial_number)
        if local_path is not None:
            try:
                FileManager.ensure_dir(local_path.parent)
                self.file_manager.save(payload, local_path)
                logger.debug(
                    f"trial_artifacts_saved_backend=local trial={trial_number}"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to save local trial artifacts: trial={trial_number} error={exc}"
                )

    def list_metrics(self) -> list[dict[str, Any]]:
        """Load all stored metrics for completed trials."""
        if self.store is not None and self.study_name:
            try:
                return run_coroutine_sync(
                    self.store.list_trial_metrics(self.study_name)
                )
            except Exception as exc:
                logger.warning(f"Failed to load metrics from Postgres: {exc}")

        return [
            payload.get("metrics", {})
            for payload in self.load_all_results()
            if isinstance(payload.get("metrics"), dict)
        ]

    def load_all_results(self) -> list[dict[str, Any]]:
        """Load every stored trial payload."""
        if self.store is not None and self.study_name:
            try:
                return run_coroutine_sync(self.store.load_all_results(self.study_name))
            except Exception as exc:
                logger.warning(f"Failed to load trial results from Postgres: {exc}")

        trials_dir = self._local_trials_dir()
        if trials_dir is None or not FileManager.exists(trials_dir):
            return []

        results: list[dict[str, Any]] = []
        for path in FileManager.glob(trials_dir, "trial_*.json"):
            try:
                payload = self.file_manager.read(path, return_native=True)
                if isinstance(payload, dict):
                    results.append(payload)
            except Exception as exc:
                logger.warning(f"Failed to read local trial artifact {path}: {exc}")
        return results
