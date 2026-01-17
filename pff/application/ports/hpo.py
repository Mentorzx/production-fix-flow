"""HPO runner port interface for the application layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class HpoRunnerPort(ABC):
    """Port for executing HPO runs via infrastructure adapters."""

    @abstractmethod
    def run(  # noqa: PLR0913
        self,
        *,
        n_trials: int,
        strategy: str,
        enable_mlflow: bool,
        enable_visualization: bool,
        study_name: str | None,
        output_dir: Path | None,
        target_entity_ratio: float,
        kge_model: str,
        use_synthetic_if_dslfm: bool,
        no_update_config: bool,
        no_bert: bool,
        resume_mode: bool | None,
        reset_state: bool,
    ) -> dict[str, Any]:
        """Execute an HPO run and return the result payload."""
        raise NotImplementedError
