"""High-level DSLFM validator interface (Facade).

Keeps orchestration thin: build manager from configs, run train/eval, and
return metrics. All filesystem/logging goes through the underlying manager
which is already aligned with utils (FileManager + logger).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pff.shared.core.logger import logger
from .core import DSLFMKGCConfig
from .manager import DSLFMKGCManager, KGCTrainingConfig


class DSLFMValidator:
    """Facade for running DSLFM-KGC training/evaluation."""

    def __init__(
        self,
        model_config: DSLFMKGCConfig,
        training_config: KGCTrainingConfig,
        relation_names: list[str] | None = None,
    ) -> None:
        self.manager = DSLFMKGCManager(
            model_config=model_config,
            training_config=training_config,
            relation_names=relation_names,
        )

    def train_and_validate(
        self,
        train_triples: np.ndarray,
        valid_triples: np.ndarray,
    ) -> dict[str, Any]:
        """Execute training + final validation."""
        logger.info("Iniciando validação DSLFM via facade")
        return self.manager.train(train_triples, valid_triples)

    def save_best(self, output_dir: Path | str) -> None:
        """Persist the best checkpoint (delegated to manager)."""
        logger.info(f"Nenhuma acao adicional de persistencia requerida em {output_dir}")
