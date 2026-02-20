"""High-level DSLFM validator interface (Facade).

Keeps orchestration thin: build manager from configs, run train/eval, and
return metrics. Filesystem and logging are delegated to the underlying manager.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pff.shared.core.logging import logger

from .dslfm_kgc import DSLFMKGCConfig
from .kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class DSLFMValidator:
    """Facade for running DSLFM-KGC training/evaluation."""

    def __init__(
        self,
        model_config: DSLFMKGCConfig,
        training_config: KGCTrainingConfig,
        persistence_port: Any | None = None,
        relation_names: list[str] | None = None,
    ) -> None:
        """Execute init.



        Args:

            model_config: Input value used by this callable.

            training_config: Input value used by this callable.

            persistence_port: Optional input value.

            relation_names: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if persistence_port is None:

            class MockPersistence:
                """Represent MockPersistence."""

                def save_checkpoint(self, data, filename):
                    """Execute save checkpoint.



                    Args:

                        data: Input value used by this callable.

                        filename: Input value used by this callable.



                    Notes:

                        Keep behavior deterministic and free of hidden side effects.

                    """

                    pass

                def load_checkpoint(self, filename, map_location=None):
                    """Execute load checkpoint.



                    Args:

                        filename: Input value used by this callable.

                        map_location: Optional input value.



                    Returns:

                        Return value produced by the callable.



                    Notes:

                        Keep behavior deterministic and free of hidden side effects.

                    """

                    return None

            persistence_port = MockPersistence()

        self.manager = DSLFMKGCManager(
            model_config=model_config,
            training_config=training_config,
            persistence_port=persistence_port,
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
